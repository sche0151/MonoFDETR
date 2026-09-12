import math
import time
import datetime
from pathlib import Path

import torch
from tqdm.auto import tqdm

from utils.logging import get_file_handler
from utils.parser import parse_args, load_config
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.accelerator_helper import build_accelerator, init_accelerator, get_mp_logger as get_logger
from lib.helpers.checkpoint_helper import CustomCheckpoint, get_resume_chekpoint_path
from lib.helpers.metric_helper import nested_to_cpu
from lib.models.configuration_monodetr import MonoDETRConfig
from lib.models.monofdetr import MonoFDETRForMultiObjectDetection as MonoDETR
from lib.models.image_processsing_monodetr import MonoDETRImageProcessor


logger = get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    cfg.with_tracking = False
    
    accelerator = build_accelerator(cfg)
    # Handle the hugingface hub repo creation
    if accelerator.is_main_process:
        if cfg.output_dir is not None:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)    
    accelerator.wait_for_everyone()
    
    logger.logger.addHandler(get_file_handler(Path(cfg.output_dir) / f"test.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    
    logger.info(f'Init accelerator...\n{accelerator.state}', main_process_only=False)
    
    logger.info("Init DataLoader...")
    _, _, test_dataloader, id2label, label2id = build_dataloader(
        cfg, workers=cfg.dataloader_num_workers, accelerator=accelerator
    )
    
    logger.info("Init Model...")
    config = MonoDETRConfig(
        label2id=label2id, id2label=id2label, **vars(cfg.model),
    )
    if hasattr(cfg, 'monodetr_model') and cfg.monodetr_model is not None:
        model = MonoDETR._load_monodetr_pretrain_model(cfg.monodetr_model, config, logger=logger)
    else:
        model = MonoDETR(config)
    image_processor = MonoDETRImageProcessor()
    
    
    # Prepare everything with our `accelerator`.
    model, test_dataloader = accelerator.prepare(
        model, test_dataloader
    )
    
    # We need to recalculate our total test steps as the size of the testing dataloader may have changed.
    cfg.max_test_steps = math.ceil(len(test_dataloader) / cfg.gradient_accumulation_steps)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    init_accelerator(accelerator, cfg)
    
    # ------------------------------------------------------------------------------------------------
    # Run testing
    # ------------------------------------------------------------------------------------------------

    total_batch_size = cfg.test_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("##################  Running testing  ##################")
    logger.info(f"  Num examples = {len(test_dataloader.dataset)}")
    logger.info(f"  Instantaneous batch size per device = {cfg.test_batch_size}")
    logger.info(f"  Total test batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total testing steps = {cfg.max_test_steps}")

    # custom checkpoint data registe to accelerator
    extra_state = CustomCheckpoint()
    
    # Potentially load in the weights and states from a previous save
    if getattr(cfg, 'pretrain_model', None):
        checkpoint_path = get_resume_chekpoint_path(cfg.pretrain_model, cfg.output_dir)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.register_for_checkpointing(extra_state)
        accelerator.load_state(checkpoint_path)
        logger.info(f"Loading Checkpoint... Best Result:{extra_state.best_result}, Best Epoch:{extra_state.best_epoch}")
    
    logger.info("***** Running evaluation *****")
    
    model.eval()
    results = {}
    for step, batch in enumerate(tqdm(test_dataloader, disable=not accelerator.is_local_main_process)):
        batch = MonoDETRImageProcessor.prepare_batch(batch, return_label=False, return_info=True)
        info = batch.pop("info")
        with torch.no_grad():
            outputs = model(**batch)
        
        # For metric computation we need to collect ground truth and predicted 3D boxes in the same format

        # 1. Collect predicted 3Dboxes, classes, scores
        # image_processor convert boxes from size_3d, box and depth predict to Real 3D box format [cx, cy, cz, h, w, l] in absolute coordinates.
        predictions = image_processor.post_process_3d_object_detection(
            outputs=outputs, 
            calibs=batch["calibs"],
            threshold=getattr(cfg, 'threshold', 0.2),
            target_sizes=batch["img_sizes"],
            img_ids=info['img_id'].tolist(), 
            top_k=50,
        )
        
        predictions = nested_to_cpu(predictions)
        results.update(predictions)

    # 2. Save the results for evaluation
    logger.info("==> Saving results...")
    results_path = Path(cfg.output_dir) / "data"
    image_processor.save_results(
        results=results,
        output_path=results_path,
        id2label=id2label,
    )
    
    if accelerator.is_main_process:
        # Wait for all processes to save the results
        time.sleep(2)
        # 3. Evaluation
        metrics = test_dataloader.dataset.eval(
            results_dir=results_path,
            logger=logger,
        )
        
        logger.info(metrics)
    
    # Save the model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(cfg.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save, safe_serialization=False)
    if accelerator.is_main_process:
        image_processor.save_pretrained(cfg.output_dir)


if __name__ == '__main__':
    main()
