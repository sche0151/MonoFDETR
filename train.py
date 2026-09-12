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
from lib.helpers.schedule_helper import build_lr_scheduler
from lib.helpers.checkpoint_helper import CustomCheckpoint, get_resume_chekpoint_path, get_checkpoint_epoch, get_checkpoint_dir
from lib.helpers.metric_helper import nested_to_cpu
from lib.helpers.huggingface_hub_helper import create_huggingface_hub_repo, upload_output_folder
from lib.models.configuration_monodetr import MonoDETRConfig
from lib.models.monofdetr import MonoFDETRForMultiObjectDetection as MonoDETR
from lib.models.image_processsing_monodetr import MonoDETRImageProcessor


logger = get_logger(__name__)


def main():
    args = parse_args()
    cfg = load_config(args, args.cfg_file)
    
    accelerator = build_accelerator(cfg)
    # Handle the hugingface hub repo creation
    if accelerator.is_main_process:
        if cfg.push_to_hub:
            api, repo_id, hub_token = create_huggingface_hub_repo(cfg)
        elif cfg.output_dir is not None:
            Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)    
    accelerator.wait_for_everyone()
    
    logger.logger.addHandler(get_file_handler(Path(cfg.output_dir) / f"train.{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"))
    
    logger.info(f'Init accelerator...\n{accelerator.state}', main_process_only=False)
    
    logger.info("Init DataLoader...")
    train_dataloader, valid_dataloader, _, id2label, label2id = build_dataloader(
        cfg, workers=cfg.dataloader_num_workers, accelerator=accelerator
    )
    
    logger.info("Init Model...")
    config = MonoDETRConfig(
        label2id=label2id, id2label=id2label, **vars(cfg.model),
    )
    if hasattr(cfg, 'pretrain_model') and cfg.pretrain_model is not None:
        model = MonoDETR.from_pretrained(cfg.pretrain_model, config=config, cache_dir=cfg.cache_dir, ignore_mismatched_sizes=True)
    elif hasattr(cfg, 'monodetr_model') and cfg.monodetr_model is not None:
        model = MonoDETR._load_monodetr_pretrain_model(cfg.monodetr_model, config, logger=logger)
    else:
        model = MonoDETR(config)
    if getattr(cfg, 'freeze_encoder', False):
        model.model.freeze_encoder()
    image_processor = MonoDETRImageProcessor()
    
    # Optimizer
    weights, biases = [], []
    for name, param in model.named_parameters():
        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]
    optimizer = torch.optim.AdamW(
        [{'params': biases, 'weight_decay': 0}, 
         {'params': weights, 'weight_decay': cfg.weight_decay}],
        lr=cfg.learning_rate,
        betas=[cfg.adam_beta1, cfg.adam_beta2],
        eps=cfg.adam_epsilon,
    )
    
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = build_lr_scheduler(
        cfg=cfg.lr_scheduler,
        optimizer=optimizer,
        num_processes=accelerator.num_processes,
        num_update_steps_per_epoch=num_update_steps_per_epoch,
        max_train_steps=cfg.max_train_steps,
    )
    
    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, valid_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, valid_dataloader, lr_scheduler
    )
    
    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / cfg.gradient_accumulation_steps)
    cfg.max_train_steps = cfg.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    cfg.num_train_epochs = math.ceil(cfg.max_train_steps / num_update_steps_per_epoch)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    init_accelerator(accelerator, cfg)
    
    # ------------------------------------------------------------------------------------------------
    # Run training with evaluation on each epoch
    # ------------------------------------------------------------------------------------------------

    total_batch_size = cfg.train_batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps

    logger.info("##################  Running training  ##################")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {cfg.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {cfg.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {cfg.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {cfg.max_train_steps}")
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(cfg.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    
    # custom checkpoint data registe to accelerator
    extra_state = CustomCheckpoint()
    
    # Potentially load in the weights and states from a previous save
    if cfg.resume_from_checkpoint:
        checkpoint_path = get_resume_chekpoint_path(cfg.resume_from_checkpoint, cfg.output_dir)
        accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")
        accelerator.register_for_checkpointing(extra_state)
        accelerator.load_state(checkpoint_path)
        logger.info(f"Loading Checkpoint... Best Result:{extra_state.best_result}, Best Epoch:{extra_state.best_epoch}")
        if extra_state.epoch >= 0:
            starting_epoch = extra_state.epoch + 1
        else:
            starting_epoch = get_checkpoint_epoch(checkpoint_path) + 1
        completed_steps = starting_epoch * num_update_steps_per_epoch
    
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)
    
    for epoch in range(starting_epoch, cfg.num_train_epochs):
        model.train()
        if cfg.with_tracking:
            total_loss = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                batch = MonoDETRImageProcessor.prepare_batch(batch, return_pixel_values_without_pd=True)
                outputs = model(**batch, output_attentions=False, output_hidden_states=False)
                loss = outputs.loss
                if cfg.with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1
                
            if step % 30 == 0:
                losses_dict = {k: v.item() for k, v in outputs.loss_dict.items()}
                losses_dict['loss'] = loss.item()
                msg = (
                    f'Epoch: [{epoch}][{step}/{len(train_dataloader)}]\t'
                    f'Loss_monodetr: {losses_dict["loss"]:.2f}\t'
                    f'loss_ce: {losses_dict["loss_ce"]:.2f}\t'
                    f'loss_bbox: {losses_dict["loss_bbox"]:.2f}\t'
                    f'loss_depth: {losses_dict["loss_depth"]:.2f}\t'
                    f'loss_dim: {losses_dict["loss_dim"]:.2f}\t'
                    f'loss_angle: {losses_dict["loss_angle"]:.2f}\t'
                    f'loss_center: {losses_dict["loss_center"]:.2f}\t'
                )
                logger.info(msg)
                if cfg.with_tracking:
                    accelerator.log(
                        {
                            "lr": lr_scheduler.get_last_lr()[0],
                            **losses_dict,
                            "epoch": epoch,
                            "step": completed_steps,
                        }, 
                        step=completed_steps,
                    )
        logger.info(f'Final Training Loss: {losses_dict["loss"]}')
        
        if epoch == cfg.num_train_epochs - 1:
            continue
        elif epoch < 60 and epoch % 5 != 0:
            continue
        elif epoch < 100 and epoch % 2 != 0:
            continue 
        
        logger.info("***** Running evaluation *****")
        
        model.eval()
        results = {}
        for step, batch in enumerate(tqdm(valid_dataloader, disable=not accelerator.is_local_main_process)):
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
        
        if accelerator.is_local_main_process:
            # Wait for all processes to save the results
            time.sleep(2)
            # 3. Evaluation
            metric = valid_dataloader.dataset.eval(
                results_dir=results_path,
                logger=logger,
            )
            
            msg = (
                f'Car_3d_moderate_R40: {metric:.2f}%\t'
            )
            logger.info(f"Final Evaluation Result: " + msg)
            eval_result = metric
            if eval_result > extra_state.best_result:
                extra_state.best_result = eval_result
                extra_state.best_epoch = epoch
                accelerator._custom_objects.clear()
                accelerator.register_for_checkpointing(extra_state)
                accelerator.save_state(get_checkpoint_dir(cfg.output_dir) / 'best')
                logger.info(f"Best Result: {extra_state.best_result}, epoch: {extra_state.best_epoch}")
            
            if cfg.with_tracking:
                accelerator.log({"eval_result": eval_result}, step=completed_steps,)
        
        # Svae model
        if cfg.push_to_hub and epoch < cfg.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(
                cfg.output_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                image_processor.save_pretrained(cfg.output_dir)
                upload_output_folder(
                    api, repo_id, hub_token, commit_message=f"Training in progress epoch {epoch}", output_dir=cfg.output_dir
                )
        
        # Save checkpoint
        extra_state.epoch = epoch
        accelerator._custom_objects.clear()
        accelerator.register_for_checkpointing(extra_state)
        accelerator.save_state(get_checkpoint_dir(cfg.output_dir) / 'latest')


if __name__ == '__main__':
    main()
