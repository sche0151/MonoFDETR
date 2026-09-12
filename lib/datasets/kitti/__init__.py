from .kitti_dataset import KITTI_Dataset

def build_dataset(cfg, accelerator=None):
    
    if hasattr(cfg, "root_dir"):
        if accelerator is None:
            train_dataset = KITTI_Dataset('train', cfg)
            # train_dataset = KITTI_Dataset('trainval', cfg)
            val_dataset = KITTI_Dataset('val', cfg)
            test_dataset = KITTI_Dataset('val', cfg)
        else:
            with accelerator.main_process_first():
                train_dataset = KITTI_Dataset('train', cfg)
                # train_dataset = KITTI_Dataset('trainval', cfg)
                val_dataset = KITTI_Dataset('val', cfg)
                test_dataset = KITTI_Dataset('val', cfg)
        return train_dataset, val_dataset, test_dataset, train_dataset.id2label, train_dataset.label2id
    elif hasattr(cfg, "name"):
        raise NotImplementedError("The dataset name is not yet implemented.")
    
    raise ValueError("You must provide either a root_dir or a name in the config to build the dataset.")
    