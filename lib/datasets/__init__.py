from .kitti import build_dataset as build_kitti_dataset


def build_dataset(cfg, accelerator=None):
    if hasattr(cfg, "root_dir") and "KITTI" in cfg.root_dir:
        return build_kitti_dataset(cfg, accelerator)
    else:
        raise ValueError(f'Unknown dataset config: {cfg}')
