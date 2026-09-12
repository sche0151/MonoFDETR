from .resnet_backbone import ResNetVisionBackbone

from .position_encoding import build_position_encoding, LearnedPositionEmbedding
from .vision_backbone import VisionBackboneModel


def build_vision_backbone(config):
    backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
    
    if backbone_model_type.startswith("resnet"):
        vision_backbone = ResNetVisionBackbone(config)
    else:
        raise ValueError(f"Not supported {backbone_model_type}")

    return vision_backbone
