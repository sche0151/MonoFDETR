import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from transformers.utils import requires_backends, logging, is_timm_available
from transformers.models.auto import AutoBackbone


logger = logging.get_logger(__name__)

if is_timm_available():
    from timm import create_model

logger = logging.get_logger(__name__)


# Copied from transformers.models.detr.modeling_detr.DetrFrozenBatchNorm2d with Detr->DeformableDetr
class FrozenBatchNorm2d(nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt, without which any other models than
    torchvision.models.resnet[18,34,50,101] produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it user-friendly
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        epsilon = self.eps
        scale = weight * (running_var + epsilon).rsqrt()
        bias = bias - running_mean * scale
        return x * scale + bias


# Copied from transformers.models.detr.modeling_detr.replace_batch_norm with Detr->DeformableDetr
def replace_batch_norm(model):
    r"""
    Recursively replace all `torch.nn.BatchNorm2d` with `DeformableDetrFrozenBatchNorm2d`.

    Args:
        model (torch.nn.Module):
            input model
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            new_module = FrozenBatchNorm2d(module.num_features)

            if not module.weight.device == torch.device("meta"):
                new_module.weight.data.copy_(module.weight)
                new_module.bias.data.copy_(module.bias)
                new_module.running_mean.data.copy_(module.running_mean)
                new_module.running_var.data.copy_(module.running_var)

            model._modules[name] = new_module

        if len(list(module.children())) > 0:
            replace_batch_norm(module)


class ResNetVisionBackbone(nn.Module):
    """
    ResNet Vision backbone, using either the AutoBackbone API or one from the timm library.

    nn.BatchNorm2d layers are replaced by DeformableDetrFrozenBatchNorm2d as defined above.

    """

    def __init__(self, config):
        super().__init__()

        self.config = config

        if config.use_torchvision_backbone:
            backbone = getattr(torchvision.models, config.backbone)(
                replace_stride_with_dilation=[False, False, False],
                pretrained=config.use_pretrained_backbone,
                norm_layer=FrozenBatchNorm2d,
            )
            for name, parameter in backbone.named_parameters():
                if config.freeze_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                    parameter.requires_grad_(False)
            if config.num_feature_levels > 1:
                return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
                self.intermediate_channel_sizes = [512, 1024, 2048]
            else:
                return_layers = {"layer4": "0"}
                self.intermediate_channel_sizes = [2048]
            self.model = IntermediateLayerGetter(backbone, return_layers=return_layers)
        else:
            if config.use_timm_backbone:
                requires_backends(self, ["timm"])
                kwargs = {}
                
                # load local pretrained model or download to .cache/huggingface from timm
                if getattr(config, 'pretrained_backbone_path', None) is not None:
                    kwargs['pretrained_cfg_overlay'] = dict(file=config.pretrained_backbone_path)
                
                if getattr(config, 'dilation', False):
                    kwargs["output_stride"] = 16
                backbone = create_model(
                    config.backbone,
                    pretrained=config.use_pretrained_backbone,
                    features_only=True,
                    out_indices=(2, 3, 4) if config.num_feature_levels > 1 else (4,),
                    in_chans=config.num_channels,
                    **kwargs,
                )
            else:
                backbone = AutoBackbone.from_config(config.backbone_config)

            # replace batch norm by frozen batch norm
            with torch.no_grad():
                replace_batch_norm(backbone)
            self.model = backbone
            self.intermediate_channel_sizes = (
                self.model.feature_info.channels() if config.use_timm_backbone else self.model.channels
            )

            backbone_model_type = config.backbone if config.use_timm_backbone else config.backbone_config.model_type
            if "resnet" in backbone_model_type:
                for name, parameter in self.model.named_parameters():
                    if config.use_timm_backbone:
                        if config.freeze_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                            parameter.requires_grad_(False)
                    else:
                        if config.freeze_backbone or "stage.1" not in name and "stage.2" not in name and "stage.3" not in name:
                            parameter.requires_grad_(False)

    # Copied from transformers.models.detr.modeling_detr.DetrConvEncoder.forward with Detr->DeformableDetr
    def forward(self, pixel_values: torch.Tensor, pixel_mask: torch.Tensor):
        # send pixel_values through the model to get list of feature maps
        if self.config.use_torchvision_backbone:
            features = self.model(pixel_values).values()
        elif self.config.use_timm_backbone:
            features = self.model(pixel_values)
        else:
            features = self.model(pixel_values).feature_maps

        out = []
        for feature_map in features:
            # downsample pixel_mask to match shape of corresponding feature_map
            mask = nn.functional.interpolate(pixel_mask[None].float(), size=feature_map.shape[-2:]).to(torch.bool)[0]
            out.append((feature_map, mask))
        return out

