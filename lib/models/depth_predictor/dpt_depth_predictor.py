from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from transformers import DepthAnythingForDepthEstimation, DPTImageProcessor
from transformers.utils import ModelOutput


@dataclass
class DepthAnythingOutput(ModelOutput):
    """
    Base Class for the outputs of the `DepthAnything` model.
    
    Args:
        predicted_depth (`torch.FloatTensor` of shape `(batch_size, height, width)`):
            Predicted depth for each pixel.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `return_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`.
            
            Hidden-states of the model at the neck module output of each layer.
    """
    
    predicted_depth: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class DPTDepthPredictorOutput(ModelOutput):
    
    depth_embed: torch.FloatTensor = None
    predicted_depth: torch.FloatTensor = None
    hidden_states: List[torch.FloatTensor] = None
    weighted_depth: torch.FloatTensor = None
    region_probs: List[torch.FloatTensor] = None


class DepthAnything(nn.Module):
    """
    Depth Anything model, using the `DepthAnythingForDepthEstimation` class.
    """
    
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        assert getattr(config, 'pretrained_dpt_path', None) is not None, "pretrained_dpt_path is required in the config."
        dpt = DepthAnythingForDepthEstimation.from_pretrained(
            config.pretrained_dpt_path,
        )
        image_processor = DPTImageProcessor.from_pretrained(
            config.pretrained_dpt_path,
        )
        self.model = dpt
        self.image_processor = image_processor
        self.hidden_size = dpt.config.fusion_hidden_size # 64
        
        for parameter in dpt.parameters():
            parameter.requires_grad_(False)
            
    def forward(
        self, 
        pixel_values: torch.Tensor,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], DepthAnythingOutput]:
        _, _, H, W = pixel_values.shape
        resize_height, resize_width = self.image_processor.size['height'], self.image_processor.size['width']
        scale_height = resize_height / H
        scale_width = resize_width / W
        
        if abs(1 - scale_width) < abs(1 - scale_height):
            scale_height = scale_width
        else:
            scale_width = scale_height
        
        new_height = round(H * scale_height / self.image_processor.ensure_multiple_of) * self.image_processor.ensure_multiple_of
        new_width = round(W * scale_width / self.image_processor.ensure_multiple_of) * self.image_processor.ensure_multiple_of
        
        pixel_values = F.interpolate(pixel_values, size=(new_height, new_width), mode='bicubic', align_corners=False)
        
        outputs = self.model.backbone(pixel_values)
        hidden_states = outputs.feature_maps
        
        patch_size = self.model.config.patch_size
        patch_height = new_height // patch_size
        patch_width = new_width // patch_size
        
        hidden_states = self.model.neck(hidden_states, patch_height, patch_width)
        
        predicted_depth = self.model.head(hidden_states, patch_height, patch_width)

        if not return_dict:
            if output_hidden_states:
                output = (predicted_depth, hidden_states)
            else:
                output = (predicted_depth,)
            return output
        
        return DepthAnythingOutput(
            predicted_depth=predicted_depth,
            hidden_states=hidden_states if output_hidden_states else None,
        )


class DPTDepthPredictor(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.depth_predictor = DepthAnything(config)
        self.hidden_state_index = -1
        self.depth_max = float(config.depth_max)
        
        # Create modules
        d_model = config.d_model
        dpt_hidden_size = self.depth_predictor.hidden_size
        depth_proj_list = []
        for _ in range(4):
            depth_proj_list.append(
                nn.Sequential(
                    nn.Conv2d(dpt_hidden_size, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
        self.depth_proj = nn.ModuleList(depth_proj_list)
        n_levels = config.num_feature_levels
        depth_fusion_list = []
        for _ in range(n_levels):
            depth_fusion_list.append(
                nn.Sequential(
                    nn.Conv2d(2 * d_model, d_model, kernel_size=1),
                    nn.GroupNorm(32, d_model),
                )
            )
        self.depth_fusion = nn.ModuleList(depth_fusion_list)
        
        self.se_blocks = nn.ModuleList([SEBlock(d_model) for _ in range(n_levels)])
        seg_pred_list = []
        for _ in range(n_levels):
            seg_pred_list.append(
                nn.Sequential(
                    nn.Conv2d(d_model, 256, kernel_size=3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(256, 1, kernel_size=3, padding=1),
                )
            )
        self.seg_pred = nn.ModuleList(seg_pred_list)
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, d_model),
            nn.ReLU(),
        )
        
        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, d_model)
        
    def forward(
        self,
        pixel_values: torch.Tensor,
        features: List[torch.Tensor],
        pixel_values_random_mix: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ):
        outputs = self.depth_predictor(pixel_values)
        output_hidden_states = outputs.hidden_states[::-1]
        predicted_depth = outputs.predicted_depth
        predicted_depth = self.normalize_depth_map(predicted_depth, pixel_values.shape[-2:])
        
        not_random_mix = pixel_values_random_mix is None or (pixel_values_random_mix == 0).all()
        if not not_random_mix:
            not_mix_mask = (pixel_values_random_mix == 0).all(dim=(1, 2, 3))
            outputs_random_mix = self.depth_predictor(pixel_values_random_mix)
            
            output_hidden_states_random_mix = outputs_random_mix.hidden_states[::-1]
            for hidden_state, hidden_state_random_mix in zip(output_hidden_states, output_hidden_states_random_mix):
                hidden_state_random_mix[not_mix_mask] = hidden_state[not_mix_mask]
                hidden_state = (hidden_state + hidden_state_random_mix) / 2.0
            
            predicted_depth_random_mix = outputs_random_mix.predicted_depth
            predicted_depth_random_mix = self.normalize_depth_map(predicted_depth_random_mix, pixel_values_random_mix.shape[-2:])
            predicted_depth_random_mix[not_mix_mask] = predicted_depth[not_mix_mask]
            predicted_depth = (predicted_depth + predicted_depth_random_mix) / 2.0
            
            del outputs_random_mix, output_hidden_states_random_mix, predicted_depth_random_mix
            
        hidden_states = []
        region_probs = []
        for i, (feature, hidden_state) in enumerate(zip(features, output_hidden_states)):
            hidden_state = hidden_state.detach()
            hidden_state = self.depth_proj[i](hidden_state)
            hidden_state = F.interpolate(hidden_state, size=feature.shape[-2:], mode='bilinear', align_corners=False)
            hidden_state = self.depth_fusion[i](torch.cat([hidden_state, feature], dim=1))
            
            seg_hidden_state = self.se_blocks[i](hidden_state)
            seg_prob = torch.sigmoid(self.seg_pred[i](seg_hidden_state))
            region_probs.append(torch.clamp(seg_prob, min=1e-5, max=1 - 1e-5))
            # enhance multi-scale features
            hidden_state = hidden_state * seg_prob
            
            hidden_states.append(hidden_state)
        
        depth_embed = self.depth_head(hidden_states[1])
        seg_embed = torch.where(region_probs[1] > 0.5, torch.ones_like(region_probs[1]), torch.zeros_like(region_probs[1]))
        
        # weighted relative depth to metric depth
        weighted_depth = self.depth_max * predicted_depth
        weighted_depth = weighted_depth.clamp(min=0, max=self.depth_max)
        depth_pos_embed_ip = self.interpolate_depth_embed(
            F.interpolate(weighted_depth.unsqueeze(1), size=depth_embed.shape[-2:], mode="bicubic", align_corners=False).squeeze(1)
        )
        depth_embed = depth_embed + depth_pos_embed_ip + seg_embed
        
        if not return_dict:
            return (depth_embed, predicted_depth, hidden_states, weighted_depth, region_probs)
        
        return DPTDepthPredictorOutput(
            depth_embed=depth_embed,
            predicted_depth=predicted_depth,
            hidden_states=hidden_states,
            weighted_depth=weighted_depth,
            region_probs=region_probs
        )
    
    def interpolate_depth_embed(self, depth):
        depth = depth.clamp(min=0, max=self.depth_max)
        pos = self.interpolate_1d(depth, self.depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos
    
    def interpolate_1d(self, coord, embed):
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=embed.num_embeddings - 1)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta
    
    def normalize_depth_map(self, depth_map, interpolate_size):
        # interpolate the predicted depth map
        depth_map = F.interpolate(
            depth_map.unsqueeze(1), size=interpolate_size, mode="bicubic", align_corners=False).squeeze(1)
        # normalize following the batch, nearest value is 0.0 and the farthest value is 1.0
        depth_map_min = depth_map.view(depth_map.shape[0], -1).min(dim=1)[0]
        depth_map_max = depth_map.view(depth_map.shape[0], -1).max(dim=1)[0]
        depth_map = 1.0 - (depth_map - depth_map_min[:, None, None]) / (depth_map_max[:, None, None] - depth_map_min[:, None, None])
        return depth_map
        

class SEBlock(nn.Module):
    """Squeeeze-and-Excitation Block, a channel-wise attention mechanism. Just with Pooling (squeeze) and FC layers (excitation) to learn the weights for each channel. Enhancing the representational power of the important channels and reducing the noise of the unrelevant background."""
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(channel, channel // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(channel // reduction, channel, kernel_size=1)

    def forward(self, x):
        w = F.adaptive_avg_pool2d(x, 1)
        w = F.relu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w
