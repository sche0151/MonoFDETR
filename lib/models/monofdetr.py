import os
import copy
import math
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

from transformers import PreTrainedModel
from transformers.utils import (
    ModelOutput, 
    is_scipy_available,
    requires_backends,
)
from transformers.pytorch_utils import meshgrid
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput

from .configuration_monodetr import MonoDETRConfig
from .depth_predictor import DDNLoss, DepthPredictor, DPTDepthPredictor
from .multi_scale_deformable_attention import MSDeformAttn, DeformAttn
from .backbone import (
    build_position_encoding, 
    build_vision_backbone, 
    VisionBackboneModel,
    LearnedPositionEmbedding
)
from utils.box_ops import generalized_box_iou, box_cxcylrtb_to_xyxy, box_cxcywh_to_xyxy

if is_scipy_available():
    from scipy.optimize import linear_sum_assignment


@dataclass
class MonoDETRDecoderOutput(ModelOutput):
    """
    Base class for outputs of the MonoDETRDecoder. This class adds three attributes to
    BaseModelOutputWithCrossAttentions, namely:
    - a stacked tensor of intermediate decoder hidden states (i.e. the output of each decoder layer)
    - a stacked tensor of intermediate reference points.
    - a stacked tensor of intermediate object 3dimensions.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        intermediate_reference_dims (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate object 3dimensions (reference 3dimensions of each layer of the decoder).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of the model at the output of each layer
            plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` and `config.add_cross_attention=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights of the decoder's cross-attention layer, after the attention softmax,
            used to compute the weighted average in the cross-attention heads.
    """

    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_reference_dims: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MonoFDETRModelOutput(ModelOutput):
    """
    Base class for outputs of the MonoFDETR encoder-decoder model.

    Args:
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        intermediate_reference_dims (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, sequence_length, hidden_size)`):
            Stacked intermediate object 3dimensions (reference 3dimensions of each layer of the decoder).
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the encoder, after the attention softmax, used to compute the weighted average in the
            self-attention heads.
        depth_logits (`torch.FloatTensor` of shape `(batch_size, num_bins + 1, img_height // 16, img_width // 16)`):
            Depth map logits predicted by the encoder_last_hidden_state.
        weighted_depth (`torch.FloatTensor` of shape `(batch_size, img_height // 16, img_width // 16)`):
            Weighted depth map predicted by the encoder_last_hidden_state.
    """
    
    init_reference_points: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    intermediate_hidden_states: torch.FloatTensor = None
    intermediate_reference_points: torch.FloatTensor = None
    intermediate_reference_dims: torch.FloatTensor = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    depth_logits: Optional[torch.FloatTensor] = None
    weighted_depth: Optional[torch.FloatTensor] = None
    region_probs: Optional[List[torch.FloatTensor]] = None


@dataclass
class MonoFDETRForMultiObjectDetectionOutput(ModelOutput):
    """
    Output type of [`MonoFDETRForMultiObjectDetectionOutput`].
    
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` are provided)):
            Total loss as a linear combination of a negative log-likehood (cross-entropy) for class prediction and a
            bounding box loss. The latter is defined as a linear combination of the L1 loss and the generalized
            scale-invariant IoU loss.
        loss_dict (`Dict`, *optional*):
            A dictionary containing the individual losses. Useful for logging.
        logits (`torch.FloatTensor` of shape `(batch_size, num_queries, num_classes + 1)`):
            Classification logits (including no-object) for all queries.
        pred_boxes (`torch.FloatTensor` of shape `(batch_size, num_queries, 6)`):
            Normalized boxes3d coordinates for all queries, represented as (center3d_x, center3d_y, l, r, b, t). These
            values are normalized in [0, 1], relative to the size of each individual image in the batch (disregarding
            possible padding). You can use [`~Mono3DVGImageProcessor.post_process_object_detection`] to retrieve the
            unnormalized bounding boxes.
        pred_3d_dim (`torch.FloatTensor` of shape `(batch_size, num_queries, 3)`):
            3D dimensions (h, w, l) for all queries.
        pred_depth (`torch.FloatTensor` of shape `(batch_size, num_queries, 2)`):
            Depth and depth_log_variance for all queries.
        pred_angle (`torch.FloatTensor` of shape `(batch_size, num_queries, 24)`):
            Angle classification first 12, and regression last 12 for all queries.
        pred_depth_map_weights (`torch.FloatTensor` of shape `(batch_size, num_queries, 2)`):
            Scale and shift from relative depth map to absolute depth map.
        pred_depth_map_logits (`torch.FloatTensor` of shape `(batch_size, num_bins + 1, img_height // 16, img_width // 16)`):
            Depth map logits for all queries.
        pred_depth_map (`torch.FloatTensor` of shape `(batch_size, img_height, img_width)`):
            Depth maps for input images.
        auxiliary_outputs (`list[Dict]`, *optional*):
            Optional, only returned when auxilary losses are activated (i.e. `config.auxiliary_loss` is set to `True`)
            and labels are provided. It is a list of dictionaries containing the two above keys (`logits` and
            `pred_boxes`) for each decoder layer.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the decoder of the model.
        decoder_hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, num_queries, hidden_size)`. Hidden-states of the decoder at the output of each layer
            plus the initial embedding outputs.
        decoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, num_queries,
            num_queries)`. Attentions weights of the decoder, after the attention softmax, used to compute the weighted
            average in the self-attention heads.
        cross_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_queries, num_heads, 4, 4)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        encoder_last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder of the model.
        encoder_attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, sequence_length, num_heads, 4,
            4)`. Attentions weights of the encoder, after the attention softmax, used to compute the weighted average
            in the self-attention heads.
        intermediate_hidden_states (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, hidden_size)`):
            Stacked intermediate hidden states (output of each layer of the decoder).
        intermediate_reference_points (`torch.FloatTensor` of shape `(batch_size, config.decoder_layers, num_queries, 4)`):
            Stacked intermediate reference points (reference points of each layer of the decoder).
        init_reference_points (`torch.FloatTensor` of shape  `(batch_size, num_queries, 4)`):
            Initial reference points sent through the Transformer decoder.
    """
    
    loss: Optional[torch.FloatTensor] = None
    loss_dict: Optional[Dict] = None
    logits: torch.FloatTensor = None
    pred_boxes: torch.FloatTensor = None
    pred_3d_dim: torch.FloatTensor = None
    pred_depth: torch.FloatTensor = None
    pred_angle: torch.FloatTensor = None
    pred_depth_map_weights: torch.FloatTensor = None
    pred_depth_map_logits: torch.FloatTensor = None
    pred_depth_map: torch.FloatTensor = None
    auxiliary_outputs: Optional[List[Dict]] = None
    init_reference_points: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    intermediate_hidden_states: Optional[torch.FloatTensor] = None
    intermediate_reference_points: Optional[torch.FloatTensor] = None
    decoder_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    decoder_attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    encoder_last_hidden_state: Optional[torch.FloatTensor] = None
    encoder_attentions: Optional[Tuple[torch.FloatTensor]] = None


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class MonoDETREncoderLayer(nn.Module):
    def __init__(self, config: MonoDETRConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MSDeformAttn(
            config, 
            num_heads=config.encoder_attention_heads, 
            n_points=config.encoder_n_points
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        reference_points=None,
        spatial_shapes=None,
        level_start_index=None,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `hidden_states`.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes of the backbone feature maps.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Apply Multi-scale Deformable Attention Module on the multi-scale feature maps.
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)

        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if self.training:
            if torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any():
                clamp_value = torch.finfo(hidden_states.dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class MonoDETRDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        # depth cross attention
        self.cross_attn_depth = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_depth_layer_norm = nn.LayerNorm(self.embed_dim)
        self.cross_attn_depth_residual = config.decoder_depth_residual
        
        # self attention
        if config.decoder_self_attn:
            self.self_attn = nn.MultiheadAttention(self.embed_dim, config.decoder_attention_heads, dropout=config.dropout, batch_first=True)
            self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
            
            self.sa_qcontent_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.sa_qpos_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.sa_kcontent_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.sa_kpos_proj = nn.Linear(self.embed_dim, self.embed_dim)
            self.sa_v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        else:
            self.self_attn = None

        # grouped-query number
        self.group_num = config.group_num
        
        # cross attention
        self.encoder_attn = MSDeformAttn(
            config,
            num_heads=config.decoder_attention_heads,
            n_points=config.decoder_n_points,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # feedforward neural networks
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        hidden_states: torch.Tensor, # bs, nq, d_model
        position_embeddings: Optional[torch.Tensor] = None, # bs, nq, d_model
        reference_points: Optional[Tensor] = None, # bs, nq, num_levels, 2
        spatial_shapes: Optional[Tensor] = None, # num_levels, 2
        level_start_index: Optional[Tensor] = None, # num_levels
        encoder_hidden_states: Optional[torch.Tensor] = None, # bs, \sum{hxw}, d_model
        encoder_attention_mask: Optional[torch.Tensor] = None, # bs, \sum{hxw}
        # For depth
        depth_embeds: Optional[Tensor] = None, # bs, H_1*W_1, d_model
        depth_attention_mask: Optional[Tensor] = None, # bs, H_1*W_1
        depth_adapt_k: Optional[Tensor] = None, # H_1*W_1, bs, d_model
        output_attentions: Optional[bool] = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(seq_len, batch, embed_dim)`.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings that are added to the queries and keys in the self-attention layer.
            reference_points (`torch.FloatTensor`, *optional*):
                Reference points.
            spatial_shapes (`torch.LongTensor`, *optional*):
                Spatial shapes.
            level_start_index (`torch.LongTensor`, *optional*):
                Level start index.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, target_len, source_len)` where padding elements are indicated by very large negative values.
            depth_embeds (`torch.FloatTensor`, *optional*):
                Depth embeddings of shape `(batch, H_1*W_1, embed_dim)`.
            depth_attention_mask (`torch.FloatTensor`, *optional*):
                Depth attention mask of shape `(batch, H_1*W_1)`.
            depth_adapt_k (`torch.FloatTensor`, *optional*):
                Depth adapt key of shape `(H_1*W_1, batch, embed_dim)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers.
        """
        residual = hidden_states
        
        # Depth Cross Attention
        hidden_states, cross_attn_depth_weights = self.cross_attn_depth(
            query=hidden_states,
            key=depth_adapt_k,
            value=depth_embeds,
            key_padding_mask=depth_attention_mask,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        if self.cross_attn_depth_residual:
            hidden_states = hidden_states + residual
        hidden_states = self.cross_attn_depth_layer_norm(hidden_states)
        if self.cross_attn_depth_residual:
            residual = hidden_states
        
        # Self-Attention
        self_attn_weights = None
        if self.self_attn is not None:
            q = k = self.with_pos_embed(hidden_states, position_embeddings)
            bs = hidden_states.size(0)
            
            q_content = self.sa_qcontent_proj(q)
            q_pos = self.sa_qpos_proj(q)
            k_content = self.sa_kcontent_proj(k)
            k_pos = self.sa_kpos_proj(k)
            # v = self.sa_v_proj(hidden_states)
            v = hidden_states
            q = q_content + q_pos
            k = k_content + k_pos
            
            if self.training:
                num_noise = q.size(1) - self.group_num * 50
                num_queries = self.group_num * 50
                q_noise = q[:, :num_noise].repeat(self.group_num, 1, 1)
                k_noise = k[:, :num_noise].repeat(self.group_num, 1, 1)
                v_noise = v[:, :num_noise].repeat(self.group_num, 1, 1)
                q = q[:, num_noise:]
                k = k[:, num_noise:]
                v = v[:, num_noise:]
                q = torch.cat(q.split(num_queries // self.group_num, dim=1), dim=0)
                k = torch.cat(k.split(num_queries // self.group_num, dim=1), dim=0)
                v = torch.cat(v.split(num_queries // self.group_num, dim=1), dim=0)
                q = torch.cat([q_noise, q], dim=1)
                k = torch.cat([k_noise, k], dim=1)
                v = torch.cat([v_noise, v], dim=1)
            
            hidden_states, self_attn_weights = self.self_attn(
                query=q,
                key=k,
                value=v,
            )
            if self.training:
                hidden_states = torch.cat(hidden_states.split(bs, dim=0), dim=1)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = hidden_states + residual
            hidden_states = self.self_attn_layer_norm(hidden_states)
            residual = hidden_states
        
        # Cross-Atrtention
        cross_attn_weights = None
        hidden_states, cross_attn_weights = self.encoder_attn(
            hidden_states=hidden_states,
            attention_mask=encoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            position_embeddings=position_embeddings,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            output_attentions=output_attentions,
        )
        
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = hidden_states + residual
        hidden_states = self.encoder_attn_layer_norm(hidden_states)
        
        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)
        
        if output_attentions:
            outputs += (cross_attn_depth_weights, self_attn_weights, cross_attn_weights)
        
        return outputs
        

class MonoDETRPreTrainedModel(PreTrainedModel):
    config_class = MonoDETRConfig
    base_model_prefix = "model"
    main_input_name = "pixel_values"
    
    def _init_weights(self, module):
        if isinstance(module, LearnedPositionEmbedding):
            nn.init.uniform_(module.row_embeddings.weight)
            nn.init.uniform_(module.column_embeddings.weight)
        elif isinstance(module, MSDeformAttn):
            module._reset_parameters()
        # For MonoDETRModel
        if hasattr(module, "reference_points") and not self.config.two_stage:
            nn.init.xavier_uniform_(module.reference_points.weight.data, gain=1.0)
            nn.init.constant_(module.reference_points.bias.data, 0.0)
        if hasattr(module, "level_embed"):
            nn.init.normal_(module.level_embed)
        if hasattr(module, "input_proj"):
            for proj in module.input_proj:
                nn.init.xavier_uniform_(proj[0].weight, gain=1.0)
                nn.init.constant_(proj[0].bias, 0.0)
        # For MonoDETRForMultiObjectDetection
        if hasattr(module, "class_embed") and hasattr(module, "bbox_embed") and hasattr(module, "depth_embed") and hasattr(module, "height_2d_err_embed"):
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            for class_embed in module.class_embed:
                class_embed.bias.data = torch.ones(self.config.num_labels) * bias_value
            if self.config.with_box_refine:
                nn.init.constant_(module.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            else:
                nn.init.constant_(module.bbox_embed.layers[-1].bias.data, -2.0)
            if self.config.init_box:
                nn.init.constant_(module.bbox_embed.layers[-1].weight.data, 0)
                nn.init.constant_(module.bbox_embed.layers[-1].bias.data, 0)
    

class MonoDETREncoder(MonoDETRPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* layers. Each layer is a [`MonoDETREncoderLayer`].
    
    The encoder updates the flattened multi-scale feature maps through multiple deformable attention layers.
    
    Args:
        config: MonoDETRConfig
    """
    
    def __init__(self, config: MonoDETRConfig):
        super().__init__(config)
        self.gradient_checkpointing = False
        
        self.dropout = config.dropout
        self.layers = nn.ModuleList([MonoDETREncoderLayer(config) for _ in range(config.encoder_layers)])
        
        # Initialize weights and apply final processing
        self.post_init()
    
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """
        Get reference points for each feature map. Used in decoder.

        Args:
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Valid ratios of each feature map.
            device (`torch.device`):
                Device on which to create the tensors.
        Returns:
            `torch.FloatTensor` of shape `(batch_size, num_queries, num_feature_levels, 2)`
        """
        reference_points_list = []
        for level, (height, width) in enumerate(spatial_shapes):
            ref_y, ref_x = meshgrid(
                torch.linspace(0.5, height - 0.5, height, dtype=valid_ratios.dtype, device=device),
                torch.linspace(0.5, width - 0.5, width, dtype=valid_ratios.dtype, device=device),
                indexing="ij",
            )
            # TODO: valid_ratios could be useless here. check https://github.com/fundamentalvision/Deformable-DETR/issues/36
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, level, 1] * height)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, level, 0] * width)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        position_embeddings=None,
        spatial_shapes=None,
        level_start_index=None,
        valid_ratios=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            spatial_shapes (`torch.LongTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of each feature map.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`):
                Starting index of each feature map.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`):
                Ratio of valid area in each feature level.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = inputs_embeds
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=inputs_embeds.device)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        for i, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    encoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_embeddings,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    output_attentions,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )

      
class MonoDETRDecoder(MonoDETRPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`MonoDETRDecoderLayer`].
    
    The decoder updates the query embeddings through multiple depth-aware cross-attention, self-attention, and cross-attention layers.
    
    Args:
        config: MonoDETRConfig
    """
    
    def __init__(self, config: MonoDETRConfig):
        super().__init__(config)
        
        self.dropout = config.dropout
        self.layers = nn.ModuleList([MonoDETRDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.gradient_checkpointing = False
        
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.dim_embed = None
        self.class_embed = None
        
        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        position_embeddings=None, # bs, nq, d_model
        reference_points=None, # bs, nq, 2
        spatial_shapes=None, # num_levels, 2
        level_start_index=None, # num_levels
        valid_ratios=None, 
        # For depth
        depth_embeds=None, # bs, H_1*W_1, d_model
        depth_attention_mask=None, # bs, H_1*W_1
        depth_adapt_k=None, # H_1*W_1, bs, d_model
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple[torch.FloatTensor], MonoDETRDecoderOutput]:
        r"""
        Args:
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`):
                The query embeddings that are passed into the decoder.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding pixel_values of the encoder. Mask values selected
                in `[0, 1]`:
                - 0 for pixels that are real (i.e. **not masked**),
                - 1 for pixels that are padding (i.e. **masked**).
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, num_queries, hidden_size)`, *optional*):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            reference_points (`torch.FloatTensor` of shape `(batch_size, num_queries, 4)` is `as_two_stage` else `(batch_size, num_queries, 2)` or , *optional*):
                Reference point in range `[0, 1]`, top-left (0,0), bottom-right (1, 1), including padding area.
            spatial_shapes (`torch.FloatTensor` of shape `(num_feature_levels, 2)`):
                Spatial shapes of the feature maps.
            level_start_index (`torch.LongTensor` of shape `(num_feature_levels)`, *optional*):
                Indexes for the start of each feature level. In range `[0, sequence_length]`.
            valid_ratios (`torch.FloatTensor` of shape `(batch_size, num_feature_levels, 2)`, *optional*):
                Ratio of valid area in each feature level.
            depth_embeds (`torch.FloatTensor` of shape `(batch_size, H_1*W_1, hidden_size)`, *optional*):
                Depth embeddings of the input depth map.
            depth_attention_mask (`torch.FloatTensor` of shape `(batch_size, H_1*W_1)`, *optional*):
                Depth attention mask of the input depth map.
            depth_adapt_k (`torch.FloatTensor` of shape `(H_1*W_1, batch_size, hidden_size)`, *optional*):
                Depth adapt key of the input depth map.

            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        
        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        intermediate = ()
        intermediate_reference_points = ()
        intermediate_reference_dims = ()
        
        for idx, decoder_layer in enumerate(self.layers):
            if reference_points.shape[-1] == 6: # x, y, l, t, r, b
                reference_points_input = (
                    reference_points[:, :, None] * torch.stack([valid_ratios[..., 0], valid_ratios[..., 1], valid_ratios[..., 0], valid_ratios[..., 0], valid_ratios[..., 1], valid_ratios[..., 1]], -1)[:, None]
                )# bs, nq, 4, 6
            else:
                if reference_points.shape[-1] != 2:
                    raise ValueError("Reference points' last dimension must be of size 2")
                reference_points_input = reference_points[:, :, None] * valid_ratios[:, None] # bs, nq, 4, 2

            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    reference_points=reference_points_input,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    depth_embeds=depth_embeds,
                    depth_attention_mask=depth_attention_mask,
                    depth_adapt_k=depth_adapt_k,
                    output_attentions=output_attentions,
                )
                
            hidden_states = layer_outputs[0] # bs, nq, d_model

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[idx](hidden_states)
                if reference_points.shape[-1] == 6:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2, f"Reference points' last dimension must be of size 2, but is {reference_points.shape[-1]}"
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[idx](hidden_states)

            intermediate += (hidden_states,)
            intermediate_reference_points += (reference_points,)
            intermediate_reference_dims += (reference_dims,)
            
            if output_attentions:
                all_self_attns += (layer_outputs[2],)
                
                if encoder_hidden_states is not None:
                    all_cross_attentions += ((layer_outputs[1], layer_outputs[3],),)
        
        # Keep batch_size as first dimension
        intermediate = torch.stack(intermediate, dim=1)
        intermediate_reference_points = torch.stack(intermediate_reference_points, dim=1)
        intermediate_reference_dims = torch.stack(intermediate_reference_dims, dim=1)
        
        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    intermediate,
                    intermediate_reference_points,
                    intermediate_reference_dims,
                    all_hidden_states,
                    all_self_attns,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return MonoDETRDecoderOutput(
            last_hidden_state=hidden_states,
            intermediate_hidden_states=intermediate,
            intermediate_reference_points=intermediate_reference_points,
            intermediate_reference_dims=intermediate_reference_dims,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class MonoFDETRModel(MonoDETRPreTrainedModel):
    def __init__(self, config: MonoDETRConfig):
        super().__init__(config)
        self.d_model = config.d_model
        
        # Create vision backbone + positional encoding
        vision_backbone = build_vision_backbone(config)
        position_embeddings = build_position_encoding(config, variant='detr')
        self.vision_backbone = VisionBackboneModel(vision_backbone, position_embeddings)
        
        # Create depth predictor
        # Extra fine-grained depth prediction
        self.extra_depth_predictor = DPTDepthPredictor(config)
        
        # Create vision input projection layers
        if config.num_feature_levels > 1:
            num_backbone_outs = len(vision_backbone.intermediate_channel_sizes)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = vision_backbone.intermediate_channel_sizes[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
            for _ in range(config.num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, self.d_model, kernel_size=3, stride=2, padding=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                )
                in_channels = self.d_model
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(vision_backbone.intermediate_channel_sizes[-1], self.d_model, kernel_size=1),
                        nn.GroupNorm(32, self.d_model),
                    )
                ]
            )

        self.encoder = MonoDETREncoder(config)
        self.decoder = MonoDETRDecoder(config)
        
        self.level_embed = nn.Parameter(torch.Tensor(config.num_feature_levels, self.d_model))
        
        # setting query
        self.num_queries = config.num_queries
        self.query_embed = nn.Embedding(self.num_queries * config.group_num, self.d_model*2)
        self.reference_points = nn.Linear(self.d_model, 2)
        
        self.post_init()

    def freeze_encoder(self):
        self.level_embed.requires_grad_(False)
        for name, param in self.input_proj.named_parameters():
            param.requires_grad_(False)
        for name, param in self.encoder.named_parameters():
            param.requires_grad_(False)

    def get_valid_ratio(self, mask):
        """Get the valid ratio of all feature maps."""

        _, height, width = mask.shape
        valid_height = torch.sum(~mask[:, :, 0], 1)
        valid_width = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_heigth = valid_height.float() / height
        valid_ratio_width = valid_width.float() / width
        valid_ratio = torch.stack([valid_ratio_width, valid_ratio_heigth], -1)
        return valid_ratio

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_without_pd: Optional[torch.FloatTensor] = None,
        pixel_values_random_mix: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MonoFDETRModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, num_channels, height, width = pixel_values.shape
        device = pixel_values.device
        
        if pixel_mask is None:
            pixel_mask = torch.zeros(((batch_size, height, width)), dtype=torch.bool, device=device)
        
        # Extract multi-scale feature maps of same resolution `config.d_model` (cf Figure 4 in paper)
        # First, sent pixel_values + pixel_mask through Backbone to obtain the features which is a list of tuples
        features, position_embeddings_list = self.vision_backbone(pixel_values, pixel_mask)
        
        # Then, apply 1x1 convolution to reduce the channel dimension to d_model (256 by default)
        sources = []
        masks = []
        for level, (source, mask) in enumerate(features):
            sources.append(self.input_proj[level](source))
            masks.append(mask)
            if mask is None:
                raise ValueError("No attention mask was provided")
        
        # Lowest resolution feature maps are obtained via 3x3 stride 2 convolutions on the final stage
        if self.config.num_feature_levels > len(sources):
            _len_sources = len(sources)
            for level in range(_len_sources, self.config.num_feature_levels):
                if level == _len_sources:
                    source = self.input_proj[level](features[-1][0])
                else:
                    source = self.input_proj[level](sources[-1])
                mask = nn.functional.interpolate(pixel_mask[None].float(), size=source.shape[-2:]).to(torch.bool)[0]
                pos_l = self.vision_backbone.position_embedding(source, mask).to(source.dtype)
                sources.append(source)
                masks.append(mask)
                position_embeddings_list.append(pos_l)
        
        # Second, prepare depth predictor inputs
        # depth_logits, depth_embeds, weighted_depth = self.depth_predictor(
        #     sources,
        #     masks[1],
        #     position_embeddings_list[1],
        # )
        # depth_embeds = depth_embeds.flatten(2).permute(0, 2, 1) # bs, H_1*W_1, d_model
        # depth_attention_mask = masks[1].flatten(1) # bs, H_1*W_1
        # depth_adapt_k = None
        
        depth_predictor_outputs = self.extra_depth_predictor(
            pixel_values=pixel_values if pixel_values_without_pd is None else pixel_values_without_pd,
            pixel_values_random_mix=pixel_values_random_mix,
            features=sources,
            return_dict=return_dict,
        )
        depth_embeds = depth_predictor_outputs[0].flatten(2).permute(0, 2, 1) # bs, H_1*W_1, d_model
        sources = depth_predictor_outputs[2]
        weighted_depth = depth_predictor_outputs[3]
        region_probs = depth_predictor_outputs[4]
        depth_attention_mask = masks[1].flatten(1) # bs, H_1*W_1
        depth_logits = None
        depth_adapt_k = None
        
        # Create queries
        if self.training:
            query_embed = self.query_embed.weight # (num_queries, d_model*2)
        else:
            # only use one group in inference
            query_embed = self.query_embed.weight[:self.num_queries] # (num_queries, d_model*2)
        
        # Third, Prepare encoder inputs (by flattening)
        source_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for level, (source, mask, pos_embed) in enumerate(zip(sources, masks, position_embeddings_list)):
            batch_size, num_channels, height, width = source.shape
            spatial_shape = (height, width)
            spatial_shapes.append(spatial_shape)
            source = source.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[level].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            source_flatten.append(source)
            mask_flatten.append(mask)
        source_flatten = torch.cat(source_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=source_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        valid_ratios = valid_ratios.float()
        
        # Then, sent source_flatten + mask_flatten + lvl_pos_embed_flatten (backbone + proj layer output) through encoder
        # Also provide spatial_shapes, level_start_index and valid_ratios
        encoder_outputs = self.encoder(
            inputs_embeds=source_flatten,
            attention_mask=mask_flatten,
            position_embeddings=lvl_pos_embed_flatten,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            output_attentions=output_attentions,
            return_dict=return_dict,
        )
        vision_embeds = encoder_outputs[0]
        
        # Fourth, prepare decoder inputs
        batch_size = vision_embeds.shape[0]
        query_embed, target = torch.split(query_embed, self.d_model, dim=1)
        query_embed = query_embed.unsqueeze(0).expand(batch_size, -1, -1) # (bs, nq, d_model)
        target = target.unsqueeze(0).expand(batch_size, -1, -1) # (bs, nq, d_model)
        reference_points = self.reference_points(query_embed).sigmoid() # (bs, nq, 2)
        init_reference_points = reference_points
        
        # Fifth, prepare decoder inputs
        decoder_outputs = self.decoder(
            inputs_embeds=target,
            position_embeddings=query_embed,
            encoder_hidden_states=vision_embeds,
            encoder_attention_mask=mask_flatten,
            reference_points=reference_points,
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            depth_embeds=depth_embeds,
            depth_attention_mask=depth_attention_mask,
            depth_adapt_k=depth_embeds if depth_adapt_k is None else depth_adapt_k,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        if not return_dict:
            depth_outputs = tuple(value for value in [depth_logits, weighted_depth] if value is not None)
            tuple_outputs = (init_reference_points,) + decoder_outputs + encoder_outputs + depth_outputs + (region_probs,)
            
            return tuple_outputs
        
        return MonoFDETRModelOutput(
            init_reference_points=init_reference_points,
            last_hidden_state=decoder_outputs.last_hidden_state,
            intermediate_hidden_states=decoder_outputs.intermediate_hidden_states,
            intermediate_reference_points=decoder_outputs.intermediate_reference_points,
            intermediate_reference_dims=decoder_outputs.intermediate_reference_dims,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_attentions=encoder_outputs.attentions,
            depth_logits=depth_logits,
            weighted_depth=weighted_depth,
            region_probs=region_probs,
        )


class MonoFDETRForMultiObjectDetection(MonoDETRPreTrainedModel):
    
    def __init__(self, config: MonoDETRConfig):
        super().__init__(config)
        
        # MonoDETR model
        self.model = MonoFDETRModel(config)
        
        # Detection heads on top
        self.class_embed = nn.Linear(config.d_model, config.num_labels)
        self.bbox_embed = MLP(config.d_model, config.d_model, 6, 3)
        self.dim_embed_3d = MLP(config.d_model, config.d_model, 3, 2)
        self.angle_embed = MLP(config.d_model, config.d_model, 24, 2)
        self.depth_embed = MLP(config.d_model, config.d_model, 2, 2)  # depth and deviation
        
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = config.decoder_layers
        if config.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.model.decoder.bbox_embed = self.bbox_embed
            self.dim_embed_3d = _get_clones(self.dim_embed_3d, num_pred)
            self.model.decoder.dim_embed = self.dim_embed_3d
            self.angle_embed = _get_clones(self.angle_embed, num_pred)
            self.depth_embed = _get_clones(self.depth_embed, num_pred)
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.dim_embed_3d = nn.ModuleList([self.dim_embed_3d for _ in range(num_pred)])
            self.angle_embed = nn.ModuleList([self.angle_embed for _ in range(num_pred)])
            self.depth_embed = nn.ModuleList([self.depth_embed for _ in range(num_pred)])
            self.model.decoder.bbox_embed = None

        # Initialize weights and apply final processing
        self.post_init()
    
    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'logits': a, 'pred_boxes': b, 
                 'pred_3d_dim': c, 'pred_angle': d, 'pred_depth': e}
                for a, b, c, d, e in zip(outputs_class[:-1], outputs_coord[:-1],
                                         outputs_3d_dim[:-1], outputs_angle[:-1], outputs_depth[:-1])]
        
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        pixel_values_without_pd: Optional[torch.FloatTensor] = None,
        pixel_values_random_mix: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        calibs: Optional[torch.FloatTensor] = None,
        img_sizes: Optional[torch.FloatTensor] = None,
        labels: Optional[List[dict]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], MonoFDETRForMultiObjectDetectionOutput]:
        r"""
        labels (`List[Dict]` of len `(batch_size,)`, *optional*):
            Labels for computing the bipartite matching loss. List of dicts, each dictionary containing at least the
            following 6 keys: 'class_labels', 'boxes_3d', 'size_3d', 'depth', 'heading_bin' and 'heading_res' (the class labels, 3D bounding boxes, 3D sizes, depths and 3D angles of an 3D object in the batch respectively).
            The class labels themselves should be a `torch.LongTensor` of len `(number of bounding boxes in the image,)`, 
            The 3D bounding boxes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 6)`, 
            The 3D sizes a `torch.FloatTensor` of shape `(number of bounding boxes in the image, 3)`,
            The depths a `torch.FloatTensor` of shape `(number of bounding boxes in the image,)`,
            The 3D angles with bin a `torch.LongTensor` of shape `(number of bounding boxes in the image,)` and res a `torch.FloatTensor` of shape `(number of bounding boxes in the image,)`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # First, sent image through DETR base model to obtain encoder + decoder outputs
        outputs = self.model(
            pixel_values,
            pixel_values_without_pd=pixel_values_without_pd,
            pixel_values_random_mix=pixel_values_random_mix,
            pixel_mask=pixel_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs.intermediate_hidden_states if return_dict else outputs[2]
        init_reference = outputs.init_reference_points if return_dict else outputs[0]
        inter_references = outputs.intermediate_reference_points if return_dict else outputs[3]
        inter_references_dim = outputs.intermediate_reference_dims if return_dict else outputs[4]
        pred_depth_map_logits = outputs.depth_logits if return_dict else outputs[-3]
        pred_region_probs = outputs.region_probs if return_dict else outputs[-1]
        
        # class logits + predicted bounding boxes + 3D dims + depths + angles
        outputs_classes = []
        outputs_coords = []
        outputs_3d_dims = []
        outputs_depths = []
        outputs_angles = []
        
        for level in range(hidden_states.shape[1]):
            if level == 0:
                reference = init_reference
            else:
                reference = inter_references[:, level - 1]
            reference = inverse_sigmoid(reference)
            delta_bbox = self.bbox_embed[level](hidden_states[:, level])
            if reference.shape[-1] == 6:
                outputs_coord_logits = delta_bbox + reference
            elif reference.shape[-1] == 2:
                delta_bbox[..., :2] += reference
                outputs_coord_logits = delta_bbox
            else:
                raise ValueError(f"reference.shape[-1] should be 6 or 2, but got {reference.shape[-1]}")

            # 3d center + 2d box
            outputs_coord = outputs_coord_logits.sigmoid()
            outputs_coords.append(outputs_coord)

            # classes
            outputs_class = self.class_embed[level](hidden_states[:, level])
            outputs_classes.append(outputs_class)

            # 3D sizes
            size3d = inter_references_dim[:, level]
            outputs_3d_dims.append(size3d)

            # angles
            outputs_angle = self.angle_embed[level](hidden_states[:, level])
            outputs_angles.append(outputs_angle)
            
            # depth_geo_height => H / H' * fy = Z
            box2d_height_norm = outputs_coord[:, :, 4] + outputs_coord[:, :, 5]
            box2d_height = torch.clamp(box2d_height_norm * img_sizes[:, 1: 2], min=1.0)
            depth_geo = size3d[:, :, 0] / box2d_height * calibs[:, 1, 1].unsqueeze(1)

            # depth_geo_err
            depth_geo_err = self.depth_embed[level](hidden_states[:, level])

            # depth average + sigma
            depth_ave = torch.cat([depth_geo.unsqueeze(-1) + depth_geo_err[:, :, 0: 1], depth_geo_err[:, :, 1: 2]], -1)
            outputs_depths.append(depth_ave)

        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        outputs_3d_dim = torch.stack(outputs_3d_dims)
        outputs_depth = torch.stack(outputs_depths)
        outputs_angle = torch.stack(outputs_angles)
        
        logits = outputs_class[-1]
        pred_boxes = outputs_coord[-1]
        pred_3d_dim = outputs_3d_dim[-1]
        pred_depth = outputs_depth[-1]
        pred_angle = outputs_angle[-1]

        loss, loss_dict, auxiliary_outputs = None, None, None
        if labels is not None:
            # First: create the matcher
            matcher = MonoDETRHungarianMatcher(
                class_cost=self.config.class_cost, center3d_cost=self.config.center3d_cost, bbox_cost=self.config.bbox_cost, giou_cost=self.config.giou_cost
            )
            # Second: create the criterion
            losses = ['labels', 'boxes', 'cardinality', 'depths', 'dims', 'angles', 'center', 'region']
            criterion = MonoFDETRLoss(
                matcher=matcher,
                num_classes=self.config.num_labels,
                focal_alpha=self.config.focal_alpha,
                losses=losses,
            )
            criterion.to(self.device)
            criterion.training = self.training
            # Third: compute the losses, based on outputs and labels
            outputs_loss = {}
            outputs_loss["logits"] = logits
            outputs_loss["pred_boxes"] = pred_boxes
            outputs_loss['pred_3d_dim'] = pred_3d_dim
            outputs_loss['pred_depth'] = pred_depth
            outputs_loss['pred_angle'] = pred_angle
            outputs_loss['pred_depth_map_logits'] = pred_depth_map_logits
            outputs_loss['pred_region_prob'] = pred_region_probs
            if self.config.auxiliary_loss:
                auxiliary_outputs = self._set_aux_loss(outputs_class, outputs_coord, outputs_3d_dim, outputs_angle, outputs_depth)
                outputs_loss["auxiliary_outputs"] = auxiliary_outputs
                
            loss_dict = criterion(outputs_loss, labels)
            # Fourth: compute total loss, as a weighted sum of the various losses
            weight_dict = {"loss_ce": self.config.cls_loss_coefficient, "loss_bbox": self.config.bbox_loss_coefficient}
            weight_dict["loss_giou"] = self.config.giou_loss_coefficient
            weight_dict['loss_dim'] = self.config.dim_loss_coefficient
            weight_dict['loss_angle'] = self.config.angle_loss_coefficient
            weight_dict['loss_depth'] = self.config.depth_loss_coefficient
            weight_dict['loss_center'] = self.config.center3d_loss_coefficient
            weight_dict['loss_depth_map'] = self.config.depth_map_loss_coefficient
            weight_dict['loss_region'] = self.config.region_loss_coefficient
            if self.config.auxiliary_loss:
                aux_weight_dict = {}
                for i in range(self.config.decoder_layers - 1):
                    aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
                weight_dict.update(aux_weight_dict)
            loss = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        
        if not return_dict:
            if auxiliary_outputs is not None:
                output = (logits, pred_boxes, pred_3d_dim, pred_depth, pred_angle, pred_depth_map_logits) + auxiliary_outputs + outputs
            else:
                output = (logits, pred_boxes, pred_3d_dim, pred_depth, pred_angle, pred_depth_map_logits) + outputs
            tuple_outputs = ((loss, loss_dict) + output) if loss is not None else output

            return tuple_outputs

        dict_outputs = MonoFDETRForMultiObjectDetectionOutput(
            loss=loss,
            loss_dict=loss_dict,
            logits=logits,
            pred_boxes=pred_boxes,
            pred_3d_dim=pred_3d_dim,
            pred_depth=pred_depth,
            pred_angle=pred_angle,
            pred_depth_map_logits=pred_depth_map_logits,
            auxiliary_outputs=auxiliary_outputs,
            last_hidden_state=outputs.last_hidden_state,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_attentions=outputs.encoder_attentions,
            intermediate_hidden_states=outputs.intermediate_hidden_states,
            intermediate_reference_points=outputs.intermediate_reference_points,
            init_reference_points=outputs.init_reference_points,
        )
        
        return dict_outputs

    @classmethod
    def _load_monodetr_pretrain_model(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        config: Optional[Union[MonoDETRConfig, dict]] = None,
        output_loading_info: bool = False,
        logger: Optional[logging.Logger] = None,
    ):
        logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Load config if we don't provide a configuration
        if config is None:
            config = MonoDETRConfig()
        elif isinstance(config, dict):
            config = MonoDETRConfig(**config)
        else:
            assert isinstance(config, MonoDETRConfig), f"config: {config} has to be of type MonoDETRConfig"
        
        config = copy.deepcopy(config) # We do not want to modify the config inplace in from_pretrained.
        model = cls(config)
        
        logger.info(f"==> Loading from MonoDETR '{pretrained_model_name_or_path}'")
        checkpoint = torch.load(pretrained_model_name_or_path, map_location="cpu", weights_only=False)
        
        new_state_dict = {}
        state_dict = checkpoint['model_state']
        model_state_dict = model.state_dict()
        for key, value in state_dict.items():
            if 'depth_predictor' in key:
                new_key = key.replace('depth_predictor', 'model.depth_predictor')
            elif 'backbone.0' in key:
                new_key = key.replace('backbone.0.body', 'model.vision_backbone.backbone.model')
            elif 'depthaware_transformer' in key:
                new_key = key.replace('depthaware_transformer', 'model')
                if 'depthaware_transformer.encoder' in key:
                    new_key = new_key.replace('norm1', 'self_attn_layer_norm')
                    new_key = new_key.replace('linear1', 'fc1')
                    new_key = new_key.replace('linear2', 'fc2')
                    new_key = new_key.replace('norm2', 'final_layer_norm')
                if 'depthaware_transformer.decoder' in key:
                    new_key = new_key.replace('cross_attn.', 'encoder_attn.')
                    new_key = new_key.replace('norm1', 'encoder_attn_layer_norm')
                    new_key = new_key.replace('norm_depth', 'cross_attn_depth_layer_norm')
                    new_key = new_key.replace('norm2', 'self_attn_layer_norm')
                    new_key = new_key.replace('linear1', 'fc1')
                    new_key = new_key.replace('linear2', 'fc2')
                    new_key = new_key.replace('norm3', 'final_layer_norm')
            elif 'input_proj' in key:
                new_key = key.replace('input_proj', 'model.input_proj')
            elif 'query_embed' in key:
                new_key = key.replace('query_embed', 'model.query_embed')
            else:
                new_key = key
            
            if new_key in model_state_dict and value.shape != model_state_dict[new_key].shape:
                logger.info(f"Skip loading parameter: {new_key}, "
                            f"required shape: {model_state_dict[new_key].shape}, "
                            f"loaded shape: {state_dict[key].shape}")
                continue
            new_state_dict[new_key] = value

        if checkpoint['model_state'] is not None:
            missing_keys, unexpected_keys = model.load_state_dict(new_state_dict,strict=False)
        logger.info("==> Done")
        
        if output_loading_info:
            return model, {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys}
        
        return model
        

# Copied from transformers.models.detr.modeling_detr.sigmoid_focal_loss
def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class MonoFDETRLoss(nn.Module):
    """ This class computes the loss for MonoFDETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
        
    Args:
        matcher (`MonoDETRHungarianMatcher`):
            Module able to compute a matching between targets and proposals.
        num_classes (`int`):
            Number of object categories, omitting the special no-object category.
        focal_alpha (`float`):
            Alpha parameter in focal loss.
        losses (`List[str]`):
            List of all the losses to be applied. See `get_loss` for a list of all available losses.
        group_num (`int`):
            Number of groups in query
    """
    
    def __init__(self, matcher, num_classes, focal_alpha, losses, group_num=11):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.ddn_loss = DDNLoss()  # for depth map
        self.group_num = group_num

    # removed logging parameter, which was part of the original implementation
    def loss_labels(self, outputs, targets, indices, num_boxes):
        """
        Classification loss (Binary focal loss) targets dicts must contain the key "class_labels" containing a tensor
        of dim [nb_target_boxes]
        """
        if "logits" not in outputs:
            raise KeyError("No logits were found in the outputs")
        source_logits = outputs["logits"]

        idx = self._get_source_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            source_logits.shape[:2], self.num_classes, dtype=torch.int64, device=source_logits.device
        )
        target_classes[idx] = target_classes_o.squeeze().long()

        target_classes_onehot = torch.zeros(
            [source_logits.shape[0], source_logits.shape[1], source_logits.shape[2] + 1],
            dtype=source_logits.dtype,
            layout=source_logits.layout,
            device=source_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(source_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2)
            * source_logits.shape[1]
        )
        losses = {"loss_ce": loss_ce}

        return losses

    @torch.no_grad()
    # Copied from transformers.models.detr.modeling_detr.DetrLoss.loss_cardinality
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """
        Compute the cardinality error, i.e. the absolute error in the number of predicted non-empty boxes.

        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients.
        """
        logits = outputs["logits"]
        device = logits.device
        target_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (logits.argmax(-1) != logits.shape[-1] - 1).sum(1)
        card_err = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
        losses = {"cardinality_error": card_err}
        return losses

    def loss_3dcenter(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the 3D center coordinates, the L1 regression loss.
        
        Targets dicts must contain the key "boxes_3d" containing a tensor of dim [nb_target_boxes, 6]. The target boxes are expected in format (cx, cy, l, r, t, b), normalized by the image size.
        """
        
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_center3d = outputs['pred_boxes'][..., :2][idx]
        target_center3d = torch.cat([t['boxes_3d'][..., :2][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_center3d = nn.functional.l1_loss(source_center3d, target_center3d, reduction='none')
        losses = {}
        losses['loss_center'] = loss_center3d.sum() / num_boxes
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes_3d" containing a tensor of dim [nb_target_boxes, 6]. The target boxes are expected in format (cx, cy, l, r, t, b), normalized by the image size.
        """
        if "pred_boxes" not in outputs:
            raise KeyError("No predicted boxes found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_boxes3d = outputs['pred_boxes'][idx]
        target_boxes3d = torch.cat([t['boxes_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        # source_boxes2d_height = outputs['pred_boxes_2d_height'][idx]
        # target_boxes2d_height = torch.cat([t['boxes_2d_h'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze(-1)

        loss_bbox = nn.functional.l1_loss(source_boxes3d[..., 2:], target_boxes3d[..., 2:], reduction="none")

        losses = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(
            generalized_box_iou(box_cxcylrtb_to_xyxy(source_boxes3d), box_cxcylrtb_to_xyxy(target_boxes3d))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        
        # loss_height = nn.functional.l1_loss(source_boxes2d_height, target_boxes2d_height, reduction="none")
        # losses["loss_height"] = loss_height.sum() / num_boxes
        
        return losses

    def loss_depths(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the depth, the Laplacian algebraic
uncertainty loss.

        Targets dicts must contain the key "depth" containing a tensor of dim [nb_target_boxes, 1].
        """
        if "pred_depth" not in outputs:
            raise KeyError("No predicted depth found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_depths = outputs['pred_depth'][idx]
        target_depths = torch.cat([t['depth'][i] for t, (_, i) in zip(targets, indices)], dim=0).squeeze()

        depth_input, depth_log_variance = source_depths[:, 0], source_depths[:, 1] 
        # depth_log_variance should be positive, 
        # but √2 * exp(-|var|) * |y-y'| + |var| will be a discontinuous function when |y-y'| > 1/√2
        # so we detect the condition and use a sectional-continuous function instead
        # depth_loss = torch.where(
        #     torch.abs(depth_input - target_depths) > 1 / 1.4142,
        #     1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance,
        #     1.4142 * torch.exp(-torch.abs(depth_log_variance)) * torch.abs(depth_input - target_depths) + torch.abs(depth_log_variance)
        # )
        # heteroscedastic aleatoric uncertainty, noted when |y-y'| < 1/√2
        # the depth_loss will be negetive, but it's ok, the backward direction is correct.
        depth_loss = 1.4142 * torch.exp(-depth_log_variance) * torch.abs(depth_input - target_depths) + depth_log_variance
        losses = {}
        losses['loss_depth'] = depth_loss.sum() / num_boxes 
        return losses
    
    def loss_dims(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the 3D size, the 3D IoU oriented loss.
        
        Targets dicts must contain the key "size_3d" containing a tensor of dim [nb_target_boxes, 3]. The target sizes are expected in format (h, w, l).
        """
        
        if "pred_3d_dim" not in outputs:
            raise KeyError("No predicted 3d_dim found in outputs")
        idx = self._get_source_permutation_idx(indices)
        source_dims = outputs['pred_3d_dim'][idx]
        target_dims = torch.cat([t['size_3d'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        dimension = target_dims.clone().detach()
        dim_loss = torch.abs(source_dims - target_dims)
        dim_loss /= dimension
        with torch.no_grad():
            compensation_weight = F.l1_loss(source_dims, target_dims) / dim_loss.mean()
        dim_loss *= compensation_weight
        losses = {}
        losses['loss_dim'] = dim_loss.sum() / num_boxes
        return losses

    def loss_angles(self, outputs, targets, indices, num_boxes):  
        """
        Compute the losses related to the angle, the MultiBin loss.
        
        Targets dicts must contain the key "heading_bin" containing a tensor of dim [nb_target_boxes, 1] and "heading_res" containing a tensor of dim [nb_target_boxes, 1].
        """
        if "pred_angle" not in outputs:
            raise KeyError("No predicted angle found in outputs")
        idx = self._get_source_permutation_idx(indices)
        heading_input = outputs['pred_angle'][idx]
        target_heading_cls = torch.cat([t['heading_bin'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_heading_res = torch.cat([t['heading_res'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        heading_input = heading_input.view(-1, 24)
        heading_target_cls = target_heading_cls.view(-1).long()
        heading_target_res = target_heading_res.view(-1)

        # classification loss
        heading_input_cls = heading_input[:, 0:12]
        cls_loss = F.cross_entropy(heading_input_cls, heading_target_cls, reduction='none')

        # regression loss
        heading_input_res = heading_input[:, 12:24]
        cls_onehot = torch.zeros(heading_target_cls.shape[0], 12).cuda().scatter_(dim=1, index=heading_target_cls.view(-1, 1), value=1)
        heading_input_res = torch.sum(heading_input_res * cls_onehot, 1)
        reg_loss = F.l1_loss(heading_input_res, heading_target_res, reduction='none')
        
        angle_loss = cls_loss + reg_loss
        losses = {}
        losses['loss_angle'] = angle_loss.sum() / num_boxes 
        return losses

    def loss_depth_map(self, outputs, targets, indices, num_boxes):
        """
        Compute the losses related to the depth map, the DDN loss.
        
        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4] and "depth" containing a tensor of dim [nb_target_boxes, 1]. The target boxes are expected in format (cx, cy, w, h), normalized by the image size.
        """
        if "pred_depth_map_logits" not in outputs:
            raise KeyError("No predicted depth map found in outputs")
        depth_map_logits = outputs['pred_depth_map_logits']

        num_gt_per_img = [len(t['boxes']) for t in targets]
        gt_boxes2d = torch.cat([t['boxes'] for t in targets], dim=0) * torch.tensor([80, 24, 80, 24], device='cuda')
        gt_boxes2d = box_cxcywh_to_xyxy(gt_boxes2d)
        gt_center_depth = torch.cat([t['depth'] for t in targets], dim=0).squeeze(dim=1)

        losses = dict()
        losses["loss_depth_map"] = self.ddn_loss(
            depth_map_logits, gt_boxes2d, num_gt_per_img, gt_center_depth)
        return losses

    def loss_region(self, outputs, targets, indices, num_boxes):
        region_probs = outputs['pred_region_prob']
        gt_region = torch.cat([t['obj_region'].unsqueeze(0) for t in targets], dim=0)

        loss = 0
        losses = dict()
        for region_prob in region_probs:
            gt_region_resized = F.interpolate(gt_region.unsqueeze(1).float(), size=region_prob.shape[2:], mode='bilinear', align_corners=True)
            # Compute intersection and union
            intersection = (region_prob * gt_region_resized).sum()
            total = region_prob.sum() + gt_region_resized.sum()
            # Compute Dice Coefficient
            dice_coef = (2. * intersection + 1) / (total + 1)
            # Compute Dice Loss
            dice_loss = 1 - dice_coef
            loss += dice_loss

        losses['loss_region'] = loss

        return losses

    def _get_source_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(source, i) for i, (source, _) in enumerate(indices)])
        source_idx = torch.cat([source for (source, _) in indices])
        return batch_idx, source_idx

    def _get_target_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(target, i) for i, (_, target) in enumerate(indices)])
        target_idx = torch.cat([target for (_, target) in indices])
        return batch_idx, target_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'depths': self.loss_depths,
            'dims': self.loss_dims,
            'angles': self.loss_angles,
            'center': self.loss_3dcenter,
            'depth_map': self.loss_depth_map,
            'region': self.loss_region,
        }
        if loss not in loss_map:
            raise ValueError(f"Loss {loss} not supported")
        return loss_map[loss](outputs, targets, indices, num_boxes)

    def forward(self, outputs, targets):
        """ 
        This performs the loss computation.
        
        Args:
             outputs (`dict`, *optional*):
                Dictionary of tensors, see the output specification of the model for the format.
             targets (`List[dict]`, *optional*):
                List of dicts, such that `len(targets) == batch_size`. The expected keys in each dict depends on the
                losses applied, see each loss' doc.
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "auxiliary_outputs"}
        group_num = self.group_num if self.training else 1

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets, group_num=group_num)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets) * group_num
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        # (Niels): comment out function below, distributed training to be added
        # if is_dist_avail_and_initialized():
        #     torch.distributed.all_reduce(num_boxes)
        # (Niels) in original implementation, num_boxes is divided by get_world_size()
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "auxiliary_outputs" in outputs:
            for i, auxiliary_outputs in enumerate(outputs["auxiliary_outputs"]):
                indices = self.matcher(auxiliary_outputs, targets, group_num=group_num)
                for loss in self.losses:
                    if loss == 'depth_map' or loss == 'region':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    l_dict = self.get_loss(loss, auxiliary_outputs, targets, indices, num_boxes)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


# Copied from transformers.models.detr.modeling_detr.DetrMLPPredictionHead
class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (MLP, also called FFN), used to predict the normalized center coordinates,
    height and width of a bounding box w.r.t. an image.

    Copied from https://github.com/facebookresearch/detr/blob/master/models/detr.py

    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = nn.functional.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class MonoDETRHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    
    Args:
        class_cost: 
            The relative weight of the classification error in the matching cost.
        center3d_cost:
            The relative weight of the L1 error of the center3d coordinates in the matching cost.
        bbox_cost: 
            The relative weight of the L1 error of the bounding box coordinates in the matching cost.
        giou_cost: 
        The relative weight of the giou loss of the bounding box in the matching cost.
    """

    def __init__(self, class_cost: float = 1, center3d_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
        super().__init__()
        requires_backends(self, ["scipy"])
        
        self.class_cost = class_cost
        self.center3d_cost = center3d_cost
        self.bbox_cost = bbox_cost
        self.giou_cost = giou_cost
        if class_cost == 0 and bbox_cost == 0 and giou_cost == 0 and center3d_cost == 0:
            raise ValueError("All costs of the Matcher can't be 0")

    @torch.no_grad()
    def forward(self, outputs, targets, group_num=11):
        """
        Args:
            outputs (`dict`):
                A dictionary that contains at least these entries:
                * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
            targets (`List[dict]`):
                A list of targets (len(targets) = batch_size), where each target is a dict containing:
                * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
                  ground-truth
                 objects in the target) containing the class labels
                * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.

        Returns:
            `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
            - index_i is the indices of the selected predictions (in order)
            - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        batch_size, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).sigmoid()  # [batch_size * num_queries, num_classes]
        out_bbox3d = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 6]
        
        # Also concat the target labels and boxes
        target_ids = torch.cat([v["labels"] for v in targets]).long()
        tgt_bbox3d = torch.cat([v["boxes_3d"] for v in targets])

        # Compute the classification cost.
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        class_cost = pos_cost_class[:, target_ids] - neg_cost_class[:, target_ids]

        # Compute the center3d L1 cost between boxes
        center3d_cost = torch.cdist(out_bbox3d[..., :2], tgt_bbox3d[..., :2], p=1)

        # Compute the L1 cost between boxes
        bbox_cost = torch.cdist(out_bbox3d[..., 2:], tgt_bbox3d[..., 2:], p=1)

        # Compute the giou cost betwen boxes
        giou_cost = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox3d), box_cxcylrtb_to_xyxy(tgt_bbox3d))

        # Final cost matrix
        cost_matrix = self.bbox_cost * bbox_cost + self.center3d_cost * center3d_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
        cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        # indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
        indices = []
        g_num_queries = num_queries // group_num
        C_list = cost_matrix.split(g_num_queries, dim=1)
        for g_i in range(group_num):
            C_g = C_list[g_i]
            indices_g = [linear_sum_assignment(c[i]) for i, c in enumerate(C_g.split(sizes, -1))]
            if g_i == 0:
                indices = indices_g
            else:
                indices = [
                    (np.concatenate([indice1[0], indice2[0] + g_num_queries * g_i]), np.concatenate([indice1[1], indice2[1]]))
                    for indice1, indice2 in zip(indices, indices_g)
                ]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

