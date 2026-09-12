from typing import Optional, Tuple, Union, Dict, Any, List
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from transformers.activations import ACT2FN
from diffusers import MarigoldDepthPipeline
from diffusers.utils import BaseOutput

from lib.models.backbone import build_position_encoding
from lib.models.configuration_monodetr import MonoDETRConfig


class DepthCrossEncoderLayer(nn.Module):
    """Cross-modal encoder layer including vision self-attention and depth-to-vision cross attention."""
    def __init__(self, config: MonoDETRConfig):
        super().__init__()
        self.embed_dim = config.d_model
        
        # image self attention
        self.self_attn = nn.MultiheadAttention(
            config.d_model, config.encoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        # depth-to-image cross attention
        self.cross_attn_depth = nn.MultiheadAttention(self.embed_dim, config.encoder_attention_heads, dropout=config.dropout, batch_first=True)
        self.cross_attn_depth_layer_norm = nn.LayerNorm(self.embed_dim)
        
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def with_pos_embed(self, tensor: torch.Tensor, position_embeddings: Optional[torch.Tensor]):
        return tensor if position_embeddings is None else tensor + position_embeddings

    def forward(
        self,
        vision_embeds: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        depth_embeds: torch.Tensor,
        depth_attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor = None,
        vision_interpolate_embeds: torch.Tensor = None,
        depth_interpolate_embeds: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            vision_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Input to the layer.
            vision_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
                Attention mask.
            depth_embeds (`torch.FloatTensor` of shape `(batch_size, depth_length, hidden_size)`):
                Input to the layer.
            text_attention_mask (`torch.FloatTensor` of shape `(batch_size, depth_length)`):
                Depth attention mask.
            position_embeddings (`torch.FloatTensor`, *optional*):
                Position embeddings, to be added to `vision_embeds`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the cross attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = vision_embeds

        # vision self attention
        vision_embeds, self_attn_weights = self.self_attn(
            query=self.with_pos_embed(vision_embeds, position_embeddings),
            key=self.with_pos_embed(vision_embeds, position_embeddings),
            value=vision_embeds,
            key_padding_mask=vision_attention_mask,
        )

        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)
        vision_embeds = residual + vision_embeds
        vision_embeds = self.self_attn_layer_norm(vision_embeds)
        vision_embeds = self.with_pos_embed(vision_embeds, vision_interpolate_embeds)

        # depth-to-image cross attention
        residual = vision_embeds
        vision_embeds, cross_attn_weights = self.cross_attn_depth(
            query=self.with_pos_embed(vision_embeds, position_embeddings),
            key=self.with_pos_embed(depth_embeds, depth_interpolate_embeds),
            value=self.with_pos_embed(depth_embeds, depth_interpolate_embeds),
            key_padding_mask=depth_attention_mask,
        )
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)
        vision_embeds = residual + vision_embeds
        vision_embeds = self.cross_attn_depth_layer_norm(vision_embeds)

        residual = vision_embeds
        vision_embeds = self.activation_fn(self.fc1(vision_embeds))
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.activation_dropout, training=self.training)

        vision_embeds = self.fc2(vision_embeds)
        vision_embeds = nn.functional.dropout(vision_embeds, p=self.dropout, training=self.training)

        vision_embeds = residual + vision_embeds
        vision_embeds = self.final_layer_norm(vision_embeds)

        if self.training:
            if torch.isinf(vision_embeds).any() or torch.isnan(vision_embeds).any():
                clamp_value = torch.finfo(vision_embeds.dtype).max - 1000
                vision_embeds = torch.clamp(vision_embeds, min=-clamp_value, max=clamp_value)

        outputs = (vision_embeds,)

        if output_attentions:
            outputs += (cross_attn_weights,)

        return outputs


class VisionDepthEncoder(nn.Module):
    """Depth-Guided Vision Encoder."""
    def __init__(self, config: MonoDETRConfig, num_layers: int = 1):
        super().__init__()
        self.layers = nn.ModuleList([DepthCrossEncoderLayer(config) for _ in range(num_layers)])
    
    def forward(
        self, 
        vision_embeds: torch.Tensor,
        vision_attention_mask: torch.Tensor,
        depth_embeds: torch.Tensor,
        depth_attention_mask: torch.Tensor,
        position_embeddings: torch.Tensor,
        vision_interpolate_embeds: torch.Tensor = None,
        depth_interpolate_embeds: torch.Tensor = None,
        output_attentions: bool = False,
    ):
        """
        Args:
            vision_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Flattened feature map (output of the backbone + projection layer) that is passed to the encoder.
            vision_attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                Mask to avoid performing attention on padding pixel features. Mask values selected in `[0, 1]`:
                - 1 for pixel features that are real (i.e. **not masked**),
                - 0 for pixel features that are padding (i.e. **masked**).
                [What are attention masks?](../glossary#attention-mask)
            depth_embeds (`torch.FloatTensor` of shape `(batch_size, depth_length, hidden_size)`):
                Input to the layer.
            depth_attention_mask (`torch.FloatTensor` of shape `(batch_size, depth_length)`):
                Text attention mask.
            position_embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
                Position embeddings that are added to the queries and keys in each self-attention layer.
            vision_interpolate_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Interpolated vision weighted map embeddings.
            depth_interpolate_embeds (`torch.FloatTensor` of shape `(batch_size, depth_length, hidden_size)`, *optional*):
                Interpolated depth weighted map embeddings.
            output_attentions (`bool`, *optional*):
                Whether or not to return the cross attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        hidden_states = vision_embeds
        
        all_attentions = () if output_attentions else None
        for layer in self.layers:
            layer_outputs = layer(
                vision_embeds=hidden_states, 
                vision_attention_mask=vision_attention_mask,
                depth_embeds=depth_embeds,
                depth_attention_mask=depth_attention_mask,
                position_embeddings=position_embeddings,
                vision_interpolate_embeds=vision_interpolate_embeds,
                depth_interpolate_embeds=depth_interpolate_embeds,
                output_attentions=output_attentions,
            )
            
            hidden_states = layer_outputs[0]
            
            if output_attentions:
                all_attentions += (layer_outputs[1],)
        
        return tuple(v for v in [hidden_states, all_attentions] if v is not None)


@dataclass
class MarigoldDepthPredictorOutput(BaseOutput):
    """
    Output of the Marigold Depth Predictor.
    
    Args:
        marigold_depth (`torch.Tensor`):
            Predicted depth maps with values in the range [0, 1] by Marigold. The shape is always [batch_size, H, W].
        depth_logits (`torch.Tensor`):
            Logits of the depth map predicted by image features. The shape is always [batch_size, num_depth_bins + 1, H/16, W/16].
        depth_embed (`torch.Tensor`):
            Embedding of the depth map. The shape is always [batch_size, dim, H/16, W/16].
        weighted_depth (`torch.Tensor`):
            Weighted depth map predicted by image features with values in the range [min_depth, max_depth]. The shape is always [batch_size, H/16, W/16].
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
    """
    marigold_depth: torch.Tensor
    depth_logits: torch.Tensor
    depth_embed: torch.Tensor
    weighted_depth: torch.Tensor
    attentions: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MarigoldDepthPipeOutput(BaseOutput):
    """
    Output of the Marigold Depth Pipeline.
    
    Args:
        prediction (`torch.Tensor`):
            Predicted depth maps with values in the range [0, 1]. The shape is always [batch_size, 1, H, W].
        depth_8_embed (`torch.Tensor`):
            Embedding of the depth map from the MidBlock's output of the VAE decoder. The shape is always [batch_size, dim, H'/8, W'/8].
        depth_8_embed_mask (`torch.Tensor`):
            Mask for the depth_8_embed tensor. The shape is always [batch_size, H'/8, W'/8]. True values indicate padding.
        depth_4_embed (`torch.Tensor`):
            Embedding of the depth map from the UpBlocks[0]'s output of the VAE decoder. The shape is always [batch_size, dim, H'/4, W'/4].
        depth_4_embed_mask (`torch.Tensor`):
            Mask for the depth_4_embed tensor. The shape is always [batch_size, H'/4, W'/4]. True values indicate padding.
        depth_2_embed (`torch.Tensor`):
            Embedding of the depth map from the UpBlocks[1]'s output of the VAE decoder. The shape is always [batch_size, dim, H'/2, W'/2].
        depth_2_embed_mask (`torch.Tensor`):
            Mask for the depth_2_embed tensor. The shape is always [batch_size, H'/2, W'/2]. True values indicate padding.
    """
    prediction: torch.Tensor
    depth_8_embed: torch.Tensor
    depth_8_embed_mask: torch.Tensor
    depth_4_embed: torch.Tensor
    depth_4_embed_mask: torch.Tensor
    depth_2_embed: torch.Tensor
    depth_2_embed_mask: torch.Tensor


class MarigoldDepthPipe(nn.Module):
    """
    Marigold Depth Predictor, using the Hugging Face diffusers library.
    """
    
    def __init__(self, config, fp16: bool = False, cache_dir: Optional[str] = 'cache'):
        super().__init__()
        
        if fp16:
            fp16_config = {
                "variant": "fp16",
                "torch_dtype": torch.float16,
            }
        else:
            fp16_config = {}
        
        pipeline = MarigoldDepthPipeline.from_pretrained(
            config.depth_predictor_type,
            cache_dir=cache_dir,
            prediction_type='depth',
            **fp16_config,
        )
        self.d_model = pipeline.vae.config['block_out_channels'][-1] # 512
        self.dtype = pipeline.dtype
        self.num_inference_steps = pipeline.default_denoising_steps # 10
        self.processing_resolution = pipeline.default_processing_resolution # 768
        self.pipe = pipeline
        
        for parameter in self.pipe.vae.parameters():
            parameter.requires_grad = False
        for parameter in self.pipe.unet.parameters():
            parameter.requires_grad = False
        for parameter in self.pipe.text_encoder.parameters():
            parameter.requires_grad = False
    
    @torch.no_grad()
    def denoising(
        self,
        image: torch.FloatTensor,
        num_inference_steps: Optional[int] = None,
        ensemble_size: int = 1,
        processing_resolution: Optional[int] = None,
        match_input_resolution: bool = True,
        resample_method_input: str = "bilinear",
        resample_method_output: str = "bilinear",
        batch_size: int = 1,
        ensembling_kwargs: Optional[Dict[str, Any]] = None,
        latents: Optional[Union[torch.Tensor, List[torch.Tensor]]] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        output_type: str = "np",
        output_uncertainty: bool = False,
    ):
        """
        Function invoked when calling the pipeline.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, `List[PIL.Image.Image]`, `List[np.ndarray]`),
                `List[torch.Tensor]`: An input image or images used as an input for the depth estimation task. For
                arrays and tensors, the expected value range is between `[0, 1]`. Passing a batch of images is possible
                by providing a four-dimensional array or a tensor. Additionally, a list of images of two- or
                three-dimensional arrays or tensors can be passed. In the latter case, all list elements must have the
                same width and height.
            num_inference_steps (`int`, *optional*, defaults to `None`):
                Number of denoising diffusion steps during inference. The default value `None` results in automatic
                selection. The number of steps should be at least 10 with the full Marigold models, and between 1 and 4
                for Marigold-LCM models.
            ensemble_size (`int`, defaults to `1`):
                Number of ensemble predictions. Recommended values are 5 and higher for better precision, or 1 for
                faster inference.
            processing_resolution (`int`, *optional*, defaults to `None`):
                Effective processing resolution. When set to `0`, matches the larger input image dimension. This
                produces crisper predictions, but may also lead to the overall loss of global context. The default
                value `None` resolves to the optimal value from the model config.
            match_input_resolution (`bool`, *optional*, defaults to `True`):
                When enabled, the output prediction is resized to match the input dimensions. When disabled, the longer
                side of the output will equal to `processing_resolution`.
            resample_method_input (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize input images to `processing_resolution`. The accepted values are:
                `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            resample_method_output (`str`, *optional*, defaults to `"bilinear"`):
                Resampling method used to resize output predictions to match the input resolution. The accepted values
                are `"nearest"`, `"nearest-exact"`, `"bilinear"`, `"bicubic"`, or `"area"`.
            batch_size (`int`, *optional*, defaults to `1`):
                Batch size; only matters when setting `ensemble_size` or passing a tensor of images.
            ensembling_kwargs (`dict`, *optional*, defaults to `None`)
                Extra dictionary with arguments for precise ensembling control. The following options are available:
                - reduction (`str`, *optional*, defaults to `"median"`): Defines the ensembling function applied in
                  every pixel location, can be either `"median"` or `"mean"`.
                - regularizer_strength (`float`, *optional*, defaults to `0.02`): Strength of the regularizer that
                  pulls the aligned predictions to the unit range from 0 to 1.
                - max_iter (`int`, *optional*, defaults to `2`): Maximum number of the alignment solver steps. Refer to
                  `scipy.optimize.minimize` function, `options` argument.
                - tol (`float`, *optional*, defaults to `1e-3`): Alignment solver tolerance. The solver stops when the
                  tolerance is reached.
                - max_res (`int`, *optional*, defaults to `None`): Resolution at which the alignment is performed;
                  `None` matches the `processing_resolution`.
            latents (`torch.Tensor`, or `List[torch.Tensor]`, *optional*, defaults to `None`):
                Latent noise tensors to replace the random initialization. These can be taken from the previous
                function call's output.
            generator (`torch.Generator`, or `List[torch.Generator]`, *optional*, defaults to `None`):
                Random number generator object to ensure reproducibility.
            output_type (`str`, *optional*, defaults to `"np"`):
                Preferred format of the output's `prediction` and the optional `uncertainty` fields. The accepted
                values are: `"np"` (numpy array) or `"pt"` (torch tensor).
            output_uncertainty (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `uncertainty` field contains the predictive uncertainty map, provided that
                the `ensemble_size` argument is set to a value above 2.
            output_latent (`bool`, *optional*, defaults to `False`):
                When enabled, the output's `latent` field contains the latent codes corresponding to the predictions
                within the ensemble. These codes can be saved, modified, and used for subsequent calls with the
                `latents` argument.
        """
        
        # 0. Resolving variables.
        device = image.device
        dtype = self.dtype
        if device != self.pipe.device:
            self.pipe.to(device)
        
        # Model-specific optimal default values leading to fast and reasonable results.
        if num_inference_steps is None:
            num_inference_steps = self.num_inference_steps
        if processing_resolution is None:
            processing_resolution = self.processing_resolution
        
        # 1. Check inputs.
        num_images = self.pipe.check_inputs(
            image,
            num_inference_steps,
            ensemble_size,
            processing_resolution,
            resample_method_input,
            resample_method_output,
            batch_size,
            ensembling_kwargs,
            latents,
            generator,
            output_type,
            output_uncertainty,
        )
        
        # 2. Prepare empty text conditioning.
        # Model invocation: self.pipe.tokenizer, self.pipe.text_encoder.
        if self.pipe.empty_text_embedding is None:
            prompt = ""
            text_inputs = self.pipe.tokenizer(
                prompt,
                padding="do_not_pad",
                max_length=self.pipe.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids.to(device)
            self.pipe.empty_text_embedding = self.pipe.text_encoder(text_input_ids)[0]  # [1,2,1024]
        
        # 3. Preprocess input images. This function loads input image or images of compatible dimensions `(H, W)`,
        # optionally downsamples them to the `processing_resolution` `(PH, PW)`, where
        # `max(PH, PW) == processing_resolution`, and pads the dimensions to `(PPH, PPW)` such that these values are
        # divisible by the latent space downscaling factor (typically 8 in Stable Diffusion). The default value `None`
        # of `processing_resolution` resolves to the optimal value from the model config. It is a recommended mode of
        # operation and leads to the most reasonable results. Using the native image resolution or any other processing
        # resolution can lead to loss of either fine details or global context in the output predictions.
        image, padding, original_resolution = self.pipe.image_processor.preprocess(
            image, processing_resolution, resample_method_input, device, dtype
        )  # [N,3,PPH,PPW]
        
        # 4. Encode input image into latent space. At this step, each of the `N` input images is represented with `E`
        # ensemble members. Each ensemble member is an independent diffused prediction, just initialized independently.
        # Latents of each such predictions across all input images and all ensemble members are represented in the
        # `pred_latent` variable. The variable `image_latent` is of the same shape: it contains each input image encoded
        # into latent space and replicated `E` times. The latents can be either generated (see `generator` to ensure
        # reproducibility), or passed explicitly via the `latents` argument. The latter can be set outside the pipeline
        # code. For example, in the Marigold-LCM video processing demo, the latents initialization of a frame is taken
        # as a convex combination of the latents output of the pipeline for the previous frame and a newly-sampled
        # noise. This behavior can be achieved by setting the `output_latent` argument to `True`. The latent space
        # dimensions are `(h, w)`. Encoding into latent space happens in batches of size `batch_size`.
        # Model invocation: self.vae.encoder.
        image_latent, pred_latent = self.pipe.prepare_latents(
            image, latents, generator, ensemble_size, batch_size
        )  # [N*E,4,h,w], [N*E,4,h,w]

        del image

        batch_empty_text_embedding = self.pipe.empty_text_embedding.to(device=device, dtype=dtype).repeat(
            batch_size, 1, 1
        )  # [B,1024,2]
        
        # 5. Process the denoising loop. All `N * E` latents are processed sequentially in batches of size `batch_size`.
        # The unet model takes concatenated latent spaces of the input image and the predicted modality as an input, and
        # outputs noise for the predicted modality's latent space. The number of denoising diffusion steps is defined by
        # `num_inference_steps`. It is either set directly, or resolves to the optimal value specific to the loaded
        # model.
        # Model invocation: self.pipe.unet.
        pred_latents = []

        for i in range(0, num_images * ensemble_size, batch_size):
            batch_image_latent = image_latent[i : i + batch_size]  # [B,4,h,w]
            batch_pred_latent = pred_latent[i : i + batch_size]  # [B,4,h,w]
            effective_batch_size = batch_image_latent.shape[0]
            text = batch_empty_text_embedding[:effective_batch_size]  # [B,2,1024]

            self.pipe.scheduler.set_timesteps(num_inference_steps, device=device)
            for t in self.pipe.scheduler.timesteps:
                batch_latent = torch.cat([batch_image_latent, batch_pred_latent], dim=1)  # [B,8,h,w]
                noise = self.pipe.unet(batch_latent, t, encoder_hidden_states=text, return_dict=False)[0]  # [B,4,h,w]
                batch_pred_latent = self.pipe.scheduler.step(
                    noise, t, batch_pred_latent, generator=generator
                ).prev_sample  # [B,4,h,w]

            pred_latents.append(batch_pred_latent)

        pred_latent = torch.cat(pred_latents, dim=0)  # [N*E,4,h,w]

        del (
            pred_latents,
            image_latent,
            batch_empty_text_embedding,
            batch_image_latent,
            batch_pred_latent,
            text,
            batch_latent,
            noise,
        )

        # 6. Decode predictions from latent into pixel space. The resulting `N * E` predictions have shape `(PPH, PPW)`,
        # which requires slight postprocessing. Decoding into pixel space happens in batches of size `batch_size`.
        # Model invocation: self.pipe.vae.decoder.
        # Add hook to capture the mid block output.
        vae_decoder_mid_outputs = []
        vae_decoder_up_0_outputs = []
        vae_decoder_up_1_outputs = []
        hooks = [
            self.pipe.vae.decoder.mid_block.register_forward_hook(
                lambda self, input, output: vae_decoder_mid_outputs.append(output)
            ),
            self.pipe.vae.decoder.up_blocks[0].register_forward_hook(
                lambda self, input, output: vae_decoder_up_0_outputs.append(output)
            ),
            self.pipe.vae.decoder.up_blocks[1].register_forward_hook(
                lambda self, input, output: vae_decoder_up_1_outputs.append(output)
            ),
        ]
        prediction = torch.cat(
            [
                self.pipe.decode_prediction(pred_latent[i : i + batch_size])
                for i in range(0, pred_latent.shape[0], batch_size)
            ],
            dim=0,
        )  # [N*E,1,PPH,PPW]
        for hook in hooks:
            hook.remove()
        depth_8_embed = torch.cat(vae_decoder_mid_outputs, dim=0).detach() # [B,512,H/8,W/8]
        depth_8_embed_mask = self.get_depth_embed_mask(depth_8_embed, padding, scale_ratio=8)
        depth_4_embed = torch.cat(vae_decoder_up_0_outputs, dim=0).detach() # [B,512,H/4,W/4]
        depth_4_embed_mask = self.get_depth_embed_mask(depth_4_embed, padding, scale_ratio=4)
        depth_2_embed = torch.cat(vae_decoder_up_1_outputs, dim=0).detach() # [B,512,H/2,W/2]
        depth_2_embed_mask = self.get_depth_embed_mask(depth_2_embed, padding, scale_ratio=2)
        
        # 7. Remove padding. The output shape is (PH, PW).
        prediction = self.pipe.image_processor.unpad_image(prediction, padding)  # [N*E,1,PH,PW]

        # 8. Ensemble and compute uncertainty (when `output_uncertainty` is set). This code treats each of the `N`
        # groups of `E` ensemble predictions independently. For each group it computes an ensembled prediction of shape
        # `(PH, PW)` and an optional uncertainty map of the same dimensions. After computing this pair of outputs for
        # each group independently, it stacks them respectively into batches of `N` almost final predictions and
        # uncertainty maps.
        uncertainty = None
        if ensemble_size > 1:
            prediction = prediction.reshape(num_images, ensemble_size, *prediction.shape[1:])  # [N,E,1,PH,PW]
            prediction = [
                self.pipe.ensemble_depth(
                    prediction[i],
                    self.pipe.scale_invariant,
                    self.pipe.shift_invariant,
                    output_uncertainty,
                    **(ensembling_kwargs or {}),
                )
                for i in range(num_images)
            ]  # [ [[1,1,PH,PW], [1,1,PH,PW]], ... ]
            prediction, uncertainty = zip(*prediction)  # [[1,1,PH,PW], ... ], [[1,1,PH,PW], ... ]
            prediction = torch.cat(prediction, dim=0)  # [N,1,PH,PW]
            if output_uncertainty:
                uncertainty = torch.cat(uncertainty, dim=0)  # [N,1,PH,PW]
            else:
                uncertainty = None

        # 9. If `match_input_resolution` is set, the output prediction and the uncertainty are upsampled to match the
        # input resolution `(H, W)`. This step may introduce upsampling artifacts, and therefore can be disabled.
        # Depending on the downstream use-case, upsampling can be also chosen based on the tolerated artifacts by
        # setting the `resample_method_output` parameter (e.g., to `"nearest"`).
        if match_input_resolution:
            prediction = self.pipe.image_processor.resize_antialias(
                prediction, original_resolution, resample_method_output, is_aa=False
            )  # [N,1,H,W]
            if uncertainty is not None and output_uncertainty:
                uncertainty = self.pipe.image_processor.resize_antialias(
                    uncertainty, original_resolution, resample_method_output, is_aa=False
                )  # [N,1,H,W]
        
        # 11. Offload all models
        self.pipe.maybe_free_model_hooks()
        
        return prediction, (depth_8_embed, depth_4_embed, depth_2_embed), (depth_8_embed_mask, depth_4_embed_mask, depth_2_embed_mask)
    
    def get_depth_embed_mask(self, depth_embed: torch.Tensor, padding: Tuple[int, int], scale_ratio: int = 8) -> torch.Tensor:
        bs, _, h, w = depth_embed.shape
        depth_embed_mask = torch.ones((bs, h, w), dtype=torch.bool, device=depth_embed.device)
        ph, pw = padding
        ph = ph // scale_ratio
        pw = pw // scale_ratio
        uh = None if ph == 0 else -ph
        uw = None if pw == 0 else -pw
        depth_embed_mask[:, :uh, :uw] = False
        return depth_embed_mask
    
    def forward(
        self, 
        pixel_values: torch.FloatTensor,
    ) -> MarigoldDepthPipeOutput:
        
        prediction, depth_embeds, depth_embed_masks = self.denoising(pixel_values)
        depth = prediction.detach() # (batch_size, 1, H, W)
          
        return MarigoldDepthPipeOutput(
            prediction=depth,
            depth_8_embed=depth_embeds[0],
            depth_8_embed_mask=depth_embed_masks[0],
            depth_4_embed=depth_embeds[1],
            depth_4_embed_mask=depth_embed_masks[1],
            depth_2_embed=depth_embeds[2],
            depth_2_embed_mask=depth_embed_masks[2]
        )
        

class MarigoldDepthPredictor(nn.Module):
    
    def __init__(self, config, position_embedding: nn.Module = None):
        super().__init__()
        depth_num_bins = int(config.num_depth_bins)
        depth_min = float(config.depth_min)
        depth_max = float(config.depth_max)
        self.depth_max = depth_max

        bin_size = 2 * (depth_max - depth_min) / (depth_num_bins * (1 + depth_num_bins))
        bin_indice = torch.linspace(0, depth_num_bins - 1, depth_num_bins)
        bin_value = (bin_indice + 0.5).pow(2) * bin_size / 2 - bin_size / 8 + depth_min
        bin_value = torch.cat([bin_value, torch.tensor([depth_max])], dim=0)
        self.depth_bin_values = nn.Parameter(bin_value, requires_grad=False)
        
        self.depth_pipeline = MarigoldDepthPipe(config)
        depth_d_model = self.depth_pipeline.d_model
        
        # Create modules
        d_model = config.d_model
        # Aggregate depth embeddings to 1/4 resolution
        # self.depth_8_embed_upsample = nn.Sequential(
        #     nn.Conv2d(depth_d_model, d_model, kernel_size=(1, 1)),
        #     nn.GroupNorm(32, d_model))
        # self.depth_4_embed_proj = nn.Sequential(
        #     nn.Conv2d(depth_d_model, d_model, kernel_size=(1, 1)),
        #     nn.GroupNorm(32, d_model))
        # self.depth_2_embed_downsample = nn.Sequential(
        #     nn.Conv2d(depth_d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
        #     nn.GroupNorm(32, d_model))
        
        # Aggregate depth embeddings to 1/8 resolution
        self.depth_8_embed_proj = nn.Sequential(
            nn.Conv2d(depth_d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.depth_4_embed_downsample = nn.Sequential(
            nn.Conv2d(depth_d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.depth_2_embed_downsample = nn.Sequential(
            nn.Conv2d(depth_d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        
        self.downsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.GroupNorm(32, d_model))
        self.proj = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        self.upsample = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(1, 1)),
            nn.GroupNorm(32, d_model))
        
        self.depth_head = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=(3, 3), padding=1),
            nn.GroupNorm(32, num_channels=d_model),
            nn.ReLU())
        
        self.depth_classifier = nn.Conv2d(d_model, depth_num_bins + 1, kernel_size=(1, 1))
        
        if position_embedding is None:
            self.position_embedding = build_position_encoding(config, 'detr')
        else:
            self.position_embedding = position_embedding
        self.depth_encoder = VisionDepthEncoder(config, num_layers=1)
        
        self.depth_pos_embed = nn.Embedding(int(self.depth_max) + 1, d_model)
        self.marigold_depth_pos_embed = nn.Embedding(1001, d_model)
    
    def freeze_vision_module(self):
        for parameter in self.proj.parameters():
            parameter.requires_grad = False
        for parameter in self.upsample.parameters():
            parameter.requires_grad = False
        for parameter in self.downsample.parameters():
            parameter.requires_grad = False
        for parameter in self.depth_head.parameters():
            parameter.requires_grad = False
        for parameter in self.depth_classifier.parameters():
            parameter.requires_grad = False
        self.depth_pos_embed.weight.requires_grad = False
        for name, parameter in self.depth_encoder.named_parameters():
            if 'self_attn' in name and 'self_attn_layer_norm' not in name:
                parameter.requires_grad = False
     
    def forward(
        self, 
        features: List[torch.FloatTensor],
        mask: torch.FloatTensor,
        pos: torch.FloatTensor,
        pixel_values: torch.FloatTensor,
        output_attentions=None,
        return_dict=None,
    ) -> MarigoldDepthPredictorOutput:
        
        assert len(features) == 4
        
        # foreground depth map
        src_16 = self.proj(features[1])
        src_32 = self.upsample(F.interpolate(features[2], size=src_16.shape[-2:], mode='bilinear'))
        src_8 = self.downsample(features[0])
        src = (src_8 + src_16 + src_32) / 3
        
        src = self.depth_head(src)
        depth_logits = self.depth_classifier(src)

        depth_probs = F.softmax(depth_logits, dim=1)
        weighted_depth = (depth_probs * self.depth_bin_values.reshape(1, -1, 1, 1)).sum(dim=1)
        depth_pos_embed_ip = self.interpolate_depth_embed(weighted_depth)
        
        # marigold depth map
        marigold_output = self.depth_pipeline(pixel_values)
        # Aggregate depth embeddings to 1/4 resolution
        # depth_4_embed = self.depth_4_embed_proj(marigold_output.depth_4_embed)
        # depth_8_embed = self.depth_8_embed_upsample(
        #     F.interpolate(marigold_output.depth_8_embed, size=depth_4_embed.shape[-2:], mode='bilinear'))
        # depth_2_embed = self.depth_2_embed_downsample(marigold_output.depth_2_embed)
        # depth_embed_mask = marigold_output.depth_4_embed_mask
        
        # Aggregate depth embeddings to 1/8 resolution
        depth_8_embed = self.depth_8_embed_proj(marigold_output.depth_8_embed)
        depth_4_embed = self.depth_4_embed_downsample(marigold_output.depth_4_embed)
        depth_2_embed = self.depth_2_embed_downsample(marigold_output.depth_2_embed)
        depth_embed_mask = marigold_output.depth_8_embed_mask
        
        depth_embed = (depth_8_embed + depth_4_embed + depth_2_embed) / 3
        
        marigold_depth = marigold_output.prediction.squeeze(1) # B, H, W
        marigold_depth_pos_embed_ip = self.interpolate_marigold_depth_embed(
            F.interpolate(marigold_output.prediction, size=depth_embed.shape[-2:], mode='bilinear').squeeze(1))
        
        # depth embeddings with depth positional encodings
        B, C, H, W = src.shape
        
        all_attentions = () if output_attentions else None
        layer_outputs = self.depth_encoder(
            vision_embeds=src.flatten(2).permute(0, 2, 1), # B, H*W, C
            vision_attention_mask=mask.flatten(1),
            depth_embeds=depth_embed.flatten(2).permute(0, 2, 1), # B, dH*dW, C
            depth_attention_mask=depth_embed_mask.flatten(1),
            position_embeddings=pos.flatten(2).permute(0, 2, 1),
            vision_interpolate_embeds=depth_pos_embed_ip.flatten(2).permute(0, 2, 1),
            depth_interpolate_embeds=marigold_depth_pos_embed_ip.flatten(2).permute(0, 2, 1),
            output_attentions=output_attentions,
        )
        
        depth_embed = layer_outputs[0].permute(0, 2, 1).reshape(B, C, H, W)
        
        if output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)
        
        if not return_dict:
            return tuple(v for v in [marigold_depth, depth_logits, depth_embed, weighted_depth, all_attentions] if v is not None)
        return MarigoldDepthPredictorOutput(
            marigold_depth=marigold_depth,
            depth_logits=depth_logits,
            depth_embed=depth_embed,
            weighted_depth=weighted_depth,
            attentions=all_attentions,
        )
    
    def interpolate_marigold_depth_embed(self, depth):
        depth = (depth * 1000).clamp(min=0, max=1000)
        pos = self.interpolate_1d(depth, self.marigold_depth_pos_embed)
        pos = pos.permute(0, 3, 1, 2)
        return pos
    
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
        
