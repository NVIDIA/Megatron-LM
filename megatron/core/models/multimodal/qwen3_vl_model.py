# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
# Modified for Qwen3-VL support.
"""Qwen3-VL multi-modal model implementation for Megatron-LM."""

import logging
from contextlib import nullcontext
from copy import deepcopy
from functools import partial
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from megatron.core import parallel_state, tensor_parallel
from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
from megatron.core.enums import Fp8Recipe
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.inference.contexts import BaseInferenceContext
from megatron.core.models.gpt import GPTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import deprecate_inference_params, log_single_rank

try:
    import transformer_engine  # pylint: disable=unused-import
    from megatron.core.extensions.transformer_engine import TEDotProductAttention
    HAVE_TE = True
    try:
        import transformer_engine_torch as tex
        HAVE_TEX = True
    except:
        HAVE_TEX = False
except:
    HAVE_TE = False

try:
    from megatron.core.extensions.transformer_engine import te_checkpoint
    HAVE_TE_CHECKPOINT = True
except:
    HAVE_TE_CHECKPOINT = False


IGNORE_INDEX = -100
DEFAULT_IMAGE_TOKEN_INDEX = -200
DEFAULT_VIDEO_TOKEN_INDEX = -300
IMAGE_TOKEN = "<image>"
VIDEO_TOKEN = "<video>"


class Qwen3VLVisionEncoder(MegatronModule):
    """
    Qwen3-VL Vision Encoder that wraps HuggingFace Qwen3-VL vision components.

    This module loads the vision encoder from a HuggingFace Qwen3-VL model
    and provides the interface expected by Megatron-LM.
    """

    def __init__(
        self,
        config: TransformerConfig,
        hf_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        freeze_vision: bool = False,
        img_h: int = 384,
        img_w: int = 384,
    ):
        super().__init__(config=config)

        self.config = config
        self.img_h = img_h
        self.img_w = img_w
        self.freeze_vision = freeze_vision

        # Load HuggingFace Qwen3-VL model to extract vision components
        self._load_hf_vision_model(hf_model_name)

    def _load_hf_vision_model(self, hf_model_name: str):
        """Load vision encoder from HuggingFace model."""
        try:
            from transformers import AutoConfig, AutoModelForVision2Seq, AutoProcessor

            # Load config first to get vision config parameters
            hf_config = AutoConfig.from_pretrained(
                hf_model_name,
                trust_remote_code=True,
            )

            # Load the full model to extract vision components
            # Use AutoModelForVision2Seq for proper Qwen3-VL loading
            hf_model = AutoModelForVision2Seq.from_pretrained(
                hf_model_name,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )

            # Extract vision encoder (visual module)
            self.visual = hf_model.visual
            self.hidden_size = hf_config.vision_config.hidden_size

            # Store vision config for preprocessing
            self.patch_size = hf_config.vision_config.patch_size
            self.temporal_patch_size = hf_config.vision_config.temporal_patch_size
            self.spatial_merge_size = hf_config.vision_config.spatial_merge_size

            # Store config values needed for get_rope_index (M-RoPE position IDs)
            self.image_token_id = hf_config.image_token_id
            self.video_token_id = hf_config.video_token_id
            self.vision_start_token_id = hf_config.vision_start_token_id

            # Clean up the rest of the model to free memory
            del hf_model.model  # Delete language model
            del hf_model

            # Replace Conv3d patch_embed with equivalent Linear layer.
            # When kernel_size == stride, Conv3d produces 1x1x1 output per patch,
            # making it equivalent to a Linear layer. cuDNN has a pathological
            # slow path for this configuration (~14s vs 0.04ms for Linear).
            self._replace_patch_embed_conv3d_with_linear()

            if self.freeze_vision:
                for param in self.visual.parameters():
                    param.requires_grad = False

        except ImportError:
            raise ImportError(
                "transformers package is required for Qwen3-VL vision encoder. "
                "Install with: pip install transformers"
            )

    def _replace_patch_embed_conv3d_with_linear(self):
        """Override patch_embed forward to use Linear math instead of Conv3d.

        The Conv3d in Qwen3VLVisionPatchEmbed uses kernel_size == stride,
        so each input patch maps to exactly one output vector. This is
        mathematically identical to a Linear layer on the flattened input.
        cuDNN hits a pathological slow path for this Conv3d configuration,
        making it ~350,000x slower than the equivalent Linear.

        We keep the Conv3d module and its weight shape unchanged for checkpoint
        compatibility, but override the forward to reshape the weight and
        use F.linear instead.
        """
        patch_embed = self.visual.patch_embed
        conv3d_proj = patch_embed.proj

        def patched_forward(hidden_states: torch.Tensor) -> torch.Tensor:
            weight = conv3d_proj.weight  # (out_ch, in_ch, T, H, W)
            bias = conv3d_proj.bias
            out_features = weight.shape[0]
            # Reshape Conv3d weight to Linear weight on the fly
            return torch.nn.functional.linear(
                hidden_states.to(dtype=weight.dtype),
                weight.reshape(out_features, -1),
                bias,
            )

        patch_embed.forward = patched_forward

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass through vision encoder.

        Aligned with ms-swift's Qwen3VL_Vit approach: pass pixel_values directly to
        the HuggingFace visual encoder without custom preprocessing. The HF processor
        should have already preprocessed pixel_values into the correct format.

        Args:
            pixel_values: Preprocessed image tensor from HuggingFace processor
                         Shape: [total_patches, C * temporal_patch_size * patch_size * patch_size]
            grid_thw: Grid dimensions tensor [N, 3] for temporal, height, width

        Returns:
            image_embeds: Vision embeddings [num_tokens, hidden_size] (2D, no batch dim)
            deepstack_visual_embeds: List of intermediate visual embeddings for deepstack
        """
        # Pass pixel_values directly to HuggingFace visual encoder (like ms-swift does)
        # The HF processor should have already preprocessed the images correctly
        image_embeds = self.visual(pixel_values, grid_thw=grid_thw)

        # Handle the output format - HF visual returns (image_embeds, deepstack_visual_embeds)
        if isinstance(image_embeds, tuple):
            image_embeds, deepstack_visual_embeds = image_embeds
        else:
            deepstack_visual_embeds = []

        return image_embeds, deepstack_visual_embeds


class Qwen3VLTransformerBlock(TransformerBlock):
    """
    TransformerBlock with DeepStack support for Qwen3-VL.

    This extends the standard TransformerBlock to inject visual features
    at intermediate layers (DeepStack mechanism).
    """

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        rotary_pos_cos: Optional[torch.Tensor] = None,
        rotary_pos_sin: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        sequence_len_offset: Optional[torch.Tensor] = None,
        # DeepStack arguments
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward with DeepStack visual feature injection.

        Args:
            visual_pos_masks: Boolean mask [seq, batch] indicating visual token positions
            deepstack_visual_embeds: Visual embeddings [num_layers, num_visual_tokens, hidden]
        """

        # Standard forward through layers with DeepStack injection
        for layer_idx, layer in enumerate(self.layers):
            # Get FP8 context if needed
            inner_fp8_context = (
                get_fp8_context(self.config, layer.layer_number - 1)
                if self.config.fp8
                else nullcontext()
            )

            with inner_fp8_context:
                hidden_states, context = layer(
                    hidden_states=hidden_states,
                    attention_mask=attention_mask,
                    context=context,
                    context_mask=context_mask,
                    rotary_pos_emb=rotary_pos_emb,
                    rotary_pos_cos=rotary_pos_cos,
                    rotary_pos_sin=rotary_pos_sin,
                    attention_bias=attention_bias,
                    inference_context=inference_context,
                    packed_seq_params=packed_seq_params,
                    sequence_len_offset=sequence_len_offset,
                )

            # DeepStack: Inject visual features at intermediate layers
            layer_number = layer.layer_number - 1
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_number < deepstack_visual_embeds.shape[0]
            ):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_number],
                )

        return hidden_states, context

    def _deepstack_process(
        self,
        hidden_states: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        visual_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add visual features to hidden states at masked positions.

        Args:
            hidden_states: [seq, batch, hidden]
            visual_pos_masks: [seq, batch] boolean mask
            visual_embeds: [num_visual_tokens, hidden]
        """
        if visual_pos_masks is None or visual_embeds is None:
            return hidden_states

        visual_embeds = visual_embeds.to(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        # Add visual embeddings to positions marked by mask
        # Flatten for indexing
        flat_hidden = hidden_states.view(-1, hidden_states.shape[-1])
        flat_mask = visual_pos_masks.view(-1)

        # Add visual features (residual connection)
        flat_hidden[flat_mask] = flat_hidden[flat_mask] + visual_embeds

        return flat_hidden.view_as(hidden_states)


class Qwen3VLModel(MegatronModule):
    """
    Qwen3-VL multi-modal model for Megatron-LM.

    This model combines:
    - Qwen3-VL vision encoder (from HuggingFace)
    - Megatron-LM GPT language model
    - DeepStack mechanism for multi-layer visual feature injection

    Args:
        language_transformer_config: Transformer config for language model
        language_transformer_layer_spec: Language model layer specification
        language_vocab_size: Vocabulary size
        language_max_sequence_length: Maximum sequence length
        vision_config: Vision encoder configuration
        hf_model_name: HuggingFace model name for loading vision encoder
        freeze_vision: Whether to freeze vision encoder weights
        freeze_language: Whether to freeze language model weights
        parallel_output: Keep outputs split across tensor parallel ranks
        share_embeddings_and_output_weights: Share embedding and output weights
        pre_process: Include embedding layer (pipeline parallel)
        post_process: Include output layer (pipeline parallel)
        add_encoder: Construct encoder (pipeline parallel)
        add_decoder: Construct decoder (pipeline parallel)
        img_h: Input image height
        img_w: Input image width
        image_token_index: Token ID for image token
        video_token_index: Token ID for video token
        num_deepstack_layers: Number of layers for DeepStack injection
    """

    def __init__(
        self,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: Optional[TransformerConfig] = None,
        hf_model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
        freeze_vision: bool = False,
        freeze_language: bool = False,
        freeze_lm_embedding: bool = False,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = 'rope',
        language_rotary_percent: float = 1.0,
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        img_h: int = 336,
        img_w: int = 336,
        language_rotary_base: int = 10000,
        image_token_index: int = DEFAULT_IMAGE_TOKEN_INDEX,
        video_token_index: int = DEFAULT_VIDEO_TOKEN_INDEX,
        num_deepstack_layers: int = 5,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        if has_config_logger_enabled(language_transformer_config):
            log_config_to_disk(language_transformer_config, locals(), prefix=type(self).__name__)

        log_single_rank(
            logging.getLogger(__name__),
            logging.WARNING,
            "Qwen3VLModel is under development. Features may be missing.",
        )

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        self.vp_stage = vp_stage

        self.image_token_index = image_token_index
        self.video_token_index = video_token_index
        self.num_deepstack_layers = num_deepstack_layers
        self.img_h = img_h
        self.img_w = img_w
        self.freeze_vision = freeze_vision
        self.freeze_language = freeze_language
        self.freeze_lm_embedding = freeze_lm_embedding

        self.encoder_hidden_state = None
        self.visual = None
        self.language_model = None

        # Vision Encoder
        if add_encoder:
            vision_config = vision_transformer_config or language_transformer_config
            self.visual = Qwen3VLVisionEncoder(
                config=vision_config,
                hf_model_name=hf_model_name,
                freeze_vision=freeze_vision,
                img_h=img_h,
                img_w=img_w,
            )
            # Override default token indices with actual values from HuggingFace config
            # (stored in vision encoder during _load_hf_vision_model)
            if hasattr(self.visual, 'image_token_id'):
                self.image_token_index = self.visual.image_token_id
                log_single_rank(
                    logging.getLogger(__name__),
                    logging.INFO,
                    f"Using image_token_index from HF config: {self.image_token_index}",
                )
            if hasattr(self.visual, 'video_token_id'):
                self.video_token_index = self.visual.video_token_id

        # Language Model (GPT)
        if add_decoder:
            self.language_model = GPTModel(
                config=language_transformer_config,
                transformer_layer_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type=language_position_embedding_type,
                rotary_percent=language_rotary_percent,
                rotary_base=language_rotary_base,
                pre_process=pre_process,
                post_process=post_process,
                pg_collection=pg_collection,
            )

            if freeze_language:
                for param in self.language_model.parameters():
                    param.requires_grad = False

            if freeze_lm_embedding:
                self.freeze_embedding()

    def set_input_tensor(self, input_tensor: torch.Tensor) -> None:
        """Set input tensor for pipeline parallelism."""
        if self.add_decoder:
            self.language_model.set_input_tensor(input_tensor)

    def freeze(
        self,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
    ):
        """Freeze model components."""
        if freeze_language_model and self.language_model is not None:
            for param in self.language_model.parameters():
                param.requires_grad = False

        if freeze_vision_model and self.visual is not None:
            for param in self.visual.parameters():
                param.requires_grad = False

    def freeze_embedding(self):
        """Freeze language model embedding layer weights.

        The embedding table is not quantized, so freezing it during QAD
        avoids unnecessary drift and saves memory (no gradient storage).
        """
        if self.language_model is not None and hasattr(self.language_model, 'embedding'):
            for param in self.language_model.embedding.parameters():
                param.requires_grad = False
            self.freeze_lm_embedding = True

    def _get_image_embeddings(
        self,
        images: torch.Tensor,
        grid_thw: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Get image embeddings from vision encoder."""
        if self.visual is None:
            return None, []

        image_embeds, deepstack_embeds = self.visual(images, grid_thw)
        return image_embeds, deepstack_embeds

    def _preprocess_data(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor],
        deepstack_embeds: Optional[List[torch.Tensor]],
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Preprocess data by inserting image embeddings at image token positions.

        For Qwen3-VL, this uses masked_scatter to replace image placeholder tokens
        with image embeddings in-place, keeping the sequence length unchanged.

        Args:
            input_ids: Input token IDs [batch, seq]
            position_ids: Position IDs [batch, seq]
            image_embeds: Image embeddings [total_patches, hidden]
            deepstack_embeds: List of intermediate embeddings for deepstack
            labels: Target labels [batch, seq] (optional)
            loss_mask: Loss mask [batch, seq] (optional)

        Returns:
            combined_embeddings: Text + image embeddings [seq, batch, hidden]
            position_ids: Position IDs (unchanged) [batch, seq]
            visual_pos_masks: Mask for visual token positions [seq, batch]
            deepstack_embeds: Passthrough
            labels: Labels (unchanged)
            loss_mask: Loss mask (unchanged)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Get text embeddings first
        # Replace image/video tokens with 0 to avoid embedding lookup errors
        input_ids_text = input_ids.clone()
        input_ids_text[input_ids_text == self.image_token_index] = 0
        input_ids_text[input_ids_text == self.video_token_index] = 0

        text_embeds = self.language_model.embedding(
            input_ids=input_ids_text,
            position_ids=position_ids,
        )
        # text_embeds: [seq, batch, hidden] or [seq/tp, batch, hidden] if sequence parallel

        # If no images, just return text embeddings as-is
        if image_embeds is None:
            # Debug: commented out
            # if parallel_state.get_data_parallel_rank() == 0:
            #     print(f"  [_preprocess_data] image_embeds is None, returning early")
            return text_embeds, position_ids, None, deepstack_embeds, labels, loss_mask

        # If sequence parallelism is enabled, gather the full sequence first
        language_config = self.language_model.config
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if language_config.sequence_parallel and tp_size > 1:
            text_embeds = tensor_parallel.gather_from_sequence_parallel_region(
                text_embeds, tensor_parallel_output_grad=True
            )
        # text_embeds: [seq, batch, hidden] (full sequence)

        # Transpose to [batch, seq, hidden] for masked_scatter
        inputs_embeds = text_embeds.transpose(0, 1).contiguous()

        # Create masks for image and video tokens
        # input_ids: [batch, seq]
        image_mask = (input_ids == self.image_token_index)
        video_mask = (input_ids == self.video_token_index)
        visual_mask = image_mask | video_mask

        # Expand mask to match embedding dimensions: [batch, seq, hidden]
        embed_dim = inputs_embeds.shape[-1]
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds)
        video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds)

        # image_embeds is [total_embeddings, hidden] or [seq, batch, hidden]
        # For Qwen3-VL: total_embeddings should equal num_image_tokens (1:1 mapping)
        if image_embeds.dim() == 2:
            # [total_embeddings, hidden] -> use directly
            image_embeds_flat = image_embeds.to(device=device, dtype=inputs_embeds.dtype)
        else:
            # [seq, batch, hidden] -> flatten
            image_embeds_flat = image_embeds.transpose(0, 1).reshape(-1, embed_dim)
            image_embeds_flat = image_embeds_flat.to(device=device, dtype=inputs_embeds.dtype)

        # Use masked_scatter to replace image token embeddings with image embeddings
        # masked_scatter replaces True positions with values from the source tensor
        # Use masked_scatter to replace image token embeddings with image embeddings.
        # masked_scatter consumes exactly as many values as True positions in the mask,
        # so no explicit count/slice is needed. We call it unconditionally (no-op when
        # mask is all-False) to avoid .sum().item() or .any() CPU-GPU sync stalls.
        inputs_embeds = inputs_embeds.masked_scatter(
            image_mask_expanded,
            image_embeds_flat,
        )
            # Debug (commented out)
            # if parallel_state.get_data_parallel_rank() == 0:
            #     after_scatter = inputs_embeds[image_mask]
            #     print(f"  [_preprocess_data] After scatter - mean: {after_scatter.mean().item():.6f}, std: {after_scatter.std().item():.6f}")
            #     print(f"  [_preprocess_data] masked_scatter done, inputs_embeds shape: {inputs_embeds.shape}")

        # Handle video embeddings similarly if present (for future use)
        # Currently treating all visual tokens as images

        # Transpose back to [seq, batch, hidden]
        combined_embeds = inputs_embeds.transpose(0, 1).contiguous()

        # Create visual position mask [seq, batch]
        visual_pos_mask = visual_mask.transpose(0, 1).contiguous()

        # Labels and loss_mask don't change since sequence length is preserved
        return combined_embeds, position_ids, visual_pos_mask, deepstack_embeds, labels, loss_mask

    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        grid_thw: Optional[torch.Tensor] = None,
        inference_context: Optional[BaseInferenceContext] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        extra_block_kwargs: Optional[dict] = None,
        runtime_gather_output: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            images: Input images [batch, channels, height, width]
            input_ids: Input token IDs [batch, seq_length]
            position_ids: Position IDs [batch, seq_length]
            attention_mask: Attention mask
            labels: Target labels for loss computation
            loss_mask: Mask for loss computation
            grid_thw: Grid dimensions for dynamic resolution
            inference_context: Inference context for generation
            packed_seq_params: Parameters for sequence packing
            extra_block_kwargs: Extra arguments for transformer blocks
            runtime_gather_output: Whether to gather output at runtime

        Returns:
            output: Model output (logits or loss depending on mode)
        """
        # Get image embeddings
        image_embeds = None
        deepstack_embeds = None

        # Debug (commented out)
        # import sys
        # print(f"  [Qwen3VLModel.forward] add_encoder={self.add_encoder}, images is None={images is None}", flush=True)
        # if images is not None:
        #     print(f"  [Qwen3VLModel.forward] images.shape={images.shape}", flush=True)

        if self.add_encoder and images is not None:
            # Skip saving activations when vision encoder is frozen (saves memory)
            if self.freeze_vision:
                with torch.no_grad():
                    image_embeds, deepstack_embeds = self._get_image_embeddings(images, grid_thw)
            else:
                image_embeds, deepstack_embeds = self._get_image_embeddings(images, grid_thw)

            # Convert deepstack to tensor if list
            if deepstack_embeds and isinstance(deepstack_embeds, list):
                deepstack_embeds = torch.stack(deepstack_embeds, dim=0)

        # Preprocess data: replace image placeholder embeddings with actual image embeddings
        # Uses masked_scatter for in-place replacement, sequence length unchanged
        # When LM embedding is frozen, skip saving activations (no trainable ops in _preprocess_data)                                                                       
        if self.freeze_lm_embedding:                                                
            with torch.no_grad():                                                   
                combined_embeds, position_ids, visual_pos_masks, deepstack_embeds, new_labels, new_loss_mask = self._preprocess_data(                                     
                    input_ids=input_ids,                                            
                    position_ids=position_ids,
                    image_embeds=image_embeds,
                    deepstack_embeds=deepstack_embeds,
                    labels=labels,
                    loss_mask=loss_mask,
                )                                                               
        else:   
            combined_embeds, position_ids, visual_pos_masks, deepstack_embeds, new_labels, new_loss_mask = self._preprocess_data(
                input_ids=input_ids,
                position_ids=position_ids,
                image_embeds=image_embeds,
                deepstack_embeds=deepstack_embeds,
                labels=labels,
                loss_mask=loss_mask,
            )

        # If sequence parallelism is enabled AND we had images, scatter combined_embeds back to SP format.
        # _preprocess_data gathered embeddings to do masked_scatter, now scatter back for TransformerBlock.
        language_config = self.language_model.config
        has_images = (self.add_encoder and images is not None)
        tp_size = parallel_state.get_tensor_model_parallel_world_size()
        if has_images and language_config.sequence_parallel and tp_size > 1:
            # Sequence length is unchanged (no expansion), should already be divisible by TP
            # but pad if necessary just in case
            seq_len = combined_embeds.shape[0]  # [seq, batch, hidden]
            if seq_len % tp_size != 0:
                pad_len = tp_size - (seq_len % tp_size)
                padding = torch.zeros(
                    pad_len, combined_embeds.shape[1], combined_embeds.shape[2],
                    dtype=combined_embeds.dtype, device=combined_embeds.device
                )
                combined_embeds = torch.cat([combined_embeds, padding], dim=0)
            combined_embeds = tensor_parallel.scatter_to_sequence_parallel_region(combined_embeds)

        # Note: DeepStack feature (visual_pos_masks, deepstack_visual_embeds) requires
        # custom TransformerBlock. Standard GPTModel doesn't support it, so we skip it for now.
        # TODO: Implement DeepStack with custom Qwen3VLTransformerBlock when needed.

        # Forward through language model
        # Note: We pass embeddings directly instead of input_ids
        # Labels and loss_mask are unchanged since sequence length is preserved

        # Debug: Check position_ids before passing to language_model (commented out)
        # import os
        # if os.environ.get("TEST_BEFORE_QUANT") == "1" and position_ids is not None and position_ids.dim() == 3:
        #     print(f"  [Qwen3VLModel.forward] Before language_model call:")
        #     print(f"    position_ids[0,0,:10] (temporal): {position_ids[0,0,:10].tolist()}")
        #     print(f"    position_ids[1,0,:10] (height): {position_ids[1,0,:10].tolist()}")
        #     print(f"    position_ids[2,0,:10] (width): {position_ids[2,0,:10].tolist()}")

        # Fix: When LM embedding is frozen, combined_embeds has requires_grad=False.
        # Activation checkpointing (te_checkpoint / CheckpointFunction) requires the input
        # to have requires_grad=True to properly propagate gradients to decoder layer parameters.
        # Without this, gradients stop at the checkpoint boundary and decoder weights get zero updates.
        if self.freeze_lm_embedding and combined_embeds.requires_grad is False:
            combined_embeds = combined_embeds.detach().requires_grad_(True)

        output = self.language_model(
            input_ids=None,  # Use decoder_input instead
            decoder_input=combined_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=new_labels,
            inference_context=inference_context,
            packed_seq_params=packed_seq_params,
            extra_block_kwargs=extra_block_kwargs,
            runtime_gather_output=runtime_gather_output,
        )

        return output, new_loss_mask

    def sharded_state_dict(
        self,
        prefix: str = '',
        sharded_offsets: tuple = (),
        metadata: Optional[dict] = None,
    ):
        """Get sharded state dict for distributed checkpointing.

        This properly wraps vision encoder parameters with ShardedTensor
        to support torch_dist checkpoint format for TP resharding.
        """
        from megatron.core.transformer.utils import make_sharded_tensors_for_checkpoint

        sharded_state_dict = {}

        if self.visual is not None:
            vision_prefix = f'{prefix}visual.'
            # Vision model parameters are replicated (not TP-sharded)
            # Use make_sharded_tensors_for_checkpoint to properly wrap them
            # for distributed checkpoint format (torch_dist)
            vision_state_dict = {}
            for name, param in self.visual.named_parameters():
                vision_state_dict[name] = param

            # Wrap vision parameters as replicated sharded tensors
            # tensor_parallel_layers_axis_map=None means all tensors are replicated
            vision_sharded_sd = make_sharded_tensors_for_checkpoint(
                vision_state_dict,
                vision_prefix,
                tensor_parallel_layers_axis_map=None,  # No TP sharding for vision
                sharded_offsets=sharded_offsets,
            )
            sharded_state_dict.update(vision_sharded_sd)

        if self.language_model is not None:
            language_prefix = f'{prefix}language_model.'
            language_sharded_sd = self.language_model.sharded_state_dict(
                prefix=language_prefix,
                sharded_offsets=sharded_offsets,
                metadata=metadata,
            )
            sharded_state_dict.update(language_sharded_sd)

        return sharded_state_dict
