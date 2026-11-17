# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/53c20f81cb575f19af34ff0ae1749fb13b07999b/src/megatron/bridge/models/nemotron_omni/modeling_nemotron_omni.py
# Source license: Apache-2.0.

"""Nemotron Omni model for processor-expanded multimodal token sequences.

Unlike MCore ``LLaVAModel``, it does
not collapse a run of image placeholders before the model and reconstruct it
inside the model. The processor-provided sequence already contains one image
placeholder per projected RADIO feature. This model replaces those positions
in place. Packing is completed by the collator; the model only applies
context-parallel sharding after media insertion.

The historical collapse/expand implementation remains available explicitly as
``NemotronOmniLlavaModel`` for compatibility with existing checkpoints, but it
is not the canonical model selected by AutoBridge.
"""

import logging
from collections import namedtuple
from typing import Optional

import torch

from megatron.bridge.models.common.base import ModelConfig
from megatron.bridge.models.logit_dtype import logit_dtype_kwarg
from megatron.bridge.utils.packed_seq_utils import get_packed_seq_cp_partition_indices
from megatron.core import tensor_parallel
from megatron.core.fp8_utils import get_fp8_align_size
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.models.multimodal.context_parallel import (
    gather_from_context_parallel_ranks_dynamic_res,
    split_to_context_parallel_ranks_dynamic_res,
)
from megatron.core.models.multimodal.llava_model import pixel_shuffle
from megatron.core.models.vision.multimodal_projector import MultimodalProjector
from megatron.core.models.vision.radio import RADIOViTModel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def _ignore_transformer_engine_extra_state(
    module: torch.nn.Module, incompatible_keys: namedtuple
) -> None:
    """Allow checkpoints produced before Transformer Engine added extra state."""

    del module
    for keys in incompatible_keys._asdict().values():
        for key in keys[::-1]:
            if "extra_state" in key:
                logging.getLogger(__name__).warning(
                    "Ignoring Transformer Engine checkpoint key %s", key
                )
                keys.remove(key)


def _build_vision_packed_seq_params(imgs_sizes: torch.Tensor, patch_dim: int) -> PackedSeqParams:
    """Build RADIO's per-image THD boundaries from image sizes."""

    sizes = imgs_sizes.tolist()
    sequence_lengths = [
        (int(height) // patch_dim) * (int(width) // patch_dim) for height, width in sizes
    ]
    cumulative_lengths = [0]
    for sequence_length in sequence_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + sequence_length)

    cu_seqlens = torch.tensor(cumulative_lengths, dtype=torch.int32, device=imgs_sizes.device)
    max_seqlen = max(sequence_lengths, default=0)
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _pixel_shuffle_dynamic_resolution(
    features: torch.Tensor, *, height: int, width: int
) -> torch.Tensor:
    """Group each spatial 2x2 patch block into the channel dimension.

    A plain reshape groups four adjacent elements in the flattened sequence,
    which is not the same operation for a row-major non-square patch grid.
    Keep the spatial permutation identical to the historical Omni LLaVA path
    and the HF/vLLM implementation.
    """

    if features.ndim != 3:
        raise ValueError(f"Expected [batch, patches, hidden] features, got {tuple(features.shape)}")
    if height * width != features.shape[1]:
        raise ValueError(
            f"Patch grid {height}x{width} does not match sequence length {features.shape[1]}"
        )
    if height % 2 or width % 2:
        raise ValueError(f"Pixel shuffle requires an even patch grid, got {height}x{width}")

    batch, _, hidden = features.shape
    shuffled = features.reshape(batch, height, width, hidden)
    shuffled = shuffled.reshape(batch, height, width // 2, hidden * 2)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    shuffled = shuffled.reshape(batch, width // 2, height // 2, hidden * 4)
    shuffled = shuffled.permute(0, 2, 1, 3).contiguous()
    return shuffled.reshape(batch, (height * width) // 4, hidden * 4)


def _pad_patch_grid_to_even(
    features: torch.Tensor, *, height: int, width: int
) -> tuple[torch.Tensor, int, int]:
    """Zero-pad a patch grid to even extents so pixel shuffle can consume it.

    Only the 1x1 placeholder images that the context-parallel split injects to
    keep every rank non-empty reach this path; real Omni frames are already a
    multiple of two in both dimensions.
    """

    if height % 2 == 0 and width % 2 == 0:
        return features, height, width

    padded_height = height + height % 2
    padded_width = width + width % 2
    grid = features.reshape(height, width, features.shape[-1])
    grid = torch.nn.functional.pad(grid, (0, 0, 0, padded_width - width, 0, padded_height - height))
    return grid.reshape(1, padded_height * padded_width, -1), padded_height, padded_width


def _postprocess_radio_embeddings(
    encoded: torch.Tensor,
    imgs_sizes: torch.Tensor,
    *,
    patch_dim: int,
    class_tokens: int,
    pad_patch_grid: bool = False,
) -> torch.Tensor:
    """Remove per-image class tokens and pixel-shuffle dynamic RADIO patch grids.

    Context-parallel shards may contain odd-sized placeholder grids, which
    require padding before pixel shuffle. Unsharded inputs retain strict grid
    validation.
    """
    sizes = [(int(height), int(width)) for height, width in imgs_sizes.tolist()]
    patch_counts = [(height // patch_dim) * (width // patch_dim) for height, width in sizes]
    chunks = torch.split(
        encoded.squeeze(0), [count + class_tokens for count in patch_counts], dim=0
    )
    shuffled = []
    for chunk, (height, width) in zip(chunks, sizes):
        features = chunk[class_tokens:].unsqueeze(0)
        grid_height = height // patch_dim
        grid_width = width // patch_dim
        if pad_patch_grid:
            features, grid_height, grid_width = _pad_patch_grid_to_even(
                features, height=grid_height, width=grid_width
            )
        shuffled.append(
            _pixel_shuffle_dynamic_resolution(
                features, height=grid_height, width=grid_width
            ).squeeze(0)
        )
    return torch.cat(shuffled, dim=0)


def _project_multimodal_embeddings(
    projection: torch.nn.Module, embeddings: torch.Tensor
) -> torch.Tensor:
    """Project media rows, padding only the temporary FP8 compute input."""
    input_shape = embeddings.shape[:-1]
    flat_embeddings = embeddings.reshape(-1, 1, embeddings.shape[-1])
    num_embeddings = flat_embeddings.shape[0]
    projection_config = getattr(projection, "config", None)
    if getattr(projection_config, "fp8", None):
        alignment = get_fp8_align_size(projection_config.fp8_recipe)
        padding = -num_embeddings % alignment
        if padding:
            flat_embeddings = torch.cat(
                (
                    flat_embeddings,
                    flat_embeddings.new_zeros((padding, 1, flat_embeddings.shape[-1])),
                ),
                dim=0,
            )

    projected = projection(flat_embeddings)[:num_embeddings]
    return projected.reshape(*input_shape, projected.shape[-1])


def patchify_dynamic_images(
    images: torch.Tensor, imgs_sizes: torch.Tensor, *, patch_dim: int
) -> torch.Tensor:
    """Convert padded processor pixels to RADIO's packed patch representation.

    Accepts ``[num_images, channels, padded_height, padded_width]`` pixels and
    returns ``[1, total_patches, channels * patch_dim**2]``. Already-patchified
    inputs (``[total_patches, C*P*P]`` or ``[1, total_patches, C*P*P]``) pass
    through unchanged.
    """

    patch_features = 3 * patch_dim * patch_dim

    if images.ndim == 2 and images.shape[-1] == patch_features:
        return images.unsqueeze(0)
    if images.ndim == 3 and images.shape[0] == 1:
        if images.shape[-1] != patch_features:
            raise ValueError(
                f"Patchified RADIO input has the wrong feature width: expected {patch_features}, got {images.shape[-1]}."
            )
        return images
    if images.ndim != 4:
        raise ValueError(
            "Dynamic-resolution RADIO input must be padded pixels [N,C,H,W] "
            "or packed patches [1,total_patches,C*P*P]; "
            f"got shape {tuple(images.shape)}."
        )
    if images.shape[0] != imgs_sizes.shape[0]:
        raise ValueError(
            f"Received {images.shape[0]} images but {imgs_sizes.shape[0]} image sizes."
        )

    patches = []
    for image, size in zip(images, imgs_sizes):
        height, width = (int(value) for value in size.tolist())
        if height % patch_dim or width % patch_dim:
            raise ValueError(
                f"Image size {(height, width)} is not divisible by patch_dim={patch_dim}."
            )
        image = image[:, :height, :width]
        channels = image.shape[0]
        rows = height // patch_dim
        columns = width // patch_dim
        image_patches = (
            image.reshape(channels, rows, patch_dim, columns, patch_dim)
            .permute(1, 3, 0, 2, 4)
            .reshape(rows * columns, channels * patch_dim * patch_dim)
        )
        patches.append(image_patches)
    return torch.cat(patches, dim=0).unsqueeze(0).contiguous()


class Nemotron35SuperVLModel(MegatronModule):
    """Nemotron Omni model whose input sequence is already media-expanded.

    The collator supplies either a dense batch or a complete MCore THD stream.
    Media insertion is one-for-one and therefore length preserving. With
    context parallelism, the model inserts media into the full stream and then
    selects the rank-local CP shard without changing packed metadata.

    Image, video, sound, and text inputs use the same one-feature-per-placeholder
    contract.
    """

    model_owns_packing = False
    model_owns_mtp_loss_mask_packing = False
    model_slices_context_parallel_inputs = True

    def __init__(
        self,
        *,
        language_transformer_config: TransformerConfig,
        language_transformer_layer_spec: ModuleSpec,
        language_vocab_size: int,
        language_max_sequence_length: int,
        vision_transformer_config: TransformerConfig,
        vision_transformer_layer_spec: ModuleSpec,
        vision_projection_config: TransformerConfig,
        vision_projection_layer_spec: ModuleSpec,
        image_token_index: int,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        language_position_embedding_type: str = "rope",
        pre_process: bool = True,
        post_process: bool = True,
        add_encoder: bool = True,
        add_decoder: bool = True,
        hybrid_layer_pattern: Optional[str] = None,
        img_h: int = 512,
        img_w: int = 512,
        patch_dim: int = 16,
        dynamic_resolution: bool = True,
        vision_class_token_len: int = 10,
        radio_force_eval_mode: bool = False,
        radio_force_cpe_eval_mode: bool = False,
        radio_interpolate_only_cpe: bool = False,
        radio_cpe_aspect_ratio_select: bool = False,
        radio_disable_cpe: bool = False,
        temporal_patch_dim: int = 1,
        separate_video_embedder: bool = False,
        temporal_ckpt_compat: bool = False,
        vision_dp_over_cp: bool = False,
        sound_model: Optional[torch.nn.Module] = None,
        sound_projection: Optional[torch.nn.Module] = None,
        sound_token_index: int = 0,
        pg_collection: Optional[ProcessGroupCollection] = None,
        vp_stage: Optional[int] = None,
        model_config: Optional[ModelConfig] = None,
    ) -> None:
        super().__init__(config=language_transformer_config)

        self.pre_process = pre_process
        self.post_process = post_process
        self.add_encoder = add_encoder
        self.add_decoder = add_decoder
        # Inference controllers inspect the top-level model for the padded
        # vocabulary size. Keep it consistent with the nested HybridModel.
        self.vocab_size = language_vocab_size
        self.image_token_index = image_token_index
        self.sound_token_index = sound_token_index
        self.patch_dim = patch_dim
        self.dynamic_resolution = dynamic_resolution
        self.sequence_parallel_lm = language_transformer_config.sequence_parallel
        self.context_parallel_lm = language_transformer_config.context_parallel_size
        self.vision_dp_over_cp = vision_dp_over_cp and self.context_parallel_lm > 1
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights
        self.encoder_hidden_state = None

        if pg_collection is None:
            pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        self.pg_collection = pg_collection

        self.language_model = None
        if add_decoder:
            # A builder-backed model config owns logit_dtype, which a provider carries on
            # language_transformer_config.
            logit_dtype = (
                language_transformer_config if model_config is None else model_config
            ).logit_dtype
            self.language_model = HybridModel(
                config=language_transformer_config,
                hybrid_stack_spec=language_transformer_layer_spec,
                vocab_size=language_vocab_size,
                max_sequence_length=language_max_sequence_length,
                parallel_output=parallel_output,
                **logit_dtype_kwarg(HybridModel, logit_dtype),
                position_embedding_type=language_position_embedding_type,
                pre_process=pre_process,
                hybrid_layer_pattern=hybrid_layer_pattern,
                post_process=post_process,
                scatter_embedding_sequence_parallel=False,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.language_model.register_load_state_dict_post_hook(
                _ignore_transformer_engine_extra_state
            )

        self.vision_model = None
        self.vision_projection = None
        if add_encoder:
            self.vision_model = RADIOViTModel(
                vision_transformer_config,
                vision_transformer_layer_spec,
                img_h=img_h,
                img_w=img_w,
                max_img_h=2048,
                max_img_w=2048,
                class_token_len=vision_class_token_len,
                patch_dim=patch_dim,
                add_class_token=True,
                embedder_bias=False,
                dynamic_resolution=dynamic_resolution,
                force_eval_mode=radio_force_eval_mode,
                force_cpe_eval_mode=radio_force_cpe_eval_mode,
                interpolate_only_cpe=radio_interpolate_only_cpe,
                cpe_aspect_ratio_select=radio_cpe_aspect_ratio_select,
                has_cpe=not radio_disable_cpe,
                temporal_patch_dim=temporal_patch_dim,
                separate_video_embedder=separate_video_embedder,
                temporal_ckpt_compat=temporal_ckpt_compat,
                pg_collection=pg_collection,
                vp_stage=vp_stage,
            )
            self.vision_projection = MultimodalProjector(
                vision_projection_config,
                vision_projection_layer_spec,
                "mlp",
                vision_transformer_config.hidden_size * 4,
                tp_group=pg_collection.tp,
            )
            self.vision_model.register_load_state_dict_post_hook(
                _ignore_transformer_engine_extra_state
            )
            self.vision_projection.register_load_state_dict_post_hook(
                _ignore_transformer_engine_extra_state
            )

        # Preserve the top-level sound-module namespace used by checkpoint
        # conversion while keeping media insertion local to this model.
        self.sound_model = sound_model
        self.sound_projection = sound_projection

        self._expose_language_model_for_cuda_graph_helper()

    def _expose_language_model_for_cuda_graph_helper(self) -> None:
        """Expose language-model fields on the multimodal root when CUDA graphs are enabled.

        MCore's CUDA graph helper discovers ``decoder`` on the top-level model,
        while Nemotron Omni stores it below ``language_model``. Properties
        expose the nested modules without registering duplicate root aliases or
        changing checkpoint keys.
        """

        llm_cuda_graph_enabled = (
            self.language_model is not None
            and getattr(self.language_model.config, "cuda_graph_impl", "none") != "none"
        )
        if not llm_cuda_graph_enabled:
            return
        assert not self.language_model.config.variable_seq_lengths, (
            "Nemotron Omni with CUDA graphs requires fixed sequence lengths "
            "(variable_seq_lengths=False). Disable variable-length inputs or turn off CUDA graphs."
        )
        self.position_embedding_type = self.language_model.position_embedding_type

    @property
    def rotary_pos_emb(self):
        """Expose the nested language model's rotary embeddings to MCore helpers."""

        return getattr(self.language_model, "rotary_pos_emb", None)

    @property
    def decoder(self):
        """Expose the nested language decoder without registering a module alias."""

        return getattr(self.language_model, "decoder", None)

    def shared_embedding_or_output_weight(self):
        """Expose the language embedding for Megatron gradient finalization."""

        if self.language_model is None:
            return None
        return self.language_model.shared_embedding_or_output_weight()

    def set_input_tensor(self, input_tensor) -> None:
        """Set the pipeline input on the language model."""

        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1, "input_tensor must contain exactly one tensor"
        if self.language_model is not None:
            self.language_model.set_input_tensor(input_tensor[0])

    def freeze(
        self,
        *,
        freeze_language_model: bool = False,
        freeze_vision_model: bool = False,
        freeze_vision_projection: bool = False,
        freeze_sound_model: bool = False,
        freeze_sound_projection: bool = False,
    ) -> None:
        """Freeze selected leaf components."""

        modules = (
            (freeze_language_model, self.language_model),
            (freeze_vision_model, self.vision_model),
            (freeze_vision_projection, self.vision_projection),
            (freeze_sound_model, self.sound_model),
            (freeze_sound_projection, self.sound_projection),
        )
        for should_freeze, module in modules:
            if should_freeze and module is not None:
                for parameter in module.parameters():
                    parameter.requires_grad = False

    @staticmethod
    def _merge_projected_media(
        language_embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        media_embeddings: torch.Tensor,
        media_token_id: int,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Replace each valid media placeholder with exactly one feature row.

        ``attention_mask`` is a token-validity mask here, not MCore's 4-D
        causal attention mask.  Requiring an exact shape match prevents a
        causal mask from broadcasting the placeholder mask to ``[B, 1, S, S]``.
        """

        media_mask = input_ids == media_token_id
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(attention_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )
            media_mask = media_mask & attention_mask.bool()

        expected_features = int(media_mask.sum().item())
        actual_features = media_embeddings.shape[0]
        if expected_features != actual_features:
            raise ValueError(
                "Expanded-sequence media alignment failed: "
                f"found {expected_features} valid placeholders for "
                f"{actual_features} projected features. Nemotron35SuperVLModel requires the processor "
                "to emit one placeholder for every projected media token before collator-owned packing. "
                "A single placeholder for multiple features usually means this batch was prepared "
                "for the legacy LLaVAModel collapse/expand path. Use expanded processor output, or "
                "load the checkpoint through the explicit NemotronOmniLlavaModelProvider/"
                "NemotronOmniLlavaBridge compatibility path."
            )

        # With batch size one, transpose can remain contiguous and
        # ``contiguous()`` may return the original autograd view. Media
        # insertion is in-place, so force independent storage for the merge.
        merged = language_embeddings.transpose(0, 1).clone()
        if actual_features > 0:
            merged[media_mask] = media_embeddings.to(dtype=merged.dtype)
        return merged.transpose(0, 1).contiguous()

    def _split_images_across_context_parallel_ranks(
        self,
        images: torch.Tensor,
        imgs_sizes: torch.Tensor,
        vision_packed_seq_params: PackedSeqParams,
        *,
        num_frames: Optional[torch.Tensor],
        temporal_patch_size: int,
    ):
        """Give each CP rank an integer share of the patched images or tubelets."""

        # The MCore splitter resolves the CP group from the global parallel
        # state, so a divergent process-group collection would silently shard
        # against the wrong ranks.
        if self.pg_collection.cp.size() != self.context_parallel_lm:
            raise ValueError(
                "Nemotron Omni vision_dp_over_cp does not match its process group: "
                f"config={self.context_parallel_lm}, group={self.pg_collection.cp.size()}."
            )

        return split_to_context_parallel_ranks_dynamic_res(
            images,
            imgs_sizes,
            vision_packed_seq_params,
            patch_dim=self.patch_dim,
            fp8_enabled=False,
            fp8_recipe=getattr(self.config, "fp8_recipe", None),
            num_frames=num_frames,
            temporal_patch_size=temporal_patch_size,
        )

    def _encode_images(
        self,
        images: torch.Tensor,
        imgs_sizes: Optional[torch.Tensor],
        vision_packed_seq_params: Optional[PackedSeqParams],
        num_frames: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Encode dynamic-resolution images and return one row per image token."""

        if self.vision_model is None or self.vision_projection is None:
            raise RuntimeError("Image data was provided on a stage without the vision encoder")

        parameter = next(self.vision_model.parameters())
        images = images.to(dtype=parameter.dtype)

        if imgs_sizes is not None and imgs_sizes.numel() > 0:
            images = self._patchify_dynamic_images(images, imgs_sizes)
            if vision_packed_seq_params is None:
                vision_packed_seq_params = _build_vision_packed_seq_params(
                    imgs_sizes, self.patch_dim
                )
            use_temporal = getattr(self.vision_model, "temporal_patch_dim", 1) > 1
            if use_temporal and num_frames is None:
                raise ValueError(
                    "num_frames is required by the configured RADIO encoder; "
                    "provide one entry per image or video item."
                )

            shard_vision = self.vision_dp_over_cp
            num_padded_ranks = 0
            if shard_vision:
                (
                    images,
                    imgs_sizes,
                    vision_packed_seq_params,
                    _has_fp8_padding,
                    num_padded_ranks,
                    local_num_frames,
                ) = self._split_images_across_context_parallel_ranks(
                    images,
                    imgs_sizes,
                    vision_packed_seq_params,
                    num_frames=num_frames,
                    temporal_patch_size=self.vision_model.temporal_patch_dim if use_temporal else 1,
                )
                if local_num_frames is not None:
                    num_frames = local_num_frames

            vision_output = self.vision_model(
                images,
                imgs_sizes=imgs_sizes,
                packed_seq_params=vision_packed_seq_params,
                num_frames=num_frames,
            )
            if use_temporal:
                encoded, imgs_sizes, _ = vision_output
            else:
                encoded = vision_output
            encoded = _postprocess_radio_embeddings(
                encoded,
                imgs_sizes,
                patch_dim=self.patch_dim,
                class_tokens=(
                    self.vision_model.class_token_len
                    if getattr(self.vision_model, "add_class_token", False)
                    else 0
                ),
                pad_patch_grid=shard_vision,
            )

            if shard_vision:
                # Project before the gather so the projector stays sharded, then
                # restore the global feature set every CP rank's media merge needs.
                projected = _project_multimodal_embeddings(self.vision_projection, encoded)
                gathered = gather_from_context_parallel_ranks_dynamic_res(
                    projected, num_padded_ranks
                )
                return gathered.contiguous()
        else:
            encoded = self.vision_model(images)
            class_tokens = self.vision_model.class_token_len
            encoded = encoded[:, class_tokens:, :]
            encoded = pixel_shuffle(encoded).reshape(-1, encoded.shape[-1] * 4)

        return _project_multimodal_embeddings(self.vision_projection, encoded).contiguous()

    def _encode_sound(
        self, sound_clips: torch.Tensor, sound_length: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Encode mel features and return valid projected rows in sample order."""

        if self.sound_model is None or self.sound_projection is None:
            raise RuntimeError("Sound data was provided on a stage without the sound encoder")
        if sound_length is None:
            raise ValueError("sound_length is required when sound_clips are provided.")
        if sound_clips.ndim < 2:
            raise ValueError(
                f"sound_clips must include batch and frame dimensions, got {tuple(sound_clips.shape)}."
            )

        parameter = next(self.sound_model.parameters())
        sound_clips = sound_clips.to(dtype=parameter.dtype)
        sound_embeddings, embedding_lengths = self.sound_model(sound_clips, sound_length)
        if sound_embeddings.ndim != 3:
            raise ValueError(
                "The sound encoder must return [batch, sequence, hidden] embeddings, "
                f"got {tuple(sound_embeddings.shape)}."
            )
        if embedding_lengths.numel() != sound_embeddings.shape[0]:
            raise ValueError(
                "The sound encoder must return one valid embedding length per sample; "
                f"got {embedding_lengths.numel()} lengths for batch size {sound_embeddings.shape[0]}."
            )

        projection_parameter = next(self.sound_projection.parameters(), None)
        if projection_parameter is not None:
            sound_embeddings = sound_embeddings.to(dtype=projection_parameter.dtype)
        projected = _project_multimodal_embeddings(
            self.sound_projection, sound_embeddings.permute(1, 0, 2).contiguous()
        ).contiguous()
        projected_by_sample = projected.permute(1, 0, 2)
        if getattr(getattr(self.sound_model, "config", None), "sound_pad_to_clip_duration", False):
            return projected_by_sample.reshape(-1, projected.shape[-1]).contiguous()

        valid_embeddings = []
        for sample_embeddings, embedding_length in zip(
            projected_by_sample, embedding_lengths, strict=True
        ):
            valid_length = int(embedding_length.item())
            if valid_length < 0 or valid_length > sample_embeddings.shape[0]:
                raise ValueError(
                    f"Sound embedding length {valid_length} is outside encoded width {sample_embeddings.shape[0]}."
                )
            valid_embeddings.append(sample_embeddings[:valid_length])
        if not valid_embeddings:
            return projected.new_empty((0, projected.shape[-1]))
        return torch.cat(valid_embeddings, dim=0).contiguous()

    def _patchify_dynamic_images(
        self, images: torch.Tensor, imgs_sizes: torch.Tensor
    ) -> torch.Tensor:
        """Convert padded processor pixels to RADIO's packed patch representation.

        The processor emits ``[num_images, channels, padded_height, padded_width]``.
        RADIO's dynamic-resolution path consumes
        ``[1, total_patches, channels * patch_dim**2]``. Keeping this conversion
        here makes raw media tensors part of the model contract and avoids an
        Omni-only NeMo-RL pre-forward adapter. Already-patchified inputs remain
        accepted for Bridge/SFT callers.
        """

        return patchify_dynamic_images(images, imgs_sizes, patch_dim=self.patch_dim)

    @staticmethod
    def _select_sequence(
        tensor: Optional[torch.Tensor], index: torch.Tensor, *, dim: int
    ) -> Optional[torch.Tensor]:
        """Select one CP shard from a token-aligned tensor."""

        if tensor is None:
            return None
        return tensor.index_select(dim, index.to(device=tensor.device))

    def _context_parallel_index(
        self,
        *,
        packed_seq_params: Optional[PackedSeqParams],
        total_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build this rank's CP index without changing sequence metadata."""

        cp_size = self.context_parallel_lm
        if cp_size <= 1:
            return None

        if self.pg_collection.cp.size() != cp_size:
            raise ValueError(
                "Nemotron Omni context-parallel configuration does not match its process group: "
                f"config={cp_size}, group={self.pg_collection.cp.size()}."
            )
        cp_rank = self.pg_collection.cp.rank()
        if packed_seq_params is not None:
            return get_packed_seq_cp_partition_indices(
                packed_seq_params,
                total_tokens=total_tokens,
                cp_size=cp_size,
                cp_rank=cp_rank,
                device=device,
                cp_group=self.pg_collection.cp,
            )

        chunks = 2 * cp_size
        if total_tokens % chunks:
            raise ValueError(
                "Dense Nemotron Omni context parallelism requires sequence length "
                f"to be divisible by 2 * CP ({chunks}); got {total_tokens}."
            )
        chunk_size = total_tokens // chunks
        first = torch.arange(
            cp_rank * chunk_size, (cp_rank + 1) * chunk_size, device=device, dtype=torch.long
        )
        mirrored_rank = chunks - cp_rank - 1
        second = torch.arange(
            mirrored_rank * chunk_size,
            (mirrored_rank + 1) * chunk_size,
            device=device,
            dtype=torch.long,
        )
        return torch.cat((first, second))

    def _apply_context_parallel_sharding(
        self,
        *,
        input_ids: Optional[torch.Tensor],
        combined_embeddings: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
        loss_mask: Optional[torch.Tensor],
        padding_mask: Optional[torch.Tensor],
        packed_seq_params: Optional[PackedSeqParams],
    ) -> tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        bool,
    ]:
        """Apply one shared CP index after length-preserving media insertion."""

        sequence_tensors = (
            (input_ids, 1),
            (combined_embeddings, 0),
            (position_ids, 1),
            (labels, 1),
            (loss_mask, 1),
            (padding_mask, 1),
        )
        full_lengths = {tensor.size(dim) for tensor, dim in sequence_tensors if tensor is not None}
        if len(full_lengths) > 1:
            raise ValueError(
                f"Nemotron Omni token-aligned tensors have inconsistent lengths: {sorted(full_lengths)}."
            )
        if not full_lengths:
            # Intermediate PP stages receive an already CP-local pipeline
            # tensor and only need the unchanged global THD metadata.
            return (
                input_ids,
                combined_embeddings,
                position_ids,
                attention_mask,
                labels,
                loss_mask,
                padding_mask,
                False,
            )

        total_tokens = full_lengths.pop()
        if packed_seq_params is not None:
            metadata_total = getattr(packed_seq_params, "total_tokens", None)
            if metadata_total is not None and int(metadata_total) != total_tokens:
                raise ValueError(
                    "Packed Nemotron Omni metadata does not match the collator-owned token stream: "
                    f"total_tokens={metadata_total}, tensor width={total_tokens}."
                )

        device = next(tensor.device for tensor, _ in sequence_tensors if tensor is not None)
        index = self._context_parallel_index(
            packed_seq_params=packed_seq_params, total_tokens=total_tokens, device=device
        )
        if index is None:
            return (
                input_ids,
                combined_embeddings,
                position_ids,
                attention_mask,
                labels,
                loss_mask,
                padding_mask,
                False,
            )

        input_ids = self._select_sequence(input_ids, index, dim=1)
        combined_embeddings = self._select_sequence(combined_embeddings, index, dim=0)
        position_ids = self._select_sequence(position_ids, index, dim=1)
        labels = self._select_sequence(labels, index, dim=1)
        loss_mask = self._select_sequence(loss_mask, index, dim=1)
        padding_mask = self._select_sequence(padding_mask, index, dim=1)

        if attention_mask is not None:
            attention_seq_dim = 1 if attention_mask.dim() == 2 else 2
            attention_mask = self._select_sequence(attention_mask, index, dim=attention_seq_dim)

        return (
            input_ids,
            combined_embeddings,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            padding_mask,
            loss_mask is not None,
        )

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        inference_context=None,
        runtime_gather_output: Optional[bool] = None,
        packed_seq_params: Optional[PackedSeqParams] = None,
        images: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        imgs_sizes: Optional[torch.Tensor] = None,
        vision_packed_seq_params: Optional[PackedSeqParams] = None,
        num_frames: Optional[torch.Tensor] = None,
        sound_clips: Optional[torch.Tensor] = None,
        sound_length: Optional[torch.Tensor] = None,
        *,
        media_token_validity_mask: torch.Tensor | None = None,
        inference_params=None,
        **kwargs,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Insert media into the expanded sequence, shard for CP, then call NemotronH.

        Returns:
            Model output, or ``(output, local_loss_mask)`` when this model
            applies a context-parallel shard to the supervision tensors.
        """

        del kwargs
        if images is None:
            images = pixel_values

        has_sound_inputs = sound_clips is not None and sound_clips.numel() > 0
        if has_sound_inputs and sound_clips.shape == torch.Size([1, 1]):
            has_sound_inputs = sound_clips[0, 0].item() != 0

        lm_input_ids = input_ids
        combined_embeddings = None
        if self.pre_process:
            if input_ids is None:
                raise ValueError("The first Nemotron Omni pipeline stage requires input_ids.")
            if images is not None and images.numel() > 0:
                image_embeddings = self._encode_images(
                    images, imgs_sizes, vision_packed_seq_params, num_frames
                )
            else:
                image_embeddings = None

            if has_sound_inputs:
                sound_embeddings = self._encode_sound(sound_clips, sound_length)
            else:
                sound_embeddings = None

            # An explicit mask from the caller wins: padding and attention masks
            # answer "is this a real token", which is a different question from
            # "is this a media anchor".  They coincide only while every media
            # token in a valid position anchors an image.  A caller whose text
            # legitimately contains the placeholder -- it is an ordinary token in
            # that vocabulary -- marks those positions here so they keep their
            # embedding instead of demanding a projected feature.
            #
            # Otherwise: MBridge collators use a 2-D attention mask as a
            # token-validity mask, while NeMo RL's dense Megatron path supplies
            # MCore's 4-D causal mask (where True means blocked).  Only the
            # former can filter media placeholders.  Padding masks are
            # unambiguous and take precedence for collator-owned packed inputs.
            if media_token_validity_mask is None:
                if padding_mask is not None:
                    media_token_validity_mask = ~padding_mask
                elif attention_mask is not None and attention_mask.dim() == input_ids.dim():
                    media_token_validity_mask = attention_mask

            if (
                media_token_validity_mask is not None
                and media_token_validity_mask.shape != input_ids.shape
            ):
                raise ValueError(
                    "The media token-validity mask must have the same shape as input_ids: "
                    f"got mask={tuple(media_token_validity_mask.shape)}, input_ids={tuple(input_ids.shape)}."
                )

            # Match LLaVAModel's execution order. Besides keeping the two
            # implementations directly comparable, this ensures that RADIO's
            # first distributed forward sees the same runtime/collective state.
            image_anchor_mask = input_ids == self.image_token_index
            if media_token_validity_mask is not None:
                image_anchor_mask &= media_token_validity_mask.bool()
            input_ids_text = input_ids.masked_fill(image_anchor_mask, 0)
            combined_embeddings = self.language_model.embedding(
                input_ids=input_ids_text, position_ids=position_ids
            )

            if image_embeddings is None:
                image_embeddings = combined_embeddings.new_empty((0, combined_embeddings.shape[-1]))

            combined_embeddings = self._merge_projected_media(
                combined_embeddings,
                input_ids,
                image_embeddings,
                self.image_token_index,
                media_token_validity_mask,
            )

            # ``sound_token_index`` is an ordinary vocabulary id.  In an
            # image-only/text-only batch it may therefore occur in generated
            # prose without denoting an audio anchor.  Audio provenance, not
            # the token id alone, decides whether sound insertion runs.
            if self.sound_token_index > 0 and has_sound_inputs:
                assert sound_embeddings is not None
                combined_embeddings = self._merge_projected_media(
                    combined_embeddings,
                    input_ids,
                    sound_embeddings,
                    self.sound_token_index,
                    media_token_validity_mask,
                )

        if packed_seq_params is not None:
            # THD tensors and their logical boundaries are final collator
            # outputs. The model may shard token-aligned tensors for CP, but
            # never rebuilds or mutates the packing metadata.
            attention_mask = None

        (
            lm_input_ids,
            combined_embeddings,
            position_ids,
            attention_mask,
            labels,
            loss_mask,
            padding_mask,
            return_sliced_loss_mask,
        ) = self._apply_context_parallel_sharding(
            input_ids=lm_input_ids,
            combined_embeddings=combined_embeddings,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            padding_mask=padding_mask,
            packed_seq_params=packed_seq_params,
        )

        language_packed_seq_params = packed_seq_params
        if (
            language_packed_seq_params is not None
            and self.context_parallel_lm == 1
            and language_packed_seq_params.cu_seqlens_q is not None
            and language_packed_seq_params.cu_seqlens_q.numel() == 2
        ):
            # A one-sample CP=1 "pack" neither concatenates sequences nor
            # shards them. Keeping PackedSeqParams here would nevertheless
            # switch Mamba to its packed-sequence kernel. Use the ordinary
            # dense path, which has the same token order and causal semantics
            # but matches vLLM prefill numerics. Multi-sample and CP runs keep
            # packed metadata so their sequence boundaries are preserved.
            language_packed_seq_params = None

        if combined_embeddings is not None and self.sequence_parallel_lm:
            combined_embeddings = tensor_parallel.scatter_to_sequence_parallel_region(
                combined_embeddings, group=self.pg_collection.tp
            ).contiguous()
        if padding_mask is not None and self.sequence_parallel_lm:
            padding_mask = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    padding_mask.transpose(0, 1).contiguous(), group=self.pg_collection.tp
                )
                .transpose(0, 1)
                .contiguous()
            )

        # Match LLaVAModel's external-embedding contract. Once media has been
        # merged into decoder embeddings, the language model must not receive
        # the pre-merge token IDs as a second input. MTP is the exception: its
        # training loss derives targets from input_ids, so retain them only
        # when MTP layers are actually enabled.
        mtp_num_layers = getattr(self.config, "mtp_num_layers", None)
        mtp_enabled = mtp_num_layers is not None and mtp_num_layers > 0
        if combined_embeddings is not None and not mtp_enabled:
            lm_input_ids = None

        output = self.language_model(
            input_ids=lm_input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=combined_embeddings,
            labels=labels,
            loss_mask=loss_mask,
            inference_context=inference_context,
            inference_params=inference_params,
            runtime_gather_output=runtime_gather_output,
            packed_seq_params=language_packed_seq_params,
            padding_mask=padding_mask,
        )
        if return_sliced_loss_mask:
            return output, loss_mask
        return output
