# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model for FSDP + EP training.

Composes a vision encoder and a ``GPTModel`` language decoder.  Designed
for FSDP + EP: always builds the **full** model on every rank (no PP
flags).  PP support is only available through the MIMO ``MimoModel``
assembly path.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig


def _cp_split_tensor(tensor, seq_dim, cp_size, cp_rank):
    """Zigzag-split *tensor* along *seq_dim* for context parallelism (BSHD).

    Splits the sequence into ``2 * cp_size`` equal chunks, then selects
    chunks ``[cp_rank, 2*cp_size - cp_rank - 1]`` and concatenates them.
    This mirrors ``megatron.core.utils.get_batch_on_this_cp_rank``.
    """
    S = tensor.shape[seq_dim]
    assert S % (2 * cp_size) == 0, (
        f"seq_len {S} not divisible by 2*cp_size={2 * cp_size}"
    )
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        2 * cp_size,
        S // (2 * cp_size),
        *tensor.shape[seq_dim + 1 :],
    )
    index = torch.zeros(2, dtype=torch.int64, device=tensor.device)
    index[0] = cp_rank
    index[1] = 2 * cp_size - cp_rank - 1
    tensor = tensor.index_select(seq_dim, index)
    tensor = tensor.view(
        *tensor.shape[:seq_dim],
        -1,
        *tensor.shape[seq_dim + 2 :],
    )
    return tensor


class _NoCPGroup:
    """Dummy size-1 process group used by the vision encoder so its
    THD RoPE ignores the language-side CP group (the language CP group
    has no business splitting variable-resolution image seqlens).
    """

    def size(self):
        return 1

    def rank(self):
        return 0


_NO_CP_GROUP = _NoCPGroup()


# Note: reported ``mtp_1 loss`` drifts ~1.3% from the CP=1 baseline under
# THD+CP. Megatron-Core's logging averages per-rank pre-divided ratios
# with op=AVG, and per-rank num_tokens are unequal after MTP rolling.
# Gradients are correct; only the *logged* value drifts.


def _thd_cp_partition_index(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Per-rank token index for THD + CP via TE's
    ``thd_get_partitioned_indices``.  Cast to int64 so the result can be
    used directly with ``index_select`` regardless of TE's return dtype.
    """
    from transformer_engine.pytorch import cpp_extensions as tex

    idx = tex.thd_get_partitioned_indices(
        cu_seqlens_padded, total_tokens, cp_size, cp_rank,
    )
    return idx.long()


class MultimodalModel(MegatronModule):
    """Base class for multimodal vision-language models.

    Composes a pre-constructed vision encoder and a ``GPTModel`` language
    decoder.  Designed for FSDP + EP; always builds the full model on
    every rank.

    Args:
        language_config: ``TransformerConfig`` for the language decoder.
        language_spec: ``ModuleSpec`` for decoder transformer layers.
        vision_encoder: Pre-constructed vision encoder module.
        vocab_size: Language model vocabulary size.
        max_sequence_length: Maximum sequence length.
        image_token_id: Token ID for image placeholder tokens.
        position_embedding_type: Position embedding type for the decoder.
        rotary_percent: Fraction of hidden dim for RoPE.
        rotary_base: Base frequency for RoPE.
        mrope_section: MRoPE channel sections.
        mtp_block_spec: Optional MTP block spec.
        parallel_output: Keep outputs split across TP ranks.
        share_embeddings_and_output_weights: Tie input/output embeddings.
    """

    def __init__(
        self,
        language_config: TransformerConfig,
        language_spec: ModuleSpec,
        vision_encoder: MegatronModule,
        vocab_size: int,
        max_sequence_length: int,
        image_token_id: int,
        position_embedding_type: str = "rope",
        rotary_percent: float = 1.0,
        rotary_base: int = 10000,
        mrope_section: list = None,
        mtp_block_spec: ModuleSpec = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
    ):
        super().__init__(config=language_config)

        self.image_token_id = image_token_id

        self.vision_model = vision_encoder
        self.language_model = GPTModel(
            config=language_config,
            transformer_layer_spec=language_spec,
            vocab_size=vocab_size,
            max_sequence_length=max_sequence_length,
            pre_process=True,
            post_process=True,
            parallel_output=parallel_output,
            share_embeddings_and_output_weights=(
                share_embeddings_and_output_weights
            ),
            position_embedding_type=position_embedding_type,
            rotary_percent=rotary_percent,
            rotary_base=rotary_base,
            mtp_block_spec=mtp_block_spec,
        )

    def set_input_tensor(self, input_tensor):
        """Route input tensors (simplified, no PP routing)."""
        if not isinstance(input_tensor, list):
            input_tensor = [input_tensor]
        assert len(input_tensor) == 1
        self.language_model.set_input_tensor(input_tensor[0])

    def _scatter_vision_embeddings(
        self,
        input_ids: Tensor,
        text_embeddings: Tensor,
        vision_embeddings: Tensor,
    ) -> Tensor:
        """Replace image-token positions with vision embeddings.

        Handles sequence parallelism (gather → scatter → re-scatter).

        Args:
            input_ids: ``[B, S]`` token IDs.
            text_embeddings: ``[S, B, D]`` (or ``[S/TP, B, D]`` with SP).
            vision_embeddings: ``[num_visual_tokens, D]``.

        Returns:
            Combined embeddings, same shape as *text_embeddings*.
        """
        sp = (
            self.config.sequence_parallel
            and parallel_state.get_tensor_model_parallel_world_size()
            > 1
        )

        if sp:
            text_embeddings = (
                tensor_parallel.gather_from_sequence_parallel_region(
                    text_embeddings, tensor_parallel_output_grad=False,
                )
            )

        combined = text_embeddings.transpose(0, 1).contiguous()
        image_mask = input_ids == self.image_token_id
        mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
        combined = combined.masked_scatter(
            mask_expanded, vision_embeddings,
        )
        combined = combined.transpose(0, 1).contiguous()

        if sp:
            combined = (
                tensor_parallel.scatter_to_sequence_parallel_region(
                    combined,
                )
            )

        return combined

    def compute_position_ids(
        self,
        input_ids: Tensor,
        image_grid_thw: Optional[Tensor] = None,
        packed_seq_params=None,
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.  ``packed_seq_params`` is
        accepted for subclass compatibility (e.g. MRoPE in THD mode).
        """
        B, S = input_ids.shape
        return (
            torch.arange(S, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
        )

    def _cp_split_for_forward(
        self,
        *,
        decoder_input,
        input_ids,
        labels,
        loss_mask,
        attention_mask,
        position_ids,
        packed_seq_params,
    ):
        """Apply CP split to model-forward inputs.

        BSHD path zigzag-splits each tensor along its seq dim.  THD path
        partitions per-sample via ``tex.thd_get_partitioned_indices`` so
        chunks line up with ``cu_seqlens_q_padded`` boundaries.
        ``position_ids`` and ``attention_mask`` are NOT split in THD —
        MRoPE returns full freqs and TE attention's
        ``_apply_rotary_pos_emb_thd`` does the per-sample CP zigzag
        itself via ``_get_thd_freqs_on_this_cp_rank``.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1:
            return (
                decoder_input, input_ids, labels, loss_mask,
                attention_mask, position_ids,
            )
        cp_rank = parallel_state.get_context_parallel_rank()

        if packed_seq_params is not None:
            total_tokens = (
                decoder_input.shape[0]
                if decoder_input is not None
                else input_ids.shape[1]
            )
            idx = _thd_cp_partition_index(
                packed_seq_params.cu_seqlens_q_padded,
                total_tokens, cp_size, cp_rank,
            )
            if decoder_input is not None:
                decoder_input = decoder_input.index_select(0, idx)
            if input_ids is not None:
                input_ids = input_ids.index_select(1, idx)
            if labels is not None:
                labels = labels.index_select(1, idx)
            if loss_mask is not None:
                loss_mask = loss_mask.index_select(1, idx)
        else:
            def _split(t, seq_dim):
                return None if t is None else _cp_split_tensor(
                    t, seq_dim=seq_dim, cp_size=cp_size, cp_rank=cp_rank,
                )
            decoder_input = _split(decoder_input, 0)
            input_ids = _split(input_ids, 1)
            labels = _split(labels, 1)
            loss_mask = _split(loss_mask, 1)
            attention_mask = _split(attention_mask, 1)

        return (
            decoder_input, input_ids, labels, loss_mask,
            attention_mask, position_ids,
        )

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        packed_seq_params=None,
        **kwargs,
    ):
        """Forward pass.

        Args:
            input_ids: ``[B, S]`` token IDs (or ``[1, T]`` in THD mode).
            position_ids: ``[3, B, S]`` for MRoPE or ``[B, S]``
                (``[3, 1, T]`` / ``[1, T]`` in THD mode).
            attention_mask: ``[B, S]`` attention mask (None in THD).
            labels: ``[B, S]`` target token IDs (``[1, T]`` in THD).
            loss_mask: ``[B, S]`` mask for loss (``[1, T]`` in THD).
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed).
            packed_seq_params: ``PackedSeqParams`` for THD attention.

        Returns:
            Loss tensor (post_process=True) or hidden states.
        """
        if position_ids is None:
            position_ids = self.compute_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                packed_seq_params=packed_seq_params,
            )

        vision_embeddings = None
        if (
            self.vision_model is not None
            and pixel_values is not None
        ):
            vision_embeddings = self.vision_model(
                pixel_values, image_grid_thw,
            )

        if decoder_input is None and self.language_model is not None:
            text_embeddings = self.language_model.embedding(
                input_ids=input_ids, position_ids=None,
            )

            if vision_embeddings is not None:
                decoder_input = self._scatter_vision_embeddings(
                    input_ids, text_embeddings, vision_embeddings,
                )
            else:
                decoder_input = text_embeddings

        (
            decoder_input, input_ids, labels, loss_mask,
            attention_mask, position_ids,
        ) = self._cp_split_for_forward(
            decoder_input=decoder_input,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
        )

        return self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            decoder_input=decoder_input,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )
