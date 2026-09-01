# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model for FSDP + EP training.

Composes a vision encoder and either a ``GPTModel`` or ``HybridModel`` language decoder. Designed
for FSDP + EP: always builds the **full** model on every rank (no PP
flags).  PP support is only available through the MIMO ``MimoModel``
assembly path.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

import contextlib
from typing import Optional

import torch
from torch import Tensor

from megatron.core import parallel_state, tensor_parallel
from megatron.core.models.gpt import GPTModel
from megatron.core.models.hybrid.hybrid_model import HybridModel
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
    assert S % (2 * cp_size) == 0, f"seq_len {S} not divisible by 2*cp_size={2 * cp_size}"
    tensor = tensor.view(
        *tensor.shape[:seq_dim], 2 * cp_size, S // (2 * cp_size), *tensor.shape[seq_dim + 1 :]
    )
    index = torch.zeros(2, dtype=torch.int64, device=tensor.device)
    index[0] = cp_rank
    index[1] = 2 * cp_size - cp_rank - 1
    tensor = tensor.index_select(seq_dim, index)
    tensor = tensor.view(*tensor.shape[:seq_dim], -1, *tensor.shape[seq_dim + 2 :])
    return tensor


class _NoCPGroup:
    """Dummy size-1 process group used to bypass MRoPE's BSHD-style
    zigzag of pre-computed THD freqs (Megatron-Core gap:
    ``MultimodalRotaryEmbedding.forward`` lacks the ``not packed_seq``
    skip that plain ``RotaryEmbedding`` has).
    """

    def size(self):
        """Pretend this group has exactly one rank."""
        return 1

    def rank(self):
        """This rank's id within the fake group is always 0."""
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

    idx = tex.thd_get_partitioned_indices(cu_seqlens_padded, total_tokens, cp_size, cp_rank)
    return idx.long()


class MultimodalModel(MegatronModule):
    """Base class for multimodal vision-language models.

    Composes a pre-constructed vision encoder and a language decoder.
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
        hybrid_stack_spec: Optional HybridModel stack spec. When supplied,
            ``language_spec`` must be ``None``.
        hybrid_layer_pattern: Unified HybridModel layer pattern.
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
        mrope_section: Optional[list] = None,
        mtp_block_spec: Optional[ModuleSpec] = None,
        parallel_output: bool = True,
        share_embeddings_and_output_weights: bool = False,
        hybrid_stack_spec: Optional[ModuleSpec] = None,
        hybrid_layer_pattern: Optional[str] = None,
    ):
        super().__init__(config=language_config)

        self.image_token_id = image_token_id

        self.vision_model = vision_encoder
        if hybrid_stack_spec is not None or hybrid_layer_pattern is not None:
            if hybrid_stack_spec is None or hybrid_layer_pattern is None:
                raise ValueError(
                    "Hybrid multimodal decoders require both hybrid_stack_spec and "
                    "hybrid_layer_pattern."
                )
            if language_spec is not None:
                raise ValueError("language_spec and hybrid_stack_spec are mutually exclusive.")
            if mtp_block_spec is not None:
                raise ValueError(
                    "HybridModel expresses MTP in hybrid_layer_pattern; mtp_block_spec is invalid."
                )
            self.language_model = HybridModel(
                config=language_config,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                hybrid_layer_pattern=hybrid_layer_pattern,
                pre_process=True,
                post_process=True,
                parallel_output=parallel_output,
                share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                position_embedding_type=position_embedding_type,
                rotary_percent=rotary_percent,
                rotary_base=rotary_base,
            )
        else:
            if language_spec is None:
                raise ValueError("GPTModel multimodal decoders require language_spec.")
            self.language_model = GPTModel(
                config=language_config,
                transformer_layer_spec=language_spec,
                vocab_size=vocab_size,
                max_sequence_length=max_sequence_length,
                pre_process=True,
                post_process=True,
                parallel_output=parallel_output,
                share_embeddings_and_output_weights=(share_embeddings_and_output_weights),
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
        vision_token_indices: Tensor = None,
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
            and parallel_state.get_tensor_model_parallel_world_size() > 1
        )

        if sp:
            text_embeddings = tensor_parallel.gather_from_sequence_parallel_region(
                text_embeddings, tensor_parallel_output_grad=False
            )

        combined = text_embeddings.transpose(0, 1).contiguous()
        if vision_token_indices is None:
            image_mask = input_ids == self.image_token_id
            num_slots = int(image_mask.sum())
            if vision_embeddings.ndim != 2 or vision_embeddings.shape[0] != num_slots:
                raise ValueError(
                    f"Found {num_slots} image-token positions but received vision embeddings "
                    f"with shape {tuple(vision_embeddings.shape)}."
                )
            mask_expanded = image_mask.unsqueeze(-1).expand_as(combined)
            combined = combined.masked_scatter(
                mask_expanded, vision_embeddings.to(device=combined.device, dtype=combined.dtype)
            )
        else:
            if vision_token_indices.ndim == 2 and vision_token_indices.shape[1] == 2:
                vision_token_indices = (
                    vision_token_indices[:, 0] * input_ids.shape[1] + vision_token_indices[:, 1]
                )
            if vision_token_indices.ndim != 1:
                raise ValueError(
                    "vision_token_indices must be flattened [N] or (batch, sequence) pairs [N, 2]."
                )
            if vision_embeddings.ndim != 2 or vision_embeddings.shape[0] != len(
                vision_token_indices
            ):
                raise ValueError(
                    f"Received {len(vision_token_indices)} vision token positions but vision "
                    f"embeddings have shape {tuple(vision_embeddings.shape)}."
                )
            flat = combined.view(-1, combined.shape[-1])
            flat = flat.index_copy(
                0,
                vision_token_indices.to(device=flat.device, dtype=torch.long),
                vision_embeddings.to(device=flat.device, dtype=flat.dtype),
            )
            combined = flat.view_as(combined)
        combined = combined.transpose(0, 1).contiguous()

        if sp:
            combined = tensor_parallel.scatter_to_sequence_parallel_region(combined)

        return combined

    def _embed_input_ids(self, input_ids: Tensor) -> Tensor:
        """Embed decoder token IDs. Subclasses may sanitize synthetic IDs."""
        return self.language_model.embedding(input_ids=input_ids, position_ids=None)

    def prepare_attention_mask(self, input_ids: Tensor, attention_mask, packed_seq_params=None):
        """Build model-specific attention metadata before CP partitioning."""
        return attention_mask

    def compute_position_ids(
        self, input_ids: Tensor, image_grid_thw: Optional[Tensor] = None, packed_seq_params=None
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.  ``packed_seq_params`` is
        accepted for subclass compatibility (e.g. MRoPE in THD mode).
        """
        B, S = input_ids.shape
        return torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

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
        padding_mask=None,
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
                decoder_input,
                input_ids,
                labels,
                loss_mask,
                attention_mask,
                position_ids,
                padding_mask,
            )
        cp_rank = parallel_state.get_context_parallel_rank()

        if packed_seq_params is not None:
            total_tokens = (
                decoder_input.shape[0] if decoder_input is not None else input_ids.shape[1]
            )
            idx = _thd_cp_partition_index(
                packed_seq_params.cu_seqlens_q_padded, total_tokens, cp_size, cp_rank
            )
            if decoder_input is not None:
                decoder_input = decoder_input.index_select(0, idx)
            if input_ids is not None:
                input_ids = input_ids.index_select(1, idx)
            if labels is not None:
                labels = labels.index_select(1, idx)
            if loss_mask is not None:
                loss_mask = loss_mask.index_select(1, idx)
            if padding_mask is not None:
                padding_mask = padding_mask.index_select(1, idx)
        else:

            def _split(t, seq_dim):
                return (
                    None
                    if t is None
                    else (
                        _cp_split_tensor(t, seq_dim=seq_dim, cp_size=cp_size, cp_rank=cp_rank)
                        if isinstance(t, Tensor)
                        else t
                    )
                )

            decoder_input = _split(decoder_input, 0)
            input_ids = _split(input_ids, 1)
            labels = _split(labels, 1)
            loss_mask = _split(loss_mask, 1)
            attention_mask = _split(attention_mask, 1)
            padding_mask = _split(padding_mask, 1)

        return (
            decoder_input,
            input_ids,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            padding_mask,
        )

    @staticmethod
    def cp_split_loss_mask(loss_mask, packed_seq_params):
        """Slice ``loss_mask`` the same way the model slices its inputs.

        Mirrors the slicing done inside :meth:`_cp_split_for_forward` so
        the loss computation outside the model can index a mask aligned
        with the model's CP-shard output. Returns ``loss_mask`` unchanged
        when ``CP <= 1``.
        """
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size <= 1 or loss_mask is None:
            return loss_mask
        cp_rank = parallel_state.get_context_parallel_rank()
        if packed_seq_params is not None:
            idx = _thd_cp_partition_index(
                packed_seq_params.cu_seqlens_q_padded, loss_mask.shape[1], cp_size, cp_rank
            )
            return loss_mask.index_select(1, idx)
        return _cp_split_tensor(loss_mask, seq_dim=1, cp_size=cp_size, cp_rank=cp_rank)

    @contextlib.contextmanager
    def _thd_mrope_no_cp_override(self, packed_seq_params):
        """Force ``rotary_pos_emb.cp_group`` to size 1 for the wrapped
        forward call so MRoPE returns full-length freqs in THD mode.
        Attention then applies per-sample CP zigzag itself via
        ``_apply_rotary_pos_emb_thd``.  Done by direct mutation rather
        than via ``packed_seq_params.cp_group`` so MTP's CP-aware roll
        (which reads that field) still sees the real CP group.
        """
        mrope = (
            getattr(self.language_model, "rotary_pos_emb", None)
            if packed_seq_params is not None
            and parallel_state.get_context_parallel_world_size() > 1
            else None
        )
        saved = getattr(mrope, "cp_group", None) if mrope is not None else None
        if mrope is not None:
            mrope.cp_group = _NO_CP_GROUP
        try:
            yield
        finally:
            if mrope is not None:
                mrope.cp_group = saved

    def forward(
        self,
        input_ids: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor = None,
        labels: Tensor = None,
        loss_mask: Tensor = None,
        padding_mask: Tensor = None,
        pixel_values: Tensor = None,
        image_grid_thw: Tensor = None,
        decoder_input: Tensor = None,
        vision_embeddings: Tensor = None,
        vision_token_indices: Tensor = None,
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
            padding_mask: ``[B, S]`` bool mask, True at collate-padded
                positions (``[1, T]`` in THD). Forwarded to the language
                decoder so MoE routing excludes padded tokens from aux
                loss / z-loss / expert-bias accumulation. Distinct from
                ``loss_mask``: only true padding, not SFT prompt tokens.
            pixel_values: Preprocessed image pixels.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.
            decoder_input: Pre-computed decoder input (skip embed).
            vision_embeddings: Pre-computed visual embeddings. This bypasses
                ``vision_model`` and is the stable injection point for MDP.
            vision_token_indices: Optional flattened ``[B*S]`` decoder positions
                for externally supplied visual rows.
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

        if vision_embeddings is not None and pixel_values is not None:
            raise ValueError("Pass either pixel_values or vision_embeddings, not both.")
        if vision_embeddings is None and self.vision_model is not None and pixel_values is not None:
            vision_embeddings = self.vision_model(pixel_values, image_grid_thw)

        if decoder_input is None and self.language_model is not None:
            text_embeddings = self._embed_input_ids(input_ids)

            if vision_embeddings is not None:
                decoder_input = self._scatter_vision_embeddings(
                    input_ids,
                    text_embeddings,
                    vision_embeddings,
                    vision_token_indices=vision_token_indices,
                )
            else:
                decoder_input = text_embeddings

        attention_mask = self.prepare_attention_mask(
            input_ids, attention_mask, packed_seq_params=packed_seq_params
        )

        (
            decoder_input,
            input_ids,
            labels,
            loss_mask,
            attention_mask,
            position_ids,
            padding_mask,
        ) = self._cp_split_for_forward(
            decoder_input=decoder_input,
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
            attention_mask=attention_mask,
            position_ids=position_ids,
            packed_seq_params=packed_seq_params,
            padding_mask=padding_mask,
        )

        with self._thd_mrope_no_cp_override(packed_seq_params):
            return self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
                padding_mask=padding_mask,
                packed_seq_params=packed_seq_params,
            )
