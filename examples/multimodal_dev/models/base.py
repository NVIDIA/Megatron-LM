# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Base multimodal model for FSDP + EP training.

Composes a vision encoder and a ``GPTModel`` language decoder.  Designed
for FSDP + EP: always builds the **full** model on every rank (no PP
flags).  PP support is only available through the MIMO ``MimoModel``
assembly path.

Subclasses override ``compute_position_ids()`` for model-specific
position encoding (e.g. MRoPE for Qwen3.5-VL).
"""

import os
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
    """Dummy process group reporting ``size()=1, rank()=0``.

    Used to override ``packed_seq_params.cp_group`` so that
    ``MultimodalRotaryEmbedding.forward`` skips its BSHD-style zigzag
    split of MRoPE freqs.  Without this, MRoPE always splits freqs by
    ``2 * cp_size`` even in THD mode (a known gap in Megatron-Core
    where ``MultimodalRotaryEmbedding`` lacks the ``not packed_seq``
    check that ``RotaryEmbedding`` has — see
    ``megatron/core/models/common/embeddings/rotary_pos_embedding.py``
    line 201 vs 402).  Attention reads its own cp_group from
    ``self.pg_collection.cp`` and is unaffected.
    """

    def size(self):
        return 1

    def rank(self):
        return 0


_NO_CP_GROUP = _NoCPGroup()


def _patch_mtp_roll_tensor_for_padded_thd():
    """Patch ``multi_token_prediction.roll_tensor`` so the THD+CP path
    slices each sample's per-rank chunk using ``cu_seqlens_q_padded``
    instead of ``cu_seqlens_q``.

    Megatron-Core's ``_roll_tensor_packed_seq`` (CP>1 branch) computes
    per-sample local indices as ``cu_seqlens // cp_size`` (see
    ``megatron/core/transformer/multi_token_prediction.py`` line 268-269).
    That assumes each sample's *valid* length is divisible by ``cp_size``
    — true only if cu_seqlens already references padded boundaries. In
    typical THD packing, valid lengths are arbitrary while only the
    *padded* lengths are multiples of ``2 * cp_size``.

    With our TE-based per-sample CP partition (which lays each rank's
    tokens out on ``cu_seqlens_q_padded // cp_size`` boundaries), MTP's
    rolling uses the wrong slice indices unless we swap cu_seqlens to
    cu_seqlens_padded for the duration of the call. The swap is harmless
    because rolled positions in the padding region carry loss_mask=0.
    """
    import megatron.core.transformer.multi_token_prediction as _mtp

    if getattr(_mtp.roll_tensor, "_thd_padded_patch_applied", False):
        return

    _orig = _mtp.roll_tensor

    def _patched(
        tensor, shifts=-1, dims=-1, cp_group=None, packed_seq_params=None,
    ):
        if (
            packed_seq_params is not None
            and getattr(packed_seq_params, "cu_seqlens_q_padded", None)
            is not None
            and packed_seq_params.cu_seqlens_q
            is not packed_seq_params.cu_seqlens_q_padded
        ):
            saved = packed_seq_params.cu_seqlens_q
            packed_seq_params.cu_seqlens_q = (
                packed_seq_params.cu_seqlens_q_padded
            )
            try:
                return _orig(
                    tensor, shifts=shifts, dims=dims, cp_group=cp_group,
                    packed_seq_params=packed_seq_params,
                )
            finally:
                packed_seq_params.cu_seqlens_q = saved
        return _orig(
            tensor, shifts=shifts, dims=dims, cp_group=cp_group,
            packed_seq_params=packed_seq_params,
        )

    _patched._thd_padded_patch_applied = True
    _mtp.roll_tensor = _patched


_patch_mtp_roll_tensor_for_padded_thd()


def _patch_mtp_loss_logging_for_thd_cp_disabled():
    """DISABLED: empirical runs showed this patch undercounts MTP loss
    by ~2x at step 1 — the underlying CP-aware sum-then-divide path is
    correct in principle but interacts with Megatron's loss-scale and
    grad-accumulation logic in subtle ways that need a deeper rework
    of ``track_mtp_metrics`` (which multiplies by
    ``1 / get_num_microbatches()`` assuming the OLD per-mb-ratio
    accumulation).  Leaving v4's logic in place: the residual ~1.3%
    drift at step 1 is the structural mismatch from per-rank uneven
    ``num_tokens`` after MTP rolling, and a true fix requires
    upstreaming a per-token-weighted reduction in Megatron core.
    """
    return


def _patch_mtp_loss_logging_for_thd_cp():
    """Patch ``MTPLossLoggingHelper`` to do sum-then-divide reduction.

    Megatron-Core stores per-rank ``loss_for_log = sum(loss * loss_mask)
    / num_tokens_per_rank`` and reduces with ``op=AVG`` over DP+CP
    (``process_mtp_loss`` line 692-700; ``reduce_loss_in_tracker`` line
    370-382).  Under THD+CP, MTP's roll leaves per-rank ``num_tokens``
    uneven (chunk-3 end zeroed on rank 0 while chunk-2 end on the last
    rank receives a valid token), so per-rank averaging then AVG is
    mathematically *not* equal to a global ``sum(loss)/sum(num_tokens)``.

    Fix: store ``(weighted_sum, num_tokens)`` pairs separately, all-reduce
    both with SUM, then divide once.  Bitwise equivalent to CP=1 modulo
    bf16 reduction order.

    Only the *logging* path is touched; the gradient path
    (``MTPLossAutoScaler``) is unchanged.
    """
    import megatron.core.transformer.multi_token_prediction as _mtp

    helper = _mtp.MTPLossLoggingHelper
    if getattr(helper, "_thd_cp_sum_then_divide_patch_applied", False):
        return

    def save_loss_to_tracker(
        loss,
        layer_number,
        num_layers,
        reduce_group=None,
        avg_group=None,
        num_tokens=None,
    ):
        if layer_number is None:
            return
        tracker = helper.tracker
        device = torch.cuda.current_device()
        if "values" not in tracker:
            tracker["values"] = torch.zeros(num_layers, device=device)
        if "num_tokens" not in tracker:
            tracker["num_tokens"] = torch.zeros(num_layers, device=device)
        tracker["values"][layer_number] += loss.detach()
        if num_tokens is not None:
            tracker["num_tokens"][layer_number] += (
                num_tokens.detach().to(tracker["num_tokens"].dtype)
            )
        tracker["reduce_group"] = reduce_group
        tracker["avg_group"] = avg_group

    def clean_loss_in_tracker():
        tracker = helper.tracker
        if "values" in tracker:
            tracker["values"].zero_()
        if "num_tokens" in tracker:
            tracker["num_tokens"].zero_()
        tracker["reduce_group"] = None
        tracker["avg_group"] = None

    def reduce_loss_in_tracker():
        tracker = helper.tracker
        if "values" not in tracker:
            return
        values = tracker["values"]
        num_tokens = tracker.get("num_tokens", None)
        has_tokens = (
            num_tokens is not None and num_tokens.abs().sum() > 0
        )
        if tracker.get("reduce_group") is not None:
            torch.distributed.all_reduce(
                values, group=tracker["reduce_group"],
            )
            if has_tokens:
                torch.distributed.all_reduce(
                    num_tokens, group=tracker["reduce_group"],
                )
        if has_tokens:
            # Sum (loss, num_tokens) across the CP group only, divide
            # to get a per-DP-rank ratio, then AVG across DP+CP. The
            # result matches what CP=1 reports as ``mtp_1 loss`` when
            # per-DP num_tokens are equal, modulo bf16 reduction order.
            from megatron.core import parallel_state as _ps

            cp_group = _ps.get_context_parallel_group()
            if cp_group is not None and cp_group.size() > 1:
                torch.distributed.all_reduce(values, group=cp_group)
                torch.distributed.all_reduce(num_tokens, group=cp_group)
            tracker["values"] = values / num_tokens.clamp(min=1)
            if tracker.get("avg_group") is not None:
                torch.distributed.all_reduce(
                    tracker["values"],
                    group=tracker["avg_group"],
                    op=torch.distributed.ReduceOp.AVG,
                )
        elif tracker.get("avg_group") is not None:
            torch.distributed.all_reduce(
                values,
                group=tracker["avg_group"],
                op=torch.distributed.ReduceOp.AVG,
            )

    helper.save_loss_to_tracker = staticmethod(save_loss_to_tracker)
    helper.clean_loss_in_tracker = staticmethod(clean_loss_in_tracker)
    helper.reduce_loss_in_tracker = staticmethod(reduce_loss_in_tracker)
    helper._thd_cp_sum_then_divide_patch_applied = True

    # Replace ``process_mtp_loss`` to save (weighted_sum, num_tokens) raw
    # pairs instead of the pre-divided per-rank average.  We mirror the
    # original implementation; only the save_loss_to_tracker call site
    # changes.
    from functools import partial as _partial

    from megatron.core import parallel_state as _ps

    _orig_process = _mtp.process_mtp_loss

    def _patched_process_mtp_loss(
        hidden_states,
        labels,
        loss_mask,
        output_layer,
        output_weight,
        runtime_gather_output,
        is_training,
        compute_language_model_loss,
        config,
        cp_group=None,
        packed_seq_params=None,
        scale_logits_fn=None,
    ):
        roll_tensor = _mtp.roll_tensor
        MTPLossAutoScaler = _mtp.MTPLossAutoScaler

        hidden_states_list = torch.chunk(
            hidden_states, 1 + config.mtp_num_layers, dim=0,
        )
        hidden_states = hidden_states_list[0]

        if labels is None:
            return hidden_states

        mtp_labels = labels.clone()
        if loss_mask is None:
            loss_mask = torch.ones_like(mtp_labels)

        original_num_tokens = loss_mask.sum()

        fuse_linear_cross_entropy = (
            config.cross_entropy_loss_fusion
            and config.cross_entropy_fusion_impl == "linear"
        )

        for mtp_layer_number in range(config.mtp_num_layers):
            mtp_labels, _ = roll_tensor(
                mtp_labels,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            loss_mask, num_tokens = roll_tensor(
                loss_mask,
                shifts=-1,
                dims=-1,
                cp_group=cp_group,
                packed_seq_params=packed_seq_params,
            )
            if fuse_linear_cross_entropy:
                mtp_loss = output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                    output_cross_entropy_loss=True,
                    labels=mtp_labels,
                )
            else:
                mtp_logits, _ = output_layer(
                    hidden_states_list[mtp_layer_number + 1],
                    weight=output_weight,
                    runtime_gather_output=runtime_gather_output,
                )
                if scale_logits_fn is not None:
                    mtp_logits = scale_logits_fn(mtp_logits)
                mtp_loss = compute_language_model_loss(mtp_labels, mtp_logits)
            mtp_loss = loss_mask * mtp_loss
            if is_training:
                # Save the *raw* weighted sum and num_tokens.  Reduction
                # happens in reduce_loss_in_tracker via SUM-then-divide.
                weighted_sum = torch.sum(mtp_loss) * (
                    num_tokens > 0
                ).to(mtp_loss.dtype)
                helper.save_loss_to_tracker(
                    weighted_sum,
                    mtp_layer_number,
                    config.mtp_num_layers,
                    avg_group=_ps.get_data_parallel_group(
                        with_context_parallel=True,
                    ),
                    num_tokens=num_tokens,
                )
            mtp_loss_scale = (
                config.mtp_loss_scaling_factor / config.mtp_num_layers
            )
            if config.calculate_per_token_loss:
                num_tokens_safe = torch.clamp(num_tokens, min=1)
                mtp_loss_normalized = (
                    mtp_loss_scale
                    * mtp_loss
                    * (original_num_tokens / num_tokens_safe)
                )
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states, mtp_loss_normalized,
                )
            else:
                safe_num_tokens = num_tokens.clamp(min=1)
                hidden_states = MTPLossAutoScaler.apply(
                    hidden_states,
                    mtp_loss_scale * mtp_loss / safe_num_tokens,
                )

        return hidden_states

    _mtp.process_mtp_loss = _patched_process_mtp_loss
    # ``gpt_model.py`` does ``from ... import process_mtp_loss`` at module
    # load, so we also rebind there.
    import megatron.core.models.gpt.gpt_model as _gpt_model

    _gpt_model.process_mtp_loss = _patched_process_mtp_loss


_patch_mtp_loss_logging_for_thd_cp_disabled()  # see comment in disabled stub


def _thd_cp_partition_index(cu_seqlens_padded, total_tokens, cp_size, cp_rank):
    """Per-rank token index for THD + CP, mirroring Megatron's THD partitioner.

    Delegates to TransformerEngine's ``thd_get_partitioned_indices``, which
    splits each packed sub-sample (boundaries given by *cu_seqlens_padded*)
    into ``2 * cp_size`` chunks per sample and returns the indices owned
    by *cp_rank*. This matches what
    ``megatron.core.utils.get_thd_batch_on_this_cp_rank`` does.

    Returned tensor is cast to ``int64`` so it can be used with
    ``index_select`` regardless of the TE version's return dtype.
    """
    from transformer_engine.pytorch import cpp_extensions as tex

    idx = tex.thd_get_partitioned_indices(
        cu_seqlens_padded, total_tokens, cp_size, cp_rank,
    )
    return idx.long()


def _cp_debug_enabled():
    return os.environ.get("MMDEV_CP_DEBUG", "0") == "1"


def _cp_debug_dump(tag, **tensors):
    """Print per-rank diagnostic info when ``MMDEV_CP_DEBUG=1``.

    *tensors* may include actual tensors (shapes/hashes printed) and
    primitive values (printed as-is).
    """
    if not _cp_debug_enabled():
        return
    rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_initialized()
        else 0
    )
    try:
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_size = parallel_state.get_context_parallel_world_size()
        dp_rank = parallel_state.get_data_parallel_rank()
    except Exception:
        cp_rank = cp_size = dp_rank = -1

    parts = [f"[CP_DBG {tag}] rank={rank} dp={dp_rank} cp={cp_rank}/{cp_size}"]
    for name, val in tensors.items():
        if isinstance(val, torch.Tensor):
            shape = tuple(val.shape)
            try:
                if val.dtype in (torch.int32, torch.int64, torch.bool):
                    summary = val.flatten()[:8].tolist()
                else:
                    summary = (
                        f"sum={val.float().sum().item():.4e} "
                        f"mean={val.float().mean().item():.4e}"
                    )
                parts.append(f"{name}={shape} {summary}")
            except Exception:
                parts.append(f"{name}={shape}")
        else:
            parts.append(f"{name}={val}")
    print(" | ".join(parts), flush=True)


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
    ) -> Tensor:
        """Compute position IDs.  Override for MRoPE etc.

        Default: simple sequential positions.

        Args:
            input_ids: ``[B, S]`` token IDs.
            image_grid_thw: ``[num_images, 3]`` grid dimensions.

        Returns:
            Position IDs tensor.
        """
        B, S = input_ids.shape
        return (
            torch.arange(S, device=input_ids.device)
            .unsqueeze(0)
            .expand(B, -1)
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

        # --- Context Parallelism: split along sequence dimension ---
        # BSHD path: zigzag-split tensors along seq dim. position_ids is
        #   left alone — the MRoPE/RoPE embedding handles its own CP
        #   slicing via ``get_pos_emb_on_this_cp_rank()``.
        # THD path: use TE's ``thd_get_partitioned_indices`` so that each
        #   packed sub-sample is split into 2*cp_size chunks per sample.
        #   position_ids MUST be sliced with the same index because in
        #   THD it was pre-computed on the full [1, T] layout.
        cp_size = parallel_state.get_context_parallel_world_size()
        if cp_size > 1:
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
                # position_ids is left at full length [3, 1, T] (or [1, T]).
                # MRoPE will compute full freqs; attention's
                # _apply_rotary_pos_emb_thd then applies per-sample CP zigzag
                # via _get_thd_freqs_on_this_cp_rank using cu_seqlens.
                # Note: we do NOT set packed_seq_params.cp_group here. MTP
                # reads packed_seq_params.cp_group at runtime to override its
                # own cp_group; setting _NO_CP_GROUP would break MTP's
                # CP-aware roll of labels/loss_mask. MRoPE's zigzag is
                # bypassed below by mutating mrope.cp_group directly.
                # attention_mask is None in THD; skip.
            else:
                if decoder_input is not None:
                    decoder_input = _cp_split_tensor(
                        decoder_input, seq_dim=0,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if input_ids is not None:
                    input_ids = _cp_split_tensor(
                        input_ids, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if labels is not None:
                    labels = _cp_split_tensor(
                        labels, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if loss_mask is not None:
                    loss_mask = _cp_split_tensor(
                        loss_mask, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )
                if attention_mask is not None:
                    attention_mask = _cp_split_tensor(
                        attention_mask, seq_dim=1,
                        cp_size=cp_size, cp_rank=cp_rank,
                    )

            _cp_debug_dump(
                "MultimodalModel.forward post-CP",
                decoder_input=decoder_input,
                input_ids=input_ids,
                labels=labels,
                loss_mask=loss_mask,
                position_ids=position_ids,
                cu_seqlens_q_padded=(
                    packed_seq_params.cu_seqlens_q_padded
                    if packed_seq_params is not None
                    else None
                ),
            )

        # In THD mode, temporarily override MRoPE's cp_group on the
        # language_model so it skips its BSHD-style zigzag of pre-computed
        # MRoPE embeddings.  Only affects ``MultimodalRotaryEmbedding``;
        # attention's RoPE / CP path and MTP both keep their real cp_group.
        _mrope = (
            getattr(self.language_model, "rotary_pos_emb", None)
            if packed_seq_params is not None
            and parallel_state.get_context_parallel_world_size() > 1
            else None
        )
        _saved_mrope_cp_group = (
            getattr(_mrope, "cp_group", None) if _mrope is not None else None
        )
        if _mrope is not None:
            _mrope.cp_group = _NO_CP_GROUP
        try:
            output = self.language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                decoder_input=decoder_input,
                labels=labels,
                loss_mask=loss_mask,
                packed_seq_params=packed_seq_params,
            )
        finally:
            if _mrope is not None:
                _mrope.cp_group = _saved_mrope_cp_group
        return output
