# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Explicit shared-prefix adapter for a HybridStack.

This module deliberately does not alter :class:`HybridStack`'s normal ``forward`` path. Call
``forward_hybrid_stack_shared_prefix`` with one packed star ``[P, C_1, ..., C_G]`` to opt in.
Mamba layers scan the prefix once and fork differentiable convolution/SSM state into the
completion branches, including CP>1 through sequence-to-head collectives. Alternative Mamba
implementations, including an uninterrupted prefix-replay parity oracle, are selected in
``_forward_mamba_layer_shared_prefix_cp``. Attention uses the exact-backward fused forest kernel.
Unsupported topology or model features fail before executing a partial forward.

CP>1 uses the model's standard zigzag sequence shards and sequence-to-head all-to-alls. The
implemented surface covers TP1 and TP>1 with sequence parallelism, CP1 and CP>1, explicit physical
padding, MoE expert-bias accounting, and full uniform activation recomputation. The capability
tokens at the end of this module name these code paths so integrations can require the exact
conjunction they use; they are contract identifiers, not evidence that a topology has been
qualified at full-model scale.
"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.ssm.mamba_branch_layout import merge_mamba_branches, pack_mamba_branches
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import (
    MAMBA_HAS_STATE_DTYPE,
    MambaMixer,
    causal_conv1d_fn,
    mamba_chunk_scan_combined,
)
from megatron.core.transformer.attention import SelfAttention
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.moe.moe_utils import router_gating_token_blocks
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_fa_min_version, make_viewless_tensor


def _mamba_state_dtype_kwargs(mixer: MambaMixer) -> dict:
    if not MAMBA_HAS_STATE_DTYPE:
        return {}
    return {"state_dtype": mixer.mamba_training_ssm_states_dtype}


# NRL_SP_MAMBA_IMPL values. ``*_training`` variants use state_fork in evaluation mode.
_MAMBA_IMPLS = (
    "ragged_state_fork",
    "ragged_state_fork_training",
    "state_fork",
    "replay_prefix",
    "replay_prefix_training",
    "packed_recurrence",
    "packed_recurrence_training",
    "packed_fused",
)


def _shared_prefix_mamba_impl() -> str:
    """Return the Mamba implementation selected by ``NRL_SP_MAMBA_IMPL``.

    ``ragged_state_fork`` is the default for every topology and root count.
    """
    implementation = os.environ.get("NRL_SP_MAMBA_IMPL", "ragged_state_fork")
    if implementation not in _MAMBA_IMPLS:
        raise ValueError(
            f"NRL_SP_MAMBA_IMPL must be one of {', '.join(map(repr, _MAMBA_IMPLS))}, "
            f"got {implementation!r}"
        )
    return implementation


def _validate_mamba_fork(mixer: MambaMixer) -> None:
    """Reject Mamba mixers and backend selections the shared-prefix paths cannot run.

    ``_validate_hybrid_stack`` calls this for every Mamba layer before any layer executes, and
    checks the mixer's TP/CP groups against the stack; the layer forwards do not repeat it.
    """
    if not mixer.rmsnorm:
        raise NotImplementedError("shared-prefix Mamba state forking requires gated RMSNorm")
    if causal_conv1d_fn is None or mamba_chunk_scan_combined is None:
        raise RuntimeError("shared-prefix Mamba state forking requires causal-conv1d and mamba-ssm")
    if _shared_prefix_mamba_impl().startswith("ragged_state_fork"):
        try:
            # Ragged imports private mamba_ssm SSD kernels; fail before any layer runs.
            from megatron.core.ssm import (  # pylint: disable=unused-import
                mamba_ragged,
                mamba_ragged_scan,
            )
        except ImportError as error:
            raise RuntimeError(
                "shared-prefix ragged Mamba cannot import its kernels; set "
                "NRL_SP_MAMBA_IMPL=state_fork to use the public mamba_ssm API"
            ) from error
        chunk_size = mixer.chunk_size
        if (
            isinstance(chunk_size, bool)
            or not isinstance(chunk_size, int)
            or chunk_size < 1
            or chunk_size & (chunk_size - 1)
        ):
            raise NotImplementedError(
                f"shared-prefix ragged Mamba requires a power-of-two chunk_size, got "
                f"{chunk_size!r}; set NRL_SP_MAMBA_IMPL=state_fork to use the padded backend"
            )
        # The forest scan keeps upstream's default state dtype (the activation dtype).
        if MAMBA_HAS_STATE_DTYPE and mixer.mamba_training_ssm_states_dtype not in (
            None,
            mixer.config.params_dtype,
        ):
            raise NotImplementedError(
                "shared-prefix ragged Mamba does not support mamba_training_ssm_states_dtype="
                f"{mixer.mamba_training_ssm_states_dtype} with params_dtype="
                f"{mixer.config.params_dtype}; unset mamba_training_ssm_states_dtype or set "
                "NRL_SP_MAMBA_IMPL=state_fork"
            )


def _prefix_conv_context(xbc: Tensor, width: int) -> Tensor:
    """Last ``width`` pre-convolution ``[batch, length, channels]`` rows, zero-padded causally."""
    if width == 0:
        return xbc[:, :0]
    padded = F.pad(xbc, (0, 0, max(0, width - xbc.shape[1]), 0))
    return padded[:, -width:].clone()


def _mamba_prefix_fork_boundary(mixer: MambaMixer, prefix_len: int) -> int:
    """Return the last scan-chunk boundary at or before ``prefix_len``.

    Restarting ``mamba_chunk_scan_combined`` at an arbitrary token changes its
    internal chunk partition.  With BF16 training state, that extra state
    boundary is numerically visible after many Hybrid layers.  Fork only at an
    ordinary Mamba chunk boundary and replay the (short) prompt tail inside
    each completion branch so the state-fork and uninterrupted scans use the
    same chunk partition.
    """
    prefix_len = int(prefix_len)
    chunk_size = getattr(mixer, "chunk_size", None)
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        message = "shared-prefix Mamba requires a positive integer chunk_size"
        raise ValueError(f"{message}, got {chunk_size!r}")
    if prefix_len < 1:
        raise ValueError("shared-prefix Mamba requires a non-empty prefix")
    return prefix_len - prefix_len % chunk_size


def _scan_mamba_projected_segment(
    mixer: MambaMixer,
    projected: Tensor,
    *,
    conv_context: Tensor | None = None,
    ssm_initial_state: Tensor | None = None,
    capture_state: bool = False,
    include_gate: bool = True,
) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None]:
    """Scan already CP-canonical, head-sharded Mamba projections for one segment batch."""
    cp = mixer.cp
    if projected.ndim != 3:
        raise ValueError(
            "shared-prefix projected Mamba segments must have shape [sequence,batch,d]"
        )
    num_groups = cp.ngroups_local_tpcp
    num_heads = cp.nheads_local_tpcp
    d_inner = cp.d_inner_local_tpcp
    branch_count = projected.shape[1]

    projected = rearrange(projected, "l b d -> b l d").contiguous()
    if include_gate:
        z, projected = projected.split([d_inner, projected.shape[-1] - d_inner], dim=-1)
    else:
        z = None
    xbc, dt = torch.split(projected, [d_inner + 2 * num_groups * mixer.d_state, num_heads], dim=-1)
    A = -torch.exp(cp.get_A_log().float())

    # Keep [batch, length, channels] storage and pass the convolution a channel-last view, as
    # MambaMixer does. causal_conv1d's channel-first backward returns wrong gradients at some
    # lengths (L % 1024 in 1..7 for bf16/fp16, L % 512 in 1..3 for fp32).
    xbc = xbc.contiguous()
    next_conv_context = _prefix_conv_context(xbc, mixer.d_conv - 1) if capture_state else None
    if conv_context is not None:
        if (
            conv_context.ndim != 3
            or conv_context.shape[1] != mixer.d_conv - 1
            or conv_context.shape[2] != xbc.shape[2]
        ):
            raise ValueError("prefix convolution state is incompatible with the branch")
        if conv_context.shape[0] not in (1, branch_count):
            raise ValueError("prefix convolution state batch is incompatible with the branch")
        repeated_context = conv_context.to(xbc.dtype).expand(branch_count, -1, -1)
        conv_input = torch.cat([repeated_context, xbc], dim=1)
        conv_output = causal_conv1d_fn(
            conv_input.transpose(1, 2),
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )[:, :, repeated_context.shape[1] :]
    else:
        conv_output = causal_conv1d_fn(
            xbc.transpose(1, 2),
            rearrange(cp.get_conv1d_weight(), "d 1 w -> d w"),
            cp.get_conv1d_bias(),
            activation=mixer.activation,
        )
    xbc = rearrange(conv_output, "b d l -> b l d").contiguous()

    x, B, C = torch.split(
        xbc, [d_inner, num_groups * mixer.d_state, num_groups * mixer.d_state], dim=-1
    )
    x = rearrange(x, "b l (h p) -> b l h p", p=mixer.headdim).contiguous()
    B = rearrange(B, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()
    C = rearrange(C, "b l (g n) -> b l g n", n=mixer.d_state).contiguous()

    initial_states = ssm_initial_state
    if initial_states is not None:
        if initial_states.shape[0] not in (1, branch_count):
            raise ValueError("prefix SSM state batch is incompatible with the branch")
        initial_states = initial_states.expand(branch_count, *initial_states.shape[1:]).contiguous()
    scan = mamba_chunk_scan_combined(
        x,
        dt.contiguous(),
        A,
        B,
        C,
        mixer.chunk_size,
        D=(
            rearrange(cp.get_D().float(), "(h p) -> h p", p=mixer.headdim)
            if mixer.D_has_hdim
            else cp.get_D()
        ),
        z=None,
        dt_bias=cp.get_dt_bias().float(),
        dt_softplus=True,
        initial_states=initial_states,
        return_final_states=capture_state,
        **_mamba_state_dtype_kwargs(mixer),
    )
    if capture_state:
        y, final_state = scan
    else:
        y, final_state = scan, None
    y = rearrange(y, "b l h p -> l b (h p)").contiguous()
    if z is not None:
        z = rearrange(z, "b l d -> l b d").contiguous()
    return y, z, next_conv_context, final_state


def _scan_mamba_shared_prefix_root(
    mixer: MambaMixer, projected: Tensor, layout: SharedPrefixLayout, *, replay_prefix: bool
) -> Tensor:
    """Scan recurrent fields for one root, retaining the established chunk boundary."""
    physical_len = projected.shape[0]
    physical_completion_lens = list(layout.completion_lens)
    # The topology validator owns physical alignment. Padding is placed after the final
    # completion;
    # scanning it as that branch's causal tail cannot affect any real-token output.
    physical_completion_lens[-1] += physical_len - layout.total_len
    fork_boundary = 0 if replay_prefix else _mamba_prefix_fork_boundary(mixer, layout.prefix_len)
    branch_prefix_len = layout.prefix_len if replay_prefix else layout.prefix_len - fork_boundary
    branches = pack_mamba_branches(
        projected,
        prefix_len=layout.prefix_len,
        tail_len=branch_prefix_len,
        completion_lens=tuple(physical_completion_lens),
    )

    prefix_y = projected.new_empty((0, 1, mixer.cp.d_inner_local_tpcp))
    if replay_prefix:
        branch_y, _, _, _ = _scan_mamba_projected_segment(mixer, branches, include_gate=False)
    else:
        conv_context = None
        final_state = None
        if fork_boundary:
            prefix_y, _, conv_context, final_state = _scan_mamba_projected_segment(
                mixer, projected[:fork_boundary], capture_state=True, include_gate=False
            )
            if conv_context is None or final_state is None:
                raise RuntimeError(
                    "shared-prefix CP Mamba scan did not return a differentiable state"
                )
        branch_y, _, _, _ = _scan_mamba_projected_segment(
            mixer,
            branches,
            conv_context=conv_context,
            ssm_initial_state=final_state,
            include_gate=False,
        )
    return merge_mamba_branches(
        prefix_y,
        branch_y,
        tail_len=branch_prefix_len,
        completion_lens=tuple(physical_completion_lens),
    )


def _forward_mamba_layer_shared_prefix_cp_impl(
    layer: MambaLayer,
    hidden_states: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    replay_prefix: bool,
    packed_recurrence: bool = False,
    ragged_state_fork: bool = False,
) -> Tensor:
    """Run a TP/SP/CP Mamba star with state-fork or uninterrupted-replay prefix handling.

    The input projection first gathers sequence-parallel TP shards, then the CP adapter converts
    local zigzag sequence shards into one canonical global sequence with TP/CP-local channels.
    State-fork scans the chunk-aligned prefix head once, expands its differentiable state, and
    replays only the unaligned prompt tail per branch. Replay is retained as a parity baseline and
    repeats the full projected prefix per branch. The packed result returns through the inverse CP
    transform and TP output-projection reduce-scatter.
    """
    mixer = layer.mixer
    cp = mixer.cp

    residual = hidden_states.float() if layer.config.fp32_residual_connection else hidden_states
    normalized = apply_module(layer.norm)(hidden_states.to(dtype=layer.config.params_dtype))
    projected, _ = mixer.in_proj(normalized)
    # Input SP gather: [physical/(TP*C), 1, hidden] -> [physical/C, 1, projection/TP].
    # Gated RMSNorm runs after the inverse CP transform, in this same local
    # token/channel order. The scan does not use z, so keep it here instead of
    # exchanging and packing it through every branch. Copy to avoid retaining
    # the entire projection allocation solely for the gate.
    local_z = projected[..., : cp.d_inner_local_tp].contiguous()
    projected = cp.pre_conv_ssm(projected, include_gate=False)
    physical_len = projected.shape[0]
    if physical_len < layout.total_len:
        raise ValueError(
            f"parallel physical length {physical_len} is shorter than shared layout "
            f"{layout.total_len}"
        )

    if ragged_state_fork:
        if replay_prefix or packed_recurrence:
            raise ValueError("Ragged state forking cannot use prefix replay")
        from megatron.core.ssm.mamba_ragged import scan_mamba_ragged_forest

        packed_y = scan_mamba_ragged_forest(mixer, projected, layout)
    elif packed_recurrence:
        if replay_prefix:
            raise ValueError("Packed recurrence and per-root replay are separate modes")
        # Keep optional convolution/scan dependencies lazy for non-Mamba models.
        from megatron.core.ssm.mamba_forest_replay import scan_mamba_forest_replay

        packed_y = scan_mamba_forest_replay(mixer, projected, layout)
    else:
        pieces = []
        for offset, root in layout.iter_roots():
            end = offset + root.total_len
            if end == layout.total_len:
                end = physical_len
            pieces.append(
                _scan_mamba_shared_prefix_root(
                    mixer, projected[offset:end], root, replay_prefix=replay_prefix
                )
            )
        packed_y = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
    # Canonical channel shards -> local zigzag sequence shards with full channels.
    packed_y = cp.post_conv_ssm(packed_y)
    packed_y = mixer.norm(packed_y, local_z)
    packed_output, output_bias = mixer.out_proj(packed_y)

    with layer.bias_dropout_add_exec_handler():
        return layer.mamba_bda(training=layer.training, fused=layer.config.bias_dropout_fusion)(
            (packed_output, output_bias), residual, layer.hidden_dropout
        )


def _forward_mamba_layer_shared_prefix_cp_state_fork(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout | SharedPrefixForestLayout
) -> Tensor:
    """Optimized CP Mamba: fork at a scan-chunk boundary and replay the prompt tail."""
    return _forward_mamba_layer_shared_prefix_cp_impl(
        layer, hidden_states, layout, replay_prefix=False
    )


def _forward_mamba_layer_shared_prefix_cp_replay(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    """Parity baseline: rescan each prefix per branch without an explicit state boundary."""
    return _forward_mamba_layer_shared_prefix_cp_impl(
        layer, hidden_states, layout, replay_prefix=True
    )


def _forward_mamba_layer_shared_prefix_cp_packed_fused_oracle(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout
) -> Tensor:
    """Diagnostic dense replay through Mamba's unchanged packed fused path.

    This deliberately gives up Mamba prefix sharing.  It reconstructs the ordinary
    branch-major packed batch from the canonical star, invokes ``MambaLayer.forward``
    with the same ``PackedSeqParams`` contract as dense NeMo-RL sequence packing, and
    then folds the result back to one shared-prefix star.  Consequently the oracle
    exercises ``mamba_split_conv1d_scan_combined(seq_idx=...)`` rather than the
    decomposed causal-convolution/chunk-scan implementation used by the optimized
    state-fork path.

    The helper is a diagnostic isolation oracle, not a fallback for the default path.
    It is selected explicitly with ``NRL_SP_MAMBA_IMPL=packed_fused``;
    ``ragged_state_fork`` is the default.
    """
    mixer = layer.mixer
    if not isinstance(mixer, MambaMixer):
        raise TypeError("shared-prefix packed-fused Mamba oracle requires MambaMixer")
    tp_group = mixer.pg_collection.tp
    cp_group = mixer.pg_collection.cp
    tp_size = tp_group.size()
    cp_size = cp_group.size()
    # _validate_hybrid_stack has checked this length against the layout and topology.
    physical_len = hidden_states.shape[0] * tp_size * cp_size

    # [star/(TP*CP),1,H] -> one canonical global star.  The oracle runs under
    # no_grad, but use the same collectives/order as the differentiable MTP
    # reconstruction so the forward topology is representative.
    cp_local_star = hidden_states
    if tp_size > 1:
        cp_local_star = tensor_parallel.gather_from_sequence_parallel_region(
            cp_local_star, tensor_parallel_output_grad=False, group=tp_group
        )
    if cp_size > 1:
        rank_order_star = tensor_parallel.gather_from_sequence_parallel_region(
            cp_local_star, tensor_parallel_output_grad=True, group=cp_group
        )
        rank_order_indices = torch.cat(
            [
                layout.cp_local_indices(physical_len, cp_size, rank, hidden_states.device)
                for rank in range(cp_size)
            ]
        )
        inverse_order = torch.empty_like(rank_order_indices)
        inverse_order[rank_order_indices] = torch.arange(
            physical_len, device=hidden_states.device, dtype=torch.long
        )
        global_star = rank_order_star.index_select(0, inverse_order)
    else:
        global_star = cp_local_star
    if global_star.shape != (physical_len, 1, hidden_states.shape[-1]):
        raise RuntimeError(
            "shared-prefix packed-fused Mamba oracle reconstructed an invalid star shape"
        )

    # Drop topology-only padding after the final completion.  Dense NeMo-RL
    # padding is per branch, and those physical branch tails are already part of
    # layout.completion_lens.
    branch_indices = layout.dense_branch_indices(hidden_states.device)
    branch_lengths = [int(indices.numel()) for indices in branch_indices]
    dense_global = torch.cat(
        [global_star.index_select(0, indices) for indices in branch_indices], dim=0
    )
    dense_total = sum(branch_lengths)
    if dense_global.shape[0] != dense_total:
        raise RuntimeError("shared-prefix packed-fused Mamba oracle built an invalid dense batch")
    if cp_size > 1 and any(length % (2 * cp_size) for length in branch_lengths):
        raise ValueError(
            "shared-prefix packed-fused Mamba oracle requires each dense branch "
            "to divide the CP zigzag quantum"
        )

    # Match NeMo-RL's per-sequence CP zigzag packing, then its TP sequence
    # parallel shard.  This ordering is distinct from the whole-star CP layout.
    if cp_size > 1:
        dense_cp_indices = []
        branch_offset = 0
        for branch_length in branch_lengths:
            dense_cp_indices.append(
                branch_offset
                + layout.cp_local_indices(
                    branch_length, cp_size, cp_group.rank(), hidden_states.device
                )
            )
            branch_offset += branch_length
        dense_cp_indices = torch.cat(dense_cp_indices)
        dense_cp_local = dense_global.index_select(0, dense_cp_indices)
    else:
        dense_cp_local = dense_global
    dense_local = dense_cp_local
    if tp_size > 1:
        dense_local = tensor_parallel.scatter_to_sequence_parallel_region(
            dense_local, group=tp_group
        )

    cumulative_lengths = [0]
    for branch_length in branch_lengths:
        cumulative_lengths.append(cumulative_lengths[-1] + branch_length)
    cu_seqlens = torch.tensor(cumulative_lengths, device=hidden_states.device, dtype=torch.int32)
    packed_seq_params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        cu_seqlens_q_padded=cu_seqlens,
        cu_seqlens_kv_padded=cu_seqlens,
        max_seqlen_q=max(branch_lengths),
        max_seqlen_kv=max(branch_lengths),
        total_tokens=dense_total,
    )
    dense_output_local = layer(
        hidden_states=dense_local, attention_mask=None, packed_seq_params=packed_seq_params
    )
    if isinstance(dense_output_local, tuple):
        dense_output_local = dense_output_local[0]

    dense_output_cp_local = dense_output_local
    if tp_size > 1:
        dense_output_cp_local = tensor_parallel.gather_from_sequence_parallel_region(
            dense_output_cp_local, tensor_parallel_output_grad=False, group=tp_group
        )
    if cp_size > 1:
        dense_rank_order = tensor_parallel.gather_from_sequence_parallel_region(
            dense_output_cp_local, tensor_parallel_output_grad=True, group=cp_group
        )
        rank_order_dense_indices = []
        for rank in range(cp_size):
            branch_offset = 0
            for branch_length in branch_lengths:
                rank_order_dense_indices.append(
                    branch_offset
                    + layout.cp_local_indices(branch_length, cp_size, rank, hidden_states.device)
                )
                branch_offset += branch_length
        rank_order_dense_indices = torch.cat(rank_order_dense_indices)
        inverse_dense_order = torch.empty_like(rank_order_dense_indices)
        inverse_dense_order[rank_order_dense_indices] = torch.arange(
            dense_total, device=hidden_states.device, dtype=torch.long
        )
        dense_output_global = dense_rank_order.index_select(0, inverse_dense_order)
    else:
        dense_output_global = dense_output_cp_local

    # One prompt output is sufficient: every dense branch has the same causal
    # prompt and the oracle is forward-only.  Completion outputs remain tied to
    # their corresponding branch.
    prefix_output = dense_output_global[: layout.prefix_len]
    completion_outputs = []
    branch_offset = 0
    for completion_length, branch_length in zip(
        layout.completion_lens, branch_lengths, strict=True
    ):
        completion_outputs.append(
            dense_output_global[
                branch_offset
                + layout.prefix_len : branch_offset
                + layout.prefix_len
                + completion_length
            ]
        )
        branch_offset += branch_length
    star_output_global = torch.cat([prefix_output, *completion_outputs], dim=0)
    trailing_padding = physical_len - layout.total_len
    if trailing_padding:
        star_output_global = torch.cat(
            [
                star_output_global,
                torch.zeros(
                    trailing_padding,
                    1,
                    star_output_global.shape[-1],
                    dtype=star_output_global.dtype,
                    device=star_output_global.device,
                ),
            ],
            dim=0,
        )
    if cp_size > 1:
        star_output_cp_local = star_output_global.index_select(
            0, layout.cp_local_indices(physical_len, cp_size, cp_group.rank(), hidden_states.device)
        )
    else:
        star_output_cp_local = star_output_global
    if tp_size > 1:
        return tensor_parallel.scatter_to_sequence_parallel_region(
            star_output_cp_local, group=tp_group
        )
    return star_output_cp_local


def _forward_mamba_layer_shared_prefix_cp(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout | SharedPrefixForestLayout
) -> Tensor:
    """Run one Mamba layer with the implementation selected by ``NRL_SP_MAMBA_IMPL``.

    ``ragged_state_fork`` is the default for every topology and root count, including TP1/CP1
    single stars. It scans each chunk-aligned prefix once, forks the state into every branch and
    pads each branch only to its own chunk boundary, so its scan work follows the real completion
    lengths rather than the longest sibling's. It is also checked for exactness:

    - it matches per-row dense Mamba to about 1e-7 in FP32 with IEEE GEMMs and, on the tested
      shapes, stays within 1.5x of the dense BF16 error
      (``tests/unit_tests/ssm/test_shared_prefix_mamba_numerics.py``);
    - its Triton kernels replay bit-exactly under deterministic mode
      (``tests/unit_tests/determinism/kernels/test_shared_prefix_mamba_kernels.py``).

    It calls private mamba_ssm SSD kernels, so stack validation fails early when they cannot be
    imported and names ``state_fork`` as the workaround.

    The other values are explicit opt-ins: ``state_fork`` uses only the public mamba_ssm API and
    pads every branch to its longest sibling; ``replay_prefix`` rescans the prefix per branch as
    a parity baseline; ``packed_recurrence`` and ``packed_fused`` are diagnostic oracles. The
    ``*_training`` variants use ``state_fork`` in evaluation mode.
    """
    implementation = _shared_prefix_mamba_impl()
    if implementation == "state_fork":
        return _forward_mamba_layer_shared_prefix_cp_state_fork(layer, hidden_states, layout)
    if implementation in ("replay_prefix_training", "replay_prefix"):
        # A training experiment: one uninterrupted scan per root can reduce
        # launch/backward overhead, at the cost of repeating the recurrent
        # prefix. Projection, attention and MoE still execute the shared layout.
        # Training mode retains state forking during evaluation. The explicit
        # replay_prefix mode also runs replay in evaluation, allowing numerical
        # logprob checks to exercise the changed training recurrence itself.
        return _forward_mamba_layer_shared_prefix_cp_impl(
            layer,
            hidden_states,
            layout,
            replay_prefix=implementation == "replay_prefix" or layer.training,
        )
    if implementation in ("packed_recurrence_training", "packed_recurrence"):
        return _forward_mamba_layer_shared_prefix_cp_impl(
            layer,
            hidden_states,
            layout,
            replay_prefix=False,
            packed_recurrence=implementation == "packed_recurrence" or layer.training,
        )
    if implementation == "packed_fused":
        if isinstance(layout, SharedPrefixForestLayout):
            raise NotImplementedError(
                "packed_fused diagnostic supports one root; forests require state_fork"
            )
        return _forward_mamba_layer_shared_prefix_cp_packed_fused_oracle(
            layer, hidden_states, layout
        )
    # ragged_state_fork (the default) or ragged_state_fork_training.
    return _forward_mamba_layer_shared_prefix_cp_impl(
        layer,
        hidden_states,
        layout,
        replay_prefix=False,
        ragged_state_fork=implementation == "ragged_state_fork" or layer.training,
    )


# MoE router load-balancing types that leave top-k routing unchanged; any loss they add is
# rejected separately through its coefficient.
_AUX_LOSS_LOAD_BALANCING_TYPES = frozenset({"none", "aux_loss", "seq_aux_loss", "global_aux_loss"})


def _has_nonzero_config_value(value) -> bool:
    """Return whether a scalar or per-layer configuration contains a nonzero value."""
    if value is None:
        return False
    if isinstance(value, (list, tuple)):
        return any(float(item) != 0.0 for item in value)
    return float(value) != 0.0


def _validate_shared_prefix_physical_length(
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    physical_len: int,
    *,
    tp_size: int,
    cp_size: int,
    sequence_parallel: bool,
) -> None:
    """Validate the global star length against its negotiated topology/padding contract."""
    physical_len = int(physical_len)
    if tp_size > 1:
        topology_multiple = 2 * tp_size * cp_size
    elif cp_size > 1:
        topology_multiple = 2 * cp_size
    else:
        topology_multiple = 1
    if isinstance(layout, SharedPrefixForestLayout):
        multiple = layout.padding_multiple or topology_multiple
        if physical_len % multiple or not 0 <= physical_len - layout.total_len < multiple:
            raise ValueError("shared-prefix forest must use minimal global topology padding")
        for root in layout.roots:
            _validate_shared_prefix_physical_length(
                root,
                ((root.total_len + multiple - 1) // multiple) * multiple,
                tp_size=tp_size,
                cp_size=cp_size,
                sequence_parallel=sequence_parallel,
            )
        return
    padding = physical_len - layout.total_len

    if layout.padding_multiple is not None:
        padding_multiple = layout.padding_multiple
        if padding_multiple % topology_multiple:
            raise ValueError(
                "shared-prefix padding_multiple must be divisible by the topology quantum: "
                f"M={padding_multiple}, Q={topology_multiple}, TP={tp_size}, CP={cp_size}"
            )
        for branch, (physical_completion, logical_completion) in enumerate(
            zip(layout.completion_lens, layout.logical_completion_lens, strict=True)
        ):
            if (layout.prefix_len + physical_completion) % padding_multiple or not (
                0 <= physical_completion - logical_completion < padding_multiple
            ):
                raise ValueError(
                    "shared-prefix physical completion span must use the minimal per-branch "
                    "padding to padding_multiple: "
                    f"branch={branch}, prefix={layout.prefix_len}, "
                    f"logical={logical_completion}, physical={physical_completion}, "
                    f"M={padding_multiple}"
                )
        if physical_len % padding_multiple:
            raise ValueError(
                "shared-prefix physical length must be divisible by padding_multiple: "
                f"physical={physical_len}, M={padding_multiple}"
            )
        if not 0 <= padding < padding_multiple:
            raise ValueError(
                "shared-prefix input must use the minimal trailing pad to padding_multiple: "
                f"physical={physical_len}, layout={layout.total_len}, M={padding_multiple}"
            )
        return

    if tp_size > 1:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP physical length must be divisible by 2 * tensor parallel "
                "size * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix TP/SP input must use the minimal trailing pad to a 2*TP*CP "
                f"multiple: physical={physical_len}, layout={layout.total_len}, "
                f"TP={tp_size}, CP={cp_size}"
            )
    elif cp_size == 1:
        if physical_len != layout.total_len:
            raise ValueError(
                f"packed sequence length {physical_len} does not match layout {layout.total_len}"
            )
    else:
        if physical_len % topology_multiple:
            raise ValueError(
                "shared-prefix CP physical length must be divisible by 2 * context parallel size"
            )
        if not 0 <= padding < topology_multiple:
            raise ValueError(
                "shared-prefix CP input must use the minimal trailing pad to a 2*CP multiple: "
                f"physical={physical_len}, layout={layout.total_len}, CP={cp_size}"
            )


def _validate_hybrid_stack(
    stack, layout: SharedPrefixLayout | SharedPrefixForestLayout, *, physical_len: int
) -> None:
    """Reject unsupported shared-prefix stack features before any layer executes.

    Only the configuration, process groups, layer types and the global physical star length are
    inspected, so a caller can run this before the embedding or any other partial forward.
    """
    # New upstream features must not silently bypass their ordinary execution.
    # These combinations need dedicated shared-layout integrations and GPU tests.
    if getattr(stack.config, "moe_num_hash_layers", 0):
        raise NotImplementedError("shared-prefix Hybrid does not yet support hash MoE routing")
    if getattr(stack.config, "quant_recipe", None) is not None:
        raise NotImplementedError("shared-prefix Hybrid does not yet support quantization recipes")
    if getattr(stack.config, "wide_residual", None) is not None:
        raise NotImplementedError("shared-prefix Hybrid does not yet support wide residual streams")
    if getattr(stack.config, "enable_mhc_connections", False):
        raise NotImplementedError("shared-prefix Hybrid does not yet support mHC connections")
    if getattr(stack.config, "attn_logit_softcapping", None) is not None:
        raise NotImplementedError("shared-prefix attention does not yet support logit softcapping")
    tp_size = stack.tp_group.size()
    sequence_parallel = bool(stack.config.sequence_parallel)
    if tp_size > 1 and not sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid TP>1 requires sequence parallelism")
    if tp_size == 1 and sequence_parallel:
        raise NotImplementedError("shared-prefix Hybrid sequence parallelism requires TP>1")
    if stack.config.tensor_model_parallel_size != tp_size:
        raise RuntimeError(
            "shared-prefix Hybrid tensor-parallel config does not match its process group"
        )
    if stack.pg_collection.tp.size() != tp_size:
        raise RuntimeError("shared-prefix Hybrid tensor-parallel groups disagree on their size")
    if stack.pp_group.size() != 1:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports PP1 only")
    cp_group = stack.pg_collection.cp
    cp_size = cp_group.size()
    if stack.config.context_parallel_size != cp_size:
        raise RuntimeError(
            "shared-prefix Hybrid context-parallel config does not match its process group"
        )
    if stack.config.recompute_granularity == "full":
        if stack.config.recompute_method != "uniform":
            raise NotImplementedError(
                'shared-prefix Hybrid full recomputation currently supports only the uniform '
                'method'
            )
        if not isinstance(stack.config.recompute_num_layers, int) or (
            stack.config.recompute_num_layers < 1
        ):
            raise ValueError(
                "shared-prefix Hybrid uniform recomputation requires recompute_num_layers >= 1"
            )
    if stack.config.fine_grained_activation_offloading:
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support fine-grained activation offloading"
        )
    if stack.config.cuda_graph_impl != "none":
        raise NotImplementedError("shared-prefix Hybrid adapter does not support CUDA graphs")
    if stack.config.fp8 or stack.config.fp4:
        raise NotImplementedError("shared-prefix Hybrid adapter currently supports fp16/bf16 only")
    _validate_shared_prefix_physical_length(
        layout, physical_len, tp_size=tp_size, cp_size=cp_size, sequence_parallel=sequence_parallel
    )
    if stack.config.attention_dropout != 0.0 or stack.config.hidden_dropout != 0.0:
        raise NotImplementedError("shared-prefix Hybrid adapter currently requires zero dropout")
    window_size = stack.config.window_size
    if window_size is not None and tuple(window_size) != (-1, -1):
        raise NotImplementedError(
            "shared-prefix Hybrid adapter does not support sliding-window attention"
        )
    if stack.config.softmax_type != "vanilla":
        raise NotImplementedError("shared-prefix fused attention supports only vanilla softmax")

    num_moe_experts = stack.config.num_moe_experts
    expert_bias_enabled = bool(getattr(stack.config, "moe_router_enable_expert_bias", False))
    if num_moe_experts is not None and num_moe_experts > 0:
        if stack.config.moe_router_force_load_balancing:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE routing"
            )
        if stack.config.moe_router_force_biased is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support randomized forced MoE router bias"
            )
        load_balancing = stack.config.moe_router_load_balancing_type
        load_balancing_types = (
            [load_balancing] if isinstance(load_balancing, str) else load_balancing
        )
        # Sinkhorn and quantile balancing replace top-k routing itself. The auxiliary-loss
        # types only add a loss term, which the coefficient check below rejects when active.
        if any(item not in _AUX_LOSS_LOAD_BALANCING_TYPES for item in load_balancing_types):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE router load balancing type "
                f"{load_balancing!r}"
            )
        aux_loss_coeffs = stack.config.moe_aux_loss_coeff
        if isinstance(load_balancing, str):
            aux_loss_coeffs = [aux_loss_coeffs]
        # As in TopKRouter.get_aux_loss_coeff, a coefficient paired with "none" adds no loss.
        if any(
            load_balancing_type != "none" and _has_nonzero_config_value(coeff)
            for load_balancing_type, coeff in zip(load_balancing_types, aux_loss_coeffs)
        ):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE auxiliary router loss"
            )
        if _has_nonzero_config_value(stack.config.moe_z_loss_coeff):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE router z-loss"
            )
        if _has_nonzero_config_value(stack.config.moe_input_jitter_eps):
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE input jitter"
            )
        if expert_bias_enabled and any(
            root.logical_completion_lens is None for _, root in layout.iter_roots()
        ):
            raise NotImplementedError(
                "shared-prefix MoE expert-bias accounting requires explicit physical "
                "branch padding and logical completion lengths"
            )
        if stack.config.moe_expert_capacity_factor is not None:
            raise NotImplementedError(
                'shared-prefix Hybrid adapter does not support MoE expert capacity or token '
                'dropping'
            )
        # Both flex backends can drop tokens over the per-rank budget. A dropped prompt row
        # would remove its expert contribution from every completion that shares it.
        if stack.config.moe_expert_rank_capacity_factor is not None:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE expert rank capacity or "
                "token dropping: the per-rank budget counts physical star rows, and a dropped "
                "prompt row is lost for every completion that shares it"
            )
        # Replay targets are recorded per dense row; shared-prefix routes physical star rows.
        if stack.config.moe_enable_routing_replay:
            raise NotImplementedError(
                "shared-prefix Hybrid adapter does not support MoE routing replay: replay "
                "targets are indexed by dense rows, while a shared-prefix forward routes "
                "physical star rows that store each prompt once"
            )
        if getattr(stack.config, "mlp_chunks_for_training", 1) != 1:
            raise NotImplementedError(
                "shared-prefix Hybrid MoE adapter does not support training MLP chunking"
            )

    has_attention = False
    for layer in stack.layers:
        if isinstance(layer, MambaLayer):
            if not isinstance(layer.mixer, MambaMixer):
                raise NotImplementedError(
                    "shared-prefix state forking only supports MambaMixer-backed Mamba layers"
                )
            _validate_mamba_fork(layer.mixer)
            if layer.mixer.pg_collection.tp.size() != tp_size:
                raise RuntimeError(
                    "shared-prefix Mamba TP helper does not match the Hybrid stack TP group"
                )
            if layer.mixer.cp.cp_size != cp_size:
                raise RuntimeError(
                    "shared-prefix Mamba CP helper does not match the Hybrid stack CP group"
                )
            # The shared path shards the star by attention zigzag and relies on the Mamba CP
            # helper to undo it; a contiguous linear layout would skip that reordering.
            if cp_size > 1 and layer.mixer.cp.sequence_is_contiguous:
                raise NotImplementedError(
                    "shared-prefix Mamba CP requires linear_cp_layout='zigzag': the shared-prefix "
                    "forward shards each star in zigzag order and relies on the Mamba CP helper "
                    "to undo it"
                )
        elif isinstance(layer, TransformerLayer):
            if not isinstance(layer.self_attention, (IdentityOp, SelfAttention)):
                raise NotImplementedError(
                    'shared-prefix Hybrid adapter supports standard self-attention and MLP/MoE '
                    'layers'
                )
            if (
                isinstance(layer.self_attention, SelfAttention)
                and layer.self_attention.checkpoint_core_attention
            ):
                raise NotImplementedError(
                    'shared-prefix fused attention does not yet support selective '
                    'core-attention recomputation'
                )
            if isinstance(layer.self_attention, SelfAttention) and (
                stack.config.qk_clip or stack.config.log_max_attention_logit
            ):
                raise NotImplementedError(
                    'shared-prefix fused attention does not yet produce QK-clipping/max-logit '
                    'statistics'
                )
            if isinstance(layer.self_attention, SelfAttention):
                has_attention = True
                if layer.self_attention.pg_collection.tp.size() != tp_size:
                    raise RuntimeError(
                        'shared-prefix attention TP helper does not match the Hybrid stack TP '
                        'group'
                    )
                if layer.self_attention.pg_collection.cp.size() != cp_size:
                    raise RuntimeError(
                        'shared-prefix attention CP helper does not match the Hybrid stack CP '
                        'group'
                    )
            if cp_size > 1 and isinstance(layer.self_attention, SelfAttention):
                from megatron.core.models.hybrid.shared_prefix_fused import (
                    _cp_kv_head_slices_for_destinations,
                )

                if stack.config.num_attention_heads % tp_size:
                    raise NotImplementedError(
                        "shared-prefix attention requires query heads divisible by TP size"
                    )
                query_heads = stack.config.num_attention_heads // tp_size
                # This is the actual local K/V tensor width. When global KV heads are fewer than
                # TP ranks, SelfAttention replicates one KV head across the relevant TP ranks.
                kv_heads = layer.self_attention.num_query_groups_per_partition
                _cp_kv_head_slices_for_destinations(query_heads, kv_heads, cp_size)
            if expert_bias_enabled and getattr(layer, "is_moe_layer", False):
                from megatron.core.transformer.moe.moe_layer import MoELayer
                from megatron.core.transformer.moe.router import TopKRouter

                if not isinstance(layer.mlp, MoELayer) or not isinstance(
                    layer.mlp.router, TopKRouter
                ):
                    raise NotImplementedError(
                        "shared-prefix expert-bias accounting requires an MCore "
                        "MoELayer with TopKRouter"
                    )
        else:
            raise NotImplementedError(
                f"shared-prefix state forking is not implemented for {type(layer).__name__}"
            )
    if has_attention:
        # The attention entry repeats this check; failing here precedes the first Mamba layer.
        from megatron.core.models.hybrid.shared_prefix_fused import _MIN_FLASH_ATTN_VERSION

        try:
            flash_attn_supported = is_fa_min_version(_MIN_FLASH_ATTN_VERSION)
        except ImportError:
            flash_attn_supported = False
        if not flash_attn_supported:
            raise RuntimeError(
                f"shared-prefix attention requires flash-attn >= {_MIN_FLASH_ATTN_VERSION} for its "
                "exact backward"
            )


def forward_hybrid_stack_shared_prefix(
    stack,
    hidden_states: Tensor,
    layout: SharedPrefixLayout | SharedPrefixForestLayout,
    *,
    rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
    position_embedding_type: str = "rope",
    exclude_sequence_padding_from_expert_bias: bool = False,
    validated_physical_len: int | None = None,
) -> Tensor:
    """Explicit exact-prompt star forward for a supported ``HybridStack`` topology.

    Normal ``HybridStack.forward`` and ``Attention.forward`` behavior is unchanged unless this
    function installs its scoped private forest descriptor. The descriptor is always removed in a
    ``finally`` block, including when a layer raises.

    ``exclude_sequence_padding_from_expert_bias`` states the caller's dense expert-bias
    convention for per-branch padding rows. The default counts them, as a dense forward with
    ``padding_mask=None`` does; ``True`` matches a dense caller that masks packed-sequence padding.
    Trailing topology padding is never counted.

    ``validated_physical_len`` is the global physical length for which the caller has already
    validated this stack and layout, as ``HybridModel.forward`` does before its embedding. The
    stack validation is then not repeated; the length implied by ``hidden_states`` must match.
    """
    if hidden_states.dtype not in (torch.float16, torch.bfloat16):
        raise TypeError("fused shared-prefix attention requires fp16 or bf16 hidden states")
    if hidden_states.ndim != 3 or hidden_states.shape[1] != 1:
        raise ValueError(
            "shared-prefix Hybrid input must have shape [sequence/(TP*CP), 1, hidden] "
            "when sequence parallelism is enabled"
        )
    cp_group = stack.pg_collection.cp
    tp_group = stack.pg_collection.tp
    tp_size = tp_group.size()
    sequence_shards = tp_size if stack.config.sequence_parallel else 1
    physical_len = hidden_states.shape[0] * cp_group.size() * sequence_shards
    if validated_physical_len is None:
        _validate_hybrid_stack(stack, layout, physical_len=physical_len)
    elif validated_physical_len != physical_len:
        raise RuntimeError(
            f"shared-prefix hidden states imply physical length {physical_len}, but the stack "
            f"was validated for {validated_physical_len}"
        )
    has_attention = any(
        isinstance(layer, TransformerLayer) and isinstance(layer.self_attention, SelfAttention)
        for layer in stack.layers
    )
    if position_embedding_type not in ("rope", "none"):
        raise NotImplementedError(
            "shared-prefix attention supports only RoPE or positionless Hybrid models"
        )
    if has_attention and position_embedding_type == "rope" and rotary_pos_emb is None:
        raise ValueError("position-aware rotary_pos_emb is required for shared-prefix attention")
    if position_embedding_type == "none" and rotary_pos_emb is not None:
        raise ValueError("positionless shared-prefix attention must not receive rotary_pos_emb")

    token_multiplicities = None
    expert_bias_enabled = bool(getattr(stack.config, "moe_router_enable_expert_bias", False))
    if expert_bias_enabled:
        token_multiplicities = layout.padded_token_multiplicities(
            physical_len,
            hidden_states.device,
            exclude_sequence_padding=exclude_sequence_padding_from_expert_bias,
        )
        if cp_group.size() > 1:
            token_multiplicities = token_multiplicities.index_select(
                0,
                layout.cp_local_indices(
                    physical_len, cp_group.size(), cp_group.rank(), hidden_states.device
                ),
            )
        if tp_size > 1:
            if token_multiplicities.numel() % tp_size:
                raise RuntimeError(
                    "shared-prefix CP-local token multiplicities must divide evenly over TP"
                )
            token_multiplicities = torch.chunk(token_multiplicities, tp_size, dim=0)[
                tp_group.rank()
            ].contiguous()
        if token_multiplicities.numel() != hidden_states.shape[0]:
            raise RuntimeError(
                "shared-prefix token multiplicity ownership does not match the local hidden rows"
            )

    @router_gating_token_blocks()
    def forward_layer_range(hidden_states: Tensor, start: int, end: int) -> Tensor:
        for layer in stack.layers[start:end]:
            moe_layer = (
                layer.mlp if expert_bias_enabled and getattr(layer, "is_moe_layer", False) else None
            )
            if moe_layer is not None:
                if moe_layer._shared_prefix_token_multiplicities is not None:
                    raise RuntimeError("nested shared-prefix MoE dispatch is not supported")
                moe_layer._shared_prefix_token_multiplicities = token_multiplicities
            try:
                if isinstance(layer, MambaLayer):
                    # Every topology and root count honors NRL_SP_MAMBA_IMPL.
                    hidden_states = _forward_mamba_layer_shared_prefix_cp(
                        layer, hidden_states, layout
                    )
                elif isinstance(layer.self_attention, IdentityOp):
                    hidden_states = layer(hidden_states=hidden_states, attention_mask=None)
                else:
                    attention = layer.self_attention
                    if attention._shared_prefix_forest is not None:
                        raise RuntimeError(
                            "nested shared-prefix attention dispatch is not supported"
                        )
                    attention._shared_prefix_forest = layout.forest
                    try:
                        hidden_states = layer(
                            hidden_states=hidden_states,
                            attention_mask=None,
                            rotary_pos_emb=rotary_pos_emb,
                        )
                    finally:
                        del attention._shared_prefix_forest
            finally:
                if moe_layer is not None:
                    del moe_layer._shared_prefix_token_multiplicities

            if isinstance(hidden_states, tuple):
                hidden_states = hidden_states[0]
        return hidden_states

    if stack.config.recompute_granularity == "full" and stack.training:
        chunk_size = stack.config.recompute_num_layers
        for start in range(0, len(stack.layers), chunk_size):
            end = min(start + chunk_size, len(stack.layers))

            def custom_forward(hidden_states: Tensor, start=start, end=end) -> Tensor:
                # Scopes live inside the callable so backward replay reconstructs them.
                return forward_layer_range(hidden_states, start, end)

            hidden_states = tensor_parallel.checkpoint(
                custom_forward, stack.config.distribute_saved_activations, hidden_states
            )
    else:
        hidden_states = forward_layer_range(hidden_states, 0, len(stack.layers))

    if stack.post_process and stack.post_layer_norm:
        hidden_states = stack.final_norm(hidden_states)
    return make_viewless_tensor(
        inp=hidden_states, requires_grad=hidden_states.requires_grad, keep_graph=True
    )


# Capability tokens are contract identifiers: each names a code path that MCore implements and
# guards. They are not qualification evidence for a topology or a model.
# The legacy scalar remains the CP1 negotiation surface. CP-aware integrations must negotiate
# against the collection so installing the CP track cannot regress existing CP1 runs.
# ``SHARED_PREFIX_CP_TRAINING_CAPABILITY`` is advertised independently so integrations can retain
# the CP1 fast path while requiring the CP>1 Hybrid forward/backward contract.
SHARED_PREFIX_TRAINING_CAPABILITY = "hybrid_star_cp1_tp1_v1"
SHARED_PREFIX_CP_TRAINING_CAPABILITY = "hybrid_star_cp_v1"
SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY = "hybrid_star_explicit_physical_padding_v1"
SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY = "hybrid_star_moe_expert_bias_v1"
SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY = "hybrid_star_full_uniform_recompute_v1"
SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp1_tp_sp_v1"
SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY = "hybrid_star_cp_tp_sp_v1"
# Target-model feature: Nemotron-H attention is positionless while Mamba remains
# state-positioned by sequence order.
SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY = "hybrid_star_positionless_attention_v1"
# MTP predictor feature: reconstruct dense attention/MLP or attention/MoE heads from
# the shared-prefix physical layout on the implemented TP/CP topologies.
SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY = "hybrid_star_mtp_dense_heads_v1"
SHARED_PREFIX_GROUP_EXECUTION_CAPABILITY = "hybrid_forest_stable_router_v1"
# Topology and feature capabilities are independent so integrations can negotiate their exact
# conjunction without inferring support from a broader aggregate token.
SHARED_PREFIX_TRAINING_CAPABILITIES = frozenset(
    {
        SHARED_PREFIX_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TRAINING_CAPABILITY,
        SHARED_PREFIX_EXPLICIT_PHYSICAL_PADDING_CAPABILITY,
        SHARED_PREFIX_MOE_EXPERT_BIAS_CAPABILITY,
        SHARED_PREFIX_FULL_RECOMPUTE_CAPABILITY,
        SHARED_PREFIX_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_CP_TP_SP_TRAINING_CAPABILITY,
        SHARED_PREFIX_POSITIONLESS_ATTENTION_CAPABILITY,
        SHARED_PREFIX_MTP_DENSE_HEADS_CAPABILITY,
        SHARED_PREFIX_GROUP_EXECUTION_CAPABILITY,
    }
)
