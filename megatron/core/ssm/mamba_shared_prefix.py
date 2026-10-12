# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix Mamba execution: one prefix scan whose state forks into every completion.

A shared-prefix layout stores each prompt once, followed by its completions
(:mod:`megatron.core.shared_prefix_layout`). :func:`forward_mamba_layer_shared_prefix` runs one
``MambaLayer`` on that layout: the prefix is scanned once and its convolution and SSM state are
forked, differentiably, into every completion branch, including at CP>1 through
sequence-to-head collectives. ``NRL_SP_MAMBA_IMPL`` selects the implementation; the default,
``ragged_state_fork``, uses the ragged forest kernels in :mod:`megatron.core.ssm.mamba_ragged`.

:func:`validate_mamba_layer_shared_prefix` rejects unsupported layers; callers run it for every
Mamba layer before any layer executes.
"""

import os

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

from megatron.core import tensor_parallel
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.shared_prefix_layout import SharedPrefixForestLayout, SharedPrefixLayout
from megatron.core.ssm.mamba_branch_layout import merge_mamba_branches, pack_mamba_branches
from megatron.core.ssm.mamba_layer import MambaLayer
from megatron.core.ssm.mamba_mixer import (
    MAMBA_HAS_STATE_DTYPE,
    MambaMixer,
    causal_conv1d_fn,
    mamba_chunk_scan_combined,
)
from megatron.core.typed_torch import apply_module


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

    :func:`validate_mamba_layer_shared_prefix` calls this and also compares the mixer's TP/CP
    sizes with the caller's; the layer forwards do not repeat either check.
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
    # The caller passes a physical length >= layout.total_len, padded after the final
    # completion; scanning that padding as the branch's causal tail cannot affect any real-token
    # output.
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
    # Callers validate this length against the layout and topology before any layer runs.
    physical_len = hidden_states.shape[0] * tp_size * cp_size

    # [star/(TP*CP),1,H] -> one canonical global star.  The oracle is a forward-only
    # diagnostic (the dispatcher rejects it with grad enabled), but it uses the same
    # collectives and order as the differentiable paths, so the forward topology
    # is representative.
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


def forward_mamba_layer_shared_prefix(
    layer: MambaLayer, hidden_states: Tensor, layout: SharedPrefixLayout | SharedPrefixForestLayout
) -> Tensor:
    """Run one Mamba layer with the implementation selected by ``NRL_SP_MAMBA_IMPL``.

    ``ragged_state_fork`` is the default for every topology and root count, including TP1/CP1
    single stars. It scans each chunk-aligned prefix once, forks the state into every branch and
    pads each branch only to its own chunk boundary, so its scan work follows the real completion
    lengths rather than the longest sibling's. It is also checked for exactness:

    - in FP32 with IEEE GEMMs its outputs match per-row dense Mamba to about 1e-7 and its
      gradients to a few 1e-6 (asserted at 5e-5 relative L2); on the tested shapes it stays
      within 1.5x of the dense BF16 error
      (``tests/unit_tests/ssm/test_shared_prefix_mamba_numerics.py``);
    - its Triton kernels replay bit-exactly under deterministic mode
      (``tests/unit_tests/determinism/kernels/test_shared_prefix_mamba_kernels.py``).

    It calls private mamba_ssm SSD kernels, so :func:`validate_mamba_layer_shared_prefix` fails
    early when they cannot be imported and names ``state_fork`` as the workaround.

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
                "packed_fused diagnostic supports one root; use another backend for forests"
            )
        if torch.is_grad_enabled():
            # A forward-only diagnostic: its backward, which routes every branch's prompt
            # gradient through one branch's prompt output, has never been validated against
            # dense rows.
            raise NotImplementedError(
                "packed_fused is a forward-only diagnostic oracle; run it under "
                "torch.no_grad() or select another NRL_SP_MAMBA_IMPL"
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


def validate_mamba_layer_shared_prefix(layer: MambaLayer, *, tp_size: int, cp_size: int) -> None:
    """Reject a Mamba layer the shared-prefix forward cannot run on the given TP/CP topology.

    Callers run this for every Mamba layer before any layer executes;
    :func:`forward_mamba_layer_shared_prefix` does not repeat it.
    """
    if not isinstance(layer.mixer, MambaMixer):
        raise NotImplementedError(
            "shared-prefix state forking only supports MambaMixer-backed Mamba layers"
        )
    _validate_mamba_fork(layer.mixer)
    if layer.mixer.pg_collection.tp.size() != tp_size:
        raise RuntimeError(
            f"shared-prefix Mamba TP helper size {layer.mixer.pg_collection.tp.size()} does not "
            f"match the caller's TP size {tp_size}"
        )
    if layer.mixer.cp.cp_size != cp_size:
        raise RuntimeError(
            f"shared-prefix Mamba CP helper size {layer.mixer.cp.cp_size} does not match the "
            f"caller's CP size {cp_size}"
        )
    # The shared path shards the star by attention zigzag and relies on the Mamba CP helper to
    # undo it; a contiguous linear layout would skip that reordering.
    if cp_size > 1 and layer.mixer.cp.sequence_is_contiguous:
        raise NotImplementedError(
            "shared-prefix Mamba CP requires linear_cp_layout='zigzag': the shared-prefix "
            "forward shards each star in zigzag order and relies on the Mamba CP helper "
            "to undo it"
        )
