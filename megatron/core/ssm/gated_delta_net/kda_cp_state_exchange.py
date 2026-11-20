# Copyright (c) 2023-2026 Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# The two pre-process functions below keep the interface and the summary kernels of
# ``chunk_gated_delta_rule_fwd_h_pre_process`` and ``chunk_gated_delta_rule_bwd_dhu_pre_process``
# in flash-linear-attention 0.5.1 (https://github.com/fla-org/flash-linear-attention,
# fla/ops/cp/chunk_delta_h.py) and replace their all-gather and merge with
# ``megatron.core.ssm.context_parallel.state_exchange``.
#
# Licensed under the MIT license; see the LICENSE file in the repository root.

"""KDA chunkwise CP: pass the boundary states with two all-gathers instead of one over all ranks.

FLA's KDA context parallelism summarizes each rank's boundary sequence as ``[delta_S | M]``
(FP32 ``[HV, K, V + K]``), all-gathers the summaries of every CP rank and composes the ranks
before (forward) or after (backward) this one. At CP 64 that moves 63 summaries, 504 MiB for
64 heads of 128 x 128, into every rank, three times per layer and step (forward, recompute and
backward). :func:`install` points FLA's KDA forward and backward at the functions here, which
compute the same summaries with FLA's kernels and pass them with
:func:`~megatron.core.ssm.context_parallel.state_exchange.exchange_boundary_state`: an all-gather
inside blocks of 8 ranks and one across the blocks, 14 summaries per rank at CP 64.

They need the CP layout of every rank, which :func:`build_cp_context` attaches to FLA's context
from the global sequence boundaries. Without it, or for 8 CP ranks or fewer, they call FLA's own
functions. The states equal FLA's up to FP32 association order.
"""

from __future__ import annotations

import torch

try:
    import triton
    from fla.ops.cp import build_cp_context as _fla_build_cp_context
    from fla.ops.cp import chunk_delta_h as _fla_cp_chunk_delta_h
    from fla.ops.cp.chunk_delta_h import (
        pre_process_bwd_kernel_merged,
        pre_process_fwd_kernel_merged,
    )

    HAVE_FLA_CP = True
except ImportError:
    HAVE_FLA_CP = False

from megatron.core.ssm.context_parallel.state_exchange import (
    block_size,
    exchange_boundary_state,
    rank_chains,
)

_CHAINS_ATTR = "megatron_rank_chains"


def build_cp_context(cu_seqlens, group, conv1d_kernel_size=None, cu_seqlens_cpu=None):
    """FLA's ``build_cp_context``, with every rank's CP layout attached for the state exchange."""
    context = _fla_build_cp_context(
        cu_seqlens=cu_seqlens,
        group=group,
        conv1d_kernel_size=conv1d_kernel_size,
        cu_seqlens_cpu=cu_seqlens_cpu,
    )
    if getattr(context, _CHAINS_ATTR, None) is None and block_size(group.size()) is not None:
        host = cu_seqlens_cpu if cu_seqlens_cpu is not None else cu_seqlens.cpu()
        chains = rank_chains(host, group.size())
        rank = group.rank()
        # FLA derives the same counts for this rank; disagreeing layouts keep FLA's exchange.
        if (chains.pre[rank], chains.post[rank]) == (context.pre_num_ranks, context.post_num_ranks):
            setattr(context, _CHAINS_ATTR, chains)
    return context


def _chains(context):
    if context is None or context.group is None or context.cu_seqlens_cpu is None:
        return None
    return getattr(context, _CHAINS_ATTR, None)


def chunk_gated_delta_rule_fwd_h_pre_process(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    chunk_size: int = 64,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    initial_state: torch.Tensor | None = None,
    context=None,
) -> torch.Tensor | None:
    """FLA's forward CP pre-process with the boundary states passed by the two-level exchange."""
    if context is None or context.group is None:
        return initial_state
    chains = _chains(context)
    if chains is None:
        return _fla_cp_chunk_delta_h.chunk_gated_delta_rule_fwd_h_pre_process(
            k=k,
            w=w,
            u=u,
            g=g,
            gk=gk,
            bg=bg,
            v=v,
            chunk_size=chunk_size,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            initial_state=initial_state,
            context=context,
        )
    assert initial_state is None, "When enable CP, the provided initial_state must be None."

    _, T, H, K = k.shape
    V, HV = u.shape[-1], u.shape[2]
    assert K <= 256, "current kernel does not support head dimension larger than 256."
    N = k.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1

    hm = k.new_zeros(HV, K, V + K, dtype=torch.float32)
    if context.post_num_ranks >= 1:
        block = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, block) + triton.cdiv(K, block), HV)
        # As in FLA: for DPLR, v provides the original v for the h contributions, while u
        # remains the WY-processed values.
        pre_process_fwd_kernel_merged[grid](
            k=k,
            v=u if v is None else v,
            w=w,
            g=g,
            gk=gk,
            bg=bg,
            u=u,
            hm=hm,
            cu_seqlens=cu_seqlens[-2:],
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=chunk_size,
            BK1=triton.next_power_of_2(K),
            BLOCK_SIZE=block,
            MULTI_SEQS=False,
        )
    incoming = exchange_boundary_state(hm, group=context.group, chains=chains)
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    initial_state = k.new_zeros(state_shape, dtype=torch.float32)
    if incoming is not None:
        initial_state[0] = incoming.transpose(-1, -2) if state_v_first else incoming
    return initial_state


def chunk_gated_delta_rule_bwd_dhu_pre_process(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    do: torch.Tensor,
    dv: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    bg: torch.Tensor | None = None,
    scale: float | None = None,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    dht: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    context=None,
    chunk_size: int = 64,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """FLA's backward CP pre-process with the state gradients passed by the two-level exchange."""
    if context is None or context.group is None:
        return dht, initial_state
    chains = _chains(context)
    if chains is None:
        return _fla_cp_chunk_delta_h.chunk_gated_delta_rule_bwd_dhu_pre_process(
            q=q,
            k=k,
            w=w,
            do=do,
            dv=dv,
            g=g,
            gk=gk,
            bg=bg,
            scale=scale,
            state_v_first=state_v_first,
            cu_seqlens=cu_seqlens,
            dht=dht,
            initial_state=initial_state,
            context=context,
            chunk_size=chunk_size,
        )
    assert dht is None, "When enable CP, the provided dht must be None."

    _, T, H, K = q.shape
    V, HV = do.shape[-1], do.shape[2]
    assert K <= 256, "current kernel does not support head dimension being larger than 256."
    N = q.shape[0] if cu_seqlens is None else len(cu_seqlens) - 1

    dhm = q.new_zeros(HV, K, V + K, dtype=torch.float32)
    if context.pre_num_ranks >= 1:
        block = 32 if K <= 64 else 64
        grid = (triton.cdiv(V, block) + triton.cdiv(K, block), HV)
        pre_process_bwd_kernel_merged[grid](
            q=q,
            k=k if bg is None else bg,
            w=w,
            g=g,
            gk=gk,
            do=do,
            dhm=dhm,
            dv=dv,
            cu_seqlens=cu_seqlens[:2],
            scale=scale,
            T=T,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BT=chunk_size,
            BK1=triton.next_power_of_2(K),
            BLOCK_SIZE=block,
            USE_BG=bg is not None,
        )
    incoming = exchange_boundary_state(dhm, group=context.group, chains=chains, reverse=True)
    state_shape = (N, HV, V, K) if state_v_first else (N, HV, K, V)
    dht = q.new_zeros(state_shape, dtype=torch.float32)
    if incoming is not None:
        dht[-1] = incoming.transpose(-1, -2) if state_v_first else incoming
    # initial_state is None in CP mode: the backward kernel needs only this rank's dht.
    return dht, None


def install() -> None:
    """Point FLA's KDA chunk forward and backward at the pre-process functions here.

    Only the names that ``fla.ops.kda`` imported change; FLA's other operators keep FLA's
    all-gather. Contexts that :func:`build_cp_context` did not build keep FLA's exchange too.
    Idempotent; a no-op without FLA.
    """
    if not HAVE_FLA_CP:
        return
    from fla.ops.kda import chunk_bwd as _fla_kda_chunk_bwd
    from fla.ops.kda import chunk_fwd as _fla_kda_chunk_fwd

    _fla_kda_chunk_fwd.chunk_gated_delta_rule_fwd_h_pre_process = (
        chunk_gated_delta_rule_fwd_h_pre_process
    )
    _fla_kda_chunk_bwd.chunk_gated_delta_rule_bwd_dhu_pre_process = (
        chunk_gated_delta_rule_bwd_dhu_pre_process
    )
