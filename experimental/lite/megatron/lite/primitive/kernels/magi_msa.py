# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Adapter between MLite attention modules and the MagiAttention MSA extension.

This is the only MLite module that imports ``magi_attention`` / ``magi_attn_extensions``.
Everything is imported lazily so that models using the ``dense`` / ``flex`` MSA backends do
not need MagiAttention installed.

Ownership split for ``msa_backend="magi"``
* MLite owns the projections, QK-norm, RoPE, the indexer projections and the output linear.
* Magi owns the CP token dispatch, the required-K / index-K communication, the ``msa_v1``
  indexer + sparse-attention kernels (MSA layers) and the varlen-causal attention of the
  dense layers (native ``calc_attn`` on the same dispatched layout).

Indexer freezing: the indexer projections run under ``torch.no_grad()`` and their parameters
carry ``requires_grad=False``; ``kl_loss_coeff=0`` only disables Magi's KL loss, it is *not*
a freeze mechanism (zero grads would still drift under decoupled weight decay).
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist

from megatron.lite.primitive.parallel import ParallelState

# msa_v1 kernel lock (MM-Sparse-Attention/training/msa_v1, Magi backends/msa_v1.py::MSAv1Property)
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
NUM_INDEX_HEADS = 4
HEAD_DIM = 128
INDEX_HEAD_DIM = 128
BLOCK_SIZE = 128
TOPK = 16
LOCAL_BLOCKS = 1
INIT_BLOCKS = 0

_MAGI: SimpleNamespace | None = None
_SINGLE_RANK_GROUPS: dict[int, dist.ProcessGroup] = {}


def _ensure_magi() -> SimpleNamespace:
    """Lazily import the Magi MSA extension and the native flexible-attention API."""
    global _MAGI
    if _MAGI is not None:
        return _MAGI
    try:
        from magi_attention.api import calc_attn
        from magi_attention.api import dispatch as native_dispatch
        from magi_attention.api import magi_attn_varlen_key
        from magi_attention.config import DistAttnConfig
        from magi_attention.meta.solver.dispatch_solver import DispatchConfig, MinHeapDispatchAlg
        from magi_attn_extensions import MSA as msa
    except ImportError as e:
        raise ImportError(
            "msa_backend='magi' requires MagiAttention (Magi-MSA branch feat/tree-attn-msa) and its "
            "extensions plus MM-Sparse-Attention (msa_v1). Install with "
            "`MAGI_ATTENTION_SKIP_CUDA_BUILD=1 pip install --no-build-isolation -e <Magi-MSA>` and "
            "`pip install --no-build-isolation -e <Magi-MSA>/extensions` (with magi_attn_ext built for the dense "
            "layers' calc_attn), plus flash_attn_cute / magi_to_hstu_cuda / create_block_mask_cuda from the Magi "
            "flash-attention fork for the fa4 dense backend (all satisfied by nvidia-cutlass-dsl==4.5.2)."
        ) from e
    _MAGI = SimpleNamespace(
        msa=msa,
        calc_attn=calc_attn,
        magi_attn_varlen_key=magi_attn_varlen_key,
        native_dispatch=native_dispatch,
        DistAttnConfig=DistAttnConfig,
        DispatchConfig=DispatchConfig,
        MinHeapDispatchAlg=MinHeapDispatchAlg,
    )
    return _MAGI


@dataclass(frozen=True)
class MagiMsaSettings:
    """Runtime knobs of the ``magi`` backend (see ``ImplConfig.magi_*``)."""

    chunk_size: int = 2048  # CP dispatch granularity (tokens); total tokens are padded to chunk_size * cp
    high_precision_reduce: bool = False  # FP32 required-dK group-reduce in backward
    dense_kernel_backend: str = "fa4"  # MAGI_ATTENTION_KERNEL_BACKEND for the dense layers' calc_attn (fa4 | sdpa_ol)
    # Ordered (writer-rank/semaphore) FP32 accumulation in the msa_v1 sparse/KL backward and score-ordered indexer
    # top-k: bitwise-reproducible dQ at the cost of the unordered atomic fast path.
    deterministic: bool = False


def validate_kernel_shapes(
    *,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
    index_n_heads: int,
    index_head_dim: int,
    block_size: int,
    topk_blocks: int,
    local_blocks: int,
) -> None:
    got = dict(
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        index_n_heads=index_n_heads,
        index_head_dim=index_head_dim,
        block_size=block_size,
        topk_blocks=topk_blocks,
        local_blocks=local_blocks,
    )
    want = dict(
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        index_n_heads=NUM_INDEX_HEADS,
        index_head_dim=INDEX_HEAD_DIM,
        block_size=BLOCK_SIZE,
        topk_blocks=TOPK,
        local_blocks=LOCAL_BLOCKS,
    )
    bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
    if bad:
        raise ValueError(
            "msa_backend='magi' uses the fixed-shape msa_v1 kernels; mismatched fields (got, required): "
            f"{bad}. Use msa_backend='flex' for other shapes."
        )


def validate_device() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        raise RuntimeError("msa_backend='magi' requires an SM100-family GPU (B200/GB300)")


def build_msa_config(settings: MagiMsaSettings, *, head_dim: int = HEAD_DIM, index_head_dim: int = INDEX_HEAD_DIM) -> Any:
    """``MsaConfig`` for the MSAV1 backend. KL/LSE coefficients are 0: the indexer is frozen (decision A)."""
    m = _ensure_magi()
    return m.msa.MsaConfig(
        kernel_backend=m.msa.MsaKernelBackend.MSAV1,
        softmax_scale=head_dim**-0.5,
        indexer_softmax_scale=index_head_dim**-0.5,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
        topk_blocks=TOPK,
        kl_loss_coeff=0.0,
        lse_loss_coeff=0.0,
        score_in_fp32=True,
        deterministic=settings.deterministic,
        high_precision_reduce=settings.high_precision_reduce,
    )


def ensure_single_process_group() -> None:
    """Initialise a world-size-1 NCCL process group (for single-GPU tests / scripts)."""
    if dist.is_initialized():
        return
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group("nccl")
        return
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0)


def msa_cp_group(ps: ParallelState) -> dist.ProcessGroup:
    """The CP group Magi should use; CP=1 needs a real world-size-one group."""
    if ps.cp_group is not None:
        return ps.cp_group
    if not dist.is_initialized():
        raise RuntimeError("msa_backend='magi' needs torch.distributed initialised (call ensure_single_process_group())")
    world = dist.get_world_size()
    group = _SINGLE_RANK_GROUPS.get(world)
    if group is None:
        if world == 1:
            group = dist.group.WORLD
        else:
            group, _ = dist.new_subgroups_by_enumeration([[r] for r in range(world)], backend="nccl")
        _SINGLE_RANK_GROUPS[world] = group
    return group


@dataclass
class MagiMsaContext:
    """Per-micro-batch state shared by every layer of a ``magi`` forward.

    ``position_ids`` are the doc-local positions of this rank's dispatched tokens (RoPE input).
    ``dense_key`` is the native flexible-attention runtime key for the full-attention (non-MSA)
    layers, built on the same dispatch layout as ``key``; ``None`` when this PP stage has no dense
    layer. It is consumed by GQAttention's existing ``attention_backend="magi"`` core (see
    ``primitive.modules.attention.magi.MagiDotProductAttention``) via ``dense_packed_seq_params()``
    -- dense layers reuse that generic backend rather than a MiniMax-M3-specific one.
    """

    key: Any
    dense_key: Any | None
    cp_size: int
    chunk_size: int
    cu_seqlens_host: tuple[int, ...]
    num_real_docs: int
    pad: int
    max_seqlen: int
    position_ids: torch.Tensor  # int64 [T_local]

    @property
    def local_tokens(self) -> int:
        return int(self.position_ids.numel())

    def rope_freqs_for(self, rotary) -> torch.Tensor:
        """``[T_local, 1, 1, rot]`` rotary table for the dispatched tokens."""
        return rotary.get_emb_for_positions(self.position_ids)

    def dense_packed_seq_params(self) -> Any:
        """``packed_seq_params`` for ``GQAttention``'s generic ``attention_backend="magi"`` path."""
        from types import SimpleNamespace

        if self.dense_key is None:
            raise RuntimeError("MagiMsaContext has no dense_key; plan_magi_batch(need_dense_key=True) is required")
        return SimpleNamespace(
            qkv_format="magi",
            magi_runtime_key=self.dense_key,
            max_seqlen_q=self.max_seqlen,
            max_seqlen_kv=self.max_seqlen,
        )


def dispatch_config(settings: MagiMsaSettings) -> Any:
    m = _ensure_magi()
    return m.DispatchConfig(chunk_size=settings.chunk_size, uneven_shard=False, alg=m.MinHeapDispatchAlg())


def plan_magi_batch(
    seq_lens_host: Sequence[int],
    *,
    ps: ParallelState,
    settings: MagiMsaSettings,
    msa_config: Any,
    need_dense_key: bool,
    num_attention_heads: int = NUM_Q_HEADS,
    num_key_value_heads: int = NUM_KV_HEADS,
    head_dim: int = HEAD_DIM,
) -> MagiMsaContext:
    """Build (cached) Magi runtimes for one packed batch.

    Total tokens are padded to a multiple of ``chunk_size * cp`` by appending the padding as an
    extra trailing document, so real documents keep their ranges and pad queries only see pad keys.
    """
    m = _ensure_magi()
    seq_lens = [int(s) for s in seq_lens_host]
    if not seq_lens or min(seq_lens) <= 0:
        raise ValueError("plan_magi_batch needs at least one non-empty sequence")
    total = sum(seq_lens)
    align = settings.chunk_size * ps.cp_size
    pad = (-total) % align
    if pad:
        seq_lens.append(pad)
    cu = [0]
    for s in seq_lens:
        cu.append(cu[-1] + s)
    cu_seqlens = tuple(cu)
    group = msa_cp_group(ps)
    dcfg = dispatch_config(settings)
    key = m.msa.make_magi_msa_key(list(cu_seqlens), group, config=msa_config, dispatch_config=dcfg)
    position_ids = m.msa.get_position_ids(key).to(torch.int64)
    dense_key = None
    if need_dense_key:
        cu_t = torch.tensor(cu_seqlens, dtype=torch.int32, device=position_ids.device)
        dense_key = m.magi_attn_varlen_key(
            cu_t,
            cu_t,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            0,
            group,
            causal=True,
            dist_attn_config=m.DistAttnConfig(dispatch_config=dcfg),
        )
    if position_ids.numel() % max(ps.tp_size, 1) != 0:
        raise ValueError(
            f"local token count {position_ids.numel()} is not divisible by tp={ps.tp_size}; "
            f"choose magi_chunk_size divisible by tp"
        )
    if dense_key is not None and os.environ.get("MLITE_MAGI_DEBUG") == "1":
        ar = torch.arange(cu_seqlens[-1], device=position_ids.device)
        if not torch.equal(m.msa.dispatch(ar, key), m.native_dispatch(ar, dense_key)):
            raise RuntimeError("Magi MSA key and native dense key produced different dispatch layouts")
    return MagiMsaContext(
        key=key,
        dense_key=dense_key,
        cp_size=ps.cp_size,
        chunk_size=settings.chunk_size,
        cu_seqlens_host=cu_seqlens,
        num_real_docs=len(seq_lens_host),
        pad=pad,
        max_seqlen=max(seq_lens),
        position_ids=position_ids,
    )


def dispatch_tokens(x: torch.Tensor, ctx: MagiMsaContext) -> torch.Tensor:
    """Global packed tensor ``[T_global, ...]`` -> this rank's dispatched tokens ``[T_local, ...]``."""
    if x.shape[0] != ctx.cu_seqlens_host[-1]:
        raise ValueError(
            f"msa_backend='magi': every CP rank must receive the identical global (padded) batch; "
            f"got {x.shape[0]} tokens, expected {ctx.cu_seqlens_host[-1]}"
        )
    return _ensure_magi().msa.dispatch(x, ctx.key)


def undispatch_tokens(x: torch.Tensor, ctx: MagiMsaContext) -> torch.Tensor:
    """This rank's dispatched tokens ``[T_local, ...]`` -> global packed order ``[T_global, ...]``."""
    return _ensure_magi().msa.undispatch(x, ctx.key)


def _check_thd(t: torch.Tensor, name: str, heads: int, dim: int) -> torch.Tensor:
    if t.dim() != 3 or t.shape[1] != heads or t.shape[2] != dim:
        raise ValueError(f"{name} must be [tokens, {heads}, {dim}], got {tuple(t.shape)}")
    if t.dtype != torch.bfloat16:
        raise TypeError(f"{name} must be bf16 for the msa_v1 kernels, got {t.dtype}")
    return t.contiguous()


def calc_msa_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    ctx: MagiMsaContext,
) -> torch.Tensor:
    """Sparse attention output ``[T_local, 64, 128]`` via Magi ``calc_msa`` (MSAV1)."""
    m = _ensure_magi()
    inputs = m.msa.MsaV1Inputs(
        q=_check_thd(q, "q", NUM_Q_HEADS, HEAD_DIM),
        k=_check_thd(k, "k", NUM_KV_HEADS, HEAD_DIM),
        v=_check_thd(v, "v", NUM_KV_HEADS, HEAD_DIM),
        index_q=_check_thd(index_q, "index_q", NUM_INDEX_HEADS, INDEX_HEAD_DIM),
        index_k=_check_thd(index_k, "index_k", 1, INDEX_HEAD_DIM),
    )
    return m.msa.calc_msa(inputs, ctx.key).output


__all__ = [
    "MagiMsaContext",
    "MagiMsaSettings",
    "build_msa_config",
    "calc_msa_v1",
    "dispatch_tokens",
    "ensure_single_process_group",
    "msa_cp_group",
    "plan_magi_batch",
    "undispatch_tokens",
    "validate_device",
    "validate_kernel_shapes",
]
