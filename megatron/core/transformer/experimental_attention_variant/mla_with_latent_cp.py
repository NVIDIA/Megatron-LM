# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Experimental MLA self-attention that exchanges latent KV over a P2P CP ring.

This module intentionally has no registration side effects. GPTModel and HybridModel opt in through
``TransformerConfig.mla_latent_cp`` and call the feature-owned, non-mutating decoder/stack
configurators during initialization. Transformer and hybrid blocks explicitly preprocess expensive
microbatch-specific backend plans before entering their layer loops. The implementation bypasses
MCore and Transformer Engine attention wrappers: only the existing MLA projection modules and RoPE
implementation are reused.
"""

from __future__ import annotations

import copy
import importlib
import importlib.metadata
import math
import os
import threading
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Any, Final, Iterator, Literal, Protocol, TypeAlias

import torch
import torch.distributed as dist
from torch import Tensor
from torch.utils.checkpoint import checkpoint

import megatron.core.tensor_parallel as mcore_tp
from megatron.core.models.common.embeddings import apply_rotary_pos_emb
from megatron.core.packed_seq_params import PackedSeqParams, resolve_cp_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel import mappings as tp_mappings
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnBackend, AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.multi_latent_attention import (
    MLASelfAttention,
    MLASelfAttentionSubmodules,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from megatron.core.transformer.transformer_block import TransformerBlockSubmodules
from megatron.core.transformer.transformer_config import MLATransformerConfig

CUDNN_FRONTEND_SOURCE_REV: Final[str] = "0a14b7181d129d30e7bad34b8c3ed0a0c995e23d"
"""Immutable source revision used to implement and qualify the cuDNN Graph adapter."""

QualifiedBackendTuple: TypeAlias = tuple[AttnBackend, str, str, tuple[int, int]]

# This feature is fail-closed. These are the complete, exact tuples backed by the checked-in
# qualification contract; wildcards, version ranges, and runtime overrides are unsupported.
QUALIFIED_BACKEND_CONFIGS: Final[tuple[QualifiedBackendTuple, ...]] = (
    (AttnBackend.fused, "1.22.1", "9.21.0", (9, 0)),
    (AttnBackend.fused, "1.26.0", "9.25.0", (10, 0)),
    (AttnBackend.flash, "4.0.0b11", "flash-attn-4==4.0.0b11", (10, 0)),
)


# -----------------------------------------------------------------------------
# Contracts and shared metadata
# -----------------------------------------------------------------------------


class LatentCPError(RuntimeError):
    """Base error for the experimental latent-CP implementation."""


class BackendNotQualifiedError(LatentCPError):
    """Raised when the exact backend/package/device tuple is not qualified."""


class BackendPlanNotSupportedError(LatentCPError):
    """Raised before P2P when a public backend reports that a phase plan is unsupported."""


@dataclass(frozen=True)
class PhaseSpec:
    """One compact attention matrix in the zigzag ring schedule."""

    phase: int
    owner: int
    kind: Literal["diagonal", "lower", "upper"]
    q_indices: Tensor
    kv_indices: Tensor
    cu_seqlens_q: Tensor
    cu_seqlens_kv: Tensor
    max_seqlen_q: int
    max_seqlen_kv: int
    causal: bool
    scatter_indices: Tensor | None = None


@dataclass(frozen=True)
class ZigZagLayout:
    """Validated view of already-zigzag THD storage."""

    cp_size: int
    cp_rank: int
    local_tokens: int
    cu_global: Tensor
    cu_full: Tensor
    cu_half: Tensor
    max_global: int
    max_full: int
    max_half: int
    front_indices: Tensor
    back_indices: Tensor
    phases: tuple[PhaseSpec, ...]


@dataclass(frozen=True)
class PayloadLease:
    """A synchronously ready latent payload and its original CP owner."""

    owner: int
    tensor: Tensor


class LatentCPLayoutAdapter(Protocol):
    """Extension seam for a future contiguous-to-zigzag layout conversion."""

    def prepare(
        self,
        local_hidden: Tensor,
        packed_seq_params: PackedSeqParams,
        cp_group: dist.ProcessGroup,
        *,
        tp_group: dist.ProcessGroup | None = None,
        sequence_parallel: bool = False,
    ) -> ZigZagLayout:
        """Validate layout metadata and return the per-rank phase plan."""
        ...


class LatentCPTransport(Protocol):
    """Extension seam for future A2A+P2P transports."""

    def iter_payloads(
        self, local_payload: Tensor, phase_plan: tuple[PhaseSpec, ...]
    ) -> Iterator[PayloadLease]:
        """Yield one ready payload lease for every phase."""
        ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"MLAWithLatentCP: {message}")


# -----------------------------------------------------------------------------
# Packed zigzag layout and phase planning
# -----------------------------------------------------------------------------


def _cu_from_lengths(lengths: Tensor) -> Tensor:
    zero = torch.zeros(1, dtype=torch.int32, device=lengths.device)
    cumulative = torch.cumsum(lengths, dim=0, dtype=torch.int32)
    return torch.cat((zero, cumulative)).contiguous()


def _packed_half_indices(local_lengths: Tensor) -> tuple[Tensor, Tensor]:
    """Return front/back row indices for per-sequence [F_r, B_r] storage."""

    device = local_lengths.device
    starts = _cu_from_lengths(local_lengths)[:-1]
    halves = local_lengths // 2
    front = [
        torch.arange(start, start + half, dtype=torch.long, device=device)
        for start, half in zip(starts.unbind(), halves.unbind())
    ]
    back = [
        torch.arange(start + half, start + length, dtype=torch.long, device=device)
        for start, half, length in zip(starts.unbind(), halves.unbind(), local_lengths.unbind())
    ]
    if not front:
        empty = torch.empty(0, dtype=torch.long, device=device)
        return empty, empty
    return torch.cat(front), torch.cat(back)


def build_zigzag_layout(
    cu_global: Tensor,
    local_tokens: int,
    cp_size: int,
    cp_rank: int,
    *,
    max_global: int | None = None,
) -> ZigZagLayout:
    """Validate packed ownership and build the three-shape zigzag phase schedule.

    cu_global always describes original global sequences. Derived cumulative lengths are
    backend-only metadata and must never be passed to RoPE.
    """

    _require(cu_global.ndim == 1 and cu_global.numel() >= 2, "cu_seqlens must be 1-D")
    _require(cu_global.dtype == torch.int32, "cu_seqlens must have dtype torch.int32")
    _require(cp_size > 0 and 0 <= cp_rank < cp_size, "invalid CP rank or size")
    _require(int(cu_global[0].item()) == 0, "cu_seqlens must start at zero")

    global_lengths = cu_global[1:] - cu_global[:-1]
    _require(bool(torch.all(global_lengths > 0).item()), "empty packed sequences are unsupported")
    if cp_size > 1:
        _require(
            bool(torch.all(torch.remainder(global_lengths, 2 * cp_size) == 0).item()),
            f"every global packed length must be divisible by 2*CP ({2 * cp_size})",
        )

    local_lengths = torch.div(global_lengths, cp_size, rounding_mode="floor")
    # CP=1 is the exact no-ring degeneration. It has only a full/full diagonal
    # phase, so no artificial half-sequence divisibility requirement is needed.
    half_lengths = (
        torch.div(global_lengths, 2 * cp_size, rounding_mode="floor")
        if cp_size > 1
        else local_lengths
    )
    cu_full = _cu_from_lengths(local_lengths)
    cu_half = _cu_from_lengths(half_lengths)
    _require(int(cu_full[-1].item()) == local_tokens, "hidden token count disagrees with metadata")

    full_indices = torch.arange(local_tokens, dtype=torch.long, device=cu_global.device)
    if cp_size > 1:
        front_indices, back_indices = _packed_half_indices(local_lengths)
    else:
        front_indices = full_indices
        back_indices = torch.empty(0, dtype=torch.long, device=cu_global.device)
    derived_max_global = int(global_lengths.max().item())
    if max_global is not None:
        _require(max_global == derived_max_global, "max_seqlen disagrees with cu_seqlens")
    max_global = derived_max_global
    max_full = int(local_lengths.max().item())
    max_half = int(half_lengths.max().item())

    phases: list[PhaseSpec] = []
    for phase in range(cp_size):
        owner = (cp_rank - phase) % cp_size
        if phase == 0:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "diagonal",
                    full_indices,
                    full_indices,
                    cu_full,
                    cu_full,
                    max_full,
                    max_full,
                    True,
                )
            )
        elif phase <= cp_rank:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "lower",
                    full_indices,
                    front_indices,
                    cu_full,
                    cu_half,
                    max_full,
                    max_half,
                    False,
                )
            )
        else:
            phases.append(
                PhaseSpec(
                    phase,
                    owner,
                    "upper",
                    back_indices,
                    full_indices,
                    cu_half,
                    cu_full,
                    max_half,
                    max_full,
                    False,
                    scatter_indices=back_indices,
                )
            )

    return ZigZagLayout(
        cp_size=cp_size,
        cp_rank=cp_rank,
        local_tokens=local_tokens,
        cu_global=cu_global,
        cu_full=cu_full,
        cu_half=cu_half,
        max_global=max_global,
        max_full=max_full,
        max_half=max_half,
        front_indices=front_indices,
        back_indices=back_indices,
        phases=tuple(phases),
    )


class AlreadyZigZagTHDAdapter:
    """V1 layout adapter: validate an input that is already zigzag-partitioned."""

    def prepare(
        self,
        local_hidden: Tensor,
        packed_seq_params: PackedSeqParams,
        cp_group: dist.ProcessGroup,
        *,
        tp_group: dist.ProcessGroup | None = None,
        sequence_parallel: bool = False,
    ) -> ZigZagLayout:
        """Validate already-zigzag THD metadata and build its phase plan."""
        _require(packed_seq_params.qkv_format == "thd", "only THD format is supported")
        _require(
            packed_seq_params.cp_partition_mode == "zigzag",
            "only an already-zigzag CP partition is supported",
        )
        cu_q = packed_seq_params.cu_seqlens_q
        cu_kv = packed_seq_params.cu_seqlens_kv
        _require(isinstance(cu_q, Tensor) and isinstance(cu_kv, Tensor), "missing cu_seqlens")
        _require(
            cu_q.is_cuda
            and cu_kv.is_cuda
            and cu_q.device == local_hidden.device
            and cu_kv.device == local_hidden.device,
            "cu_seqlens must be CUDA tensors colocated with hidden_states",
        )
        _require(
            cu_q.dtype == torch.int32 and cu_kv.dtype == torch.int32,
            "both Q and KV cu_seqlens must have dtype torch.int32",
        )
        _require(cu_q.is_contiguous() and cu_kv.is_contiguous(), "cu_seqlens must be contiguous")
        _require(torch.equal(cu_q, cu_kv), "self-attention requires equal Q/KV cu_seqlens")
        for padded, valid, name in (
            (packed_seq_params.cu_seqlens_q_padded, cu_q, "Q"),
            (packed_seq_params.cu_seqlens_kv_padded, cu_kv, "KV"),
        ):
            _require(
                padded is None or torch.equal(padded, valid),
                f"{name} inter-sequence/tail padding is unsupported",
            )
        max_q = packed_seq_params.max_seqlen_q
        max_kv = packed_seq_params.max_seqlen_kv
        _require(
            isinstance(max_q, int)
            and not isinstance(max_q, bool)
            and isinstance(max_kv, int)
            and not isinstance(max_kv, bool)
            and max_q > 0
            and max_kv > 0,
            "Q and KV max_seqlen must be positive Python integers",
        )
        _require(max_q == max_kv, "self-attention requires equal Q/KV max_seqlen")
        cp_size = dist.get_world_size(cp_group)
        cp_rank = dist.get_rank(cp_group)
        local_tokens = local_hidden.size(0)
        if sequence_parallel:
            _require(tp_group is not None, "sequence parallelism requires a TP group")
            local_tokens *= dist.get_world_size(tp_group)
        return build_zigzag_layout(cu_q, local_tokens, cp_size, cp_rank, max_global=max_q)


# -----------------------------------------------------------------------------
# Differentiable latent P2P transport
# -----------------------------------------------------------------------------


class _LatentRingExchange(torch.autograd.Function):
    """One explicit-group clockwise ring hop with the exact reverse backward hop."""

    @staticmethod
    def forward(
        ctx: Any, payload: Tensor, cp_group: dist.ProcessGroup, previous_peer: int, next_peer: int
    ) -> Tensor:
        """Send one payload clockwise and receive the preceding owner's payload."""
        receive = torch.empty_like(payload)
        operations = [
            dist.P2POp(dist.isend, payload, next_peer, group=cp_group),
            dist.P2POp(dist.irecv, receive, previous_peer, group=cp_group),
        ]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        ctx.cp_group = cp_group
        ctx.previous_peer = previous_peer
        ctx.next_peer = next_peer
        return receive

    @staticmethod
    def backward(ctx: Any, grad_receive: Tensor) -> tuple[Tensor, None, None, None]:
        """Route the received-payload gradient through the reverse ring hop."""
        grad_receive = grad_receive.contiguous()
        grad_payload = torch.empty_like(grad_receive)
        operations = [
            dist.P2POp(dist.isend, grad_receive, ctx.previous_peer, group=ctx.cp_group),
            dist.P2POp(dist.irecv, grad_payload, ctx.next_peer, group=ctx.cp_group),
        ]
        for work in dist.batch_isend_irecv(operations):
            work.wait()
        return grad_payload, None, None, None


class P2PRingTransport:
    """Synchronous wait-at-each-hop v1 transport."""

    def __init__(self, cp_group: dist.ProcessGroup):
        self.cp_group = cp_group
        self.group_ranks = tuple(dist.get_process_group_ranks(cp_group))
        self.rank = dist.get_rank(cp_group)
        self.size = dist.get_world_size(cp_group)
        _require(len(self.group_ranks) == self.size, "invalid CP peer list")
        self.previous_peer = self.group_ranks[(self.rank - 1) % self.size]
        self.next_peer = self.group_ranks[(self.rank + 1) % self.size]

    def iter_payloads(
        self, local_payload: Tensor, phase_plan: tuple[PhaseSpec, ...]
    ) -> Iterator[PayloadLease]:
        """Yield the local payload followed by each synchronous clockwise hop."""
        _require(len(phase_plan) == self.size, "phase-plan length must equal CP size")
        for phase_index, phase in enumerate(phase_plan):
            expected_owner = (self.rank - phase_index) % self.size
            _require(phase.phase == phase_index, "phase-plan indices must be contiguous")
            _require(
                phase.owner == expected_owner, "phase-plan owner order disagrees with the P2P ring"
            )

        payload = local_payload
        for phase_index, phase in enumerate(phase_plan):
            yield PayloadLease(owner=phase.owner, tensor=payload)
            if phase_index + 1 < self.size:
                payload = _LatentRingExchange.apply(
                    payload, self.cp_group, self.previous_peer, self.next_peer
                )


# -----------------------------------------------------------------------------
# FP32 partial-output merge and backend-independent gradient correction
# -----------------------------------------------------------------------------


def scatter_upper_phase(
    output: Tensor, lse: Tensor, back_indices: Tensor, local_tokens: int
) -> tuple[Tensor, Tensor]:
    """Functionally scatter an upper rectangular phase into full local Q rows."""

    output_full = torch.zeros(
        (local_tokens, *output.shape[1:]), dtype=torch.float32, device=output.device
    ).index_copy(0, back_indices, output.float())
    lse_full = torch.full(
        (local_tokens, lse.size(1)), -torch.inf, dtype=torch.float32, device=lse.device
    ).index_copy(0, back_indices, lse.float())
    return output_full, lse_full


def merge_attention_partials(
    output_a: Tensor, lse_a: Tensor, output_b: Tensor, lse_b: Tensor
) -> tuple[Tensor, Tensor]:
    """Stable FP32 online-softmax merge for two attention partials."""

    _require(output_a.dtype == output_b.dtype == torch.float32, "partial outputs must be FP32")
    _require(lse_a.dtype == lse_b.dtype == torch.float32, "partial LSE must be FP32")
    merged_lse = torch.logaddexp(lse_a, lse_b)
    valid_a = torch.isfinite(lse_a) & torch.isfinite(merged_lse)
    valid_b = torch.isfinite(lse_b) & torch.isfinite(merged_lse)
    delta_a = torch.where(valid_a, lse_a - merged_lse, torch.full_like(lse_a, -torch.inf))
    delta_b = torch.where(valid_b, lse_b - merged_lse, torch.full_like(lse_b, -torch.inf))
    weight_a = torch.exp(delta_a)
    weight_b = torch.exp(delta_b)
    merged_output = output_a * weight_a.unsqueeze(-1) + output_b * weight_b.unsqueeze(-1)
    return merged_output, merged_lse


def cudnn_backward_proxy(
    partial_output: Tensor, grad_output: Tensor, grad_lse: Tensor
) -> tuple[Tensor, Tensor]:
    """Construct cuDNN's corrected BF16 o/dO inputs in FP32.

    For safe rows, dot(G_i, O_corr) encodes the missing LSE gradient in public sdpa_backward.
    Zero and tiny rows use zero correction.
    """

    partial_output = partial_output.float()
    grad_output = grad_output.float()
    grad_lse = grad_lse.float()
    norm2 = torch.sum(grad_output * grad_output, dim=-1)
    threshold = math.sqrt(torch.finfo(torch.float32).tiny)
    safe = torch.isfinite(norm2) & (norm2 >= threshold) & torch.isfinite(grad_lse)
    denominator = torch.where(safe, norm2, torch.ones_like(norm2))
    coefficient = torch.where(safe, grad_lse / denominator, torch.zeros_like(grad_lse))
    raw_correction = coefficient.unsqueeze(-1) * grad_output
    correction = torch.where(safe.unsqueeze(-1), raw_correction, torch.zeros_like(raw_correction))
    return partial_output - correction, grad_output


# -----------------------------------------------------------------------------
# Direct FA4 and cuDNN Frontend adapters
# -----------------------------------------------------------------------------


class DirectAttentionAdapter(Protocol):
    """Direct packed-attention backend boundary used by each CP phase."""

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Validate and prepare every phase before ring communication starts."""
        ...

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Return canonical FP32 output and LSE for one packed phase."""
        ...


class FA4Adapter:
    """Thin lazy adapter for public flash_attn.cute.flash_attn_varlen_func."""

    def __init__(self) -> None:
        try:
            module = importlib.import_module("flash_attn.cute")
            self._function = getattr(module, "flash_attn_varlen_func")
        except (ImportError, AttributeError) as error:
            raise BackendNotQualifiedError(
                "FA4 requires public flash_attn.cute.flash_attn_varlen_func"
            ) from error

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Accept phase metadata; FA4 requires no persistent plan construction."""
        del num_heads, qk_dim, v_dim, phases, scale

    @staticmethod
    def _canonical_lse(lse: Tensor, tokens: int, heads: int) -> Tensor:
        if lse.shape == (tokens, heads):
            return lse.float()
        if lse.shape == (heads, tokens):
            return lse.transpose(0, 1).contiguous().float()
        raise LatentCPError(
            f"FA4 returned unsupported LSE shape {tuple(lse.shape)}; "
            f"expected {(tokens, heads)} or {(heads, tokens)}"
        )

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Execute one phase through the public FA4 varlen API."""
        _require(
            cu_q.dtype == torch.int32 and cu_kv.dtype == torch.int32,
            "FA4 cu_seqlens must have dtype torch.int32",
        )
        _require(
            cu_q.is_contiguous() and cu_kv.is_contiguous(), "FA4 cu_seqlens must be contiguous"
        )
        _require(
            cu_q.device == q.device and cu_kv.device == k.device,
            "FA4 cu_seqlens must be colocated with their Q/K tensors",
        )
        result = self._function(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_kv,
            max_seqlen_q=max_q,
            max_seqlen_k=max_kv,
            softmax_scale=scale,
            causal=causal,
            return_lse=True,
        )
        if not isinstance(result, tuple) or len(result) < 2:
            raise LatentCPError("FA4 return_lse=True did not return (output, LSE)")
        output, lse = result[:2]
        _require(output.dtype == torch.bfloat16, "FA4 phase output must be BF16")
        return output.float(), self._canonical_lse(lse, q.size(0), q.size(1))


class _CudnnUid(IntEnum):
    Q = 0
    K = 1
    V = 2
    O = 3
    STATS = 4
    DQ = 5
    DK = 6
    DV = 7
    DO = 8
    SEQ_Q = 9
    SEQ_KV = 10
    Q_OFFSET = 11
    K_OFFSET = 12
    V_OFFSET = 13
    O_OFFSET = 14
    STATS_OFFSET = 15


@dataclass(frozen=True)
class _CudnnPlanKey:
    process_id: int
    device_index: int
    frontend_version: str
    runtime_version: str
    dtype: torch.dtype
    sm: tuple[int, int]
    batch: int
    heads: int
    qk_dim: int
    v_dim: int
    max_q: int
    max_kv: int
    capacity_q: int
    capacity_kv: int
    causal: bool
    scale: float


@dataclass(frozen=True)
class _CudnnPlan:
    forward_graph: Any
    backward_graph: Any
    key: _CudnnPlanKey


def _packed_bshd_stride(shape: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Pinned cuDNN Frontend ragged BSHD stride for a logical BHSD shape."""

    _, heads, sequence, dimension = shape
    return (sequence * heads * dimension, dimension, heads * dimension, 1)


def _aligned_token_capacity(tokens: int) -> int:
    return max(64, ((tokens + 63) // 64) * 64)


def _pad_token_rows(tensor: Tensor, capacity: int) -> Tensor:
    _require(tensor.size(0) <= capacity, "cuDNN token capacity is too small")
    padded = torch.empty((capacity, *tensor.shape[1:]), dtype=tensor.dtype, device=tensor.device)
    padded[: tensor.size(0)].copy_(tensor)
    return padded


class _CudnnSDPAFunction(torch.autograd.Function):
    """Public cuDNN Graph SDPA with explicit support for the outer LSE gradient."""

    @staticmethod
    def forward(
        ctx: Any,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
        adapter: "CudnnFusedAttentionAdapter",
    ) -> tuple[Tensor, Tensor]:
        """Execute public cuDNN Graph SDPA and save its backward inputs."""
        raw_output, stats = adapter._execute_forward(
            q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale
        )
        ctx.save_for_backward(q, k, v, raw_output, stats, cu_q, cu_kv)
        ctx.max_q = max_q
        ctx.max_kv = max_kv
        ctx.causal = causal
        ctx.scale = scale
        ctx.adapter = adapter
        return raw_output.float(), stats.float()

    @staticmethod
    def backward(
        ctx: Any, grad_output: Tensor | None, grad_lse: Tensor | None
    ) -> tuple[Tensor, Tensor, Tensor, None, None, None, None, None, None, None]:
        """Execute corrected public cuDNN Graph SDPA backward."""
        q, k, v, raw_output, stats, cu_q, cu_kv = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(raw_output, dtype=torch.float32)
        if grad_lse is None:
            grad_lse = torch.zeros(
                raw_output.shape[:2], dtype=torch.float32, device=raw_output.device
            )
        corrected_output, local_grad_output = cudnn_backward_proxy(
            raw_output, grad_output, grad_lse
        )
        dq, dk, dv = ctx.adapter._execute_backward(
            q,
            k,
            v,
            corrected_output.to(torch.bfloat16),
            local_grad_output.to(torch.bfloat16),
            stats,
            cu_q,
            cu_kv,
            ctx.max_q,
            ctx.max_kv,
            ctx.causal,
            ctx.scale,
        )
        return dq, dk, dv, None, None, None, None, None, None, None


def _resolve_cudnn_frontend_version(cudnn: Any) -> str:
    try:
        version = importlib.metadata.version("nvidia-cudnn-frontend")
    except importlib.metadata.PackageNotFoundError:
        version = getattr(cudnn, "__version__", None)
    if version is None:
        raise BackendNotQualifiedError("cuDNN Frontend package version metadata is missing")
    return str(version)


class CudnnFusedAttentionAdapter:
    """Direct public cuDNN Frontend Graph adapter for fully ragged BF16 SDPA.

    One adapter (and therefore one public cuDNN handle and plan cache) is shared by all latent-CP
    layers in a process on one CUDA device. Plan building and handle stream mutation are protected
    by a reentrant lock. Invocation workspaces and graph variant packs remain call-local.
    """

    def __init__(
        self,
        expected_identity: QualifiedBackendTuple | None = None,
        *,
        device_index: int | None = None,
    ) -> None:
        try:
            self.cudnn = importlib.import_module("cudnn")
        except ImportError as error:
            raise BackendNotQualifiedError(
                "fused latent CP requires the public nvidia-cudnn-frontend package"
            ) from error

        self.process_id = os.getpid()
        self.device_index = (
            torch.cuda.current_device() if device_index is None else int(device_index)
        )
        self.frontend_version = _resolve_cudnn_frontend_version(self.cudnn)
        self.runtime_version = str(self.cudnn.backend_version_string())
        with torch.cuda.device(self.device_index):
            capability = torch.cuda.get_device_capability(self.device_index)
            self._handle = self.cudnn.create_handle()
        self.identity: QualifiedBackendTuple = (
            AttnBackend.fused,
            self.frontend_version,
            self.runtime_version,
            capability,
        )
        if expected_identity is not None and self.identity != expected_identity:
            self.cudnn.destroy_handle(self._handle)
            self._handle = None
            raise BackendNotQualifiedError(
                f"cuDNN adapter identity {self.identity!r} != {expected_identity!r}"
            )
        self._plans: dict[_CudnnPlanKey, _CudnnPlan] = {}
        self._execution_lock = threading.RLock()

    def __del__(self) -> None:
        handle = getattr(self, "_handle", None)
        cudnn = getattr(self, "cudnn", None)
        if (
            handle is not None
            and cudnn is not None
            and getattr(self, "process_id", None) == os.getpid()
        ):
            try:
                with torch.cuda.device(self.device_index):
                    cudnn.destroy_handle(handle)
            except Exception:
                # Interpreter shutdown may unload CUDA before module destruction.
                pass
            self._handle = None

    def _assert_bound_device(self, device: torch.device) -> None:
        _require(os.getpid() == self.process_id, "a forked process cannot reuse a cuDNN adapter")
        _require(device.type == "cuda", "cuDNN plans require a CUDA device")
        device_index = torch.cuda.current_device() if device.index is None else device.index
        _require(
            device_index == self.device_index,
            "cuDNN adapter/handle cannot be reused across CUDA devices",
        )

    def _plan_key_from_metadata(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        cu_q: Tensor,
        cu_kv: Tensor,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> _CudnnPlanKey:
        self._assert_bound_device(device)
        _require(dtype == torch.bfloat16, "cuDNN adapter supports BF16 only")
        _require(
            cu_q.device == device and cu_kv.device == device,
            "cuDNN metadata must be on the bound CUDA device",
        )
        total_q = int(cu_q[-1].item())
        total_kv = int(cu_kv[-1].item())
        _require(total_q > 0 and total_kv > 0, "cuDNN phase totals must be positive")
        return _CudnnPlanKey(
            process_id=self.process_id,
            device_index=self.device_index,
            frontend_version=self.frontend_version,
            runtime_version=self.runtime_version,
            dtype=dtype,
            sm=torch.cuda.get_device_capability(self.device_index),
            batch=cu_q.numel() - 1,
            heads=num_heads,
            qk_dim=qk_dim,
            v_dim=v_dim,
            max_q=max_q,
            max_kv=max_kv,
            capacity_q=_aligned_token_capacity(total_q),
            capacity_kv=_aligned_token_capacity(total_kv),
            causal=causal,
            scale=float(scale),
        )

    @staticmethod
    def _metadata(
        cu_q: Tensor, cu_kv: Tensor, heads: int, qk_dim: int, v_dim: int
    ) -> dict[_CudnnUid, Tensor]:
        def canonical_rank4(tensor: Tensor) -> Tensor:
            return tensor.contiguous().view(-1, 1, 1, 1)

        return {
            _CudnnUid.SEQ_Q: canonical_rank4((cu_q[1:] - cu_q[:-1]).to(torch.int32)),
            _CudnnUid.SEQ_KV: canonical_rank4((cu_kv[1:] - cu_kv[:-1]).to(torch.int32)),
            _CudnnUid.Q_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads * qk_dim),
            _CudnnUid.K_OFFSET: canonical_rank4(cu_kv.to(torch.int64) * heads * qk_dim),
            _CudnnUid.V_OFFSET: canonical_rank4(cu_kv.to(torch.int64) * heads * v_dim),
            _CudnnUid.O_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads * v_dim),
            _CudnnUid.STATS_OFFSET: canonical_rank4(cu_q.to(torch.int64) * heads),
        }

    def _new_graph(self, key: _CudnnPlanKey) -> Any:
        _require(
            key.process_id == self.process_id
            and key.device_index == self.device_index
            and key.frontend_version == self.frontend_version
            and key.runtime_version == self.runtime_version
            and key.dtype == torch.bfloat16,
            "cuDNN plan key is not bound to this adapter identity",
        )
        with torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle, stream=torch.cuda.current_stream(self.device_index).cuda_stream
            )
            return self.cudnn.pygraph(
                io_data_type=self.cudnn.data_type.BFLOAT16,
                intermediate_data_type=self.cudnn.data_type.FLOAT,
                compute_data_type=self.cudnn.data_type.FLOAT,
                handle=self._handle,
                sm_version=key.sm[0] * 10 + key.sm[1],
            )

    def _ragged_descriptors(
        self, graph: Any, key: _CudnnPlanKey, *, backward: bool
    ) -> dict[_CudnnUid, Any]:
        q_shape = (key.batch, key.heads, key.max_q, key.qk_dim)
        k_shape = (key.batch, key.heads, key.max_kv, key.qk_dim)
        v_shape = (key.batch, key.heads, key.max_kv, key.v_dim)
        o_shape = (key.batch, key.heads, key.max_q, key.v_dim)
        stats_shape = (key.batch, key.heads, key.max_q, 1)
        tensor_specs: list[tuple[_CudnnUid, tuple[int, ...], tuple[int, ...], Any]] = [
            (_CudnnUid.Q, q_shape, _packed_bshd_stride(q_shape), self.cudnn.data_type.BFLOAT16),
            (_CudnnUid.K, k_shape, _packed_bshd_stride(k_shape), self.cudnn.data_type.BFLOAT16),
            (_CudnnUid.V, v_shape, _packed_bshd_stride(v_shape), self.cudnn.data_type.BFLOAT16),
        ]
        if backward:
            tensor_specs.extend(
                [
                    (
                        _CudnnUid.O,
                        o_shape,
                        _packed_bshd_stride(o_shape),
                        self.cudnn.data_type.BFLOAT16,
                    ),
                    (
                        _CudnnUid.DO,
                        o_shape,
                        _packed_bshd_stride(o_shape),
                        self.cudnn.data_type.BFLOAT16,
                    ),
                    (
                        _CudnnUid.STATS,
                        stats_shape,
                        _packed_bshd_stride(stats_shape),
                        self.cudnn.data_type.FLOAT,
                    ),
                ]
            )
        tensors = {
            uid: graph.tensor(uid=int(uid), dim=shape, stride=stride, data_type=dtype)
            for uid, shape, stride, dtype in tensor_specs
        }
        for uid in (
            _CudnnUid.Q_OFFSET,
            _CudnnUid.K_OFFSET,
            _CudnnUid.V_OFFSET,
            _CudnnUid.O_OFFSET,
            _CudnnUid.STATS_OFFSET,
        ):
            tensors[uid] = graph.tensor(
                uid=int(uid),
                dim=(key.batch + 1, 1, 1, 1),
                stride=(1, 1, 1, 1),
                data_type=self.cudnn.data_type.INT64,
            )
        tensors[_CudnnUid.SEQ_Q] = graph.tensor(
            uid=int(_CudnnUid.SEQ_Q),
            dim=(key.batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=self.cudnn.data_type.INT32,
        )
        tensors[_CudnnUid.SEQ_KV] = graph.tensor(
            uid=int(_CudnnUid.SEQ_KV),
            dim=(key.batch, 1, 1, 1),
            stride=(1, 1, 1, 1),
            data_type=self.cudnn.data_type.INT32,
        )
        tensors[_CudnnUid.Q].set_ragged_offset(tensors[_CudnnUid.Q_OFFSET])
        tensors[_CudnnUid.K].set_ragged_offset(tensors[_CudnnUid.K_OFFSET])
        tensors[_CudnnUid.V].set_ragged_offset(tensors[_CudnnUid.V_OFFSET])
        if backward:
            tensors[_CudnnUid.O].set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
            tensors[_CudnnUid.DO].set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
            tensors[_CudnnUid.STATS].set_ragged_offset(tensors[_CudnnUid.STATS_OFFSET])
        return tensors

    def _build_forward_graph(self, key: _CudnnPlanKey) -> Any:
        graph = self._new_graph(key)
        tensors = self._ragged_descriptors(graph, key, backward=False)
        output, stats = graph.sdpa(
            name="mla_latent_cp_sdpa_forward",
            q=tensors[_CudnnUid.Q],
            k=tensors[_CudnnUid.K],
            v=tensors[_CudnnUid.V],
            generate_stats=True,
            attn_scale=key.scale,
            use_padding_mask=True,
            seq_len_q=tensors[_CudnnUid.SEQ_Q],
            seq_len_kv=tensors[_CudnnUid.SEQ_KV],
            diagonal_band_right_bound=0 if key.causal else None,
            diagonal_alignment=self.cudnn.diagonal_alignment.TOP_LEFT,
        )
        o_shape = (key.batch, key.heads, key.max_q, key.v_dim)
        output.set_uid(int(_CudnnUid.O)).set_output(True).set_dim(o_shape).set_stride(
            _packed_bshd_stride(o_shape)
        )
        output.set_ragged_offset(tensors[_CudnnUid.O_OFFSET])
        stats_shape = (key.batch, key.heads, key.max_q, 1)
        stats.set_uid(int(_CudnnUid.STATS)).set_output(True).set_data_type(
            self.cudnn.data_type.FLOAT
        ).set_dim(stats_shape).set_stride(_packed_bshd_stride(stats_shape))
        stats.set_ragged_offset(tensors[_CudnnUid.STATS_OFFSET])
        self._build_graph(graph)
        return graph

    def _build_backward_graph(self, key: _CudnnPlanKey) -> Any:
        graph = self._new_graph(key)
        tensors = self._ragged_descriptors(graph, key, backward=True)
        dq, dk, dv = graph.sdpa_backward(
            name="mla_latent_cp_sdpa_backward",
            q=tensors[_CudnnUid.Q],
            k=tensors[_CudnnUid.K],
            v=tensors[_CudnnUid.V],
            o=tensors[_CudnnUid.O],
            dO=tensors[_CudnnUid.DO],
            stats=tensors[_CudnnUid.STATS],
            attn_scale=key.scale,
            use_padding_mask=True,
            seq_len_q=tensors[_CudnnUid.SEQ_Q],
            seq_len_kv=tensors[_CudnnUid.SEQ_KV],
            max_total_seq_len_q=key.capacity_q,
            max_total_seq_len_kv=key.capacity_kv,
            diagonal_band_right_bound=0 if key.causal else None,
            diagonal_alignment=self.cudnn.diagonal_alignment.TOP_LEFT,
        )
        q_shape = (key.batch, key.heads, key.max_q, key.qk_dim)
        k_shape = (key.batch, key.heads, key.max_kv, key.qk_dim)
        v_shape = (key.batch, key.heads, key.max_kv, key.v_dim)
        for tensor, uid, shape, offset_uid in (
            (dq, _CudnnUid.DQ, q_shape, _CudnnUid.Q_OFFSET),
            (dk, _CudnnUid.DK, k_shape, _CudnnUid.K_OFFSET),
            (dv, _CudnnUid.DV, v_shape, _CudnnUid.V_OFFSET),
        ):
            tensor.set_uid(int(uid)).set_output(True).set_dim(shape).set_stride(
                _packed_bshd_stride(shape)
            )
            tensor.set_ragged_offset(tensors[offset_uid])
        self._build_graph(graph)
        return graph

    def _build_graph(self, graph: Any) -> None:
        try:
            graph.validate()
            graph.build_operation_graph()
            graph.create_execution_plans([self.cudnn.heur_mode.A, self.cudnn.heur_mode.FALLBACK])
            graph.check_support()
            graph.build_plans()
        except self.cudnn.cudnnGraphNotSupportedError as error:
            raise BackendPlanNotSupportedError(str(error)) from error

    def _prepare_plan(self, key: _CudnnPlanKey) -> _CudnnPlan:
        with self._execution_lock:
            plan = self._plans.get(key)
            if plan is None:
                plan = _CudnnPlan(
                    forward_graph=self._build_forward_graph(key),
                    backward_graph=self._build_backward_graph(key),
                    key=key,
                )
                self._plans[key] = plan
            return plan

    def _get_prepared_plan(self, key: _CudnnPlanKey) -> _CudnnPlan:
        with self._execution_lock:
            plan = self._plans.get(key)
        if plan is None:
            raise BackendPlanNotSupportedError(
                "cuDNN phase plan was not prepared before transformer block execution"
            )
        return plan

    def prepare(
        self,
        *,
        num_heads: int,
        qk_dim: int,
        v_dim: int,
        phases: tuple[PhaseSpec, ...],
        scale: float,
    ) -> None:
        """Build or reuse public cuDNN Graph plans for every phase."""
        device = torch.device("cuda", self.device_index)
        for phase in phases:
            key = self._plan_key_from_metadata(
                device=device,
                dtype=torch.bfloat16,
                cu_q=phase.cu_seqlens_q,
                cu_kv=phase.cu_seqlens_kv,
                num_heads=num_heads,
                qk_dim=qk_dim,
                v_dim=v_dim,
                max_q=phase.max_seqlen_q,
                max_kv=phase.max_seqlen_kv,
                causal=phase.causal,
                scale=scale,
            )
            self._prepare_plan(key)

    def _execution_key(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> _CudnnPlanKey:
        _require(q.dtype == k.dtype == v.dtype == torch.bfloat16, "cuDNN Q/K/V must all be BF16")
        _require(q.device == k.device == v.device, "cuDNN Q/K/V must use one CUDA device")
        _require(
            q.ndim == k.ndim == v.ndim == 3
            and q.size(1) == k.size(1) == v.size(1)
            and q.size(2) == k.size(2),
            "invalid cuDNN THD Q/K/V shapes",
        )
        _require(
            q.size(0) == int(cu_q[-1].item()) and k.size(0) == v.size(0) == int(cu_kv[-1].item()),
            "cuDNN tensor rows disagree with cumulative lengths",
        )
        return self._plan_key_from_metadata(
            device=q.device,
            dtype=q.dtype,
            cu_q=cu_q,
            cu_kv=cu_kv,
            num_heads=q.size(1),
            qk_dim=q.size(2),
            v_dim=v.size(2),
            max_q=max_q,
            max_kv=max_kv,
            causal=causal,
            scale=scale,
        )

    def _execute_forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        key = self._execution_key(q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale)
        plan = self._get_prepared_plan(key)
        metadata = self._metadata(cu_q, cu_kv, key.heads, key.qk_dim, key.v_dim)
        q_buffer = _pad_token_rows(q, key.capacity_q)
        k_buffer = _pad_token_rows(k, key.capacity_kv)
        v_buffer = _pad_token_rows(v, key.capacity_kv)
        o_buffer = torch.empty(
            (key.capacity_q, key.heads, key.v_dim), dtype=torch.bfloat16, device=q.device
        )
        stats_buffer = torch.empty(
            (key.capacity_q, key.heads, 1), dtype=torch.float32, device=q.device
        )
        pack = {
            int(_CudnnUid.Q): q_buffer,
            int(_CudnnUid.K): k_buffer,
            int(_CudnnUid.V): v_buffer,
            int(_CudnnUid.O): o_buffer,
            int(_CudnnUid.STATS): stats_buffer,
            **{int(uid): tensor for uid, tensor in metadata.items()},
        }
        workspace = torch.empty(
            plan.forward_graph.get_workspace_size(), dtype=torch.uint8, device=q.device
        )
        with self._execution_lock, torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle, stream=torch.cuda.current_stream(self.device_index).cuda_stream
            )
            plan.forward_graph.execute(pack, workspace, self._handle)
        return o_buffer[: q.size(0)], stats_buffer[: q.size(0), :, 0]

    def _execute_backward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        output: Tensor,
        grad_output: Tensor,
        stats: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        key = self._execution_key(q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale)
        plan = self._get_prepared_plan(key)
        metadata = self._metadata(cu_q, cu_kv, key.heads, key.qk_dim, key.v_dim)
        q_buffer = _pad_token_rows(q, key.capacity_q)
        k_buffer = _pad_token_rows(k, key.capacity_kv)
        v_buffer = _pad_token_rows(v, key.capacity_kv)
        o_buffer = _pad_token_rows(output, key.capacity_q)
        do_buffer = _pad_token_rows(grad_output, key.capacity_q)
        stats_buffer = _pad_token_rows(stats.unsqueeze(-1), key.capacity_q)
        dq_buffer = torch.empty_like(q_buffer)
        dk_buffer = torch.empty_like(k_buffer)
        dv_buffer = torch.empty_like(v_buffer)
        pack = {
            int(_CudnnUid.Q): q_buffer,
            int(_CudnnUid.K): k_buffer,
            int(_CudnnUid.V): v_buffer,
            int(_CudnnUid.O): o_buffer,
            int(_CudnnUid.DO): do_buffer,
            int(_CudnnUid.STATS): stats_buffer,
            int(_CudnnUid.DQ): dq_buffer,
            int(_CudnnUid.DK): dk_buffer,
            int(_CudnnUid.DV): dv_buffer,
            **{int(uid): tensor for uid, tensor in metadata.items()},
        }
        workspace = torch.empty(
            plan.backward_graph.get_workspace_size(), dtype=torch.uint8, device=q.device
        )
        with self._execution_lock, torch.cuda.device(self.device_index):
            self.cudnn.set_stream(
                handle=self._handle, stream=torch.cuda.current_stream(self.device_index).cuda_stream
            )
            plan.backward_graph.execute(pack, workspace, self._handle)
        return dq_buffer[: q.size(0)], dk_buffer[: k.size(0)], dv_buffer[: v.size(0)]

    def forward_phase(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        cu_q: Tensor,
        cu_kv: Tensor,
        max_q: int,
        max_kv: int,
        causal: bool,
        scale: float,
    ) -> tuple[Tensor, Tensor]:
        """Execute one phase through the differentiable cuDNN Graph wrapper."""
        return _CudnnSDPAFunction.apply(q, k, v, cu_q, cu_kv, max_q, max_kv, causal, scale, self)


_CUDNN_ADAPTER_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_CUDNN_ADAPTER_CACHE: dict[tuple[int, int, QualifiedBackendTuple], CudnnFusedAttentionAdapter] = {}


def _shared_cudnn_adapter(runtime_identity: QualifiedBackendTuple) -> CudnnFusedAttentionAdapter:
    """Return the process/device-scoped adapter shared by every local MLA layer."""

    cache_key = (os.getpid(), torch.cuda.current_device(), runtime_identity)
    with _CUDNN_ADAPTER_CACHE_LOCK:
        adapter = _CUDNN_ADAPTER_CACHE.get(cache_key)
        if adapter is None:
            adapter = CudnnFusedAttentionAdapter(runtime_identity, device_index=cache_key[1])
            _CUDNN_ADAPTER_CACHE[cache_key] = adapter
        return adapter


# -----------------------------------------------------------------------------
# Runtime qualification and shared backend construction
# -----------------------------------------------------------------------------


def _runtime_backend_tuple(backend: AttnBackend) -> QualifiedBackendTuple:
    """Resolve the exact immutable qualification key without running numerical probes."""

    if not torch.cuda.is_available():
        raise BackendNotQualifiedError("latent CP attention requires CUDA")
    capability = torch.cuda.get_device_capability()
    if backend is AttnBackend.fused:
        try:
            cudnn = importlib.import_module("cudnn")
            frontend_version = _resolve_cudnn_frontend_version(cudnn)
            runtime_version = str(cudnn.backend_version_string())
        except (ImportError, AttributeError) as error:
            raise BackendNotQualifiedError(
                "unable to resolve cuDNN Frontend package/runtime versions"
            ) from error
        if not runtime_version:
            raise BackendNotQualifiedError("incomplete cuDNN qualification metadata")
        return backend, frontend_version, runtime_version, capability
    if backend is AttnBackend.flash:
        try:
            version = importlib.metadata.version("flash-attn-4")
            importlib.import_module("flash_attn.cute")
        except (importlib.metadata.PackageNotFoundError, ImportError) as error:
            raise BackendNotQualifiedError(
                "FA4 requires the exact flash-attn-4 distribution and flash_attn.cute"
            ) from error
        return backend, version, f"flash-attn-4=={version}", capability
    raise BackendNotQualifiedError(f"unsupported backend {backend!r}")


def _qualified_backend_adapter(
    backend: AttnBackend, runtime_tuple: QualifiedBackendTuple | None = None
) -> tuple[DirectAttentionAdapter, QualifiedBackendTuple]:
    if runtime_tuple is None:
        runtime_tuple = _runtime_backend_tuple(backend)
    if runtime_tuple not in QUALIFIED_BACKEND_CONFIGS:
        raise BackendNotQualifiedError(
            "unqualified latent-CP backend tuple "
            f"{runtime_tuple!r}; qualified tuples are {QUALIFIED_BACKEND_CONFIGS!r}"
        )
    if backend is AttnBackend.fused:
        return _shared_cudnn_adapter(runtime_tuple), runtime_tuple
    if backend is AttnBackend.flash:
        return FA4Adapter(), runtime_tuple
    raise BackendNotQualifiedError(f"unsupported backend {backend!r}")


def _build_local_latent_norm(
    *, config: MLATransformerConfig, hidden_size: int, eps: float
) -> torch.nn.Module:
    """Build a token-local norm while preserving SP parameter-gradient synchronization."""

    norm_config = copy.copy(config)
    norm_config.sequence_parallel = False
    norm = WrappedTorchNorm(config=norm_config, hidden_size=hidden_size, eps=eps)
    if config.sequence_parallel:
        for parameter in norm.parameters():
            setattr(parameter, "sequence_parallel", True)
    return norm


def _validate_supported_submodules(submodules: MLASelfAttentionSubmodules) -> None:
    expected_column = (
        "linear_q_proj",
        "linear_q_down_proj",
        "linear_q_up_proj",
        "linear_kv_down_proj",
        "linear_kv_up_proj",
    )
    for name in expected_column:
        _require(
            getattr(submodules, name) is ColumnParallelLinear,
            f"{name} must use the local MCore ColumnParallelLinear spec",
        )
    _require(
        submodules.linear_proj is RowParallelLinear,
        "linear_proj must use the local MCore RowParallelLinear spec",
    )
    _require(
        submodules.linear_gate in (None, ColumnParallelLinear),
        "linear_gate must use the local MCore ColumnParallelLinear spec",
    )
    _require(
        submodules.q_layernorm is _build_local_latent_norm
        and submodules.kv_layernorm is _build_local_latent_norm,
        "Q/KV norms must use the local latent-CP norm builder",
    )
    _require(submodules.linear_qkv_down_proj is None, "fused MLA down projection is unsupported")
    _require(
        submodules.core_attention is IdentityOp,
        "core_attention must be IdentityOp; use make_mla_with_latent_cp_spec",
    )


# -----------------------------------------------------------------------------
# Latent-CP MLA module
# -----------------------------------------------------------------------------


class MLAWithLatentCP(MLASelfAttention):
    """Training-only THD MLA whose P2P CP ring exchanges normalized latent KV."""

    def __init__(
        self,
        config: MLATransformerConfig,
        submodules: MLASelfAttentionSubmodules,
        layer_number: int,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        cp_comm_type: str | None = None,
        pg_collection: ProcessGroupCollection | None = None,
        pp_layer_offset: int | None = None,
        is_mtp_layer: bool = False,
        name: str | None = None,
    ) -> None:
        _require(pg_collection is not None, "an explicit ProcessGroupCollection is required")
        _require(
            hasattr(pg_collection, "tp")
            and pg_collection.tp is not None
            and hasattr(pg_collection, "cp")
            and pg_collection.cp is not None,
            "explicit non-null TP and CP process groups are required",
        )
        _require(not is_mtp_layer, "MTP layers are unsupported in v1")
        _validate_supported_submodules(submodules)
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            attn_mask_type=attn_mask_type,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
            pp_layer_offset=pp_layer_offset,
            is_mtp_layer=is_mtp_layer,
            name=name,
        )
        self._cp_comm_type = cp_comm_type if cp_comm_type is not None else config.cp_comm_type
        self._layout_adapter: LatentCPLayoutAdapter = AlreadyZigZagTHDAdapter()
        self._parameter_dtypes_validated = False
        self._validate_initial_config()
        self._validate_projection_groups()
        self._backend_adapter: DirectAttentionAdapter
        self._backend_runtime_tuple: QualifiedBackendTuple
        self._backend_adapter, self._backend_runtime_tuple = _qualified_backend_adapter(
            self.config.attention_backend
        )

    def _validate_initial_config(self) -> None:
        config = self.config
        tp_size = dist.get_world_size(self.pg_collection.tp)
        cp_size = dist.get_world_size(self.pg_collection.cp)
        _require(isinstance(config, MLATransformerConfig), "MLATransformerConfig is required")
        _require(config.multi_latent_attention, "multi_latent_attention=True is required")
        _require(config.mla_latent_cp, "mla_latent_cp=True is required")
        _require(config.qk_layernorm, "standalone Q/KV layer norms must be enabled")
        _require(not config.add_bias_linear, "all MLA projection biases must be disabled")
        _require(config.rotary_percent == 1.0, "partial rotary dimensions are unsupported")
        _require(
            self.attn_mask_type is AttnMaskType.causal, "only causal self-attention is supported"
        )
        _require(cp_size == 1 or self._cp_comm_type == "p2p", "CP>1 requires cp_comm_type='p2p'")
        _require(
            config.tensor_model_parallel_size == tp_size,
            "configured TP size disagrees with the injected TP group",
        )
        _require(
            config.context_parallel_size == cp_size,
            "configured CP size disagrees with the injected CP group",
        )
        _require(config.q_lora_rank is not None, "a nonzero q_lora_rank is required")
        _require(config.q_lora_rank > 0, "q_lora_rank must be positive")
        _require(config.normalization == "RMSNorm", "only RMSNorm projection specs are supported")
        _require(tp_size == 1 or config.sequence_parallel, "TP>1 requires sequence_parallel=True")
        _require(config.num_attention_heads % tp_size == 0, "attention heads must divide TP")
        _require(
            config.num_query_groups == config.num_attention_heads, "v1 requires Hq=Hkv (no GQA)"
        )
        _require(
            config.qk_head_dim == 128
            and config.qk_pos_emb_head_dim == 64
            and config.v_head_dim == 128,
            "v1 requires qk content/rope/value dimensions 128/64/128",
        )
        _require(config.rope_type in ("rope", "yarn"), "only rope and yarn are supported")
        _require(
            not self.use_rope or not config.apply_rope_fusion,
            "fused RoPE is unsupported when this layer applies RoPE",
        )
        _require(config.attention_dropout == 0.0, "attention dropout must be zero")
        _require(config.bf16 and not config.fp16, "v1 requires BF16 and rejects FP16")
        _require(config.fp8 is None and config.fp4 is None, "FP8 and FP4 are unsupported")
        _require(not config.cache_mla_latents, "inference latent caching is unsupported")
        _require(
            config.recompute_granularity is None and config.recompute_modules in (None, []),
            "outer/selective recompute is unsupported; set recompute_modules=[]",
        )
        _require(
            not config.fine_grained_activation_offloading,
            "fine-grained activation offload is unsupported",
        )
        _require(
            not config.cpu_offloading and config._cpu_offloading_context is None,
            "CPU offloading is unsupported",
        )
        _require(
            config.cuda_graph_impl == "none"
            and not config.enable_cuda_graph
            and not config.external_cuda_graph,
            "CUDA graph execution is unsupported",
        )
        _require(
            config.attention_backend in (AttnBackend.fused, AttnBackend.flash),
            "attention_backend must be fused (cuDNN) or flash (FA4)",
        )

    def _validate_projection_groups(self) -> None:
        for name in (
            "linear_q_down_proj",
            "linear_q_up_proj",
            "linear_kv_down_proj",
            "linear_kv_up_proj",
            "linear_proj",
        ):
            module = getattr(self, name)
            _require(
                getattr(module, "tp_group", None) is self.pg_collection.tp,
                f"{name} does not retain the injected TP process group",
            )

        if self.linear_gate is not None:
            _require(
                isinstance(self.linear_gate, ColumnParallelLinear),
                "linear_gate must be a local MCore ColumnParallelLinear",
            )
            _require(
                self.linear_gate.tp_group is self.pg_collection.tp,
                "linear_gate does not retain the injected TP process group",
            )
            _require(
                not self.linear_gate.gather_output
                and not self.linear_gate.skip_bias_add
                and self.linear_gate.bias is None
                and not self.linear_gate.explicit_expert_comm,
                "linear_gate must be a bias-free non-expert sharded projection",
            )

        output_projection = self.linear_proj
        _require(
            output_projection.input_is_parallel
            and output_projection.skip_bias_add
            and output_projection.sequence_parallel == self.config.sequence_parallel
            and not output_projection.explicit_expert_comm,
            "linear_proj must be a non-expert row-parallel projection with parallel input",
        )
        _require(output_projection.bias is None, "linear_proj bias is unsupported")

    def _validate_forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None,
        inference_context: Any,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None,
        rotary_pos_cos: Tensor | None,
        rotary_pos_sin: Tensor | None,
        rotary_pos_cos_sin: Tensor | None,
        attention_bias: Tensor | None,
        packed_seq_params: PackedSeqParams | None,
        position_ids: Tensor | None,
        sequence_len_offset: int | None,
        inference_params: Any,
    ) -> PackedSeqParams:
        _require(self.training, "v1 is training-only")
        _require(hidden_states.ndim == 3, "hidden_states must have shape [T, 1, hidden]")
        _require(hidden_states.size(1) == 1, "THD requires the singleton batch axis")
        _require(hidden_states.is_cuda, "activations must be CUDA tensors")
        _require(
            hidden_states.device.index == torch.cuda.current_device(),
            "hidden_states must use the current CUDA device",
        )
        _require(hidden_states.dtype == torch.bfloat16, "activations must be BF16")
        _require(attention_mask is None, "explicit attention masks are unsupported")
        _require(key_value_states is None, "cross attention is unsupported")
        _require(inference_context is None and inference_params is None, "inference is unsupported")
        _require(rotary_pos_emb is None, "external rotary_pos_emb is unsupported")
        _require(
            rotary_pos_cos is None and rotary_pos_sin is None and rotary_pos_cos_sin is None,
            "flash-decoding/fused rotary inputs are unsupported",
        )
        _require(attention_bias is None, "attention bias is unsupported")
        _require(position_ids is None, "external position_ids are unsupported")
        _require(sequence_len_offset is None, "sequence offsets are unsupported")
        _require(isinstance(packed_seq_params, PackedSeqParams), "PackedSeqParams is required")
        local_tokens = hidden_states.size(0)
        if self.config.sequence_parallel:
            local_tokens *= dist.get_world_size(self.pg_collection.tp)
        _require(
            packed_seq_params.total_tokens in (None, local_tokens),
            "total_tokens must equal the pre-SP local THD token count",
        )
        _require(
            packed_seq_params.pad_between_seqs in (None, False),
            "inter-sequence padding is unsupported",
        )
        if not self._parameter_dtypes_validated:
            for name, parameter in self.named_parameters():
                if parameter.is_floating_point():
                    _require(parameter.dtype == torch.bfloat16, f"parameter {name} must be BF16")
            self._parameter_dtypes_validated = True
        return packed_seq_params

    def _microbatch_layout(
        self, hidden_states: Tensor, packed_seq_params: PackedSeqParams
    ) -> tuple[dist.ProcessGroup, ZigZagLayout]:
        """Build the cheap per-microbatch layout using the scheduler-selected CP group."""

        cp_group = resolve_cp_group(self.pg_collection.cp, packed_seq_params)
        layout = self._layout_adapter.prepare(
            hidden_states,
            packed_seq_params,
            cp_group,
            tp_group=self.pg_collection.tp,
            sequence_parallel=self.config.sequence_parallel,
        )
        return cp_group, layout

    def _preprocess_backend(
        self, hidden_states: Tensor, packed_seq_params: PackedSeqParams
    ) -> None:
        """Prepare expensive backend plans before the transformer layer loop."""

        _require(isinstance(packed_seq_params, PackedSeqParams), "PackedSeqParams is required")
        _, layout = self._microbatch_layout(hidden_states, packed_seq_params)
        self._backend_adapter.prepare(
            num_heads=self.num_attention_heads_per_partition,
            qk_dim=self.q_head_dim,
            v_dim=self.config.v_head_dim,
            phases=layout.phases,
            scale=self.softmax_scale,
        )

    def _explicit_output_projection(self, core_output: Tensor) -> tuple[Tensor, Tensor | None]:
        """Apply the inherited row-sharded weight without an implicit TP-group lookup."""

        projection = self.linear_proj
        _require(core_output.dtype == torch.bfloat16, "linear_proj input must be BF16")
        _require(projection.bias is None, "linear_proj bias is unsupported")
        _require(
            not self.config.cpu_offloading and self.config._cpu_offloading_context is None,
            "CPU offloading is unsupported",
        )
        _require(
            projection.weight.requires_grad, "frozen linear_proj weights are unsupported in v1"
        )
        output_parallel = mcore_tp.linear_with_grad_accumulation_and_async_allreduce(
            input=core_output,
            weight=projection.weight,
            bias=None,
            gradient_accumulation_fusion=projection.gradient_accumulation_fusion,
            allreduce_dgrad=False,
            sequence_parallel=False,
            grad_output_buffer=None,
            wgrad_deferral_limit=0,
            tp_group=self.pg_collection.tp,
        )
        if self.config.sequence_parallel:
            output = mcore_tp.reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.pg_collection.tp
            )
        else:
            output = mcore_tp.reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.pg_collection.tp
            )
        return output, None

    def _latent_cp_down_projection(self, hidden_states: Tensor) -> tuple[Tensor, Tensor]:
        """Run local projection modules and gather every shard with the injected TP group."""

        q_compressed, _ = self.linear_q_down_proj(hidden_states)
        kv_combined, _ = self.linear_kv_down_proj(hidden_states)
        expected_kv = self.config.kv_lora_rank + self.config.qk_pos_emb_head_dim
        if q_compressed.size(-1) != self.config.q_lora_rank:
            q_compressed = tp_mappings.gather_from_tensor_model_parallel_region(
                q_compressed, group=self.pg_collection.tp
            )
        if kv_combined.size(-1) != expected_kv:
            kv_combined = tp_mappings.gather_from_tensor_model_parallel_region(
                kv_combined, group=self.pg_collection.tp
            )
        _require(
            q_compressed.size(-1) == self.config.q_lora_rank,
            "Q down-projection gather produced the wrong size",
        )
        _require(
            kv_combined.size(-1) == expected_kv, "KV down-projection gather produced the wrong size"
        )
        return q_compressed, kv_combined

    def _project_query_and_payload(
        self,
        hidden_states: Tensor,
        packed_seq_params: PackedSeqParams,
        layout: ZigZagLayout,
        cp_group: dist.ProcessGroup | None = None,
    ) -> tuple[Tensor, Tensor]:
        if cp_group is None:
            cp_group = self.pg_collection.cp
        q_compressed, kv_combined = self._latent_cp_down_projection(hidden_states)
        q_compressed = q_compressed.squeeze(1)
        kv_combined = kv_combined.squeeze(1)
        kv_compressed, k_rope_raw = torch.split(
            kv_combined, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        if self.config.sequence_parallel:
            q_compressed = tp_mappings.scatter_to_sequence_parallel_region(
                q_compressed, group=self.pg_collection.tp
            )
            kv_compressed = tp_mappings.scatter_to_sequence_parallel_region(
                kv_compressed, group=self.pg_collection.tp
            )
        q_compressed = self.q_layernorm(q_compressed)
        kv_compressed = self.kv_layernorm(kv_compressed)

        q, _ = self.linear_q_up_proj(q_compressed)
        q = q.view(q.size(0), self.num_attention_heads_per_partition, self.q_head_dim)
        q_content, q_rope = torch.split(
            q, [self.config.qk_head_dim, self.config.qk_pos_emb_head_dim], dim=-1
        )
        k_rope = k_rope_raw.unsqueeze(1)

        if self.use_rope:
            rotary_seq_len = self.rotary_pos_emb.get_rotary_seq_len(
                None, None, hidden_states, self.config, packed_seq_params
            )
            mscale = 1.0
            if self.config.rope_type == "rope":
                rotary = self.rotary_pos_emb(rotary_seq_len, packed_seq=True, cp_group=cp_group)
            else:
                rotary, mscale = self.rotary_pos_emb(
                    rotary_seq_len, packed_seq=True, cp_group=cp_group
                )
            q_rope = apply_rotary_pos_emb(
                q_rope,
                rotary,
                config=self.config,
                cu_seqlens=layout.cu_global,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                max_seqlen=layout.max_global,
            )
            k_rope = apply_rotary_pos_emb(
                k_rope,
                rotary,
                config=self.config,
                cu_seqlens=layout.cu_global,
                mscale=mscale,
                cp_group=cp_group,
                mla_rotary_interleaved=True,
                max_seqlen=layout.max_global,
            )
        if self.config.sequence_parallel:
            k_rope = tp_mappings.scatter_to_sequence_parallel_region(
                k_rope, group=self.pg_collection.tp
            )
        query = torch.cat((q_content, q_rope), dim=-1).contiguous()
        payload = torch.cat((kv_compressed, k_rope.squeeze(1)), dim=-1).contiguous()
        _require(query.dtype == torch.bfloat16, "query projection must remain BF16")
        _require(payload.dtype == torch.bfloat16, "latent ring payload must remain BF16")
        return query, payload

    def _phase_attention(
        self, query: Tensor, payload: Tensor, phase: PhaseSpec, backend: DirectAttentionAdapter
    ) -> tuple[Tensor, Tensor]:
        latent, k_rope = torch.split(
            payload, [self.config.kv_lora_rank, self.config.qk_pos_emb_head_dim], dim=-1
        )
        latent = latent.contiguous()
        k_rope = k_rope.contiguous()
        expanded, _ = self.linear_kv_up_proj(latent)
        if self.config.sequence_parallel:
            k_rope = tp_mappings.gather_from_sequence_parallel_region(
                k_rope, tensor_parallel_output_grad=True, group=self.pg_collection.tp
            )
        expanded = expanded.view(
            expanded.size(0),
            self.num_attention_heads_per_partition,
            self.config.qk_head_dim + self.config.v_head_dim,
        )
        expanded = expanded.index_select(0, phase.kv_indices)
        k_rope = k_rope.index_select(0, phase.kv_indices)
        k_content, value = torch.split(
            expanded, [self.config.qk_head_dim, self.config.v_head_dim], dim=-1
        )
        key = torch.cat(
            (k_content, k_rope.unsqueeze(1).expand(-1, self.num_attention_heads_per_partition, -1)),
            dim=-1,
        ).contiguous()
        output, lse = backend.forward_phase(
            query.contiguous(),
            key,
            value.contiguous(),
            phase.cu_seqlens_q,
            phase.cu_seqlens_kv,
            phase.max_seqlen_q,
            phase.max_seqlen_kv,
            phase.causal,
            self.softmax_scale,
        )
        _require(output.dtype == torch.float32, "backend canonical output must be FP32")
        _require(lse.dtype == torch.float32, "backend canonical LSE must be FP32")
        return output, lse

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None,
        key_value_states: Tensor | None = None,
        inference_context: Any = None,
        rotary_pos_emb: Tensor | tuple[Tensor, Tensor] | None = None,
        rotary_pos_cos: Tensor | None = None,
        rotary_pos_sin: Tensor | None = None,
        rotary_pos_cos_sin: Tensor | None = None,
        attention_bias: Tensor | None = None,
        packed_seq_params: PackedSeqParams | None = None,
        position_ids: Tensor | None = None,
        sequence_len_offset: int | None = None,
        *,
        inference_params: Any = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run latent-P2P context-parallel MLA for one packed THD input."""
        packed_seq_params = self._validate_forward(
            hidden_states,
            attention_mask,
            key_value_states,
            inference_context,
            rotary_pos_emb,
            rotary_pos_cos,
            rotary_pos_sin,
            rotary_pos_cos_sin,
            attention_bias,
            packed_seq_params,
            position_ids,
            sequence_len_offset,
            inference_params,
        )
        effective_cp_group, layout = self._microbatch_layout(hidden_states, packed_seq_params)
        backend = self._backend_adapter
        query, local_payload = self._project_query_and_payload(
            hidden_states, packed_seq_params, layout, effective_cp_group
        )
        transport: LatentCPTransport = P2PRingTransport(effective_cp_group)

        merged_output: Tensor | None = None
        merged_lse: Tensor | None = None
        lease_count = 0
        leases = transport.iter_payloads(local_payload, layout.phases)
        for phase, lease in zip(layout.phases, leases, strict=True):
            lease_count += 1
            _require(lease.owner == phase.owner, "transport owner order disagrees with plan")
            q_phase = query.index_select(0, phase.q_indices)
            payload_phase = lease.tensor

            def run_phase(
                q_input: Tensor,
                payload_input: Tensor,
                phase_spec: PhaseSpec = phase,
                phase_backend: DirectAttentionAdapter = backend,
            ) -> tuple[Tensor, Tensor]:
                return self._phase_attention(q_input, payload_input, phase_spec, phase_backend)

            partial_output, partial_lse = checkpoint(
                run_phase, q_phase, payload_phase, use_reentrant=False, preserve_rng_state=False
            )
            if phase.scatter_indices is not None:
                partial_output, partial_lse = scatter_upper_phase(
                    partial_output, partial_lse, phase.scatter_indices, layout.local_tokens
                )
            if merged_output is None:
                merged_output, merged_lse = partial_output, partial_lse
            else:
                merged_output, merged_lse = merge_attention_partials(
                    merged_output, merged_lse, partial_output, partial_lse
                )

        _require(
            lease_count == len(layout.phases),
            "transport did not yield exactly one lease per CP phase",
        )

        if merged_output is None:
            raise LatentCPError("zigzag phase plan unexpectedly produced no attention output")
        # This is the one and only post-backend FP32-to-BF16 cast.
        core_output = merged_output.to(torch.bfloat16).reshape(
            layout.local_tokens, 1, self.num_attention_heads_per_partition * self.config.v_head_dim
        )
        if self.linear_gate is not None:
            core_output = self._project_and_apply_mla_output_gate(core_output, hidden_states)
        return self._explicit_output_projection(core_output)


def preprocess_mla_latent_cp(
    block: torch.nn.Module, hidden_states: Tensor, packed_seq_params: PackedSeqParams | None
) -> None:
    """Prepare every latent-CP attention layer before a block enters its layer loop.

    Backend qualification is construction-time state. This hook owns only expensive,
    microbatch-specific plan preparation; it does not cache forward state or run collectives.
    """

    latent_layers = tuple(
        module for module in block.modules() if isinstance(module, MLAWithLatentCP)
    )
    if not latent_layers:
        return
    _require(isinstance(packed_seq_params, PackedSeqParams), "PackedSeqParams is required")
    for layer in latent_layers:
        layer._preprocess_backend(hidden_states, packed_seq_params)


# -----------------------------------------------------------------------------
# Non-mutating model-spec integration
# -----------------------------------------------------------------------------


def make_mla_with_latent_cp_spec(base_mla_spec: ModuleSpec) -> ModuleSpec:
    """Return a non-mutating opt-in copy of a supported local MLA attention spec."""

    from megatron.core.transformer.dot_product_attention import DotProductAttention

    _require(isinstance(base_mla_spec, ModuleSpec), "base_mla_spec must be a ModuleSpec")
    _require(
        base_mla_spec.module is MLASelfAttention,
        "base_mla_spec must be layer_spec.submodules.self_attention from local MLA",
    )
    _require(
        isinstance(base_mla_spec.submodules, MLASelfAttentionSubmodules),
        "base_mla_spec has incompatible submodules",
    )
    original = base_mla_spec.submodules
    expected_column = (
        original.linear_q_proj,
        original.linear_q_down_proj,
        original.linear_q_up_proj,
        original.linear_kv_down_proj,
        original.linear_kv_up_proj,
    )
    _require(
        all(module is ColumnParallelLinear for module in expected_column),
        "base MLA spec must use local MCore ColumnParallelLinear projections",
    )
    _require(
        original.linear_proj is RowParallelLinear,
        "base MLA spec must use local MCore RowParallelLinear output",
    )
    _require(
        original.q_layernorm is WrappedTorchNorm and original.kv_layernorm is WrappedTorchNorm,
        "base MLA spec must use standalone WrappedTorchNorm Q/KV norms",
    )
    _require(
        original.core_attention is DotProductAttention,
        "base MLA spec must use the local MCore core-attention placeholder",
    )
    _require(original.linear_qkv_down_proj is None, "fused MLA down projection is unsupported")
    _require(
        base_mla_spec.params.get("attn_mask_type") is AttnMaskType.causal,
        "base MLA spec must be causal",
    )

    latent_submodules = replace(
        original,
        q_layernorm=_build_local_latent_norm,
        kv_layernorm=_build_local_latent_norm,
        core_attention=IdentityOp,
    )
    return replace(
        base_mla_spec,
        module=MLAWithLatentCP,
        params=dict(base_mla_spec.params),
        submodules=latent_submodules,
        metainfo=dict(base_mla_spec.metainfo),
    )


def get_mla_with_latent_cp_spec() -> ModuleSpec:
    """Build the feature-owned local MLA attention spec used by model integration."""

    return ModuleSpec(
        module=MLAWithLatentCP,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=MLASelfAttentionSubmodules(
            linear_q_proj=ColumnParallelLinear,
            linear_q_down_proj=ColumnParallelLinear,
            linear_q_up_proj=ColumnParallelLinear,
            linear_kv_down_proj=ColumnParallelLinear,
            linear_kv_up_proj=ColumnParallelLinear,
            core_attention=IdentityOp,
            linear_gate=ColumnParallelLinear,
            linear_proj=RowParallelLinear,
            q_layernorm=_build_local_latent_norm,
            kv_layernorm=_build_local_latent_norm,
        ),
        metainfo={"fuse_input_layernorm": False},
    )


def _replace_transformer_layer_attention(layer_spec: ModuleSpec) -> tuple[ModuleSpec, bool]:
    """Replace one ordinary MLA attention slot without mutating its layer spec."""

    _require(isinstance(layer_spec, ModuleSpec), "decoder layers must use ModuleSpec")
    layer_submodules = layer_spec.submodules
    _require(layer_submodules is not None, "decoder layer spec must define submodules")
    attention_spec = getattr(layer_submodules, "self_attention", None)
    if not isinstance(attention_spec, ModuleSpec) or attention_spec.module is not MLASelfAttention:
        return layer_spec, False
    return (
        replace(
            layer_spec,
            params=dict(layer_spec.params),
            metainfo=dict(layer_spec.metainfo),
            submodules=replace(
                layer_submodules,
                self_attention=get_mla_with_latent_cp_spec(),
                sharded_state_dict_keys_map=dict(layer_submodules.sharded_state_dict_keys_map),
            ),
        ),
        True,
    )


def configure_mla_latent_cp_decoder(
    decoder_spec: ModuleSpec | TransformerBlockSubmodules,
) -> ModuleSpec | TransformerBlockSubmodules:
    """Return a non-mutating GPT decoder spec with latent CP attention."""

    replaced = 0
    if isinstance(decoder_spec, ModuleSpec):
        configured_spec, changed = _replace_transformer_layer_attention(decoder_spec)
        replaced = int(changed)
    elif isinstance(decoder_spec, TransformerBlockSubmodules):
        _require(decoder_spec.layer_specs is not None, "decoder block must define layer specs")
        configured_layers = []
        for layer_spec in decoder_spec.layer_specs:
            configured_layer, changed = _replace_transformer_layer_attention(layer_spec)
            configured_layers.append(configured_layer)
            replaced += int(changed)
        configured_spec = replace(decoder_spec, layer_specs=configured_layers)
    else:
        raise LatentCPError(
            "latent CP requires a ModuleSpec or TransformerBlockSubmodules decoder spec"
        )
    _require(replaced > 0, "decoder spec contains no ordinary MLA attention slot")
    return configured_spec


def configure_mla_latent_cp_hybrid_stack(stack_spec: ModuleSpec) -> ModuleSpec:
    """Return a non-mutating Hybrid stack with latent CP in its ordinary attention slot."""

    _require(isinstance(stack_spec, ModuleSpec), "hybrid stack must use ModuleSpec")
    stack_submodules = stack_spec.submodules
    _require(stack_submodules is not None, "hybrid stack spec must define submodules")
    mla_layer = getattr(stack_submodules, "mla_layer", None)
    _require(
        isinstance(mla_layer, ModuleSpec),
        "hybrid stack must provide its ordinary MLA layer template",
    )
    latent_layer, replaced = _replace_transformer_layer_attention(mla_layer)
    _require(replaced, "hybrid MLA layer template has no ordinary MLA attention slot")
    return replace(
        stack_spec,
        params=dict(stack_spec.params),
        metainfo=dict(stack_spec.metainfo),
        submodules=replace(stack_submodules, attention_layer=latent_layer),
    )
