# Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Portions of this file follow the MIT-licensed FLA v0.5.1 GDR autograd
# contract. The SM100 forward implementation is local to Megatron Core.

"""Autograd orchestration for the in-tree SM100 CuTe DSL GDR kernel."""

from __future__ import annotations

import importlib.util
import inspect
import os
import weakref
from dataclasses import dataclass
from typing import Any

import torch
from fla.ops.common.chunk_delta_h import (
    chunk_gated_delta_rule_bwd_dhu,
    chunk_gated_delta_rule_fwd_h,
)
from fla.ops.common.chunk_o import chunk_bwd_dqkwg, chunk_bwd_dv_local, chunk_fwd_o
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
from fla.ops.gated_delta_rule.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.ops.utils.constant import RCP_LN2
from fla.ops.utils.index import prepare_chunk_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard

try:
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
except ModuleNotFoundError as exc:
    if exc.name != "fla.ops.gated_delta_rule.chunk_fwd":
        raise
    chunk_gated_delta_rule_fwd_intra = None

_CHUNK_SIZE = 64
_BACKEND_ENV = "MCORE_GDN_INTERNAL_BACKEND"
_FUSED_BWD_HEADS = 64
_FUSED_BWD_HEAD_DIM = 128
_fused_bwd_zero_dht_cache: dict[tuple[str, int | None, int], torch.Tensor] = {}


@dataclass(frozen=True)
class _PackedChunkMetadata:
    _cu_seqlens_ref: weakref.ReferenceType
    chunk_indices: torch.Tensor
    chunk_offsets: torch.Tensor

    @property
    def cu_seqlens(self) -> torch.Tensor:
        tensor = self._cu_seqlens_ref()
        if tensor is None:
            raise RuntimeError("cu_seqlens owner is no longer alive")
        return tensor


@dataclass
class _PackedChunkMetadataEntry:
    owner: weakref.ReferenceType
    version: int
    chunk_size: int
    metadata: _PackedChunkMetadata


_packed_chunk_metadata_cache: dict[int, _PackedChunkMetadataEntry] = {}


def _clear_packed_chunk_metadata_cache_for_test() -> None:
    _packed_chunk_metadata_cache.clear()


def _packed_chunk_metadata(
    cu_seqlens: torch.Tensor | None, cu_seqlens_cpu: torch.Tensor | None
) -> _PackedChunkMetadata | None:
    if cu_seqlens is None:
        return None
    key = id(cu_seqlens)
    version = cu_seqlens._version
    cached = _packed_chunk_metadata_cache.get(key)
    if (
        cached is not None
        and cached.owner() is cu_seqlens
        and cached.version == version
        and cached.chunk_size == _CHUNK_SIZE
    ):
        return cached.metadata

    chunk_indices = prepare_chunk_indices(
        cu_seqlens, _CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu
    )
    chunk_offsets = (cu_seqlens // _CHUNK_SIZE).contiguous()

    def remove_cached_owner(owner_ref):
        current = _packed_chunk_metadata_cache.get(key)
        if current is not None and current.owner is owner_ref:
            _packed_chunk_metadata_cache.pop(key, None)

    owner_ref = weakref.ref(cu_seqlens, remove_cached_owner)
    metadata = _PackedChunkMetadata(
        _cu_seqlens_ref=owner_ref, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
    )
    _packed_chunk_metadata_cache[key] = _PackedChunkMetadataEntry(
        owner=owner_ref, version=version, chunk_size=_CHUNK_SIZE, metadata=metadata
    )
    return metadata


def _call_fla_compat(function, **kwargs):
    """Call an FLA primitive across compatible v0.5.x signatures."""
    try:
        parameters = inspect.signature(function).parameters
    except (TypeError, ValueError):
        parameters = {}
    kwargs = dict(kwargs)
    if kwargs.get("use_exp2") and "use_exp2" not in parameters and kwargs.get("g") is not None:
        kwargs["g"] = kwargs["g"] / RCP_LN2
    if parameters:
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        if not accepts_kwargs:
            kwargs = {name: value for name, value in kwargs.items() if name in parameters}
        else:
            for name in ("use_cute", "use_exp2"):
                if name not in parameters:
                    kwargs.pop(name, None)
    return function(**kwargs)


def _fla_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
):
    """Recompose the FLA v0.5.1 backward from its public kernel primitives."""
    common = {"cu_seqlens": cu_seqlens, "chunk_indices": chunk_indices}
    w, u = _call_fla_compat(recompute_w_u_fwd, k=k, v=v, beta=beta, A=A, g=g, **common)
    h, v_new, _ = _call_fla_compat(
        chunk_gated_delta_rule_fwd_h,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        transpose_state_layout=False,
        **common,
    )
    dv = _call_fla_compat(chunk_bwd_dv_local, q=q, k=k, g=g, do=do, scale=scale, **common)
    dh, _dh0, dv = _call_fla_compat(
        chunk_gated_delta_rule_bwd_dhu,
        q=q,
        k=k,
        w=w,
        g=g,
        h0=None,
        dht=None,
        do=do,
        dv=dv,
        scale=scale,
        transpose_state_layout=False,
        use_cute=False,
        **common,
    )
    dq, dk, dw, dg = _call_fla_compat(
        chunk_bwd_dqkwg,
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        transpose_state_layout=False,
        use_cute=False,
        **common,
    )
    dk2, dv, db, dg2 = _call_fla_compat(
        prepare_wy_repr_bwd, k=k, v=v, beta=beta, g=g, A=A, dw=dw, du=dv, use_cute=False, **common
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(
        dg, chunk_size=_CHUNK_SIZE, reverse=True, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    return dq, dk, dv, db, dg


def _recompute_fused_bwd_h(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> torch.Tensor:
    """Recompute the recurrent state exactly as in latest-main's verified path."""
    w, u = _call_fla_compat(
        recompute_w_u_fwd,
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
    )
    h, _v_new, _ = _call_fla_compat(
        chunk_gated_delta_rule_fwd_h,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=False,
    )
    return h


def _fused_bwd_support_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    trust_device_cu_seqlens: bool = False,
) -> str | None:
    """Return why the verified latest-main fused backward cannot run."""
    if q.dtype != torch.bfloat16:
        return "fused backward requires bf16 inputs"
    if any(tensor.dtype != q.dtype for tensor in (k, v, do)):
        return "q, k, v, and do dtypes differ"
    if q.ndim != 4 or tuple(q.shape[2:]) != (_FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM):
        return "fused backward requires BTHD with H=64 and D=128"
    if any(tensor.shape != q.shape for tensor in (k, v, do)):
        return "q, k, v, and do shapes differ"
    if any(not tensor.is_contiguous() for tensor in (q, k, v, do)):
        return "q, k, v, and do must be contiguous"
    scalar_shape = q.shape[:-1]
    if g.shape != scalar_shape or beta.shape != scalar_shape:
        return "g and beta must match q's BTH shape"
    if A.dtype != torch.bfloat16 or A.shape != (*scalar_shape, _CHUNK_SIZE):
        return "A must be bf16 with shape BTH64"
    if cu_seqlens is None:
        num_sequences = q.shape[0]
        if num_sequences < 1:
            return "dense fused backward requires a positive batch size"
        if q.shape[1] % _CHUNK_SIZE:
            return "sequence length must be divisible by 64"
    else:
        if q.shape[0] != 1:
            return "packed fused backward requires batch size 1"
        if (
            cu_seqlens.dtype != torch.int32
            or not cu_seqlens.is_contiguous()
            or cu_seqlens.numel() < 2
        ):
            return "cu_seqlens must be contiguous int32 with at least two offsets"
        offsets = _host_cu_seqlens(cu_seqlens, cu_seqlens_cpu)
        if offsets is not None:
            bounds = offsets.detach().tolist()
            if bounds[0] != 0 or bounds[-1] != q.shape[1]:
                return "cu_seqlens bounds do not match the packed token count"
            if any(
                end <= start or (end - start) % _CHUNK_SIZE
                for start, end in zip(bounds, bounds[1:])
            ):
                return "every packed sequence length must be a positive multiple of 64"
            num_sequences = len(bounds) - 1
        elif trust_device_cu_seqlens:
            if q.shape[1] % _CHUNK_SIZE:
                return "packed token count must be divisible by 64"
            num_sequences = cu_seqlens.numel() - 1
        else:
            return "packed cu_seqlens host metadata is required for validation"
    if dht is not None and (
        dht.dtype != torch.float32
        or dht.shape != (num_sequences, _FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM, _FUSED_BWD_HEAD_DIM)
    ):
        return f"dht must be float32 with shape ({num_sequences}, 64, 128, 128)"
    return None


def _fused_bwd_zero_dht(device: torch.device, num_sequences: int) -> torch.Tensor:
    device = torch.device(device)
    device_index = device.index
    if device.type == "cuda" and device_index is None:
        device_index = torch.cuda.current_device()
    key = (device.type, device_index, num_sequences)
    cached = _fused_bwd_zero_dht_cache.get(key)
    if cached is None or cached.device != device:
        cached = torch.zeros(
            (num_sequences, _FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM, _FUSED_BWD_HEAD_DIM),
            dtype=torch.float32,
            device=device,
        )
        _fused_bwd_zero_dht_cache[key] = cached
    return cached


def _prepare_fused_bwd_h(
    h: torch.Tensor, *, total_chunks: int, num_heads: int, head_size: int
) -> torch.Tensor:
    """Normalize saved or recomputed chunk states to the fused-backward layout."""
    return (
        h.detach()
        .reshape(total_chunks, num_heads, head_size, head_size)
        .to(torch.bfloat16)
        .contiguous()
    )


def _call_fused_gdr_bwd_cute(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
):
    from .kernels.fused_gdr_bwd_cute import fused_gdr_bwd

    batch_size, seqlen, num_heads, head_size = q.shape
    total_tokens = batch_size * seqlen
    launch_cu_seqlens = (
        _dense_cu_seqlens(batch_size, seqlen, q.device) if cu_seqlens is None else cu_seqlens
    )
    num_sequences = launch_cu_seqlens.numel() - 1
    if h is None:
        h = _recompute_fused_bwd_h(
            k=k, v=v, g=g, beta=beta, A=A, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
        )
    launch_h = _prepare_fused_bwd_h(
        h, total_chunks=total_tokens // _CHUNK_SIZE, num_heads=num_heads, head_size=head_size
    ).unsqueeze(0)
    launch_g = g.detach().reshape(1, total_tokens, num_heads).to(torch.float32).contiguous()
    launch_beta = beta.detach().reshape(1, total_tokens, num_heads).to(torch.float32).contiguous()
    launch_dht = (
        _fused_bwd_zero_dht(q.device, num_sequences) if dht is None else dht.detach().contiguous()
    )

    dq, dk, dv, dg, db, _dh0 = fused_gdr_bwd(
        q=q.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        k=k.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        v=v.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        a=A.detach().reshape(1, total_tokens, num_heads, _CHUNK_SIZE).contiguous(),
        g=launch_g,
        beta=launch_beta,
        do=do.detach().reshape(1, total_tokens, num_heads, head_size).contiguous(),
        dht=launch_dht,
        h=launch_h,
        scale=scale,
        cu_seqlens=launch_cu_seqlens,
        chunk_offsets=chunk_offsets,
        chunk_size=_CHUNK_SIZE,
        state_v_first=False,
    )
    return (
        dq.reshape_as(q),
        dk.reshape_as(k),
        dv.reshape_as(v),
        db.reshape_as(beta).to(beta.dtype),
        dg.reshape_as(g).to(g.dtype),
    )


def _backend_mode() -> str:
    mode = os.environ.get(_BACKEND_ENV, "auto").lower()
    if mode not in {"auto", "cute", "fla"}:
        raise ValueError(f"{_BACKEND_ENV} must be auto|cute|fla, got {mode!r}")
    return mode


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _host_cu_seqlens(
    cu_seqlens: torch.Tensor, cu_seqlens_cpu: torch.Tensor | None
) -> torch.Tensor | None:
    if cu_seqlens_cpu is not None:
        return cu_seqlens_cpu
    if cu_seqlens.device.type == "cpu":
        return cu_seqlens
    return None


def _aligned_sequence_lengths(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: torch.Tensor | None,
    *,
    trust_device_cu_seqlens: bool = False,
) -> bool:
    offsets = _host_cu_seqlens(cu_seqlens, cu_seqlens_cpu)
    if offsets is None:
        return trust_device_cu_seqlens
    if offsets.numel() < 2:
        return False
    lengths = offsets[1:] - offsets[:-1]
    return bool((lengths > 0).all().item() and (lengths % _CHUNK_SIZE == 0).all().item())


def _cutedsl_support_reason(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    use_beta_sigmoid_in_kernel: bool,
    allow_neg_eigval: bool,
    state_v_first: bool,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    cp_context: object | None,
    kwargs: dict[str, Any],
    trust_device_cu_seqlens: bool = False,
) -> str | None:
    if not _module_available("cutlass") or not _module_available("cuda.bindings.driver"):
        return "CuTe DSL runtime is not installed"
    if not q.is_cuda:
        return "inputs are not CUDA tensors"
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        return "q, k, v, g, and beta must be on the same CUDA device"
    if not torch.version.cuda or int(torch.version.cuda.split(".", 1)[0]) < 13:
        return "CUDA 13 or newer is required"
    if torch.cuda.get_device_capability(q.device)[0] != 10:
        return "device is not SM100"
    if q.dtype not in (torch.bfloat16, torch.float16):
        return f"dtype {q.dtype} is not supported"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q, k, and v dtypes differ"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return "q, k, and v must use BTHD layout"
    if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
        return "q, k, and v must be contiguous"
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        return "q, k, and v token dimensions differ"
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        return "grouped value attention is not yet enabled for the CuTe path"
    if q.shape[-1] != 128 or k.shape[-1] != 128 or v.shape[-1] != 128:
        return "head dimensions must all equal 128"
    if tuple(g.shape) != tuple(beta.shape) or tuple(g.shape) != tuple(q.shape[:3]):
        return "g and beta shapes must match q's BTH dimensions"
    if initial_state is not None or output_final_state:
        return "initial or final state is not yet supported"
    if use_qk_l2norm_in_kernel:
        return "in-kernel QK L2 normalization is not supported"
    if use_beta_sigmoid_in_kernel or allow_neg_eigval:
        return "in-kernel beta transformation is not supported"
    if state_v_first or "transpose_state_layout" in kwargs:
        return "V-first state layout is not supported"
    if cp_context is not None:
        return "context parallel is not supported"
    extra_kwargs = dict(kwargs)
    if extra_kwargs.get("use_gate_in_kernel", False):
        return "in-kernel gate activation is not supported"
    for name in ("use_gate_in_kernel", "head_first", "use_cute"):
        if extra_kwargs.get(name) is False:
            extra_kwargs.pop(name)
    for name in ("A_log", "dt_bias", "g_input"):
        if extra_kwargs.get(name) is None:
            extra_kwargs.pop(name, None)
    if extra_kwargs:
        return f"extra options are not supported: {sorted(extra_kwargs)}"
    if cu_seqlens is None:
        if q.shape[1] % _CHUNK_SIZE:
            return "sequence length must be a multiple of 64"
    else:
        if q.shape[0] != 1:
            return "packed variable length inputs require batch size 1"
        if not _aligned_sequence_lengths(
            cu_seqlens,
            cu_seqlens_cpu,
            trust_device_cu_seqlens=trust_device_cu_seqlens,
        ):
            return "every packed sequence length must be a positive multiple of 64"
    return None


def _dense_cu_seqlens(batch_size: int, seqlen: int, device: torch.device) -> torch.Tensor:
    return torch.arange(0, (batch_size + 1) * seqlen, seqlen, device=device, dtype=torch.int32)


def _reshape_bthd_to_thd(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], *tensor.shape[2:]).contiguous()


def _reshape_bth_to_th(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(tensor.shape[0] * tensor.shape[1], tensor.shape[2]).contiguous()


def _can_use_fused_bwd_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    trust_device_cu_seqlens: bool = False,
) -> bool:
    if q.dtype != torch.bfloat16 or q.ndim != 4:
        return False
    if tuple(q.shape[2:]) != (_FUSED_BWD_HEADS, _FUSED_BWD_HEAD_DIM):
        return False
    if any(tensor.dtype != q.dtype or tensor.shape != q.shape for tensor in (k, v)):
        return False
    if g.shape != q.shape[:-1] or beta.shape != q.shape[:-1]:
        return False
    if any(not tensor.is_contiguous() for tensor in (q, k, v)):
        return False
    if cu_seqlens is None:
        return q.shape[0] >= 1 and q.shape[1] % _CHUNK_SIZE == 0
    return (
        q.shape[0] == 1
        and cu_seqlens.dtype == torch.int32
        and cu_seqlens.is_contiguous()
        and cu_seqlens.numel() >= 2
        and _aligned_sequence_lengths(
            cu_seqlens,
            cu_seqlens_cpu,
            trust_device_cu_seqlens=trust_device_cu_seqlens,
        )
    )


def _fla_forward_for_fused_bwd(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    packed_metadata: _PackedChunkMetadata | None = None,
    save_fused_bwd_state: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Prepare the exact FLA state consumed by latest-main's fused backward."""
    chunk_indices = None
    if cu_seqlens is not None:
        if packed_metadata is not None:
            chunk_indices = packed_metadata.chunk_indices
        else:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, _CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu
            )
    g = chunk_local_cumsum(
        g, chunk_size=_CHUNK_SIZE, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    intra_kwargs = {
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "cu_seqlens": cu_seqlens,
        "chunk_indices": chunk_indices,
        "chunk_size": _CHUNK_SIZE,
        "use_exp2": True,
    }
    if chunk_gated_delta_rule_fwd_intra is not None:
        w, u, A = _call_fla_compat(chunk_gated_delta_rule_fwd_intra, **intra_kwargs)
    else:
        A = _call_fla_compat(
            chunk_scaled_dot_kkt_fwd,
            k=k,
            g=g,
            beta=beta,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_size=_CHUNK_SIZE,
            output_dtype=torch.float32,
            use_exp2=True,
        )
        A = _call_fla_compat(
            solve_tril,
            A=A,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            output_dtype=k.dtype,
        )
        w, u = _call_fla_compat(recompute_w_u_fwd, A=A, **intra_kwargs)
    h, v_new, _ = _call_fla_compat(
        chunk_gated_delta_rule_fwd_h,
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=None,
        output_final_state=False,
        save_new_value=True,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=False,
    )
    output = _call_fla_compat(
        chunk_fwd_o,
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        use_exp2=True,
        transpose_state_layout=False,
    )
    saved_h = None
    if save_fused_bwd_state:
        saved_h = _prepare_fused_bwd_h(
            h,
            total_chunks=q.shape[0] * q.shape[1] // _CHUNK_SIZE,
            num_heads=q.shape[2],
            head_size=q.shape[3],
        )
    return g, output, A, saved_h, chunk_indices


def _cutedsl_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    cu_seqlens_cpu: torch.Tensor | None,
    packed_metadata: _PackedChunkMetadata | None = None,
    save_fused_bwd_state: bool = False,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    from .kernels.fused_gdr_fwd_cute import chunk_gated_delta_rule_prefill_cute

    batch_size, seqlen, num_heads, head_size = q.shape
    chunk_indices = None
    if cu_seqlens is not None:
        if packed_metadata is not None:
            chunk_indices = packed_metadata.chunk_indices
        else:
            chunk_indices = prepare_chunk_indices(
                cu_seqlens, _CHUNK_SIZE, cu_seqlens_cpu=cu_seqlens_cpu
            )
    chunk_offsets = packed_metadata.chunk_offsets if packed_metadata is not None else None
    g = chunk_local_cumsum(
        g, chunk_size=_CHUNK_SIZE, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices
    )
    launch_cu_seqlens = (
        _dense_cu_seqlens(batch_size, seqlen, q.device)
        if cu_seqlens is None
        else cu_seqlens.to(device=q.device, dtype=torch.int32).contiguous()
    )
    flat_q = _reshape_bthd_to_thd(q.detach())
    flat_k = _reshape_bthd_to_thd(k.detach())
    flat_v = _reshape_bthd_to_thd(v.detach())
    flat_g = _reshape_bth_to_th(g.detach())
    flat_beta = _reshape_bth_to_th(beta.detach())
    flat_A = torch.empty(
        (batch_size * seqlen, num_heads, _CHUNK_SIZE), dtype=q.dtype, device=q.device
    )
    output = torch.empty(
        (batch_size * seqlen, num_heads, head_size), dtype=q.dtype, device=q.device
    )
    flat_h = None
    checkpoint_cu_starts = None
    if save_fused_bwd_state:
        flat_h = torch.empty(
            (batch_size * seqlen // _CHUNK_SIZE, num_heads, head_size, head_size),
            dtype=q.dtype,
            device=q.device,
        )
        checkpoint_cu_starts = (
            chunk_offsets if chunk_offsets is not None else launch_cu_seqlens // _CHUNK_SIZE
        )
    chunk_gated_delta_rule_prefill_cute(
        q=flat_q,
        k=flat_k,
        v=flat_v,
        g=flat_g,
        beta=flat_beta,
        scale=scale,
        cu_seqlens=launch_cu_seqlens,
        output=output,
        output_A=flat_A,
        assume_valid_cu_seqlens=True,
        output_h=flat_h,
        checkpoint_every_n_tokens=_CHUNK_SIZE if save_fused_bwd_state else 0,
        checkpoint_cu_starts=checkpoint_cu_starts,
        gate_is_log_cumsum=True,
    )
    return (
        g,
        output.reshape(batch_size, seqlen, num_heads, head_size),
        flat_A.reshape(batch_size, seqlen, num_heads, _CHUNK_SIZE),
        flat_h,
        chunk_indices,
        chunk_offsets,
    )


def _cutedsl_backward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    do: torch.Tensor,
    dht: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
    chunk_offsets: torch.Tensor | None = None,
    h: torch.Tensor | None = None,
    cu_seqlens_cpu: torch.Tensor | None = None,
    trust_device_cu_seqlens: bool = False,
):
    reason = _fused_bwd_support_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        do=do,
        dht=dht,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        trust_device_cu_seqlens=trust_device_cu_seqlens,
    )
    if reason is None:
        return _call_fused_gdr_bwd_cute(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            do=do,
            dht=dht,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            h=h,
        )
    if _backend_mode() == "cute":
        raise RuntimeError(f"Internal fused CuTe DSL GDR backward is unavailable: {reason}")
    return _fla_backward(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        A=A,
        scale=scale,
        do=do,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )


class InternalChunkGatedDeltaRuleFunction(torch.autograd.Function):
    """Use the local CuTe DSL forward or the verified fused-backward pipeline."""

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        cu_seqlens: torch.LongTensor | None,
        cu_seqlens_cpu: torch.LongTensor | None,
        recompute_h: bool,
    ):
        mode = _backend_mode()
        trust_device_cu_seqlens = True
        use_fused_bwd = _can_use_fused_bwd_forward(
            q, k, v, g, beta, cu_seqlens, cu_seqlens_cpu, trust_device_cu_seqlens
        )
        save_fused_bwd_state = use_fused_bwd and not recompute_h
        saved_h = None
        packed_metadata = _packed_chunk_metadata(cu_seqlens, cu_seqlens_cpu)
        chunk_offsets = packed_metadata.chunk_offsets if packed_metadata is not None else None
        if mode != "cute" and use_fused_bwd:
            g, output, A, saved_h, chunk_indices = _fla_forward_for_fused_bwd(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                packed_metadata=packed_metadata,
                save_fused_bwd_state=save_fused_bwd_state,
            )
        else:
            g, output, A, saved_h, chunk_indices, chunk_offsets = _cutedsl_forward(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                scale=scale,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_seqlens_cpu,
                packed_metadata=packed_metadata,
                save_fused_bwd_state=save_fused_bwd_state,
            )
        ctx.save_for_backward(
            q, k, v, g, beta, A, saved_h, cu_seqlens, chunk_indices, chunk_offsets
        )
        ctx.scale = scale
        ctx.cu_seqlens_cpu = cu_seqlens_cpu
        ctx.trust_device_cu_seqlens = trust_device_cu_seqlens
        return output.to(q.dtype), None

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor | None):
        q, k, v, g, beta, A, saved_h, cu_seqlens, chunk_indices, chunk_offsets = ctx.saved_tensors
        dq, dk, dv, db, dg = _cutedsl_backward(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            h=saved_h,
            cu_seqlens_cpu=ctx.cu_seqlens_cpu,
            trust_device_cu_seqlens=ctx.trust_device_cu_seqlens,
        )
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, None, None, None


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: object | None = None,
    recompute_h: bool = False,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Dispatch supported SM100 inputs to CuTe DSL and otherwise use FLA."""
    if use_beta_sigmoid_in_kernel:
        raise ValueError("The internal GDR backend does not support in-kernel beta sigmoid.")
    if allow_neg_eigval:
        raise ValueError("The internal GDR backend does not support negative eigenvalues.")
    transpose_state_layout = kwargs.pop("transpose_state_layout", None)
    if transpose_state_layout is not None:
        if state_v_first and not bool(transpose_state_layout):
            raise ValueError("state_v_first conflicts with transpose_state_layout=False.")
        state_v_first = bool(transpose_state_layout)

    mode = _backend_mode()
    if mode == "fla":
        return _call_fla_compat(
            fla_chunk_gated_delta_rule,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            transpose_state_layout=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )
    reason = _cutedsl_support_reason(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        cp_context=cp_context,
        kwargs=kwargs,
        trust_device_cu_seqlens=True,
    )
    if reason is not None:
        if mode == "cute":
            raise RuntimeError(f"Internal CuTe DSL GDR path is unavailable: {reason}")
        return _call_fla_compat(
            fla_chunk_gated_delta_rule,
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            transpose_state_layout=state_v_first,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            cp_context=cp_context,
            **kwargs,
        )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    return InternalChunkGatedDeltaRuleFunction.apply(
        q, k, v, g, beta, scale, cu_seqlens, cu_seqlens_cpu, recompute_h
    )
