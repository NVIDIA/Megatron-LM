"""Host-side compilation, caching, and launch support for fused GDR backward."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .kernel import _BT, FusedGdrBwdKernel


@dataclass(frozen=True)
class _CompileCacheKey:
    """All properties that can change one compiled device specialization."""

    input_dtypes: tuple[str, ...]
    output_dtypes: tuple[str, ...]
    heads: int
    grouped_heads: int
    num_sequences: int
    use_dht: bool
    state_v_first: bool
    device_index: int
    capability: tuple[int, int]
    uniform_sequence_length: int = 0
    enable_varlen_tail: bool = False
    enable_iket: bool = False


@dataclass(frozen=True)
class _CompiledKernelArtifacts:
    """Published artifacts that never own compile-call tensor descriptors."""

    compiled: Any
    kernel: FusedGdrBwdKernel


@dataclass(frozen=True, slots=True)
class _PreparedFusedGdrBwdLaunch:
    """Immutable strong owner for one fixed-stream compiled replay."""

    artifacts: _CompiledKernelArtifacts
    launch_args: tuple[Any, ...]
    torch_stream: Any
    tensor_refs: tuple[Any, ...]

    def __call__(self) -> None:
        self.artifacts.compiled(*self.launch_args)


@dataclass
class _CompileCacheEntry:
    """One independently synchronized specialization slot."""

    lock: Any = field(default_factory=threading.Lock)
    artifacts: _CompiledKernelArtifacts | None = None


_COMPILE_CACHE_LOCK = threading.Lock()
_COMPILE_CACHE: dict[_CompileCacheKey, _CompileCacheEntry] = {}


def _clear_compile_cache_for_test() -> None:
    """Drop published and in-flight entries between isolated tests."""

    with _COMPILE_CACHE_LOCK:
        _COMPILE_CACHE.clear()


def _get_or_compile(
    key: _CompileCacheKey, build: Callable[[], _CompiledKernelArtifacts]
) -> _CompiledKernelArtifacts:
    """Return one atomically published build per specialization key."""

    while True:
        with _COMPILE_CACHE_LOCK:
            entry = _COMPILE_CACHE.get(key)
            if entry is None:
                entry = _CompileCacheEntry()
                _COMPILE_CACHE[key] = entry

        artifacts = entry.artifacts
        if artifacts is not None:
            return artifacts

        with entry.lock:
            artifacts = entry.artifacts
            if artifacts is not None:
                return artifacts

            with _COMPILE_CACHE_LOCK:
                if _COMPILE_CACHE.get(key) is not entry:
                    continue

            try:
                artifacts = build()
            except BaseException:
                with _COMPILE_CACHE_LOCK:
                    if _COMPILE_CACHE.get(key) is entry:
                        _COMPILE_CACHE.pop(key)
                raise

            with _COMPILE_CACHE_LOCK:
                if _COMPILE_CACHE.get(key) is not entry:
                    continue
                entry.artifacts = artifacts
            return artifacts


def _cutlass_dtype(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return cutlass.BFloat16
    if dtype == torch.float32:
        return cutlass.Float32
    if dtype == torch.int32:
        return cutlass.Int32
    raise TypeError(f"unsupported fused GDR CuTeDSL dtype: {dtype}")


def _mark_dynamic_mode(tensor: torch.Tensor, *, mode: int = 1, divisibility: int):
    descriptor = from_dlpack(tensor, assumed_align=16)
    descriptor.mark_compact_shape_dynamic(
        mode=mode, stride_order=tuple(range(tensor.ndim)), divisibility=divisibility
    )
    return descriptor


def _mark_dynamic_mode_many(tensors: tuple[torch.Tensor, ...], divisibility: int):
    return tuple(_mark_dynamic_mode(tensor, divisibility=divisibility) for tensor in tensors)


def _resolve_device_index(device: torch.device) -> int:
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return int(index)


def _build_compile_descriptors(
    input_tensors: tuple[torch.Tensor, ...],
    output_tensors: tuple[torch.Tensor, ...],
    *,
    enable_varlen_tail: bool,
):
    q, k, v, a, g, beta, do, dht, h, cu_seqlens, chunk_offsets = input_tensors
    dq, dk, dv, dg, db, dh0 = output_tensors
    token_divisibility = 1 if enable_varlen_tail else _BT
    token_inputs = _mark_dynamic_mode_many(
        (q, k, v, a, g, beta, do), token_divisibility
    )
    token_outputs = _mark_dynamic_mode_many(
        (dq, dk, dv, dg, db), token_divisibility
    )
    return (
        *token_inputs,
        from_dlpack(dht, assumed_align=16),
        _mark_dynamic_mode(h, divisibility=1),
        from_dlpack(cu_seqlens, assumed_align=4),
        from_dlpack(chunk_offsets, assumed_align=4),
        *token_outputs,
        from_dlpack(dh0, assumed_align=16),
    )


def _validate_launch_devices(tensors, torch_stream) -> int:
    device_index = _resolve_device_index(tensors[0].device)
    for tensor in tensors[1:]:
        tensor_device_index = _resolve_device_index(tensor.device)
        if tensor_device_index != device_index:
            raise ValueError(
                f"tensor device cuda:{tensor_device_index} does not match "
                f"q device cuda:{device_index}"
            )
    stream_device_index = _resolve_device_index(torch_stream.device)
    if stream_device_index != device_index:
        raise ValueError(
            f"torch_stream device cuda:{stream_device_index} does not match "
            f"tensor device cuda:{device_index}"
        )
    return device_index


def _make_compile_key(
    input_tensors,
    output_tensors,
    *,
    heads: int,
    grouped_heads: int,
    num_sequences: int,
    uniform_sequence_length: int,
    device_index: int,
    capability: tuple[int, int],
    enable_varlen_tail: bool,
    enable_iket: bool,
) -> _CompileCacheKey:
    return _CompileCacheKey(
        input_dtypes=tuple(str(tensor.dtype) for tensor in input_tensors),
        output_dtypes=tuple(str(tensor.dtype) for tensor in output_tensors),
        heads=heads,
        grouped_heads=grouped_heads,
        num_sequences=num_sequences,
        use_dht=True,
        state_v_first=False,
        uniform_sequence_length=uniform_sequence_length,
        device_index=device_index,
        capability=capability,
        enable_varlen_tail=enable_varlen_tail,
        enable_iket=enable_iket,
    )


def _compile_artifacts(
    *,
    input_tensors,
    output_tensors,
    scale: float,
    driver_stream,
    heads: int,
    grouped_heads: int,
    num_sequences: int,
    uniform_sequence_length: int,
    enable_varlen_tail: bool,
    enable_iket: bool,
) -> _CompiledKernelArtifacts:
    q, _k, _v, _a, _g, _beta, _do, dht, *_ = input_tensors
    kernel = FusedGdrBwdKernel(
        io_dtype=_cutlass_dtype(q.dtype),
        acc_dtype=_cutlass_dtype(dht.dtype),
        heads=heads,
        grouped_heads=grouped_heads,
        num_sequences=num_sequences,
        use_dht=True,
        state_v_first=False,
        uniform_sequence_length=uniform_sequence_length,
        enable_varlen_tail=enable_varlen_tail,
        enable_iket=enable_iket,
    )
    compiled = cute.compile(
        kernel,
        *_build_compile_descriptors(
            input_tensors,
            output_tensors,
            enable_varlen_tail=enable_varlen_tail,
        ),
        scale,
        driver_stream,
        options="--enable-tvm-ffi --opt-level 2",
    )
    return _CompiledKernelArtifacts(compiled=compiled, kernel=kernel)


def prepare_fused_gdr_bwd_launch(
    *, q, k, v, a, g, beta, do, dht, h, scale, metadata, outputs, torch_stream
) -> _PreparedFusedGdrBwdLaunch:
    """Bind one compiled CuTe DSL replay to tensors and a fixed torch stream."""

    dq, dk, dv, dg, db, dh0 = outputs
    cu_seqlens = metadata.cu_seqlens
    chunk_offsets = metadata.chunk_offsets
    heads = v.shape[2]
    grouped_heads = q.shape[2]
    if grouped_heads != heads:
        raise ValueError("grouped_heads must equal heads; GQA head mapping is unsupported")

    uniform_sequence_length = getattr(metadata, "uniform_sequence_length", 0)
    enable_varlen_tail = getattr(metadata, "has_partial_chunks", False)
    num_sequences = metadata.num_sequences
    enable_iket = os.environ.get("FUSED_GDR_BWD_ENABLE_IKET") == "1"
    input_tensors = (q, k, v, a, g, beta, do, dht, h, cu_seqlens, chunk_offsets)
    output_tensors = (dq, dk, dv, dg, db, dh0)
    tensor_refs = input_tensors + output_tensors
    device_index = _validate_launch_devices(tensor_refs, torch_stream)
    capability = tuple(torch.cuda.get_device_capability(q.device))
    driver_stream = cuda.CUstream(torch_stream.cuda_stream)
    key = _make_compile_key(
        input_tensors,
        output_tensors,
        heads=heads,
        grouped_heads=grouped_heads,
        num_sequences=num_sequences,
        uniform_sequence_length=uniform_sequence_length,
        device_index=device_index,
        capability=capability,
        enable_varlen_tail=enable_varlen_tail,
        enable_iket=enable_iket,
    )

    def build() -> _CompiledKernelArtifacts:
        return _compile_artifacts(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            scale=scale,
            driver_stream=driver_stream,
            heads=heads,
            grouped_heads=grouped_heads,
            num_sequences=num_sequences,
            uniform_sequence_length=uniform_sequence_length,
            enable_varlen_tail=enable_varlen_tail,
            enable_iket=enable_iket,
        )

    try:
        artifacts = _get_or_compile(key, build)
    finally:
        del build

    return _PreparedFusedGdrBwdLaunch(
        artifacts=artifacts,
        launch_args=tensor_refs + (scale, driver_stream),
        torch_stream=torch_stream,
        tensor_refs=tensor_refs,
    )


def launch_fused_gdr_bwd(*, q, k, v, a, g, beta, do, dht, h, scale, metadata, outputs) -> None:
    """Prepare and immediately replay on the caller's current torch stream."""

    prepared = prepare_fused_gdr_bwd_launch(
        q=q,
        k=k,
        v=v,
        a=a,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
        h=h,
        scale=scale,
        metadata=metadata,
        outputs=outputs,
        torch_stream=torch.cuda.current_stream(device=q.device),
    )
    prepared()


__all__ = ["launch_fused_gdr_bwd", "prepare_fused_gdr_bwd_launch"]
