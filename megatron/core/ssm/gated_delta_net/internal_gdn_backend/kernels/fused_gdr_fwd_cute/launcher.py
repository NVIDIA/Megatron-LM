"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

# This local mcore_gdn_opt SM100 launcher is referred from FlashInfer commit
# e8d31317bedb4efd52559a2234f4cb9e83428cb9; keep the license above
# when updating.

CuTeDSL Fused Chunk GDN Forward SM100 Adapter
=============================================

Bridges FlashInfer's PyTorch-based ``chunk_gated_delta_rule()`` API to the
CuTe DSL chunked GDN kernel for SM100 (Blackwell).

Follows the same compile-once-cache-and-replay pattern used by the decode
kernels in ``gdn_decode_pretranspose.py``.

State layout: ``[N, H, V, K]``.
"""

import functools
from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import from_dlpack

from .kernel import GatedDeltaNetChunkedKernel


@functools.cache
def get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
@functools.cache
def _get_compiled_cache(
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_A: bool,
    store_g: bool,
    store_h: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16")


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    else:
        raise ValueError(f"Unsupported state dtype {torch_dtype}, expected float32 or bfloat16")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def cutedsl_fused_chunk_gdn_fwd_sm100(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    scale: float,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_A: Optional[torch.Tensor] = None,
    output_g: Optional[torch.Tensor] = None,
    output_h: Optional[torch.Tensor] = None,
    gate_is_log_decay: bool = False,
) -> None:
    """Execute the Blackwell chunked GDN forward kernel derived from FlashInfer prefill.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, raw Megatron log-decay forget gate
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        cu_checkpoints: ``(num_seqs + 1,)`` int32 per-sequence chunk prefix
            (``sum(ceil(len_i / 64))``); required with ``output_h``
        output_A: optional ``(total_tokens, HO, 64)`` float16/bfloat16 side output
        output_g: optional ``(total_tokens, HO)`` float32 transformed-g side output
        output_h: optional ``(total_chunks, HO, DK, DK)`` float16/bfloat16 FLA-layout
            per-chunk input-state side output (one slot per 64-token chunk)
        gate_is_log_decay: must be True; the kernel only implements the raw
            Megatron log-decay gate, which it prefix-sums with the base-2
            conversion in-kernel

    Sequence lengths do not need to be multiples of 64: the kernel bounds every
    per-sequence TMA descriptor at the logical sequence end, neutralizes the
    gate/beta rows of a partial final chunk, and predicates the side-output
    stores of that chunk. Zero-length sequences (trailing fixed-capacity
    ``cu_seqlens`` entries) are skipped.
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    if not gate_is_log_decay:
        raise NotImplementedError(
            "the fused GDR forward kernel only implements raw log-decay gates; "
            "pass gate_is_log_decay=True"
        )
    store_A = output_A is not None
    store_g = output_g is not None
    store_h = output_h is not None
    if store_h and cu_checkpoints is None:
        raise ValueError("cu_checkpoints must be provided when output_h is requested")
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_h is not None:
        state_torch_dtype = output_h.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    _cu_checkpoints = cu_checkpoints if store_h else None

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_A,
        store_g,
        store_h,
    )

    if "compiled" not in cache:
        # --- First call: compile the kernel ---
        num_sm = get_num_sm(q.device)
        max_active_clusters = num_sm

        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(128, 64, 128),
            mma_tiler_qkv=(128, 64, 64),
            mma_tiler_kv=(128, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_A=store_A,
            store_g=store_g,
            store_h=store_h,
        )

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        gate_out_cute = None
        if store_g:
            gate_out_cute = from_dlpack(output_g, assumed_align=16)
            gate_out_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1), divisibility=1)
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        A_cute = None
        if store_A:
            A_cute = from_dlpack(output_A, assumed_align=16)
            A_cute.mark_compact_shape_dynamic(mode=0, stride_order=(0, 1, 2), divisibility=1)
        h_cute = None
        if store_h:
            h_cute = from_dlpack(output_h, assumed_align=16)
            h_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(_initial_state, assumed_align=16)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        cu_checkpoints_cute = None
        if store_h:
            cu_checkpoints_cute = from_dlpack(
                _cu_checkpoints, assumed_align=4
            ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(num_sm)
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            gate_cute,
            beta_cute,
            gate_out_cute,
            o_cute,
            A_cute,
            h_cute,
            cu_seqlens_cute,
            s_in_cute,
            cu_checkpoints_cute,
            scale,
            workspace_cute,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(num_sm)
    ws_key = f"workspace_{q.device.index}"
    if ws_key not in cache or cache[ws_key].size(0) < workspace_size:
        cache[ws_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[ws_key]

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output_g if store_g else None,
        output,
        output_A if store_A else None,
        output_h if store_h else None,
        cu_seqlens,
        _initial_state,
        _cu_checkpoints,
        scale,
        workspace,
        stream,
    )
