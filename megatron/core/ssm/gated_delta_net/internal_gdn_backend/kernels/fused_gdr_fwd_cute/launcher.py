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
    store_final_state: bool,
    enable_checkpoints: bool,
    input_A: bool,
    store_A: bool,
    store_v_new: bool,
    store_w: bool,
    store_h: bool,
    w_rhs_precomputed: bool,
    training_side_outputs_only: bool,
    gate_is_log_cumsum: bool,
    enable_varlen_tail: bool,
    enable_timeline: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(
            f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16"
        )


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    else:
        raise ValueError(
            f"Unsupported state dtype {torch_dtype}, expected float32 or bfloat16"
        )


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
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_checkpoints: Optional[torch.Tensor] = None,
    input_A: Optional[torch.Tensor] = None,
    output_A: Optional[torch.Tensor] = None,
    output_v_new: Optional[torch.Tensor] = None,
    output_w: Optional[torch.Tensor] = None,
    output_h: Optional[torch.Tensor] = None,
    w_rhs: Optional[torch.Tensor] = None,
    training_side_outputs_only: bool = False,
    gate_is_log_cumsum: bool = False,
    debug_timing: Optional[torch.Tensor] = None,
    enable_varlen_tail: bool = False,
) -> None:
    """Execute the Blackwell chunked GDN forward kernel derived from FlashInfer prefill.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: store intermediate state every N tokens (0 = disabled)
        cu_checkpoints: ``(num_seqs + 1,)`` int32, cumulative checkpoint counts
        output_checkpoints: ``(total_checkpoints, HO, DK, DK)`` float32/bfloat16, or None
        input_A: optional saved unscaled ``A`` side input, shape ``(total_tokens, HO, 64)``
        output_A: optional ``(total_tokens, HO, 64)`` float16/bfloat16 side output
        output_v_new: optional ``(total_tokens, HO, DK)`` float16/bfloat16 side output
        output_w: optional ``(total_tokens, HO, DK)`` float16/bfloat16 side output
        output_h: optional ``(total_chunks, HO, DK, DK)`` float16/bfloat16 FLA-layout h side output
        w_rhs: optional ``(total_tokens, HO, DK)`` float16/bfloat16 precomputed
            RHS used to compute output_w; if omitted, the kernel computes it
            from k and gate cumprod
        training_side_outputs_only: when True, skip the forward-only Q/O path and materialize only training side outputs
        gate_is_log_cumsum: when True, gate is already chunk-local cumulative log2 ``g``
        debug_timing: optional ``(6, 2, 32)`` int64 tensor for debug globaltimer tags
        enable_varlen_tail: compile native partial-tile handling for packed lengths
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0 and output_checkpoints is not None
    use_input_A = input_A is not None
    store_A = output_A is not None
    store_v_new = output_v_new is not None
    store_w = output_w is not None
    store_h = output_h is not None
    w_rhs_precomputed = w_rhs is not None
    enable_timeline = debug_timing is not None
    if enable_timeline:
        if tuple(debug_timing.shape) != (6, 2, 32):
            raise ValueError(f"debug_timing shape mismatch: expected (6, 2, 32), got {tuple(debug_timing.shape)}")
        if debug_timing.dtype != torch.int64:
            raise ValueError("debug_timing must be torch.int64")
        if not debug_timing.is_cuda or not debug_timing.is_contiguous():
            raise ValueError("debug_timing must be a contiguous CUDA tensor")
    if (enable_checkpoints or store_h) and cu_checkpoints is None:
        raise ValueError("cu_checkpoints must be provided when checkpoints or output_h are requested")
    if store_h and checkpoint_every_n_tokens <= 0:
        raise ValueError("checkpoint_every_n_tokens must be positive when output_h is requested")
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    elif output_h is not None:
        state_torch_dtype = output_h.dtype
    elif output_checkpoints is not None:
        state_torch_dtype = output_checkpoints.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    B = cu_seqlens.size(0) - 1
    _output_state = output_state if store_final_state else None

    cache = _get_compiled_cache(
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_input_A,
        store_A,
        store_v_new,
        store_w,
        store_h,
        w_rhs_precomputed,
        training_side_outputs_only,
        gate_is_log_cumsum,
        enable_varlen_tail,
        enable_timeline,
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
            store_final_state=store_final_state,
            enable_checkpoints=enable_checkpoints,
            input_A=use_input_A,
            store_A=store_A,
            store_v_new=store_v_new,
            store_w=store_w,
            store_h=store_h,
            w_rhs_precomputed=w_rhs_precomputed,
            training_side_outputs_only=training_side_outputs_only,
            gate_is_log_cumsum=gate_is_log_cumsum,
            enable_varlen_tail=enable_varlen_tail,
            is_persistent=True,
            enable_timeline=enable_timeline,
        )

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        w_rhs_cute = None
        if w_rhs_precomputed:
            w_rhs_cute = from_dlpack(w_rhs, assumed_align=16)
            w_rhs_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=1
            )
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        input_A_cute = None
        if use_input_A:
            input_A_cute = from_dlpack(input_A, assumed_align=16)
            input_A_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=1
            )
        A_cute = None
        if store_A:
            A_cute = from_dlpack(output_A, assumed_align=16)
            A_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=1
            )
        v_new_cute = None
        if store_v_new:
            v_new_cute = from_dlpack(output_v_new, assumed_align=16)
            v_new_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=1
            )
        w_cute = None
        if store_w:
            w_cute = from_dlpack(output_w, assumed_align=16)
            w_cute.mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2), divisibility=1
            )
        h_cute = None
        if store_h:
            h_cute = from_dlpack(output_h, assumed_align=16)
            h_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()
        debug_timing_cute = None
        if enable_timeline:
            debug_timing_cute = from_dlpack(debug_timing, assumed_align=8).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(_initial_state, assumed_align=16)
            s_in_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(_output_state, assumed_align=16)
            s_out_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )

        s_checkpoints_cute = None
        cu_checkpoints_cute = None
        if enable_checkpoints:
            s_checkpoints_cute = from_dlpack(output_checkpoints, assumed_align=16)
            s_checkpoints_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
        if enable_checkpoints or store_h:
            cu_checkpoints_cute = from_dlpack(
                cu_checkpoints, assumed_align=4
            ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            w_rhs_cute,
            gate_cute,
            beta_cute,
            o_cute,
            input_A_cute,
            A_cute,
            v_new_cute,
            w_cute,
            h_cute,
            cu_seqlens_cute,
            s_in_cute,
            s_out_cute,
            s_checkpoints_cute,
            cu_checkpoints_cute,
            checkpoint_every_n_tokens,
            scale,
            workspace_cute,
            debug_timing_cute,
            stream,
            options="--enable-tvm-ffi --opt-level 2",
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, B, HQ, HV, True
    )
    ws_key = f"workspace_{q.device.index}"
    if ws_key not in cache or cache[ws_key].size(0) < workspace_size:
        cache[ws_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[ws_key]

    stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)
    compiled(
        q,
        k,
        v,
        w_rhs if w_rhs_precomputed else None,
        gate,
        beta,
        output,
        input_A if use_input_A else None,
        output_A if store_A else None,
        output_v_new if store_v_new else None,
        output_w if store_w else None,
        output_h if store_h else None,
        cu_seqlens,
        _initial_state,
        _output_state,
        output_checkpoints,
        cu_checkpoints,
        checkpoint_every_n_tokens,
        scale,
        workspace,
        debug_timing if enable_timeline else None,
        stream,
    )
