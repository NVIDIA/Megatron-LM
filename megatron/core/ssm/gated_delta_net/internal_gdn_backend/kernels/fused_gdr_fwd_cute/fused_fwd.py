# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Megatron Core SM100 GDN prefill wrapper.

This local wrapper is referred from FlashInfer commit e8d31317bedb4efd52559a2234f4cb9e83428cb9, but is
trimmed to the Megatron Core GDN training path so Megatron-LM does not need
to import FlashInfer.
"""

# This wrapper is referred from FlashInfer commit e8d31317bedb4efd52559a2234f4cb9e83428cb9
# and trimmed for Megatron Core.
from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch

_BT = 64


def _cuda_major() -> int:
    return int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0


def _check_cuda_sm100(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise NotImplementedError("Megatron GDN prefill CuTe kernel requires CUDA tensors")
    if _cuda_major() < 13:
        raise NotImplementedError("Megatron GDN prefill CuTe kernel requires CUDA 13+")
    if torch.cuda.get_device_capability(tensor.device)[0] != 10:
        raise NotImplementedError("Megatron GDN prefill CuTe kernel is SM100/B200 only")


def _require_contiguous(name: str, tensor: torch.Tensor) -> torch.Tensor:
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _validate_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int) -> torch.Tensor:
    if cu_seqlens.ndim != 1:
        raise ValueError("cu_seqlens must be a 1D tensor")
    if int(cu_seqlens[-1].item()) != total_tokens:
        raise ValueError(
            "cu_seqlens final value must match total tokens: "
            f"{int(cu_seqlens[-1].item())} != {total_tokens}"
        )
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if not bool((lengths % _BT == 0).all().item()):
        raise NotImplementedError(
            "Megatron GDN prefill CuTe kernel requires every sequence length to be a 64 multiple"
        )
    return cu_seqlens.to(device=cu_seqlens.device, dtype=torch.int32).contiguous()


def _prepare_cu_seqlens_for_launch(
    cu_seqlens: torch.Tensor, total_tokens: int, *, device: torch.device, assume_valid: bool = False
) -> torch.Tensor:
    if assume_valid:
        if cu_seqlens.ndim != 1:
            raise ValueError("cu_seqlens must be a 1D tensor")
        if cu_seqlens.device != device:
            raise ValueError("trusted cu_seqlens must already be on the launch device")
        if cu_seqlens.dtype != torch.int32:
            raise ValueError("trusted cu_seqlens must already be int32")
        return _require_contiguous("cu_seqlens", cu_seqlens)
    return _validate_cu_seqlens(cu_seqlens.to(device=device), total_tokens)


def chunk_gated_delta_rule_prefill_cute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_A: Optional[torch.Tensor] = None,
    output_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    output_h: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    assume_valid_cu_seqlens: bool = False,
    gate_is_log_cumsum: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Run the local SM100 GDN prefill kernel.

    The accepted signature mirrors the FlashInfer API used by Megatron, but the
    implementation is local to Megatron Core and only supports the SM100 path.
    """
    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("QK L2 norm inside the GDN prefill kernel is not supported")
    if initial_state is not None:
        raise NotImplementedError("initial_state is out of scope for the local mcore prefill path")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must be 3D THD tensors")
    _check_cuda_sm100(q)
    q = _require_contiguous("q", q)
    k = _require_contiguous("k", k)
    v = _require_contiguous("v", v)
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise NotImplementedError(f"Megatron GDN prefill supports bf16/fp16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, and v must have the same dtype")

    total_tokens, num_q_heads, head_size = q.shape
    if k.shape[0] != total_tokens or v.shape[0] != total_tokens:
        raise ValueError("q, k, and v must have the same token dimension")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head size")
    if head_size != 128:
        raise NotImplementedError(f"Megatron GDN prefill requires head_size=128, got {head_size}")

    num_v_heads = v.shape[1]
    num_o_heads = max(num_q_heads, num_v_heads)
    if cu_seqlens is None:
        raise ValueError("cu_seqlens is required for the local mcore GDN prefill path")
    cu_i32 = _prepare_cu_seqlens_for_launch(
        cu_seqlens, total_tokens, device=q.device, assume_valid=assume_valid_cu_seqlens
    )

    if output is None:
        output = torch.empty((total_tokens, num_o_heads, head_size), dtype=q.dtype, device=q.device)
    else:
        if tuple(output.shape) != (total_tokens, num_o_heads, head_size):
            raise ValueError("output shape mismatch for mcore GDN prefill")
        if output.dtype != q.dtype:
            raise ValueError("output dtype must match q dtype")
        output = _require_contiguous("output", output)

    if output_A is not None:
        expected_A_shape = (total_tokens, num_o_heads, _BT)
        if tuple(output_A.shape) != expected_A_shape:
            raise ValueError(
                f"output_A shape mismatch: expected {expected_A_shape}, got {tuple(output_A.shape)}"
            )
        if output_A.dtype != q.dtype:
            raise ValueError("output_A dtype must match q dtype")
        output_A = _require_contiguous("output_A", output_A)

    if output_h is not None:
        expected_h_shape = (total_tokens // _BT, num_o_heads, head_size, head_size)
        if tuple(output_h.shape) != expected_h_shape:
            raise ValueError(
                f"output_h shape mismatch: expected {expected_h_shape}, got {tuple(output_h.shape)}"
            )
        if output_h.dtype != q.dtype:
            raise ValueError("output_h dtype must match q dtype")
        output_h = _require_contiguous("output_h", output_h)

    if checkpoint_every_n_tokens < 0:
        raise ValueError("checkpoint_every_n_tokens must be non-negative")
    if output_h is not None and checkpoint_every_n_tokens != _BT:
        raise ValueError(
            "output_h requires checkpoint_every_n_tokens=64 because it stores every chunk state"
        )
    if checkpoint_every_n_tokens > 0:
        if checkpoint_every_n_tokens % _BT != 0:
            raise ValueError("checkpoint_every_n_tokens must be a multiple of 64")
        if checkpoint_cu_starts is None:
            raise ValueError("checkpoint_cu_starts is required when checkpointing is enabled")
        if state_checkpoints is None and output_h is None:
            raise ValueError(
                "state_checkpoints or output_h is required when checkpointing is enabled"
            )
    elif state_checkpoints is not None or checkpoint_cu_starts is not None or output_h is not None:
        raise ValueError("checkpoint tensors must be None when checkpoint_every_n_tokens is 0")

    if output_h is not None and output_h.dtype != torch.bfloat16:
        output_h.index_fill_(0, (cu_i32[:-1] // _BT).to(torch.long), 0)

    if output_final_state and output_state is None:
        output_state = torch.empty(
            (cu_i32.numel() - 1, num_o_heads, head_size, head_size),
            dtype=torch.float32,
            device=q.device,
        )
    if not output_final_state:
        output_state = None

    gate = (
        g
        if g is not None
        else torch.ones((total_tokens, num_o_heads), dtype=torch.float32, device=q.device)
    )
    update = (
        beta
        if beta is not None
        else torch.ones((total_tokens, num_o_heads), dtype=torch.float32, device=q.device)
    )
    gate = gate.float().contiguous()
    update = update.float().contiguous()
    if tuple(gate.shape) != (total_tokens, num_o_heads):
        raise ValueError("g shape mismatch for mcore GDN prefill")
    if tuple(update.shape) != (total_tokens, num_o_heads):
        raise ValueError("beta shape mismatch for mcore GDN prefill")

    cu_checkpoints = (
        checkpoint_cu_starts.to(device=q.device, dtype=torch.int32).contiguous()
        if checkpoint_cu_starts is not None
        else None
    )
    kernel_scale = scale if scale is not None else 1.0 / math.sqrt(head_size)

    from .launcher import cutedsl_fused_chunk_gdn_fwd_sm100

    cutedsl_fused_chunk_gdn_fwd_sm100(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=update,
        output=output,
        cu_seqlens=cu_i32,
        initial_state=None,
        output_state=output_state,
        scale=kernel_scale,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        cu_checkpoints=cu_checkpoints,
        output_checkpoints=state_checkpoints,
        output_A=output_A,
        output_h=output_h,
        gate_is_log_cumsum=gate_is_log_cumsum,
    )
    return (output, output_state) if output_final_state else output
