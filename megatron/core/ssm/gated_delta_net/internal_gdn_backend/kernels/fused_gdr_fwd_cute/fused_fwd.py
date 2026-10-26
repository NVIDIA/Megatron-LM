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
from typing import Optional

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
        raise NotImplementedError(
            "Megatron GDN prefill CuTe kernel requires a Blackwell data-center GPU (SM10x)"
        )


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
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError("cu_seqlens must start at 0")
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    if bool((lengths < 0).any().item()):
        raise ValueError("cu_seqlens must be non-decreasing")
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
    cu_seqlens: Optional[torch.Tensor] = None,
    use_qk_l2norm_in_kernel: bool = False,
    output: Optional[torch.Tensor] = None,
    output_A: Optional[torch.Tensor] = None,
    output_g: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    output_h: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    assume_valid_cu_seqlens: bool = False,
    gate_is_log_decay: bool = False,
) -> torch.Tensor:
    """Run the local SM100 GDN prefill kernel.

    The accepted signature mirrors the FlashInfer API used by Megatron, but the
    implementation is local to Megatron Core and only supports the SM100 path.
    The kernel implements only the raw Megatron log-decay gate, so
    ``gate_is_log_decay`` must be True. ``checkpoint_every_n_tokens`` is validated
    (``output_h`` stores one state per 64-token chunk and therefore requires 64)
    but not forwarded: the kernel's chunk size fixes the checkpoint stride.
    """
    if not gate_is_log_decay:
        raise NotImplementedError(
            "Megatron GDN prefill CuTe kernel only implements raw log-decay gates; "
            "pass gate_is_log_decay=True"
        )
    if use_qk_l2norm_in_kernel:
        raise NotImplementedError("QK L2 norm inside the GDN prefill kernel is not supported")
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

    if initial_state is not None:
        expected_state_shape = (cu_i32.numel() - 1, num_o_heads, head_size, head_size)
        if tuple(initial_state.shape) != expected_state_shape:
            raise ValueError(
                f"initial_state shape mismatch: expected {expected_state_shape}, "
                f"got {tuple(initial_state.shape)}"
            )
        if initial_state.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("initial_state dtype must be torch.float32 or torch.bfloat16")
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same device as q")
        initial_state = _require_contiguous("initial_state", initial_state)

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

    if output_g is not None:
        expected_g_shape = (total_tokens, num_o_heads)
        if tuple(output_g.shape) != expected_g_shape:
            raise ValueError(
                f"output_g shape mismatch: expected {expected_g_shape}, got {tuple(output_g.shape)}"
            )
        if output_g.dtype != torch.float32:
            raise ValueError("output_g dtype must be torch.float32")
        output_g = _require_contiguous("output_g", output_g)

    if output_h is not None:
        # The chunk count is sum(ceil(len_i / 64)) over the packed sequences, which the
        # caller supplies as ``checkpoint_cu_starts[-1]``; only the per-chunk dims are
        # checked here so the hot path does not synchronize with the device.
        if output_h.ndim != 4 or tuple(output_h.shape[1:]) != (num_o_heads, head_size, head_size):
            raise ValueError(
                "output_h shape mismatch: expected "
                f"(num_chunks, {num_o_heads}, {head_size}, {head_size}), got {tuple(output_h.shape)}"
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
        if checkpoint_cu_starts.ndim != 1 or checkpoint_cu_starts.numel() != cu_i32.numel():
            raise ValueError(
                "checkpoint_cu_starts must be the per-sequence chunk prefix with the same "
                f"length as cu_seqlens ({cu_i32.numel()}), got {tuple(checkpoint_cu_starts.shape)}"
            )
        if output_h is None:
            raise ValueError("output_h is required when checkpointing is enabled")
    elif checkpoint_cu_starts is not None or output_h is not None:
        raise ValueError("checkpoint tensors must be None when checkpoint_every_n_tokens is 0")

    if output_h is not None and not assume_valid_cu_seqlens:
        expected_chunks = int(checkpoint_cu_starts[-1].item())
        if output_h.shape[0] != expected_chunks:
            raise ValueError(
                f"output_h must hold {expected_chunks} chunk states, got {output_h.shape[0]}"
            )
    if output_h is not None and output_h.dtype != torch.bfloat16:
        # Non-bf16 states are initialized by the wrapper: zero the first chunk slot of
        # every sequence (the bf16 TMA path does this in-kernel). A zero-length sequence
        # owns no slot and its start offset equals the next sequence's (or the total), so
        # redirect it to slot 0, which is a legitimate first slot that is zeroed anyway.
        starts = checkpoint_cu_starts.to(device=output_h.device, dtype=torch.long)
        owns_chunks = (starts[1:] - starts[:-1]) > 0
        first_chunk_slots = torch.where(owns_chunks, starts[:-1], torch.zeros_like(starts[:-1]))
        output_h.index_fill_(0, first_chunk_slots, 0)

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
        initial_state=initial_state,
        scale=kernel_scale,
        cu_checkpoints=cu_checkpoints,
        output_A=output_A,
        output_g=output_g,
        output_h=output_h,
        gate_is_log_decay=gate_is_log_decay,
    )
    return output
