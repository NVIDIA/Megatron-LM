# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Autograd bridge to cuDNN Frontend's saved-state Engram gate on SM100."""

from __future__ import annotations

import math
from collections import OrderedDict
from typing import Any

import torch
from torch.autograd.function import once_differentiable


class _CudnnEngramFunction(torch.autograd.Function):
    """Each forward owns its saved state; backward owns its scratch and outputs."""

    @staticmethod
    def forward(ctx, x, kv, weight, mask, owner):
        """Run the gate while retaining private scalar state for this invocation."""
        output = torch.empty_like(x)
        saved = torch.empty((x.shape[0], 4, 4), device=x.device, dtype=torch.float32)
        forward, backward = owner.plans(x, kv, weight, mask, saved)
        stream = torch.cuda.current_stream(x.device).cuda_stream
        forward.execute(x, kv, weight, mask, output, saved, current_stream=stream)
        ctx.save_for_backward(x, kv, weight, saved)
        ctx.backward_plan = backward
        owner.forward_calls += 1
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        """Return all gate gradients with scratch owned by this backward call."""
        x, kv, weight, saved = ctx.saved_tensors
        grad_x, grad_kv, grad_weight = (
            torch.empty_like(x),
            torch.empty_like(kv),
            torch.empty_like(weight),
        )
        plan = ctx.backward_plan
        workspace = torch.empty(plan.scratch_workspace_bytes(), device=x.device, dtype=torch.uint8)
        plan.execute(
            x,
            kv,
            weight,
            saved,
            grad_output.contiguous(),
            grad_x,
            grad_kv,
            grad_weight,
            workspace,
            current_stream=torch.cuda.current_stream(x.device).cuda_stream,
        )
        return grad_x, grad_kv, grad_weight, None, None


class CudnnEngramGate:
    """Prepared metadata plans for BF16 four-stream gates with H=5120.

    Plans contain no per-forward activation state. Every invocation retains its
    own input, packed KV, normalization-weight product and saved gate scalars.
    Torch autograd applies the product rule to the original q/k parameters.
    Layout conversions are explicit here and belong to the measured module cost.
    CUDA Graph capture and higher-order differentiation are not supported.
    """

    def __init__(self, eps: float):
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("cuDNN Engram requires finite positive normalization epsilon")
        self.eps = eps
        self._plans: OrderedDict[tuple[int, int], tuple[Any, Any]] = OrderedDict()
        self.forward_calls = 0
        self.plan_builds = 0

    def plans(self, x, kv, weight, mask, saved):
        """Cache shape/device plans, while autograd retains plans evicted from this cache."""
        key = (x.device.index, x.shape[0])
        if key not in self._plans:
            try:
                import cudnn
            except ImportError as error:
                raise RuntimeError(
                    "cuDNN Engram requires cuDNN Frontend saved-state Engram APIs; "
                    "see docs/models/engram_cudnn.md for the validated dependency commit"
                ) from error
            if not all(
                hasattr(cudnn, name)
                for name in ("EngramGateSavedForward", "EngramGateSavedBackward")
            ):
                raise RuntimeError(
                    "cuDNN Engram requires cuDNN Frontend saved-state Engram APIs; "
                    "the default Megatron dependency pin does not provide them. "
                    "See docs/models/engram_cudnn.md for the validated dependency commit"
                )

            forward = cudnn.EngramGateSavedForward(
                x, kv, weight, mask, eps=self.eps, backend="frost"
            )
            backward = cudnn.EngramGateSavedBackward(x, kv, weight, saved, x, backend="frost")
            forward.compile()
            backward.compile()
            self._plans[key] = (forward, backward)
            self.plan_builds += 1
            if len(self._plans) > 8:
                self._plans.popitem(last=False)
        self._plans.move_to_end(key)
        return self._plans[key]

    @torch.compiler.disable
    def __call__(self, hidden, kv, q_weight, k_weight, token_mask=None):
        """Run a sequence-major gate and preserve gradients of the original parameters."""
        if hidden.ndim != 3 or hidden.shape[-1] != 4 * 5120:
            raise ValueError("cuDNN Engram requires [sequence,batch,4*5120] hidden states")
        if hidden.device.type != "cuda" or hidden.dtype != torch.bfloat16:
            raise ValueError("cuDNN Engram requires CUDA BF16 hidden states")
        if torch.cuda.get_device_capability(hidden.device) != (10, 0):
            raise ValueError("cuDNN Engram currently requires SM100")
        with torch.cuda.device(hidden.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cuDNN Engram integration does not support CUDA Graph capture")
        length, batch, _ = hidden.shape
        tokens = length * batch
        if tokens <= 0 or tokens > 8192 or tokens % 64:
            raise ValueError("cuDNN Engram requires 64..8192 tokens in multiples of 64")
        if kv.shape != (length, batch, 5 * 5120) or kv.dtype != torch.bfloat16:
            raise ValueError("cuDNN Engram requires BF16 packed keys/value matching hidden states")
        for parameter in (q_weight, k_weight):
            if parameter.shape != (4, 5120) or parameter.dtype not in (
                torch.bfloat16,
                torch.float32,
            ):
                raise ValueError("cuDNN Engram requires BF16/FP32 [4,5120] q/k weights")
        if any(t.device != hidden.device for t in (kv, q_weight, k_weight)):
            raise ValueError("cuDNN Engram operands must reside on the same CUDA device")
        if token_mask is None:
            mask = torch.ones(tokens, device=hidden.device, dtype=torch.bool)
        else:
            if (
                token_mask.shape != (batch, length)
                or token_mask.dtype != torch.bool
                or token_mask.device != hidden.device
            ):
                raise ValueError("cuDNN Engram token mask must be CUDA bool [batch,sequence]")
            mask = token_mask.transpose(0, 1).contiguous().view(tokens)
        x = hidden.contiguous().view(tokens, 4, 5120)
        packed = kv.contiguous().view(tokens, 5 * 5120)
        weight = (q_weight.float() * k_weight.float()).contiguous()
        return _CudnnEngramFunction.apply(x, packed, weight, mask, self).view_as(hidden)
