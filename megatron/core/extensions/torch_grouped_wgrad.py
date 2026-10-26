# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Opt-in BF16 grouped linear with TE forward/dgrad and PyTorch weight gradients.

The weights remain owned by TEGroupedLinear, preserving parameter names and
checkpoint/parallel metadata. This eager-only path does not patch TE globals.
"""

from itertools import accumulate
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def _grouped_mm() -> Optional[Callable[..., torch.Tensor]]:
    """Resolve the public operator or its PyTorch 2.9 compatibility name."""
    return getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)


def torch_grouped_wgrad_eligible(
    module: Any, x: torch.Tensor, m_splits: Union[torch.Tensor, Sequence[int]]
) -> bool:
    """Whether this invocation can bypass TE's legacy BF16 autograd wrapper.

    Unsupported invocations stay on the original TE path. In particular,
    device-resident routing metadata must not be copied back to the CPU here.
    """
    config = module.config
    if not getattr(config, "moe_use_torch_grouped_wgrad", False):
        return False
    if (
        _grouped_mm() is None
        or not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.ndim < 2
        or x.shape[-1] == 0
        or x.numel() // x.shape[-1] > torch.iinfo(torch.int32).max
        or torch.cuda.get_device_capability(x.device) != (9, 0)
        or not 0 < module.num_gemms < 1024
    ):
        return False
    if isinstance(m_splits, torch.Tensor):
        if (
            m_splits.device.type != "cpu"
            or m_splits.ndim != 1
            or m_splits.dtype not in (torch.int32, torch.int64)
        ):
            return False
    elif not isinstance(m_splits, (list, tuple)):
        return False
    if len(m_splits) != module.num_gemms:
        return False
    if (
        module.use_bias
        or module.delay_wgrad_compute
        or module.fuse_wgrad_accumulation
        or getattr(module, "use_grouped_tensor", False)
        or getattr(module, "single_grouped_weight", False)
        or getattr(config, "fp8", None)
        or getattr(config, "fp4", None)
        or getattr(config, "quant_recipe", None)
        or getattr(config, "cpu_offloading", False)
        or getattr(config, "fine_grained_activation_offloading", False)
        or getattr(config, "use_custom_fsdp", False)
        or getattr(config, "use_torch_fsdp2", False)
        or getattr(config, "expert_tensor_parallel_size", 1) != 1
        or getattr(config, "expert_gtp_weight_remat_size", 1) not in (None, 1)
        or getattr(config, "cuda_graph_impl", "none") != "none"
        or torch.cuda.is_current_stream_capturing()
    ):
        return False
    if torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") != torch.bfloat16:
        return False
    if hasattr(module, "is_debug_iter") and module.is_debug_iter():
        return False
    from transformer_engine.pytorch import cpp_extensions, cpu_offload

    if getattr(cpu_offload, "is_cpu_offload_enabled", lambda: False)():
        return False
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager

    if not hasattr(cpp_extensions, "general_grouped_gemm"):
        return False

    if FP8GlobalStateManager.is_fp8_enabled() or FP8GlobalStateManager.is_fp8_calibration():
        return False
    weights: list[Any] = [getattr(module, f"weight{i}", None) for i in range(module.num_gemms)]
    return all(
        type(weight) is torch.nn.Parameter
        and weight.device == x.device
        and weight.dtype == torch.bfloat16
        and weight.ndim == 2
        and weight.is_contiguous()
        and weight.shape == weights[0].shape
        and weight.shape[1] == x.shape[-1]
        and weight.shape[0] > 0
        and weight.shape[0] % 8 == 0
        and weight.shape[1] % 8 == 0
        for weight in weights
    )


class _TorchGroupedLinear(torch.autograd.Function):
    """Use existing packed activation storage and a single grouped dW call."""

    @staticmethod
    def forward(
        ctx: Any, x: torch.Tensor, counts: Tuple[int, ...], *weights: torch.Tensor
    ) -> torch.Tensor:
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

        packed_x = x.reshape(-1, x.shape[-1])
        if len(counts) != len(weights) or min(counts) < 0 or sum(counts) != packed_x.shape[0]:
            raise ValueError("Expert token counts must partition the grouped-linear input.")
        out_features = weights[0].shape[0]
        output = x.new_empty((packed_x.shape[0], out_features))
        if packed_x.shape[0]:
            general_grouped_gemm(
                list(weights),
                list(packed_x.split(counts)),
                [output],
                [None] * len(weights),
                x.dtype,
                m_splits=list(counts),
                single_output=True,
            )
        # Metadata is available on the host already. A pinned, nonblocking copy
        # avoids a new CUDA stream synchronization in the weight-gradient path.
        offsets = None
        if any(ctx.needs_input_grad[2:]) and packed_x.shape[0]:
            host_offsets = torch.tensor(
                list(accumulate(counts)), dtype=torch.int32, pin_memory=True
            )
            offsets = host_offsets.to(x.device, non_blocking=True)
        ctx.counts = counts
        ctx.input_shape = x.shape
        ctx.save_for_backward(packed_x, offsets, *weights)
        return output.reshape(*x.shape[:-1], out_features)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        from transformer_engine.pytorch.cpp_extensions import general_grouped_gemm

        packed_x, offsets, *weights = ctx.saved_tensors
        counts = ctx.counts
        grad_y = grad_output.reshape(-1, weights[0].shape[0]).contiguous()
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.empty_like(packed_x)
            if packed_x.shape[0]:
                general_grouped_gemm(
                    weights,
                    list(grad_y.split(counts)),
                    [grad_input],
                    [None] * len(weights),
                    packed_x.dtype,
                    layout="NN",
                    m_splits=list(counts),
                    grad=True,
                    single_output=True,
                )
            grad_input = grad_input.reshape(ctx.input_shape)
        grad_weights: list[Optional[torch.Tensor]] = [None] * len(weights)
        if any(ctx.needs_input_grad[2:]):
            if packed_x.shape[0]:
                grouped_mm = _grouped_mm()
                if grouped_mm is None:
                    raise RuntimeError("PyTorch grouped matmul is unavailable during backward.")
                packed_dw = grouped_mm(grad_y.t(), packed_x, offs=offsets)
            else:
                packed_dw = packed_x.new_zeros((len(weights), *weights[0].shape))
            grad_weights = [
                grad if needed else None
                for grad, needed in zip(packed_dw.unbind(0), ctx.needs_input_grad[2:])
            ]
        return grad_input, None, *grad_weights


def torch_grouped_linear(
    x: torch.Tensor, m_splits: Union[torch.Tensor, Sequence[int]], weights: Sequence[torch.Tensor]
) -> torch.Tensor:
    """Run an eligible grouped linear without changing TE-owned parameter storage."""
    counts = tuple(m_splits.tolist()) if isinstance(m_splits, torch.Tensor) else tuple(m_splits)
    return _TorchGroupedLinear.apply(x, counts, *weights)
