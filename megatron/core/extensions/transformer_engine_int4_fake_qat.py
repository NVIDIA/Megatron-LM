# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import os

import torch

from megatron.core.model_parallel_config import ModelParallelConfig

INT4_FAKE_QAT_FLAG = "OPEN_TRAINING_INT4_FAKE_QAT_FLAG"
INT4_GROUP_SIZE = "OPEN_TRAINING_INT4_GROUP_SIZE"


def ceil_div(x: int, y: int) -> int:
    """Return ``ceil(x / y)`` for integer inputs."""
    return (x + y - 1) // y


class _FakeInt4QuantizationSTE(torch.autograd.Function):
    """Straight-through estimator for per-group signed int4 fake quantization."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group_size: int) -> torch.Tensor:
        """Fake-quantize ``x`` to signed int4 values, then dequantize it."""
        m, n = x.shape
        block_size_m, block_size_n = 1, group_size

        m_padded = ceil_div(m, block_size_m) * block_size_m
        n_padded = ceil_div(n, block_size_n) * block_size_n

        x_padded = torch.zeros((m_padded, n_padded), dtype=x.dtype, device=x.device)
        x_padded[:m, :n] = x

        x_view = x_padded.view(
            m_padded // block_size_m, block_size_m, n_padded // block_size_n, block_size_n
        )

        x_max = x_view.abs().float().amax(dim=(1, 3), keepdim=True)
        q_max = 7
        x_scale = x_max / q_max
        x_scale = x_scale.clamp(min=1e-5)

        x_div = x_view / x_scale
        x_round = torch.round(x_div)
        x_q_clamped = x_round.clamp(-q_max, q_max)
        x_dequant_view = x_q_clamped * x_scale

        x_dequant_full = x_dequant_view.view_as(x_padded)
        x_out = x_dequant_full[:m, :n].contiguous().to(x.dtype)

        return x_out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        """Propagate gradients through the fake-quantization op unchanged."""
        return grad_output, None


def fake_int4_quantization_ste(x: torch.Tensor, group_size: int) -> torch.Tensor:
    """Apply signed int4 fake quantization with a straight-through gradient."""
    x_out = _FakeInt4QuantizationSTE.apply(x, group_size)

    if hasattr(x, 'main_grad'):
        x_out.main_grad = x.main_grad

    return x_out


def _validate_int4_fake_qat_support(
    config: ModelParallelConfig, delay_wgrad_compute: bool, weight_tensors: list[torch.Tensor]
) -> None:
    """Reject int4 fake-QAT paths that replace TE weights with unsafe STE tensors."""
    if config.gradient_accumulation_fusion:
        raise RuntimeError(
            f"{INT4_FAKE_QAT_FLAG}=1 is not supported with "
            "gradient_accumulation_fusion because TE fused wgrad accumulation mutates "
            "Python attributes on the original weight tensors."
        )
    if delay_wgrad_compute:
        raise RuntimeError(
            f"{INT4_FAKE_QAT_FLAG}=1 is not supported with delayed wgrad compute because "
            "the delayed TE path mutates Python attributes on the original weight tensors."
        )
    if any(
        hasattr(weight, "__fsdp_param__") or hasattr(weight, "get_main_grad")
        for weight in weight_tensors
    ):
        raise RuntimeError(
            f"{INT4_FAKE_QAT_FLAG}=1 is not supported with Megatron FSDP because "
            "FSDP patches weight tensors with main-gradient attributes and methods."
        )


def maybe_fake_quantize_int4_weight_tensors(
    config: ModelParallelConfig, delay_wgrad_compute: bool, weight_tensors: list[torch.Tensor]
) -> list[torch.Tensor]:
    """Optionally apply env-var-gated int4 fake QAT to TE grouped-linear weight tensors."""
    if os.getenv(INT4_FAKE_QAT_FLAG, "0") != "1":
        return weight_tensors

    _validate_int4_fake_qat_support(config, delay_wgrad_compute, weight_tensors)
    group_size = int(os.getenv(INT4_GROUP_SIZE, "128"))

    return [fake_int4_quantization_ste(weight, group_size) for weight in weight_tensors]
