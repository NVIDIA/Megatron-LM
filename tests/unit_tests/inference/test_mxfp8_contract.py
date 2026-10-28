# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Specification-level contracts for standalone inference MXFP8 storage and GEMM."""

from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

import megatron.core.inference.quantization.utils as quantization_utils
from megatron.core.inference.quantization.mxfp8_quantize import (
    MXFP8_BLOCK_SIZE,
    MXFP8_SCALE_COL_BLOCK,
    MXFP8_SCALE_ROW_BLOCK,
    mxfp8_quantize,
    mxfp8_quantize_into,
    mxfp8_quantize_rows,
)
from megatron.core.inference.quantization.mxfp8_tensor import (
    HAVE_FLASHINFER,
    MXFP8Tensor,
    validate_mxfp8_tensor,
)
from megatron.core.inference.symmetric_memory import SymmetricMemoryManager
from megatron.core.tensor_parallel.inference_layers import (
    InferenceColumnParallelLinear,
    InferenceRowParallelLinear,
    _apply_linear,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.internal,
]


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _reference_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Independently apply the finite MXFP8 RCEIL definition to logical blocks."""
    rows, cols = x.shape
    groups = x.float().reshape(rows, cols // MXFP8_BLOCK_SIZE, MXFP8_BLOCK_SIZE)
    max_abs = groups.abs().amax(dim=-1)
    exponent = (
        torch.ceil(torch.log2(max_abs / torch.finfo(torch.float8_e4m3fn).max))
        .clamp(min=-127, max=127)
        .add(127)
        .to(torch.uint8)
    )
    # E8M0 byte 0 is the finite scale 2**-127, not a zero-scale encoding.
    # Zero blocks share that byte but force a zero multiplier to avoid 0 * huge.
    multiplier = torch.where(
        max_abs == 0, torch.zeros_like(max_abs), torch.exp2(127 - exponent.to(torch.float32))
    )
    payload = torch.clamp(
        groups * multiplier.unsqueeze(-1),
        min=-torch.finfo(torch.float8_e4m3fn).max,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)
    return payload.reshape_as(x), exponent


def _swizzle_scales(logical: torch.Tensor, storage_rows: int | None = None) -> torch.Tensor:
    """Map logical E8M0 bytes to the cuBLAS SWIZZLE_32_4_4 byte layout."""
    rows, cols = logical.shape
    storage_rows = rows if storage_rows is None else storage_rows
    padded_rows = _ceil_div(storage_rows, MXFP8_SCALE_ROW_BLOCK) * MXFP8_SCALE_ROW_BLOCK
    padded_cols = _ceil_div(cols, MXFP8_SCALE_COL_BLOCK) * MXFP8_SCALE_COL_BLOCK
    physical = torch.zeros(padded_rows * padded_cols, device=logical.device, dtype=torch.uint8)
    row = torch.arange(rows, device=logical.device)[:, None]
    col = torch.arange(cols, device=logical.device)[None, :]
    n_col_blocks = padded_cols // MXFP8_SCALE_COL_BLOCK
    offset = (
        ((row // 128) * n_col_blocks + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    physical[offset] = logical
    return physical


def _logical_scale_bytes(tensor: MXFP8Tensor) -> torch.Tensor:
    """Decode either supported backend scale layout to logical exponent bytes."""
    rows, cols = tensor.data.shape
    scale_cols = cols // MXFP8_BLOCK_SIZE
    if tensor.scale.ndim == 2:
        return tensor.scale[:rows, :scale_cols].view(torch.uint8)
    padded_cols = _ceil_div(scale_cols, MXFP8_SCALE_COL_BLOCK) * MXFP8_SCALE_COL_BLOCK
    row = torch.arange(rows, device=tensor.device)[:, None]
    col = torch.arange(scale_cols, device=tensor.device)[None, :]
    n_col_blocks = padded_cols // MXFP8_SCALE_COL_BLOCK
    offset = (
        ((row // 128) * n_col_blocks + col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + col % 4
    )
    return tensor.scale.view(torch.uint8)[offset]


def _dequantize(tensor: MXFP8Tensor) -> torch.Tensor:
    scales = torch.exp2(_logical_scale_bytes(tensor).to(torch.float32) - 127)
    values = tensor.data.float().reshape(tensor.shape[0], -1, MXFP8_BLOCK_SIZE)
    return (values * scales.unsqueeze(-1)).reshape(tensor.shape)


def _assert_backend_gemm_close(actual: torch.Tensor, reference: torch.Tensor, label: str) -> None:
    """Report backend accumulation error before applying the BF16 output bound."""
    actual_f32, reference_f32 = actual.float(), reference.float()
    absolute_error = (actual_f32 - reference_f32).abs()
    significant = reference_f32.abs() >= 0.1
    max_relative = (
        (absolute_error[significant] / reference_f32[significant].abs()).max().item()
        if significant.any()
        else 0.0
    )
    print(
        f"{label}: max_abs_error={absolute_error.max().item():.8f}, "
        f"max_relative_error_for_abs_reference_ge_0.1={max_relative:.8f}"
    )
    # Both sides use the exact same quantized operands. The allowance covers only
    # backend accumulation order and final BF16 rounding over 256-term dot products.
    torch.testing.assert_close(actual_f32, reference_f32, rtol=0.02, atol=0.02)


def _boundary_input(rows: int, cols: int) -> torch.Tensor:
    """Give every row/group a distinct range while retaining exact binary values."""
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[:, None, None]
    group = torch.arange(cols // 32, device="cuda", dtype=torch.int64)[None, :, None]
    lane = torch.arange(32, device="cuda", dtype=torch.float32)[None, None, :]
    exponent = ((row + 3 * group) % 19 - 12).to(torch.float32)
    signs = torch.where((lane.to(torch.int64) + row + group) % 2 == 0, 1.0, -1.0)
    values = signs * torch.exp2(exponent) * (0.5 + lane / 64)
    result = values.reshape(rows, cols)
    result[0, :32] = 0
    if rows > 32:
        result[32, 32:64] = torch.finfo(torch.float32).tiny / 4
    result[-1, -32:] = torch.linspace(-1.0e30, 1.0e30, 32, device="cuda")
    return result


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize(
    "backend,dispatch_name",
    [("triton", "_mm_mxfp8_torch"), ("flashinfer", "_mm_mxfp8_flashinfer")],
    ids=["torch-scaled-mm", "flashinfer"],
)
def test_direct_linear_backend_matches_dequantized_matmul(
    monkeypatch, backend: str, dispatch_name: str
):
    """Execute activation quantization and the selected real GEMM on identical values."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native MXFP8 GEMM requires Blackwell")
    if backend == "flashinfer" and not HAVE_FLASHINFER:
        pytest.skip("FlashInfer is unavailable")
    if backend == "triton" and not quantization_utils.HAVE_TORCH_SCALED_MM:
        pytest.skip("torch scaled_mm block scaling is unavailable")

    torch.manual_seed(2301)
    x = torch.randn(7, 1, 256, device="cuda", dtype=torch.bfloat16) / 4
    weight_source = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16) / 4
    weight = MXFP8Tensor.from_bf16(weight_source, backend=backend)
    observed = {}

    real_from_bf16 = MXFP8Tensor.from_bf16.__func__

    def tracked_from_bf16(cls, value, group_size=32, backend="flashinfer"):
        result = real_from_bf16(cls, value, group_size=group_size, backend=backend)
        observed["activation"] = result
        return result

    real_dispatch = getattr(quantization_utils, dispatch_name)

    def tracked_dispatch(activation, dispatched_weight, out=None):
        observed["dispatch"] = dispatch_name
        return real_dispatch(activation, dispatched_weight, out=out)

    monkeypatch.setattr(MXFP8Tensor, "from_bf16", classmethod(tracked_from_bf16))
    monkeypatch.setattr(quantization_utils, dispatch_name, tracked_dispatch)

    output_storage = torch.full(
        (x.shape[0] + 2, weight.shape[0]), float("nan"), device="cuda", dtype=torch.bfloat16
    )
    output_buffer = output_storage[: x.shape[0]]
    output_ptr = output_buffer.data_ptr()
    output = _apply_linear(x, weight, SimpleNamespace(), out=output_buffer)
    activation = observed["activation"]
    reference = (_dequantize(activation) @ _dequantize(weight).T).to(torch.bfloat16)
    bf16_weight = torch.randn(96, 256, device="cuda", dtype=torch.bfloat16) / 4
    bf16_output = _apply_linear(x, bf16_weight, SimpleNamespace())

    assert observed["dispatch"] == dispatch_name
    assert torch.count_nonzero(activation.data.view(torch.uint8)) > 0
    assert torch.count_nonzero(_logical_scale_bytes(activation)) > 0
    assert output.data_ptr() == output_ptr
    assert torch.isnan(output_storage[x.shape[0] :]).all()
    _assert_backend_gemm_close(output.squeeze(1), reference, backend)
    assert torch.equal(bf16_output, torch.matmul(x, bf16_weight.T))


@pytest.mark.launch_on_gb200
def test_tp_sequence_parallel_production_linear_matches_quantized_reference(monkeypatch):
    """Exercise production TP/SP collectives and both MXFP8 linears on one step."""
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("native MXFP8 GEMM requires Blackwell")
    if not quantization_utils.HAVE_TORCH_SCALED_MM:
        pytest.skip("torch scaled_mm block scaling is unavailable")
    if Utils.world_size != 2:
        pytest.skip("requires an exact two-rank launch")
    Utils.initialize_distributed()

    world_size = dist.get_world_size()
    Utils.initialize_model_parallel(tensor_model_parallel_size=world_size)
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=256,
            num_attention_heads=8,
            tensor_model_parallel_size=world_size,
            sequence_parallel=True,
            params_dtype=torch.bfloat16,
            bf16=True,
            transformer_impl="inference_optimized",
            normalization="RMSNorm",
            add_bias_linear=False,
            inference_disable_triton_nvls_kernels=True,
        )
        column = InferenceColumnParallelLinear(
            256,
            256,
            config=config,
            init_method=lambda value: torch.nn.init.normal_(value, std=0.01),
            gather_output=False,
            bias=False,
            skip_bias_add=False,
            is_expert=False,
        )
        row = InferenceRowParallelLinear(
            256,
            256,
            config=config,
            init_method=lambda value: torch.nn.init.normal_(value, std=0.01),
            bias=False,
            input_is_parallel=True,
            skip_bias_add=False,
            is_expert=False,
        )
        column.eval()
        row.eval()

        generator = torch.Generator(device="cuda").manual_seed(9042)
        full_input = torch.randn(
            10, 1, 256, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        full_column_weight = torch.randn(
            256, 256, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        full_row_weight = torch.randn(
            256, 256, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        local_input = full_input.chunk(world_size, dim=0)[dist.get_rank()].contiguous()
        local_column_weight = full_column_weight.chunk(world_size, dim=0)[
            dist.get_rank()
        ].contiguous()
        local_row_weight = full_row_weight.chunk(world_size, dim=1)[dist.get_rank()].contiguous()
        quantized_column_weight = MXFP8Tensor.from_bf16(local_column_weight, backend="triton")
        quantized_row_weight = MXFP8Tensor.from_bf16(local_row_weight, backend="triton")
        del column._parameters["weight"]
        del row._parameters["weight"]
        column.weight = quantized_column_weight
        row.weight = quantized_row_weight

        observed_activations = []
        real_from_bf16 = MXFP8Tensor.from_bf16.__func__

        def tracked_from_bf16(cls, value, group_size=32, backend="flashinfer"):
            result = real_from_bf16(cls, value, group_size=group_size, backend=backend)
            observed_activations.append(result)
            return result

        dispatches = []
        real_dispatch = quantization_utils._mm_mxfp8_torch

        def tracked_dispatch(activation, dispatched_weight, out=None):
            dispatches.append((activation, dispatched_weight))
            return real_dispatch(activation, dispatched_weight, out=out)

        monkeypatch.setattr(MXFP8Tensor, "from_bf16", classmethod(tracked_from_bf16))
        monkeypatch.setattr(quantization_utils, "_mm_mxfp8_torch", tracked_dispatch)

        intermediate, column_bias = column(local_input)
        output, row_bias = row(intermediate)
        column_activation, row_activation = observed_activations
        column_reference = (
            _dequantize(column_activation) @ _dequantize(quantized_column_weight).T
        ).to(torch.bfloat16)
        local_row_reference = (
            _dequantize(row_activation) @ _dequantize(quantized_row_weight).T
        ).to(torch.bfloat16)
        dist.all_reduce(local_row_reference)
        row_reference = local_row_reference.chunk(world_size, dim=0)[dist.get_rank()].contiguous()

        assert column_bias is None and row_bias is None
        assert len(observed_activations) == 2
        assert len(dispatches) == 2
        assert dispatches[0][0] is column_activation
        assert dispatches[0][1] is quantized_column_weight
        assert dispatches[1][0] is row_activation
        assert dispatches[1][1] is quantized_row_weight
        assert column_activation.shape == full_input.squeeze(1).shape
        assert row_activation.shape == intermediate.squeeze(1).shape
        assert intermediate.shape == (full_input.shape[0], 1, local_column_weight.shape[0])
        assert output.shape == local_input.shape
        assert torch.count_nonzero(column_activation.data.view(torch.uint8)) > 0
        assert torch.count_nonzero(row_activation.data.view(torch.uint8)) > 0
        _assert_backend_gemm_close(
            intermediate.squeeze(1), column_reference, "tp-column-all-gather"
        )
        _assert_backend_gemm_close(output.squeeze(1), row_reference, "sp-row-reduce-scatter")
    finally:
        SymmetricMemoryManager.destroy()
        Utils.destroy_model_parallel()
