# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Compile and replay the CI image's legacy block-pointer compatibility lowering."""

import pytest
import torch
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource


@triton.jit
def _copy_advance(x, y, N: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    source = tl.make_block_ptr(x, (N,), (STRIDE,), (-BLOCK,), (BLOCK,), (0,))
    target = tl.make_block_ptr(y, (3 * BLOCK,), (1,), (0,), (BLOCK,), (0,))
    for _ in range(3):
        values = tl.load(source, boundary_check=(0,), padding_option="zero")
        tl.store(target, values)
        source = tl.advance(source, (BLOCK,))
        target = tl.advance(target, (BLOCK,))


@triton.jit
def _copy_2d(
    x,
    y,
    ROWS: tl.constexpr,
    COLS: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    ORDER: tl.constexpr,
    PADDING: tl.constexpr,
):
    row, col = tl.program_id(0) * 4, tl.program_id(1) * 4
    source = tl.make_block_ptr(x, (ROWS, COLS), (S0, S1), (row, col), (4, 4), ORDER)
    values = tl.load(source, boundary_check=(0, 1), padding_option=PADDING)
    target = tl.make_block_ptr(y, (8, 8), (8, 1), (row, col), (4, 4), (1, 0))
    tl.store(target, values.to(tl.float32))


@triton.jit
def _copy_3d(x, y):
    i, j, k = tl.program_id(0) * 2, tl.program_id(1) * 2, tl.program_id(2) * 4
    source = tl.make_block_ptr(x, (3, 3, 5), (15, 5, 1), (i, j, k), (2, 2, 4), (2, 1, 0))
    values = tl.load(source, boundary_check=(0, 1, 2), padding_option="zero")
    target = tl.make_block_ptr(y, (4, 4, 8), (32, 8, 1), (i, j, k), (2, 2, 4), (2, 1, 0))
    tl.store(target, values)


@triton.jit
def _plain_copy(x, y, N: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets, mask=offsets < N, other=0.0)
    tl.store(y + offsets, values, mask=offsets < N)


@pytest.mark.parametrize("arch", [90, 100])
@pytest.mark.parametrize("dtype", ["fp32", "fp16", "bf16", "fp8e4nv"])
def test_block_pointer_lowering_compiles(arch, dtype):
    """Exercise real compiler lowering without requiring a local GPU or driver."""
    target = GPUTarget("cuda", arch, 32)
    cases = [
        (_copy_advance, {"N": 19, "STRIDE": 2, "BLOCK": 16}),
        (_copy_2d, {"ROWS": 5, "COLS": 7, "S0": 1, "S1": 5, "ORDER": (0, 1), "PADDING": "zero"}),
        (_copy_3d, {}),
        (_plain_copy, {"N": 19, "BLOCK": 32}),
    ]
    for kernel, constants in cases:
        compiled = triton.compile(
            ASTSource(kernel, {"x": f"*{dtype}", "y": "*fp32"}, constexprs=constants), target=target
        )
        assert "tt.load" in compiled.asm["ttir"]
        assert "tt.store" in compiled.asm["ttir"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_strided_negative_offset_and_advance_replay(dtype):
    values = torch.arange(38, device="cuda", dtype=dtype)[::2]
    expected = torch.zeros(48, device="cuda", dtype=torch.float32)
    expected[16:35] = values.float()
    for _ in range(3):
        output = torch.full_like(expected, float("nan"))
        _copy_advance[(1,)](values, output, values.numel(), values.stride(0), 16)
        assert torch.equal(output, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("padding", ["zero", "nan"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float8_e4m3fn])
def test_boundary_padding_and_order_replay(transposed, padding, dtype):
    if dtype == torch.float8_e4m3fn and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("native FP8 requires Hopper or newer")
    values = torch.arange(35, device="cuda", dtype=torch.float32).reshape(5, 7).to(dtype)
    if transposed:
        values = values.t().contiguous().t()
    fill = 0.0 if padding == "zero" else float("nan")
    expected = torch.full((8, 8), fill, device="cuda")
    expected[:5, :7] = values.float()
    for _ in range(3):
        output = torch.empty_like(expected)
        _copy_2d[(2, 2)](
            values, output, 5, 7, *values.stride(), (0, 1) if transposed else (1, 0), padding
        )
        torch.testing.assert_close(output, expected, rtol=0, atol=0, equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_three_dimensional_boundary_replay():
    values = torch.arange(45, device="cuda", dtype=torch.float32).reshape(3, 3, 5)
    expected = torch.zeros((4, 4, 8), device="cuda")
    expected[:3, :3, :5] = values
    for _ in range(3):
        output = torch.empty_like(expected)
        _copy_3d[(2, 2, 2)](values, output)
        assert torch.equal(output, expected)
