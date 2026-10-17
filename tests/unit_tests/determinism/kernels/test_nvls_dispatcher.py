# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the NVLS combine epilogue, including the intermediate bf16 rounding."""

import pytest
import torch

triton = pytest.importorskip("triton")
import triton.language as tl

from megatron.core.inference.communication.torch_symm_triton.variable_collectives import (
    _f32_bits_to_bf16_bits,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded


@triton.jit
def _combine_epilogue(values, addend, output, size, HAS_ADDEND: tl.constexpr):
    offsets = tl.program_id(0) * 128 + tl.arange(0, 128)
    bits = tl.load(values + offsets, offsets < size, other=0).to(tl.uint32, bitcast=True)
    addend_bits = tl.load(addend + offsets, offsets < size, other=0).to(tl.uint16, bitcast=True)
    result = _f32_bits_to_bf16_bits(bits, addend_bits.to(tl.uint32), HAS_ADDEND)
    tl.store(output + offsets, result.to(tl.uint16).to(tl.bfloat16, bitcast=True), offsets < size)


@pytest.mark.parametrize("with_shared", [False, True])
@pytest.mark.parametrize("size", [127, 1024, 2688])
def test_nvls_combine_epilogue_replay(size, with_shared):
    """Fusing cast and add must retain the rounding between the two operations."""
    seeded()
    values = torch.randn(size, device="cuda") * 100
    ties = (torch.arange(32, device="cuda", dtype=torch.int32) << 16) + 0x3F808000
    values[:32] = ties.view(torch.float32)
    values[32:36] = torch.tensor([3.4e38, -3.4e38, 1e-40, -0.0], device="cuda")
    addend = torch.randn(size, device="cuda", dtype=torch.bfloat16)

    def run(x, a):
        output = torch.empty_like(a)
        _combine_epilogue[(triton.cdiv(size, 128),)](x, a, output, size, with_shared)
        return output

    expected = values.bfloat16()
    if with_shared:
        expected = expected + addend
    assert torch.equal(run(values, addend).view(torch.int16), expected.view(torch.int16))
    assert_replays_bit_exact(run, (values, addend), backward=False, replays=4)
