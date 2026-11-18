# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The vectorized router BF16/FP16 -> FP32 cast must be a bit-exact ``Tensor.to``.

``router_fast_cast`` widens the router GEMM operands with a Triton copy kernel. Widening a
16-bit float is exact, so every input bit pattern must map to the FP32 bits ``Tensor.to``
produces, on whole blocks, on the ragged tail copied outside the kernel, under replay with
side-stream contention, for ineligible inputs (which use ``Tensor.to``), and inside CUDA-graph
capture, where a specialization that never ran eagerly uses ``Tensor.to`` instead of compiling.
"""

import pytest
import torch

from megatron.core.transformer.moe import router_fast_cast
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and router_fast_cast.HAVE_TRITON),
    reason="needs a GPU and Triton",
)

# (elements per program, warps, split-store variant): the default and the extremes of the knobs.
LAUNCH_CONFIGS = [(4096, 8, False), (4096, 8, True), (256, 1, False), (16384, 16, True)]


def _key(dtype, block, num_warps, split_store):
    return (torch.cuda.current_device(), dtype, block, num_warps, split_store)


def _every_bit_pattern(dtype):
    """All 65536 values of a 16-bit float type: NaN payloads, signed zeros, subnormals, inf."""
    return (
        torch.arange(-(1 << 15), 1 << 15, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(dtype)
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("block,num_warps,split_store", LAUNCH_CONFIGS)
def test_every_bit_pattern_widens_exactly(dtype, block, num_warps, split_store):
    """Whole blocks and a ragged tail give the FP32 bits of Tensor.to for every input."""
    patterns = _every_bit_pattern(dtype)
    for source in (patterns, torch.cat([patterns, patterns[:77]])):
        widened = router_fast_cast.cast_fp32_with_config(source, block, num_warps, split_store)
        assert bytes_equal(widened, source.to(torch.float32)), (dtype, block, source.numel())
    key = _key(dtype, block, num_warps, split_store)
    assert key in router_fast_cast._WARM_KEYS and key not in router_fast_cast._FAILED_KEYS


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_router_sized_cast_replays_under_contention(dtype):
    """The default launch shape on a router-input-sized operand with a ragged tail."""
    seeded()
    source = torch.randn(8192, 5123, device="cuda").to(dtype)
    outputs, _ = assert_replays_bit_exact(
        lambda tensor: router_fast_cast.fast_cast(tensor, torch.float32),
        (source,),
        backward=False,
        replays=3,
        contention=True,
        what=f"router fast cast[{dtype}]",
    )
    assert bytes_equal(outputs["out"], source.to(torch.float32))


def test_ineligible_inputs_use_tensor_to():
    """Inputs the single kernel specialization cannot take give exactly Tensor.to."""
    seeded()
    source = torch.randn(64, 4096, device="cuda").to(torch.bfloat16)
    cases = {
        "transposed": source.t(),
        "misaligned": source.view(-1)[1:4097],
        "below_one_block": source[:1, :100],
        "records_autograd": source.clone().requires_grad_(True),
    }
    for name, tensor in cases.items():
        assert bytes_equal(
            router_fast_cast.fast_cast(tensor, torch.float32), tensor.to(torch.float32)
        ), name
    assert router_fast_cast.fast_cast(source, torch.bfloat16) is source
    assert bytes_equal(router_fast_cast.fast_cast(source, torch.float64), source.to(torch.float64))
    widened = source.to(torch.float32)
    assert router_fast_cast.fast_cast(widened, torch.float32) is widened


def test_cold_specialization_inside_capture_uses_tensor_to():
    """Capture never compiles: a specialization first seen inside capture records Tensor.to,
    and once warm the captured kernel replays new inputs bit-exactly."""
    seeded()
    block, num_warps, split_store = 1024, 2, True
    key = _key(torch.bfloat16, block, num_warps, split_store)
    router_fast_cast._WARM_KEYS.discard(key)
    source = torch.randn(3, 4101, device="cuda").to(torch.bfloat16)
    fallbacks = router_fast_cast.capture_fallbacks

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cold_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(cold_graph, stream=stream):
        cold = router_fast_cast.cast_fp32_with_config(source, block, num_warps, split_store)
    assert router_fast_cast.capture_fallbacks == fallbacks + 1
    assert key not in router_fast_cast._WARM_KEYS

    eager = router_fast_cast.cast_fp32_with_config(source, block, num_warps, split_store)
    assert key in router_fast_cast._WARM_KEYS
    assert bytes_equal(eager, source.to(torch.float32))
    stream.wait_stream(torch.cuda.current_stream())
    warm_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(warm_graph, stream=stream):
        warm = router_fast_cast.cast_fp32_with_config(source, block, num_warps, split_store)
    assert router_fast_cast.capture_fallbacks == fallbacks + 1

    source.copy_(torch.randn(source.shape, device="cuda").to(torch.bfloat16))
    cold_graph.replay()
    warm_graph.replay()
    torch.cuda.synchronize()
    expected = source.to(torch.float32)
    assert bytes_equal(cold, expected) and bytes_equal(warm, expected)
