# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The copy-free grouped output projection of DSv4 attention (MCORE_DSV4_GROUP_PROJ_BMM).

The legacy (non-TE) grouped projection runs as strided-batched GEMMs on the token-major tensors.
It must give torch.einsum's output, input gradient and weight gradient bit for bit, and replay
bit-exactly, eagerly and in CUDA graphs.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant import (
    deepseek_v4_hybrid_attention as dsv4_attention,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    run_once,
    seeded,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

# tokens per CP rank, micro-batch, groups, group width, LoRA rank
_PROXY_SHAPE = (4096, 1, 8, 4096, 1024)


def _einsum_projection(x, weight):
    """The legacy grouped projection that MCORE_DSV4_GROUP_PROJ_BMM=0 keeps."""
    groups, width = x.shape[-2:]
    return torch.einsum("...gd,grd->...gr", x, weight.reshape(groups, -1, width))


def _bmm_projection(x, weight):
    return dsv4_attention._GroupedProjectionBMM.apply(x, weight)


def _inputs(shape, input_grad=True, weight_grad=True):
    """BF16 input [tokens, batch, groups, width], weight [groups * rank, width], output grad."""
    tokens, batch, groups, width, rank = shape
    x = torch.randn(tokens, batch, groups, width, device="cuda").bfloat16()
    weight = (torch.randn(groups * rank, width, device="cuda") * width**-0.5).bfloat16()
    grad = torch.randn(tokens, batch, groups, rank, device="cuda").bfloat16()
    return x.requires_grad_(input_grad), weight.requires_grad_(weight_grad), grad


@pytest.mark.parametrize(
    "shape,input_grad,weight_grad",
    [
        (_PROXY_SHAPE, True, True),
        ((37, 2, 4, 96, 48), True, True),
        ((5, 3, 2, 64, 32), True, False),
        ((5, 3, 2, 64, 32), False, True),
    ],
    ids=["64k_cp16_proxy", "odd_shape", "input_grad_only", "weight_grad_only"],
)
def test_grouped_projection_bmm_matches_einsum(shape, input_grad, weight_grad):
    """The strided-batched GEMMs reproduce einsum's output and input/weight gradients bit for
    bit, and replay bit-exactly under contention."""
    seeded()
    x, weight, grad = _inputs(shape, input_grad, weight_grad)
    grad_outputs = {"out": grad}
    expected = run_once(_einsum_projection, (x, weight), grad_outputs)
    actual = assert_replays_bit_exact(
        _bmm_projection,
        (x, weight),
        grad_outputs=grad_outputs,
        contention=True,
        what="DSv4 grouped projection BMM",
    )
    for got, want in zip(actual, expected):
        assert got.keys() == want.keys()
        for name, tensor in want.items():
            assert bytes_equal(got[name], tensor), name


def test_grouped_projection_dispatch(monkeypatch):
    """_apply_linear_o_group_proj takes the strided-batched GEMMs only while the switch is on
    and only for nonempty contiguous inputs; every path matches einsum bit for bit."""
    seeded()
    groups, width, rank = 4, 64, 32
    weight = torch.nn.Parameter(
        (torch.randn(groups * rank, width, device="cuda") * width**-0.5).bfloat16()
    )
    module = SimpleNamespace(
        _uses_batched_linear=False,
        _linear_o_group_proj_weight=weight,
        output_projection_local_groups=groups,
        config=SimpleNamespace(output_projection_lora_rank=rank),
    )
    calls = []
    apply = dsv4_attention._GroupedProjectionBMM.apply

    def counted_apply(*args):
        calls.append(args[0].shape)
        return apply(*args)

    monkeypatch.setattr(dsv4_attention._GroupedProjectionBMM, "apply", counted_apply)
    x = torch.randn(16, 2, groups, width, device="cuda").bfloat16()
    cases = {
        "contiguous": (x, 1),
        "strided": (x.transpose(0, 1).contiguous().transpose(0, 1), 0),
        "empty": (x[:0], 0),
    }
    project = dsv4_attention.DSv4HybridAttention._apply_linear_o_group_proj
    for switch in (False, True):
        monkeypatch.setattr(dsv4_attention, "_GROUP_PROJ_BMM", switch)
        for name, (value, bmm_calls) in cases.items():
            calls.clear()
            output = project(module, value)
            assert len(calls) == (bmm_calls if switch else 0), (switch, name)
            assert bytes_equal(output, _einsum_projection(value, weight)), (switch, name)


def test_grouped_projection_bmm_replays_in_cuda_graph():
    """Forward and backward captured once, as inside captured attention layers, replay on new
    values in the same storage bit for bit with eager runs."""
    seeded()
    shape = (512, 1, 8, 1024, 256)
    x, weight, grad = _inputs(shape)
    static_x = x.detach().clone().requires_grad_()
    static_weight = weight.detach().clone().requires_grad_()
    static_grad = grad.clone()

    def step():
        out = _bmm_projection(static_x, static_weight)
        return (out, *torch.autograd.grad(out, (static_x, static_weight), static_grad))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for _ in range(2):
        new_x, new_weight, new_grad = _inputs(shape)
        with torch.no_grad():
            static_x.copy_(new_x)
            static_weight.copy_(new_weight)
            static_grad.copy_(new_grad)
        graph.replay()
        outputs, grads = run_once(_bmm_projection, (new_x, new_weight), {"out": new_grad})
        expected = (outputs["out"], grads["in[0]"], grads["in[1]"])
        for name, got, want in zip(("output", "input grad", "weight grad"), captured, expected):
            assert bytes_equal(got, want), name
