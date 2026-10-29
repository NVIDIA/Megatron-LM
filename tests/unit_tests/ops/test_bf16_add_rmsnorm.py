# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact forward comparisons with the TE native training RMSNorm contract."""

import os

import pytest
import torch

pytest.importorskip("triton")
te = pytest.importorskip("transformer_engine.pytorch")

from megatron.core.fusions.fused_bias_dropout import (
    bias_dropout_add_fused_train,
    bias_dropout_add_unfused,
)
from megatron.core.ops.normalization.bf16_add_rmsnorm import fused_bf16_add_rmsnorm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@pytest.fixture(autouse=True)
def cuda_device():
    """Give each torchrun worker its own GPU, including standalone kernel runs."""
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))


def make_inputs(shape, seed=123):
    """Generate independent reproducible inputs on each test rank."""
    torch.manual_seed(seed + int(os.environ.get("LOCAL_RANK", "0")))
    x = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], dtype=torch.bfloat16, device="cuda")
    return x, residual, weight


def training_reference(x, residual, weight, eps=1e-5, zero=False, compiled_add=False):
    """Run the training BDA and actual TE module with autograd enabled."""
    norm = te.RMSNorm(x.shape[-1], eps=eps, dtype=torch.bfloat16, zero_centered_gamma=zero)
    norm.train()
    with torch.no_grad():
        norm.weight.copy_(weight)
    with torch.enable_grad():
        add = bias_dropout_add_fused_train if compiled_add else bias_dropout_add_unfused(True)
        summed = add((x.detach().requires_grad_(), None), residual, 0.0)
        output = norm(summed)
    return output.detach(), summed.detach()


def assert_bits(actual, expected):
    """Check signed zero and NaN bits as well as ordinary finite values."""
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype == torch.bfloat16
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize(
    "shape", [(2688,), (1, 2688), (7, 2688), (32, 2, 2688), (257, 2688), (2120, 2688), (4096, 2688)]
)
@pytest.mark.parametrize("eps,zero", [(1e-5, False), (1e-6, False), (1e-5, True)])
def test_matches_training(shape, eps, zero):
    x, residual, weight = make_inputs(shape)
    before = [value.clone() for value in (x, residual, weight)]
    actual = fused_bf16_add_rmsnorm(x, residual, weight, eps, zero)
    expected = training_reference(x, residual, weight, eps, zero)
    for value, ref in zip(actual, expected):
        assert_bits(value, ref)
    for value, ref in zip((x, residual, weight), before):
        assert_bits(value, ref)
    assert actual[0].data_ptr() != actual[1].data_ptr()
    assert all(y.data_ptr() not in (x.data_ptr(), residual.data_ptr()) for y in actual)


@pytest.mark.parametrize(
    "hidden", [8, 64, 128, 256, 512, 768, 1024, 1536, 2048, 2688, 4096, 6144, 8192]
)
@pytest.mark.parametrize("rows", [1, 2, 4, 7, 32, 128, 257, 2048])
@pytest.mark.parametrize("zero", [False, True])
def test_width_and_row_dispatch(hidden, rows, zero):
    """Cover general and tuned TE dispatch on both sides of each row-count boundary."""
    x, residual, weight = make_inputs((rows, hidden))
    actual = fused_bf16_add_rmsnorm(x, residual, weight, zero_centered_gamma=zero)
    expected = training_reference(x, residual, weight, zero=zero)
    for value, ref in zip(actual, expected):
        assert_bits(value, ref)


@pytest.mark.parametrize(
    "case", ["ties", "cancel", "zeros", "tiny", "subnormal", "large", "offset", "overflow"]
)
@pytest.mark.parametrize(
    "rows,hidden",
    [
        (33, 2688),
        (32, 128),
        (33, 128),
        (32, 768),
        (33, 768),
        (32, 4096),
        (33, 4096),
        (32, 8192),
        (33, 8192),
    ],
)
def test_rounding_and_extreme_inputs(case, rows, hidden):
    x, residual, weight = make_inputs((rows, hidden))
    if case == "ties":
        x.fill_(1.0)
        x[:, 1::2] = 1.0078125
        residual.fill_(1.0 / 256)
    elif case == "cancel":
        residual.copy_(-x)
        residual[:, ::2] *= 1.0078125
    elif case == "zeros":
        x.zero_()
        residual.zero_()
        x[:, ::2] = -0.0
        residual[:, ::2] = -0.0
    elif case == "tiny":
        x *= 1e-20
        residual *= 1e-20
    elif case == "subnormal":
        x = (torch.arange(hidden, device="cuda").float() % 127 * 2.0**-133).bfloat16()
        x = x.expand(rows, -1).contiguous()
        residual = -x.roll(1, -1)
    elif case == "large":
        x *= 1e18
        residual *= 1e18
    elif case == "offset":
        x += 1024
        residual *= 0.01
    elif case == "overflow":
        x.fill_(torch.finfo(torch.bfloat16).max)
        residual.copy_(x)
    actual = fused_bf16_add_rmsnorm(x, residual, weight)
    expected = training_reference(x, residual, weight)
    for value, ref in zip(actual, expected):
        assert_bits(value, ref)


def test_compiled_training_add():
    x, residual, weight = make_inputs((257, 2688))
    actual = fused_bf16_add_rmsnorm(x, residual, weight)
    expected = training_reference(x, residual, weight, compiled_add=True)
    for value, ref in zip(actual, expected):
        assert_bits(value, ref)


@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("shape", [(257, 2688), (128, 8192)])
def test_cuda_graph_replay_and_weight_refit(inplace, shape):
    x, residual, weight = make_inputs(shape)
    fused_bf16_add_rmsnorm(x, residual, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused_bf16_add_rmsnorm(x, residual, weight, inplace=inplace)
    pointers = tuple(t.data_ptr() for t in actual)
    for seed in (17, 18, 19):
        for target, new_value in zip((x, residual, weight), make_inputs(x.shape, seed)):
            target.copy_(new_value)
        expected = training_reference(x, residual, weight)
        graph.replay()
        for value, ref in zip(actual, expected):
            assert_bits(value, ref)
        assert tuple(t.data_ptr() for t in actual) == pointers


def test_empty_input():
    x, residual, weight = make_inputs((0, 2688))
    y, summed = fused_bf16_add_rmsnorm(x, residual, weight)
    assert y.shape == summed.shape == x.shape
    assert y.dtype == summed.dtype == torch.bfloat16


def test_inplace_residual():
    x, residual, weight = make_inputs((257, 2688))
    expected = training_reference(x, residual, weight)
    residual_before, weight_before = residual.clone(), weight.clone()
    actual = fused_bf16_add_rmsnorm(x, residual, weight, inplace=True)
    assert actual[1] is x
    for value, ref in zip(actual, expected):
        assert_bits(value, ref)
    assert_bits(residual, residual_before)
    assert_bits(weight, weight_before)


@pytest.mark.parametrize("alias", ["residual", "weight"])
def test_rejects_inplace_storage_alias(alias):
    x, residual, weight = make_inputs((2, 2688))
    if alias == "residual":
        residual = x
    else:
        weight = x[0]
    with pytest.raises(ValueError, match="share storage"):
        fused_bf16_add_rmsnorm(x, residual, weight, inplace=True)


@pytest.mark.parametrize(
    "invalid",
    [
        "width",
        "shape",
        "dtype",
        "weight_dtype",
        "layout",
        "cpu",
        "epsilon_zero",
        "epsilon_nan",
        "epsilon_inf",
        "grad",
    ],
)
def test_rejects_unsupported_contract(invalid):
    x, residual, weight = make_inputs((2, 2688))
    eps = 1e-5
    if invalid == "width":
        x = x[:, :255].contiguous()
        residual = residual[:, :255].contiguous()
        weight = weight[:255]
    elif invalid == "shape":
        residual = residual[:1]
    elif invalid == "dtype":
        residual = residual.float()
    elif invalid == "weight_dtype":
        weight = weight.float()
    elif invalid == "layout":
        x = x.t().contiguous().t()
    elif invalid == "cpu":
        weight = weight.cpu()
    elif invalid.startswith("epsilon"):
        eps = {"epsilon_zero": 0.0, "epsilon_nan": float("nan"), "epsilon_inf": float("inf")}[
            invalid
        ]
    elif invalid == "grad":
        x.requires_grad_()
    with pytest.raises(RuntimeError if invalid == "grad" else ValueError):
        fused_bf16_add_rmsnorm(x, residual, weight, eps)
    if invalid == "grad":
        with torch.no_grad():
            fused_bf16_add_rmsnorm(x, residual, weight)
