# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay TP reductions across row ownership, shapes, graphs and TE training."""

import importlib

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.tensor_parallel.inference_layers import (
    InferenceRowParallelLinear,
    inference_reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.tensor_parallel.mappings import _reduce_scatter_along_first_dim_ordered
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.custom_layers.batch_invariant_kernels import set_batch_invariant_mode
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import bytes_equal
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(params=[1, 2, 4])
def tp_group(request):
    if Utils.world_size % request.param:
        pytest.skip("requires a world size divisible by TP size")
    Utils.initialize_model_parallel(request.param, 1)
    model_parallel_cuda_manual_seed(123)
    yield parallel_state.get_tensor_model_parallel_group()
    Utils.destroy_model_parallel()


def _partials(rank, dtype):
    generator = torch.Generator().manual_seed(812 + rank)
    values = torch.randn(512, 2, 64, generator=generator).to(dtype)
    # Low-precision pairwise sums depend on who owns a row after scattering.
    values[..., :8] = (2048, -2048, 1, 1)[rank]
    return values


def _reference(group, dtype):
    accumulation_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    result = torch.zeros(512, 2, 64, dtype=accumulation_dtype)
    for rank in range(group.size()):
        result += _partials(rank, dtype).to(accumulation_dtype)
    return result.to(dtype).cuda()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_reduce_scatter_along_first_dim_ordered_shape_and_owner_invariance(tp_group, dtype):
    original = _partials(tp_group.rank(), dtype).cuda()
    reference = _reference(tp_group, dtype)
    for rows in (512, 256, 128, 64):
        # A noncontiguous view also exercises packing before communication.
        x = original[:rows, :, ::2]
        expected = reference[:rows, :, ::2].chunk(tp_group.size())[tp_group.rank()]
        for _ in range(3):
            actual = _reduce_scatter_along_first_dim_ordered(x, tp_group)
            assert bytes_equal(actual, expected)
        assert bytes_equal(original, _partials(tp_group.rank(), dtype).cuda())
    empty = _reduce_scatter_along_first_dim_ordered(original[:0], tp_group)
    assert empty.shape == original[:0].shape
    if tp_group.size() == 1:
        assert _reduce_scatter_along_first_dim_ordered(original, tp_group) is original


def test_inference_reduce_scatter_graph_and_shape_invariance(tp_group):
    x = _partials(tp_group.rank(), torch.bfloat16).cuda()
    reference = _reference(tp_group, x.dtype)
    config = TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=4)
    with set_batch_invariant_mode(True, backend="te_native"), torch.no_grad():
        for rows in (512, 128, 64):
            expected = reference[:rows].chunk(tp_group.size())[tp_group.rank()]
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    inference_reduce_scatter_to_sequence_parallel_region(x[:rows], tp_group, config)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = inference_reduce_scatter_to_sequence_parallel_region(
                    x[:rows], tp_group, config
                )
            for _ in range(3):
                graph.replay()
                assert bytes_equal(result, expected)


def test_te_training_and_inference_share_ordered_reduction(tp_group):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=512,
        num_attention_heads=8,
        tensor_model_parallel_size=tp_group.size(),
        sequence_parallel=tp_group.size() > 1,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
    )
    layer = InferenceRowParallelLinear(
        128 * tp_group.size(),
        512,
        config=config,
        init_method=torch.nn.init.ones_,
        bias=False,
        input_is_parallel=True,
        skip_bias_add=True,
        is_expert=False,
        tp_group=tp_group,
    ).cuda()
    value = (16, -16, 1 / 128, 1 / 128)[tp_group.rank()]
    x = torch.full((64, 1, 128), value, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    expected_value = sum((2048, -2048, 1, 1)[: tp_group.size()])
    with set_batch_invariant_mode(True, backend="te_native"):
        training_output = layer.train()(x)[0]
        assert bytes_equal(training_output, torch.full_like(training_output, expected_value))
        training_output.sum().backward()
        assert bytes_equal(x.grad, torch.full_like(x, 512))
        assert bytes_equal(layer.weight.grad, torch.full_like(layer.weight, 64 * value))
        with torch.no_grad():
            inference_output = layer.eval()(x.detach())[0]
        assert bytes_equal(training_output, inference_output)


def test_te_collective_aliases_output_async_and_restoration(tp_group):
    names = (
        "distributed",
        "module.linear",
        "module.layernorm_linear",
        "module.layernorm_mlp",
        "ops.basic.basic_linear",
    )
    modules = [importlib.import_module(f"transformer_engine.pytorch.{name}") for name in names]
    originals = [module.reduce_scatter_along_first_dim for module in modules]
    x = _partials(tp_group.rank(), torch.bfloat16).cuda()
    expected = _reference(tp_group, x.dtype).chunk(tp_group.size())[tp_group.rank()]
    for _ in range(2):
        with set_batch_invariant_mode(True, backend="te_native"):
            for module in modules:
                output = torch.empty_like(expected)
                result, work = module.reduce_scatter_along_first_dim(
                    x, tp_group, async_op=True, output=output
                )
                assert result is (x if tp_group.size() == 1 else output) and work is None
                assert bytes_equal(result, expected)
                if tp_group.size() == dist.get_world_size():
                    result, work = module.reduce_scatter_along_first_dim(x, None)
                    assert work is None and bytes_equal(result, expected)
        for module, original in zip(modules, originals):
            assert module.reduce_scatter_along_first_dim is original
