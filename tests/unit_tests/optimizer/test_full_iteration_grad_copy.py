# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Gradient values, update order and storage aliasing across full-CG phases."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    FullIterationGradCopy,
    get_shared_capture_stream,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer import Float16OptimizerWithFloat16Params
from tests.unit_tests.test_utilities import Utils


def make_optimizer(dtype, reuse=True, precision_aware=False):
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    # Converted shards must use the large-block allocator pool, like F/B scratch.
    shard_size = 512 * 1024
    model = torch.nn.Parameter(torch.zeros(shard_size + 11, device="cuda", dtype=dtype))
    model.main_grad = torch.empty_like(model)
    master = torch.nn.Parameter(torch.zeros(shard_size, device="cuda"))
    fp32 = torch.nn.Parameter(torch.zeros(64, device="cuda"))
    fp32.main_grad = torch.empty_like(fp32)
    fp32_master = torch.nn.Parameter(torch.zeros_like(fp32))
    optimizer.is_stub_optimizer = False
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=precision_aware
    )
    optimizer.model_float16_groups = [[model]]
    optimizer.model_fp32_groups = [[fp32]]
    optimizer.shard_fp32_from_float16_groups = [[master]]
    optimizer.shard_fp32_groups = [[fp32_master]]
    optimizer.shard_float16_groups = [
        [torch.nn.Parameter(torch.zeros(shard_size, device="cuda", dtype=dtype))]
    ]
    optimizer._get_model_param_range_map = lambda param: {
        "param": Range(7, shard_size + 7) if param is model else Range(0, 64)
    }
    optimizer._full_iteration_grad_copy = FullIterationGradCopy() if reuse else None
    return optimizer


def pairs(optimizer):
    if isinstance(optimizer, Float16OptimizerWithFloat16Params):
        for models, masters in zip(optimizer.float16_groups, optimizer.fp32_from_float16_groups):
            yield from zip(models, masters)
        for models in optimizer.fp32_from_fp32_groups:
            yield from ((model, model) for model in models)
        return
    for models, masters in (
        (optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups),
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
    ):
        for model_group, master_group in zip(models, masters):
            yield from zip(model_group, master_group)


def make_layerwise_optimizer(dtype):
    optimizer = Float16OptimizerWithFloat16Params.__new__(Float16OptimizerWithFloat16Params)
    model = torch.nn.Parameter(torch.zeros(512 * 1024, device="cuda", dtype=dtype))
    model.main_grad = torch.empty_like(model)
    master = torch.nn.Parameter(torch.zeros_like(model, dtype=torch.float32))
    fp32 = torch.nn.Parameter(torch.zeros(64, device="cuda"))
    fp32.main_grad = torch.empty_like(fp32)
    optimizer.is_stub_optimizer = False
    optimizer.float16_groups = [[model]]
    optimizer.fp32_from_float16_groups = [[master]]
    optimizer.fp32_from_fp32_groups = [[fp32]]
    optimizer._full_iteration_grad_copy = FullIterationGradCopy()
    return optimizer


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("factory", [make_optimizer, make_layerwise_optimizer])
def test_full_iteration_gradient_storage_reuse(dtype, factory):
    """Changing and poisoned replays preserve exact gradients and eager Adam updates."""
    Utils.initialize_distributed()
    old_graph = FullCudaGraphWrapper.cuda_graph["training"]
    FullCudaGraphWrapper.cuda_graph["training"] = None
    optimizers = [factory(dtype), factory(dtype)]
    masters = [master for optimizer in optimizers for _, master in pairs(optimizer)]
    references = [torch.nn.Parameter(master.detach().clone()) for master in masters]
    adam = torch.optim.Adam(masters, lr=0.01, foreach=False)
    reference_adam = torch.optim.Adam(references, lr=0.01, foreach=False)
    value = torch.ones((), device="cuda")
    stream = get_shared_capture_stream()
    scratch_ranges = []

    def forward_backward():
        scratch = torch.empty(4 * 1024 * 1024, device="cuda")
        scratch.fill_(value)
        if torch.cuda.is_current_stream_capturing():
            scratch_ranges.append((scratch.data_ptr(), scratch.numel() * scratch.element_size()))
        for optimizer in optimizers:
            for model, _ in pairs(optimizer):
                model.main_grad.copy_(scratch[: model.numel()])

    def update_and_check():
        reference_index = 0
        for optimizer in optimizers:
            optimizer._copy_model_grads_to_main_grads()
            for model, master in pairs(optimizer):
                expected = model.main_grad
                if isinstance(optimizer, DistributedOptimizer):
                    span = optimizer._get_model_param_range_map(model)["param"]
                    expected = expected[span.start : span.end]
                expected = expected.float()
                assert torch.equal(master.grad.view(torch.uint8), expected.view(torch.uint8))
                references[reference_index].grad = expected.clone()
                reference_index += 1
        adam.step()
        reference_adam.step()
        for master, reference in zip(masters, references):
            assert torch.equal(
                master.detach().view(torch.uint8), reference.detach().view(torch.uint8)
            )

    graph = None
    try:
        # Warmup allocations must belong to the same stream as F/B capture.
        for iteration in range(3):
            value.fill_(iteration + Utils.rank + 1)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                forward_backward()
            torch.cuda.current_stream().wait_stream(stream)
            update_and_check()
            address = masters[0].grad.data_ptr()
            segment = next(
                s
                for s in torch.cuda.memory_snapshot()
                if s["address"] <= address < s["address"] + s["total_size"]
            )
            assert segment["stream"] == stream.cuda_stream
            for optimizer in optimizers:
                optimizer.zero_grad()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            forward_backward()
        FullCudaGraphWrapper.cuda_graph["training"] = graph
        for iteration in range(36):
            # Retained handles must not make their old values observable by F/B.
            for optimizer in optimizers:
                for _, grad in optimizer._full_iteration_grad_copy.outputs:
                    grad.fill_(float("nan"))
            value.fill_((iteration + 1) * (-1 if iteration % 2 else 1) + Utils.rank)
            graph.replay()
            update_and_check()
            for optimizer in optimizers:
                assert optimizer._full_iteration_grad_copy.graph.pool() == graph.pool()
                optimizer.zero_grad()
                assert all(master.grad is None for _, master in pairs(optimizer))
        converted = [optimizer._full_iteration_grad_copy.outputs[0][1] for optimizer in optimizers]
        assert converted[0].data_ptr() != converted[1].data_ptr()
        assert any(
            start <= grad.data_ptr() < start + size
            for grad in converted
            for start, size in scratch_ranges
        )
        # Replacing a captured input must fail before replay can read stale storage.
        model = next(pairs(optimizers[0]))[0]
        model.main_grad = torch.empty_like(model)
        with pytest.raises(RuntimeError, match="gradient storage changed"):
            optimizers[0]._copy_model_grads_to_main_grads()
    finally:
        torch.cuda.synchronize()
        for optimizer in optimizers:
            optimizer._full_iteration_grad_copy = None
            optimizer.zero_grad()
        FullCudaGraphWrapper.cuda_graph["training"] = old_graph
        if graph is not None:
            graph.reset()


@pytest.mark.parametrize("precision_aware", [False, True])
def test_eager_gradient_copy_remains_unchanged(precision_aware):
    """Ordinary FP32 conversion and precision-aware aliases retain their semantics."""
    Utils.initialize_distributed()
    optimizer = make_optimizer(torch.bfloat16, reuse=False, precision_aware=precision_aware)
    for model, _ in pairs(optimizer):
        model.main_grad.fill_(3 + Utils.rank)
    optimizer._copy_model_grads_to_main_grads()
    model = optimizer.model_float16_groups[0][0]
    if precision_aware:
        grad = optimizer.shard_float16_groups[0][0].decoupled_grad
        assert grad.data_ptr() == model.main_grad[7:].data_ptr()
        assert grad.dtype == torch.bfloat16
    else:
        grad = optimizer.shard_fp32_from_float16_groups[0][0].grad
        assert grad.dtype == torch.float32
    torch.testing.assert_close(grad, torch.full_like(grad, 3 + Utils.rank), rtol=0, atol=0)


def test_gradient_copy_rebuilds_after_training_graph_reset(monkeypatch):
    """Recapturing training may replace its pool and its persistent gradient inputs."""
    Utils.initialize_distributed()
    monkeypatch.setattr(FullCudaGraphWrapper, "cuda_graph", {"training": None})
    copier = FullIterationGradCopy()
    master = torch.nn.Parameter(torch.zeros(1024, device="cuda"))
    stream = get_shared_capture_stream()
    previous_conversion = None
    for value in (2.0, 5.0):
        FullCudaGraphWrapper.cuda_graph["training"] = None
        source = torch.full((1024,), value, device="cuda", dtype=torch.bfloat16)
        master.grad = None

        def copy():
            master.grad = source.float()

        copier.copy(copy, [(source, master)])
        assert copier.graph is None
        torch.testing.assert_close(master.grad, source.float())
        master.grad = None
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            source.fill_(value)
        FullCudaGraphWrapper.cuda_graph["training"] = graph
        graph.replay()
        copier.copy(copy, [(source, master)])
        assert copier.graph is not previous_conversion
        assert copier.graph.pool() == graph.pool()
        torch.testing.assert_close(master.grad, source.float())
        previous_conversion = copier.graph
