# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""``fused_main_grad_copy`` gives the FP32 main grads of the per-parameter ``.float()`` copy."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Parameter shapes laid out in one grad buffer with 64-element padding, like a DDP bucket.
SHAPES = [(7168,), (1536, 7168), (576, 7168), (7168, 2048), (17,), (24576, 512), (128, 7168)]


def _numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


def make_optimizer(dtype, fused, shard_lo_frac=0.2, shard_hi_frac=0.85, buffers=2):
    """A DistributedOptimizer shell whose rank shard cuts parameters in the middle.

    ``buffers`` grad buffers hold the same parameter layout; one more FP32 parameter has its own
    FP32 grad.
    """
    torch.manual_seed(1)
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.is_stub_optimizer = False
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.config = SimpleNamespace(
        use_precision_aware_optimizer_no_fp8_or_ds_fp8=False, fused_main_grad_copy=fused
    )
    optimizer._cudagraphed_grad_copy = None
    ranges = {}
    models, masters = [], []
    for _ in range(buffers):
        offsets, total = [], 0
        for shape in SHAPES:
            offsets.append(total)
            total += -(-_numel(shape) // 64) * 64
        # Odd start: the buffer itself is a view at an unaligned storage offset.
        storage = torch.randn(total + 3, device="cuda", dtype=dtype)
        grad_buffer = storage[3:]
        lo = int(total * shard_lo_frac) + 5
        hi = int(total * shard_hi_frac) + 3
        for shape, offset in zip(SHAPES, offsets):
            n = _numel(shape)
            start, end = max(lo, offset) - offset, min(hi, offset + n) - offset
            if start >= end:
                continue
            model = torch.nn.Parameter(torch.empty(shape, device="cuda", dtype=dtype))
            model.main_grad = grad_buffer[offset : offset + n].view(shape)
            models.append(model)
            masters.append(torch.nn.Parameter(torch.empty(end - start, device="cuda")))
            ranges[model] = Range(start, end)
    fp32_model = torch.nn.Parameter(torch.empty(1000, device="cuda"))
    fp32_model.main_grad = torch.randn(1000, device="cuda")
    fp32_master = torch.nn.Parameter(torch.empty(990, device="cuda"))
    ranges[fp32_model] = Range(10, 1000)
    optimizer.model_float16_groups = [models[: len(models) // 2], models[len(models) // 2 :]]
    optimizer.shard_fp32_from_float16_groups = [
        masters[: len(masters) // 2],
        masters[len(masters) // 2 :],
    ]
    optimizer.model_fp32_groups = [[fp32_model]]
    optimizer.shard_fp32_groups = [[fp32_master]]
    optimizer._get_model_param_range_map = lambda param: {"param": ranges[param]}
    return optimizer


def model_master_pairs(optimizer):
    for models, masters in (
        (optimizer.model_float16_groups, optimizer.shard_fp32_from_float16_groups),
        (optimizer.model_fp32_groups, optimizer.shard_fp32_groups),
    ):
        for model_group, master_group in zip(models, masters):
            yield from zip(model_group, master_group)


def expected_grads(optimizer):
    out = []
    for model, _ in model_master_pairs(optimizer):
        span = optimizer._get_model_param_range_map(model)["param"]
        out.append(model.main_grad.view(-1)[span.start : span.end].float())
    return out


def assert_grads_equal(optimizer, expected):
    for (model, master), ref in zip(model_master_pairs(optimizer), expected):
        assert master.grad is not None and master.grad.dtype == torch.float32
        assert master.grad.shape == master.shape
        assert torch.equal(master.grad.view(torch.int32), ref.view(torch.int32))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_main_grad_copy_matches_per_param_float(dtype):
    optimizer = make_optimizer(dtype, fused=True)
    for step in range(3):
        for model, _ in model_master_pairs(optimizer):
            model.main_grad.normal_()
        expected = expected_grads(optimizer)
        optimizer._copy_model_grads_to_main_grads()
        assert_grads_equal(optimizer, expected)
    # One run per grad buffer plus the FP32 parameter.
    assert len(optimizer._main_grad_runs) == 3


def test_fused_main_grad_copy_fp32_grad_aliases_buffer():
    """FP32 grads are not converted: the main grad is a view of the grad buffer, as .float()."""
    optimizer = make_optimizer(torch.bfloat16, fused=True)
    optimizer._copy_model_grads_to_main_grads()
    fp32_model = optimizer.model_fp32_groups[0][0]
    grad = optimizer.shard_fp32_groups[0][0].grad
    assert grad.data_ptr() == fp32_model.main_grad[10:].data_ptr()


def test_fused_main_grad_copy_cuda_graph():
    """The plan can be built during capture, and replays follow the grad buffer's values."""
    optimizer = make_optimizer(torch.bfloat16, fused=True)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        optimizer._copy_model_grads_to_main_grads_eager()
    outputs = [master.grad for _, master in model_master_pairs(optimizer)]
    for _ in range(3):
        for model, _ in model_master_pairs(optimizer):
            model.main_grad.normal_()
        expected = expected_grads(optimizer)
        graph.replay()
        torch.cuda.synchronize()
        for out, ref in zip(outputs, expected):
            assert torch.equal(out.view(torch.int32), ref.view(torch.int32))
    graph.reset()


def test_runs_follow_a_moved_grad_buffer():
    """A grad buffer moved to new storage rebuilds the plan."""
    optimizer = make_optimizer(torch.bfloat16, fused=True, buffers=1)
    optimizer._copy_model_grads_to_main_grads()
    first = optimizer._main_grad_runs
    models = optimizer.model_float16_groups[0] + optimizer.model_float16_groups[1]
    old_storage = models[0].main_grad
    new_buffer = torch.empty(
        old_storage.untyped_storage().nbytes() // 2, device="cuda", dtype=torch.bfloat16
    )
    for model in models:
        offset = model.main_grad.storage_offset()
        model.main_grad = new_buffer[offset : offset + model.numel()].view(model.shape)
        model.main_grad.normal_()
    expected = expected_grads(optimizer)
    optimizer._copy_model_grads_to_main_grads()
    assert optimizer._main_grad_runs is not first
    assert_grads_equal(optimizer, expected)


def _train(fused):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.transformer_config import TransformerConfig

    torch.manual_seed(7)
    model = torch.nn.Module()
    # Sizes that make the data-parallel shard boundaries cut parameters.
    model.weights = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.randn(rows, 136, device="cuda", dtype=torch.bfloat16))
            for rows in (257, 33, 1000, 7)
        ]
    )
    model.bias = torch.nn.Parameter(torch.randn(1031, device="cuda"))
    ddp = DistributedDataParallel(
        TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True),
        DistributedDataParallelConfig(use_distributed_optimizer=True, bucket_size=20000),
        model,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            fused_main_grad_copy=fused,
            clip_grad=1.0,
        ),
        [ddp],
    )
    for step in range(3):
        torch.manual_seed(100 + step)
        optimizer.zero_grad()
        for param in model.parameters():
            param.main_grad.copy_(torch.randn_like(param.main_grad))
        assert optimizer.step()[0]
    return [param.detach().clone() for param in model.parameters()]


def test_fused_main_grad_copy_end_to_end():
    """Adam steps through DDP and the distributed optimizer give bit-identical parameters."""
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel()
    try:
        reference = _train(False)
        fused = _train(True)
        for a, b in zip(reference, fused):
            assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    finally:
        Utils.destroy_model_parallel()
