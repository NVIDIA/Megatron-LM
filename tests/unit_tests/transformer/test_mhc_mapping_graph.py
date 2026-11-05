# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Exact native-versus-graphed mHC outputs, gradients, and replay lifetimes."""

import copy
import gc
import os
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from megatron.core.tensor_parallel.random import CheckpointWithoutOutputManager
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def mapping_graph_environment():
    """Use the normal distributed runner and isolate compiler specializations."""
    Utils.initialize_model_parallel(1, 1)
    torch.manual_seed(814)
    torch._dynamo.reset()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    yield
    torch.cuda.synchronize()
    torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch._dynamo.reset()
    Utils.destroy_model_parallel()


def _pair(
    dtype=torch.bfloat16, precision=(False, False), batch=1, streams=4, frozen=(), prepare=True
):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        use_cpu_initialization=True,
        enable_mhc_connections=True,
        mhc_num_residual_streams=streams,
        mhc_sinkhorn_iterations=5,
        mhc_norm_eps_inside_sqrt=precision[0],
        mhc_keep_mappings_in_fp32=precision[1],
    )
    reference = HyperConnectionModule(config, 1).cuda()
    for name, parameter in reference.named_parameters():
        if name in frozen:
            parameter.requires_grad_(False)
    candidate = copy.deepcopy(reference)
    sample = torch.randn(8, batch, streams * 16, device="cuda", dtype=dtype)
    if prepare:
        candidate.enable_mapping_cuda_graphs(sample)
    return reference, candidate, sample


def _exact(actual, expected):
    if expected is None:
        assert actual is None
        return
    assert actual is not None and actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
    )


def _compare(reference, candidate, sample, used=(0, 1, 2), requires_grad=True, operation=None):
    x = torch.empty_strided(sample.shape, sample.stride(), device=sample.device, dtype=sample.dtype)
    # empty_like/clone normalize nondense views; keep the explicit native strides.
    y = torch.empty_strided(sample.shape, sample.stride(), device=sample.device, dtype=sample.dtype)
    x.copy_(sample.detach()).requires_grad_(requires_grad)
    y.copy_(sample.detach()).requires_grad_(requires_grad)
    ref_outputs = reference.compute_mappings(x)
    actual = candidate.compute_mappings(y) if operation is None else operation(y)
    for output, expected in zip(actual, ref_outputs):
        _exact(output, expected)
    grads = tuple(torch.randn_like(ref_outputs[index]) for index in used)
    torch.autograd.backward(tuple(ref_outputs[index] for index in used), grads)
    torch.autograd.backward(tuple(actual[index] for index in used), grads)
    _exact(y.grad, x.grad)
    for parameter, expected in zip(candidate.parameters(), reference.parameters()):
        _exact(parameter.grad, expected.grad)
    return actual


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("precision", [(False, False), (True, False), (False, True), (True, True)])
def test_mhc_mapping_graph_exact_updates(dtype, precision):
    """Repeated first-order VJPs consume current inputs, weights, and upstream gradients."""
    reference, candidate, sample = _pair(dtype, precision)
    original_keys = tuple(reference.state_dict())
    for update in range(3):
        outputs = _compare(reference, candidate, sample * (update + 1))
        saved = tuple(t.detach().clone() for t in outputs)
        with torch.no_grad():
            for module in (reference, candidate):
                for parameter in module.parameters():
                    parameter.add_(parameter.grad, alpha=-0.001)
            expected = reference.compute_mappings(sample * 2)
            actual = candidate.compute_mappings(sample * 2)
            for value, target in zip(actual, expected):
                _exact(value, target)
        for value, target in zip(outputs, saved):
            _exact(value, target)
        reference.zero_grad(set_to_none=True)
        candidate.zero_grad(set_to_none=True)
    assert tuple(candidate.state_dict()) == original_keys
    assert candidate._mapping_cuda_graph.replays["forward"] == 3
    assert candidate._mapping_cuda_graph.replays["backward"] == 3
    assert candidate._mapping_cuda_graph.replays["evaluation"] == 3


@pytest.mark.parametrize("batch", [1, 2, 4])
@pytest.mark.parametrize("streams", [2, 4])
def test_mhc_mapping_graph_native_shapes(batch, streams):
    reference, candidate, sample = _pair(batch=batch, streams=streams)
    _compare(reference, candidate, sample)
    assert candidate._mapping_cuda_graph.replays["backward"] == 1


@pytest.mark.parametrize("scale", [0.0, 0.0001, 3.0])
def test_mhc_mapping_graph_input_ranges(scale):
    reference, candidate, sample = _pair(precision=(True, True))
    _compare(reference, candidate, sample * scale)
    assert candidate._mapping_cuda_graph.replays["backward"] == 1


@pytest.mark.parametrize("used", [(0,), (1,), (2,), (0, 2)])
def test_mhc_mapping_graph_unused_outputs(used):
    """Unused branches preserve absent parameter gradients rather than materializing zeros."""
    reference, candidate, sample = _pair()
    _compare(reference, candidate, sample, used=used)
    assert candidate._mapping_cuda_graph.replays["backward_fallback"] == 1


@pytest.mark.parametrize("input_grad", [False, True])
@pytest.mark.parametrize("frozen", [("mapping_proj.weight",), ("alpha_pre", "alpha_res", "bias")])
def test_mhc_mapping_graph_frozen_parameters(input_grad, frozen):
    reference, candidate, sample = _pair(frozen=frozen)
    _compare(reference, candidate, sample, requires_grad=input_grad)
    assert candidate._mapping_cuda_graph.replays["fallback"] == int(not input_grad)


@pytest.mark.parametrize("reentrant", [False, True])
def test_mhc_mapping_graph_checkpoint(reentrant):
    reference, candidate, sample = _pair()
    _compare(
        reference,
        candidate,
        sample,
        operation=lambda x: checkpoint(candidate.compute_mappings, x, use_reentrant=reentrant),
    )
    graph = candidate._mapping_cuda_graph
    assert graph.replays["backward"] == int(reentrant)
    assert bool(graph.replays["fallback"]) == (not reentrant)


@pytest.mark.parametrize(
    "change", ["shape", "dtype", "stride", "parameter", "configuration", "frozen"]
)
def test_mhc_mapping_graph_changed_metadata_requires_recapture(change):
    reference, candidate, sample = _pair()
    if change == "shape":
        sample = sample[:4]
    elif change == "dtype":
        sample = sample.float()
    elif change == "stride":
        sample = torch.stack((sample, sample), dim=-1)[..., 0]
    elif change == "parameter":
        for module in (reference, candidate):
            module.mapping_proj.weight = torch.nn.Parameter(
                module.mapping_proj.weight.detach().clone()
            )
    elif change == "configuration":
        reference.sinkhorn_iterations += 1
        candidate.sinkhorn_iterations += 1
    elif change == "frozen":
        reference.bias.requires_grad_(False)
        candidate.bias.requires_grad_(False)
    graph = candidate._mapping_cuda_graph
    with pytest.raises(RuntimeError, match="metadata changed"):
        candidate.compute_mappings(sample.requires_grad_())
    assert not any(graph.replays.values())
    candidate.disable_mapping_cuda_graphs()
    assert candidate._mapping_cuda_graph is None


def test_mhc_mapping_graph_outstanding_forwards_and_evaluation():
    """Additional microbatches and evaluation cannot overwrite a pending training context."""
    reference, candidate, sample = _pair()
    ref_inputs = [sample.clone().requires_grad_(), (sample * 2).requires_grad_()]
    inputs = [x.detach().clone().requires_grad_() for x in ref_inputs]
    expected = [reference.compute_mappings(x) for x in ref_inputs]
    actual = [candidate.compute_mappings(x) for x in inputs]
    with pytest.raises(RuntimeError, match="pending"):
        candidate.disable_mapping_cuda_graphs()
    with torch.inference_mode():
        for a, b in zip(
            candidate.compute_mappings(sample * 3), reference.compute_mappings(sample * 3)
        ):
            _exact(a, b)
    for index in (1, 0):
        incoming = tuple(torch.randn_like(y) for y in expected[index])
        torch.autograd.backward(expected[index], incoming)
        torch.autograd.backward(actual[index], incoming)
        _exact(inputs[index].grad, ref_inputs[index].grad)
    for p, q in zip(candidate.parameters(), reference.parameters()):
        _exact(p.grad, q.grad)
    assert candidate._mapping_cuda_graph.replays["fallback"] == 2
    candidate.disable_mapping_cuda_graphs()
    assert candidate._mapping_cuda_graph is None


def test_mhc_mapping_graph_released_without_backward():
    _, candidate, sample = _pair()
    sample.requires_grad_()
    output = candidate.compute_mappings(sample)
    del output
    gc.collect()
    candidate.compute_mappings(sample)
    assert candidate._mapping_cuda_graph.replays["forward"] == 2


@pytest.mark.parametrize("autocast", [False, True])
def test_mhc_mapping_graph_streams_and_autocast(autocast):
    with torch.autocast("cuda", enabled=autocast, dtype=torch.bfloat16):
        reference, candidate, sample = _pair()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.autocast("cuda", enabled=autocast, dtype=torch.bfloat16):
            a = reference.compute_mappings(sample.requires_grad_())
            x = sample.detach().clone().requires_grad_()
            b = candidate.compute_mappings(x)
        incoming = tuple(torch.randn_like(y) for y in a)
        torch.autograd.backward(a, incoming)
        torch.autograd.backward(b, incoming)
    torch.cuda.current_stream().wait_stream(stream)
    for p, q in zip(reference.parameters(), candidate.parameters()):
        _exact(p.grad, q.grad)
    _exact(sample.grad, x.grad)


def test_mhc_mapping_graph_parameter_hooks_and_prepare_state():
    reference, candidate, sample = _pair()
    candidate.disable_mapping_cuda_graphs()
    calls = {name: 0 for name, _ in candidate.named_parameters()}

    def record(name, grad):
        calls[name] += 1
        return grad

    for name, parameter in candidate.named_parameters():
        parameter.register_hook(lambda grad, name=name: record(name, grad))
    for expected, parameter in zip(reference.parameters(), candidate.parameters()):
        expected.grad = torch.randn_like(expected)
        parameter.grad = expected.grad.clone()
    rng = torch.cuda.get_rng_state()
    candidate.enable_mapping_cuda_graphs(sample)
    assert torch.equal(rng, torch.cuda.get_rng_state())
    assert not any(calls.values())
    for expected, parameter in zip(reference.parameters(), candidate.parameters()):
        _exact(parameter.grad, expected.grad)
    _compare(reference, candidate, sample)
    assert all(value == 1 for value in calls.values())


@pytest.mark.parametrize(
    "case",
    ["cpu", "empty", "stride", "shape", "dtype", "fused", "outer_graph", "warmup", "variable"],
)
def test_mhc_mapping_graph_prepare_guards(case):
    _, candidate, sample = _pair(prepare=False)
    warmup = 3
    if case == "cpu":
        sample = sample.cpu()
    elif case == "empty":
        sample = sample[:0]
    elif case == "stride":
        sample = torch.stack((sample, sample), dim=-1)[..., 0]
    elif case == "shape":
        sample = torch.randn(8, 1, 12, device="cuda", dtype=sample.dtype)
    elif case == "dtype":
        sample = sample.to(torch.int64)
    elif case == "fused":
        candidate.config.use_fused_mhc = True
    elif case == "outer_graph":
        candidate.config.cuda_graph_impl = "local"
    elif case == "warmup":
        warmup = 0
    elif case == "variable":
        candidate.config.variable_seq_lengths = True
    with pytest.raises(ValueError):
        candidate.enable_mapping_cuda_graphs(sample, warmup_steps=warmup)
    assert candidate._mapping_cuda_graph is None


def test_mhc_mapping_graph_prepare_restores_parameter_bindings(monkeypatch):
    _, candidate, sample = _pair(prepare=False)
    parameters = tuple(candidate.parameters())
    original = candidate._compute_mappings

    def failed_warmup(_):
        raise RuntimeError("injected warmup failure")

    monkeypatch.setattr(candidate, "_compute_mappings", failed_warmup)
    with pytest.raises(RuntimeError, match="injected warmup"):
        candidate.enable_mapping_cuda_graphs(sample)
    assert all(actual is expected for actual, expected in zip(candidate.parameters(), parameters))
    assert candidate._mapping_cuda_graph is None
    monkeypatch.setattr(candidate, "_compute_mappings", original)
    candidate.enable_mapping_cuda_graphs(sample)


@pytest.mark.parametrize("context", ["inference", "saved_tensors"])
def test_mhc_mapping_graph_prepare_context_guards(context):
    _, candidate, sample = _pair(prepare=False)
    scope = (
        torch.inference_mode()
        if context == "inference"
        else torch.autograd.graph.saved_tensors_hooks(lambda tensor: tensor, lambda tensor: tensor)
    )
    with scope, pytest.raises(ValueError):
        candidate.enable_mapping_cuda_graphs(sample)
    assert candidate._mapping_cuda_graph is None


def test_mhc_mapping_graph_noncurrent_device():
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two visible CUDA devices")
    device = torch.cuda.current_device()
    other = (device + 1) % torch.cuda.device_count()
    reference, candidate, sample = _pair(prepare=False)
    reference = reference.cuda(other)
    candidate = candidate.cuda(other)
    sample = sample.cuda(other)
    candidate.enable_mapping_cuda_graphs(sample)
    _compare(reference, candidate, sample)
    assert torch.cuda.current_device() == device


def test_mhc_mapping_graph_weight_version_guard():
    _, candidate, sample = _pair()
    output = candidate.compute_mappings(sample.requires_grad_())
    with torch.no_grad():
        candidate.bias.add_(0.1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        sum(x.sum() for x in output).backward()


class _MappingsOnly(torch.nn.Module):
    """Exercise ordinary DDP/optimizer hooks outside the captured operation."""

    def __init__(self, module, checkpointed=False) -> None:
        super().__init__()
        self.module = module
        self.checkpointed = checkpointed

    def forward(self, x):
        if self.checkpointed:
            return checkpoint(self.module.compute_mappings, x, use_reentrant=True)
        return self.module.compute_mappings(x)


@pytest.mark.parametrize("checkpointed", [False, True])
def test_mhc_mapping_graph_pytorch_ddp_falls_back(checkpointed):
    """Keep ordinary DDP's native bucket order, including backward recomputation."""
    reference, candidate, sample = _pair()
    candidate.disable_mapping_cuda_graphs()
    kwargs = {"device_ids": [torch.cuda.current_device()], "broadcast_buffers": False}
    ref_ddp = torch.nn.parallel.DistributedDataParallel(
        _MappingsOnly(reference, checkpointed), **kwargs
    )
    ddp = torch.nn.parallel.DistributedDataParallel(
        _MappingsOnly(candidate, checkpointed), **kwargs
    )
    candidate.enable_mapping_cuda_graphs(sample)
    optimizers = [torch.optim.SGD(model.parameters(), lr=0.001) for model in (ref_ddp, ddp)]
    for step in range(3):
        for microbatch in range(2):
            x = sample * (1 + int(os.environ.get("RANK", "0")) + step + microbatch)
            a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
            expected, actual = ref_ddp(a), ddp(b)
            for y, z in zip(expected, actual):
                _exact(y, z)
            grads = tuple(torch.randn_like(y) for y in expected)
            torch.autograd.backward(expected, grads)
            torch.autograd.backward(actual, grads)
            _exact(a.grad, b.grad)
        for p, q in zip(reference.parameters(), candidate.parameters()):
            _exact(p.grad, q.grad)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
    assert candidate._mapping_cuda_graph.replays["backward"] == 0
    assert candidate._mapping_cuda_graph.replays["fallback"] == (12 if checkpointed else 6)


class _NativeBlock(torch.nn.Module):
    """Keep the mapping-to-MLP-to-residual gradient join outside graph capture."""

    def __init__(self, module: HyperConnectionModule) -> None:
        super().__init__()
        self.module = module
        self.linear = torch.nn.Linear(16, 16, dtype=torch.bfloat16, device="cuda")

    def forward(self, x, managed=False):
        manager = CheckpointWithoutOutputManager() if managed else None
        aggregate, h_res, h_post, residual = self.module(
            x, mhc_recompute_manager=manager, return_residual=True
        )
        output = self.module.fused_h_res_h_post_bda(
            h_res, residual, h_post, (self.linear(aggregate), None), 0.0, self.training, False
        )
        if manager is not None:
            manager.discard_all_outputs_and_register_unified_recompute(output)
        return output


@pytest.mark.parametrize("mode", ["direct", "reentrant", "managed"])
def test_mhc_mapping_graph_full_block_gradient_join(mode):
    reference, candidate, sample = _pair(precision=(True, True))
    ref_block = _NativeBlock(reference)
    block = _NativeBlock(candidate)
    block.linear.load_state_dict(ref_block.linear.state_dict())
    for step in range(3):
        a, b = (sample * (step + 1)).requires_grad_(), (sample * (step + 1)).requires_grad_()

        def run(model, x):
            if mode == "reentrant":
                return checkpoint(model, x, use_reentrant=True)
            return model(x, managed=mode == "managed")

        expected, actual = run(ref_block, a), run(block, b)
        _exact(actual, expected)
        incoming = torch.randn_like(actual)
        expected.backward(incoming)
        actual.backward(incoming)
        _exact(a.grad, b.grad)
        for p, q in zip(ref_block.parameters(), block.parameters()):
            _exact(p.grad, q.grad)
            with torch.no_grad():
                p.add_(p.grad, alpha=-0.001)
                q.add_(q.grad, alpha=-0.001)
        ref_block.zero_grad(set_to_none=True)
        block.zero_grad(set_to_none=True)
    assert candidate._mapping_cuda_graph.replays["backward"] == 3


@pytest.mark.parametrize("overlap", [False, True])
@pytest.mark.parametrize("mixed_execution", [False, True])
def test_mhc_mapping_graph_megatron_ddp(overlap, mixed_execution):
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    reference, candidate, sample = _pair()
    candidate.disable_mapping_cuda_graphs()
    config = DistributedDataParallelConfig(overlap_grad_reduce=overlap, bucket_size=1_000_000)
    ref_ddp = DistributedDataParallel(reference.config, config, _MappingsOnly(reference))
    ddp = DistributedDataParallel(candidate.config, config, _MappingsOnly(candidate))
    ref_ddp.zero_grad_buffer()
    ddp.zero_grad_buffer()
    candidate.enable_mapping_cuda_graphs(sample, ddp_model=ddp)
    assert all(torch.count_nonzero(p.main_grad) == 0 for p in candidate.parameters())
    for step in range(3):
        for microbatch in range(2):
            x = sample * (1 + int(os.environ.get("RANK", "0")) + step + microbatch)
            a, b = x.clone().requires_grad_(), x.clone().requires_grad_()
            native = mixed_execution and microbatch == 1 and Utils.rank % 2
            hooks = (
                torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t)
                if native
                else nullcontext()
            )
            with hooks:
                if microbatch == 0:
                    with ref_ddp.no_sync(), ddp.no_sync():
                        expected, actual = ref_ddp(a), ddp(b)
                        incoming = tuple(torch.randn_like(y) for y in expected)
                        torch.autograd.backward(expected, incoming)
                        torch.autograd.backward(actual, incoming)
                else:
                    expected, actual = ref_ddp(a), ddp(b)
                    incoming = tuple(torch.randn_like(y) for y in expected)
                    torch.autograd.backward(expected, incoming)
                    torch.autograd.backward(actual, incoming)
            for value, target in zip(actual, expected):
                _exact(value, target)
            _exact(a.grad, b.grad)
        ref_ddp.finish_grad_sync()
        ddp.finish_grad_sync()
        for p, q in zip(reference.parameters(), candidate.parameters()):
            assert p.grad is None and q.grad is None
            _exact(p.main_grad, q.main_grad)
            with torch.no_grad():
                p.add_(p.main_grad, alpha=-0.001)
                q.add_(q.main_grad, alpha=-0.001)
        ref_ddp.zero_grad_buffer()
        ddp.zero_grad_buffer()
    fallbacks = 3 if mixed_execution and Utils.rank % 2 else 0
    assert candidate._mapping_cuda_graph.replays["backward"] == 6 - fallbacks
    assert candidate._mapping_cuda_graph.replays["fallback"] == fallbacks


@pytest.mark.parametrize(
    "case",
    ["owner_missing", "wrong_owner", "multi_bucket", "overlap_param_gather", "grad_buffer_offload"],
)
def test_mhc_mapping_graph_megatron_ddp_prepare_guards(case, monkeypatch):
    """Reject unsupported storage/collective lifecycles before any graph capture."""
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    reference, candidate, sample = _pair(prepare=False)
    config = DistributedDataParallelConfig(
        overlap_grad_reduce=case == "multi_bucket", bucket_size=128
    )
    ddp = DistributedDataParallel(candidate.config, config, _MappingsOnly(candidate))
    ddp.zero_grad_buffer()
    owner = ddp
    if case == "owner_missing":
        owner = None
    elif case == "wrong_owner":
        owner = DistributedDataParallel(
            reference.config, copy.deepcopy(config), _MappingsOnly(reference)
        )
    elif case == "multi_bucket":
        assert len(ddp.bucket_groups) > 1
    else:
        monkeypatch.setattr(ddp.ddp_config, case, True)
    with pytest.raises(ValueError):
        candidate.enable_mapping_cuda_graphs(sample, ddp_model=owner)
    assert candidate._mapping_cuda_graph is None
    assert all(torch.count_nonzero(p.main_grad) == 0 for p in candidate.parameters())
