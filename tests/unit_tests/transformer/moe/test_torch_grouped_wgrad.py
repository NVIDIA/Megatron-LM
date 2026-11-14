# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Numerical, autograd, dispatch and checkpoint contracts of grouped dW."""

import argparse
import copy
from unittest.mock import patch

import pytest
import torch
from torch.utils.checkpoint import checkpoint

pytest.importorskip("transformer_engine")

from megatron.core.extensions import transformer_engine as wrappers
from megatron.core.extensions.torch_grouped_wgrad import (
    _grouped_mm,
    torch_grouped_linear,
    torch_grouped_wgrad_eligible,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import init_method_normal
from megatron.training.arguments import _add_network_size_args
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or _grouped_mm() is None,
    reason="requires CUDA and PyTorch grouped matmul",
)


@pytest.fixture(autouse=True)
def model_groups():
    """Use the repository's real distributed initialization even on one GPU."""
    Utils.initialize_model_parallel()
    model_parallel_cuda_manual_seed(123)
    yield
    Utils.destroy_model_parallel()


def _layer(groups, enabled, *, bias=False, dtype=torch.bfloat16):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_moe_experts=groups,
        moe_grouped_gemm=True,
        moe_use_torch_grouped_wgrad=enabled,
        params_dtype=dtype,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_accumulation_fusion=False,
        use_cpu_initialization=False,
    )
    return wrappers.TEGroupedLinear(
        groups,
        64,
        128,
        parallel_mode="column",
        config=config,
        init_method=init_method_normal(0.02),
        bias=bias,
        skip_bias_add=False,
        is_expert=True,
        pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
    )


def _close(actual, expected):
    assert actual is not None and expected is not None
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.005)
    a, b = actual.float(), expected.float()
    assert (a - b).norm() <= 0.005 * b.norm().clamp_min(1e-8)


@pytest.mark.parametrize("counts", [[0, 1, 17, 129], [32, 32, 32, 32], [0, 0, 0, 0], [193]])
@pytest.mark.parametrize("recompute", [False, True])
def test_matches_native_te(counts, recompute):
    """Same TE weights and inputs match forward, dgrad and every expert dW."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("fast dispatch currently targets SM90")
    baseline = _layer(len(counts), False)
    candidate = _layer(len(counts), True)
    candidate.load_state_dict(copy.deepcopy(baseline.state_dict()))
    torch.manual_seed(1234)
    data = torch.randn(sum(counts), 64, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16) * 0.01
    splits = torch.tensor(counts, dtype=torch.int64)
    snapshots = []
    for module in (baseline, candidate):
        x = data.clone().requires_grad_()
        assert torch_grouped_wgrad_eligible(module, x, splits) == (module is candidate)
        with patch.object(wrappers, "torch_grouped_linear", wraps=torch_grouped_linear) as dispatch:
            call = lambda inp: module(inp, splits)[0]
            y = checkpoint(call, x, use_reentrant=False) if recompute else call(x)
            y.backward(dy)
        assert (dispatch.call_count > 0) == (module is candidate)
        snapshots.append((y.detach(), x.grad, [p.grad for p in module.parameters()]))
    _close(snapshots[1][0], snapshots[0][0])
    _close(snapshots[1][1], snapshots[0][1])
    for actual, expected in zip(snapshots[1][2], snapshots[0][2]):
        _close(actual, expected)


@pytest.mark.parametrize("input_grad", [False, True])
def test_outstanding_forwards_and_gradient_accumulation(input_grad):
    """Offsets and saved inputs belong to each forward, not to mutable module state."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("fast dispatch currently targets SM90")
    module = _layer(4, True)
    torch.manual_seed(937)
    inputs = []
    upstream = []
    outputs = []
    refs = [torch.zeros_like(w, dtype=torch.float32) for w in module.parameters()]
    splits_list = [[0, 1, 17, 129], [64, 0, 93, 1]]
    for counts in splits_list:
        x = torch.randn(
            sum(counts), 64, device="cuda", dtype=torch.bfloat16, requires_grad=input_grad
        )
        dy = torch.randn(sum(counts), 128, device="cuda", dtype=torch.bfloat16) * 0.01
        outputs.append(module(x, torch.tensor(counts))[0])
        inputs.append(x)
        upstream.append(dy)
        for ref, xp, yp in zip(refs, x.detach().split(counts), dy.split(counts)):
            ref.add_((yp.float().t() @ xp.float()).bfloat16().float())
    # A later config change cannot switch the backward of either saved graph.
    module.config.moe_use_torch_grouped_wgrad = False
    for y, dy in zip(outputs, upstream):
        y.backward(dy)
    for weight, ref in zip(module.parameters(), refs):
        _close(weight.grad, ref.bfloat16())
    assert all((x.grad is not None) == input_grad for x in inputs)


@pytest.mark.parametrize("bias,dtype", [(True, torch.bfloat16), (False, torch.float16)])
def test_unsupported_invocations_keep_te(bias, dtype):
    """Opting in cannot route unsupported bias or FP16 inputs to the fast path."""
    module = _layer(4, True, bias=bias, dtype=dtype)
    x = torch.randn(16, 64, device="cuda", dtype=dtype, requires_grad=True)
    counts = torch.tensor([0, 3, 5, 8])
    assert not torch_grouped_wgrad_eligible(module, x, counts)
    with patch.object(wrappers, "torch_grouped_linear", side_effect=AssertionError("must use TE")):
        y, _ = module(x, counts)
        y.sum().backward()
    assert all(p.grad is not None for p in module.parameters())


@pytest.mark.parametrize(
    "owner,name,value",
    [
        ("module", "fuse_wgrad_accumulation", True),
        ("module", "delay_wgrad_compute", True),
        ("module", "use_grouped_tensor", True),
        ("module", "single_grouped_weight", True),
        ("config", "cpu_offloading", True),
        ("config", "fine_grained_activation_offloading", True),
        ("config", "expert_tensor_parallel_size", 2),
        ("config", "expert_gtp_weight_remat_size", 2),
        ("config", "cuda_graph_impl", "local"),
    ],
)
def test_unsupported_configuration_guard(owner, name, value, monkeypatch):
    """Advanced training modes retain the original TE implementation."""
    module = _layer(4, True)
    monkeypatch.setattr(module if owner == "module" else module.config, name, value, raising=False)
    x = torch.empty(16, 64, device="cuda", dtype=torch.bfloat16)
    assert not torch_grouped_wgrad_eligible(module, x, torch.tensor([0, 3, 5, 8]))


def test_cuda_metadata_is_not_copied_to_host():
    """GPU-resident counts belong to TE's device-metadata implementation."""
    module = _layer(4, True)
    x = torch.empty(16, 64, device="cuda", dtype=torch.bfloat16)
    assert not torch_grouped_wgrad_eligible(module, x, torch.tensor([0, 3, 5, 8], device="cuda"))


def test_checkpoint_parameter_names_unchanged():
    """The fast path does not replace the TE-owned expert parameters."""
    baseline = _layer(4, False)
    candidate = _layer(4, True)
    state = copy.deepcopy(baseline.state_dict())
    candidate.load_state_dict(state, strict=True)
    assert list(candidate.state_dict()) == list(state)
    assert [n for n, _ in candidate.named_parameters()] == [f"weight{i}" for i in range(4)]


def test_invalid_counts_rejected():
    """The new entry point rejects a partition that does not match its input."""
    x = torch.empty(10, 64, device="cuda", dtype=torch.bfloat16)
    weights = [torch.empty(128, 64, device="cuda", dtype=torch.bfloat16)] * 2
    with pytest.raises(ValueError, match="partition"):
        torch_grouped_linear(x, [3, 3], weights)


def test_cli_flag_and_default():
    """The configuration field is exposed by the standard training parser."""
    parser = argparse.ArgumentParser()
    _add_network_size_args(parser)
    assert parser.parse_args([]).moe_use_torch_grouped_wgrad is False
    assert parser.parse_args(["--moe-use-torch-grouped-wgrad"]).moe_use_torch_grouped_wgrad is True


def test_fp16_autocast_keeps_te():
    """BF16 storage must not bypass an enclosing FP16 activation autocast."""
    module = _layer(4, True)
    x = torch.randn(16, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    counts = torch.tensor([0, 3, 5, 8])
    with torch.autocast("cuda", dtype=torch.float16):
        assert not torch_grouped_wgrad_eligible(module, x, counts)
        with patch.object(
            wrappers, "torch_grouped_linear", side_effect=AssertionError("must use TE")
        ):
            output, _ = module(x, counts)
    assert output.dtype == torch.float16
    output.sum().backward()


def test_debug_iteration_keeps_te(monkeypatch):
    """TE debugging must retain the original quantizer/inspection machinery."""
    module = _layer(4, True)
    monkeypatch.setattr(module, "is_debug_iter", lambda: True, raising=False)
    x = torch.empty(16, 64, device="cuda", dtype=torch.bfloat16)
    assert not torch_grouped_wgrad_eligible(module, x, torch.tensor([0, 3, 5, 8]))


def test_multidimensional_input_and_noncontiguous_gradient():
    """Flatten/restore preserves input shape and accepts strided upstream gradients."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 path")
    baseline = _layer(4, False)
    candidate = _layer(4, True)
    candidate.load_state_dict(copy.deepcopy(baseline.state_dict()))
    data = torch.randn(3, 49, 64, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(3, 128, 49, device="cuda", dtype=torch.bfloat16).transpose(1, 2) * 0.01
    counts = torch.tensor([0, 1, 17, 129])
    snapshots = []
    for module in (baseline, candidate):
        x = data.clone().requires_grad_()
        y, _ = module(x, counts)
        y.backward(dy)
        snapshots.append((y.detach(), x.grad, [p.grad for p in module.parameters()]))
    _close(snapshots[1][0], snapshots[0][0])
    _close(snapshots[1][1], snapshots[0][1])
    for a, b in zip(snapshots[1][2], snapshots[0][2]):
        _close(a, b)


def test_frozen_expert_weights_keep_none_gradients():
    """Mixed frozen/trainable experts retain autograd's None-gradient contract."""
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("SM90 path")
    module = _layer(4, True)
    module.weight0.requires_grad_(False)
    module.weight2.requires_grad_(False)
    x = torch.randn(147, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y, _ = module(x, torch.tensor([0, 1, 17, 129]))
    y.sum().backward()
    assert module.weight0.grad is None and module.weight2.grad is None
    assert module.weight1.grad is not None and module.weight3.grad is not None
    assert x.grad is not None
