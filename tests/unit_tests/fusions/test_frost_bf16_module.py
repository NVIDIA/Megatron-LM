# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Actual TEGroupedMLP/DDP training parity for the optional Frost expert backend."""

from contextlib import nullcontext

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.extensions.transformer_engine import (
    TEColumnParallelGroupedLinear,
    TERowParallelGroupedLinear,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.experts import GroupedMLPSubmodules, TEGroupedMLP
from tests.unit_tests.fusions.test_frost_bf16_experts import _config
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="Frost BF16 expert kernels require SM100",
)


def _interleave(value: torch.Tensor) -> torch.Tensor:
    gate, up = value.chunk(2, dim=0)
    return torch.stack(
        (gate.view(-1, 32, *value.shape[1:]), up.view(-1, 32, *value.shape[1:])), dim=1
    ).reshape_as(value)


def _canonical(value: torch.Tensor) -> torch.Tensor:
    blocks = value.view(-1, 2, 32, *value.shape[1:])
    return torch.cat(
        (blocks[:, 0].reshape(-1, *value.shape[1:]), blocks[:, 1].reshape(-1, *value.shape[1:])),
        dim=0,
    )


def _assert_error(actual: torch.Tensor, expected: torch.Tensor, limit: float) -> None:
    actual, expected = actual.detach().float(), expected.detach().float()
    assert torch.isfinite(actual).all()
    delta = actual - expected
    relative_l2 = delta.norm() / expected.norm().clamp_min(1e-20)
    max_scaled = delta.abs().max() / expected.abs().max().clamp_min(1e-20)
    assert relative_l2 <= limit, float(relative_l2)
    assert max_scaled <= 2 * limit, float(max_scaled)


def _make_arm(backend: str, experts: int = 2, hidden: int = 256, intermediate: int = 256):
    model_parallel_cuda_manual_seed(4122)
    config = _config(
        num_moe_experts=experts,
        hidden_size=hidden,
        ffn_hidden_size=intermediate,
        moe_ffn_hidden_size=intermediate,
        num_attention_heads=4,
        moe_bf16_expert_backend=backend,
        moe_mlp_glu_interleave_size=32 if backend == "frost" else None,
        bias_activation_fusion=True,
    )
    groups = ProcessGroupCollection.use_mpu_process_groups()
    module = TEGroupedMLP(
        experts,
        config,
        GroupedMLPSubmodules(TEColumnParallelGroupedLinear, TERowParallelGroupedLinear),
        groups,
    )
    ddp = DistributedDataParallel(
        config,
        DistributedDataParallelConfig(
            use_distributed_optimizer=False,
            overlap_grad_reduce=True,
            grad_reduce_in_fp32=True,
            average_in_collective=False,
        ),
        module,
        pg_collection=groups,
    )
    return module, ddp


def _run(module, ddp, inputs, counts, probabilities, upstream, inflight: bool):
    ddp.zero_grad_buffer()
    module.set_is_first_microbatch()
    pending, outputs = [], []
    for index, (value, prob, gradient) in enumerate(zip(inputs, probabilities, upstream)):
        x, p = value.detach().requires_grad_(), prob.detach().requires_grad_()
        context = nullcontext() if index == len(inputs) - 1 else ddp.no_sync()
        with context:
            y, bias = ddp(x, counts[index], p)
            assert bias is None
            before = y.detach().clone()
            if inflight:
                pending.append((index, x, p, y, before, gradient))
            else:
                y.backward(gradient)
                assert torch.equal(y, before)
                outputs.append((before, x.grad.clone(), p.grad.clone()))
    for index, x, p, y, before, gradient in pending:
        with nullcontext() if index == len(inputs) - 1 else ddp.no_sync():
            y.backward(gradient)
        assert torch.equal(y, before)
        outputs.append((before, x.grad.clone(), p.grad.clone()))
    ddp.finish_grad_sync()
    gradients = {name: p.main_grad.clone() for name, p in module.named_parameters()}
    return outputs, gradients


@pytest.mark.parametrize("inflight", [False, True])
@pytest.mark.parametrize("host_counts", [False, True])
def test_actual_grouped_mlp_accumulation(monkeypatch, inflight, host_counts):
    """Match real TE output/dX/dprob/main_grad with changing inputs and an empty expert."""
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_USE_FUSED_GROUPED_GEMM", "0")
    Utils.initialize_model_parallel()
    try:
        torch.manual_seed(4122)
        native, native_ddp = _make_arm("transformer_engine")
        frost, frost_ddp = _make_arm("frost")
        with torch.no_grad():
            for (name, source), (_, target) in zip(
                native.named_parameters(), frost.named_parameters()
            ):
                target.copy_(_interleave(source) if name.startswith("linear_fc1") else source)
        counts = [
            torch.tensor(x, dtype=torch.int64, device="cpu" if host_counts else "cuda")
            for x in ([257, 129], [386, 0])
        ]
        for scale in (1.0, 8.0):
            inputs = [
                torch.randn(386, 256, device="cuda", dtype=torch.bfloat16) * scale for _ in counts
            ]
            probs = [torch.rand(386, device="cuda", dtype=torch.float32) for _ in counts]
            upstream = [torch.randn_like(x) / 128 for x in inputs]
            reference = _run(native, native_ddp, inputs, counts, probs, upstream, inflight)
            actual = _run(frost, frost_ddp, inputs, counts, probs, upstream, inflight)
            for values, expected in zip(actual[0], reference[0]):
                for index, (a, b) in enumerate(zip(values, expected)):
                    _assert_error(a, b, 0.01 if index == 0 else 0.03)
            for name, expected in reference[1].items():
                value = actual[1][name]
                _assert_error(
                    _canonical(value) if name.startswith("linear_fc1") else value, expected, 0.03
                )
        owner = frost._frost_bf16_ops[0]
        assert owner.forward_calls == owner.backward_calls == 4
        assert owner.max_inflight == (2 if inflight else 1)
        assert len(owner.compiled_plans) == 6
        assert all(not plan.in_use for pool in owner.pools.values() for plan in pool)
        assert all(
            plan.a.untyped_storage().nbytes() == 0 for pool in owner.pools.values() for plan in pool
        )
    finally:
        Utils.destroy_model_parallel()
