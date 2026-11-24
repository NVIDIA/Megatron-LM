# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Compare physical expert replication against an owner-only BF16 MoK backward."""

import os

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _RoutedMoK(torch.nn.Module):
    """Register expert weights with DDP and pass the same logical routes to both backends."""

    def __init__(self, layer):
        super().__init__()
        self.experts = layer.experts
        self.megakernel_experts = layer.megakernel_experts

    def forward(self, x, gates, mask):
        return self.megakernel_experts(x, gates, mask)


def _module(groups, replicas):
    config = TransformerConfig(
        num_layers=1,
        hidden_size=256,
        num_attention_heads=4,
        num_moe_experts=8,
        expert_model_parallel_size=4,
        expert_tensor_parallel_size=1,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_router_dtype="fp32",
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_token_dispatcher_type="flex",
        moe_flex_dispatcher_backend="hybridep",
        moe_megakernel_backend="mok",
        moe_virtual_expert_load_balance=replicas,
        moe_megakernel_backend_config=dict(
            fwd_num_comm_sms=12,
            bwd_num_comm_sms=12,
            minibatch_size=256,
            macrobatch_size=4096,
            schedule_capacity_multiplier=2.0,
        ),
        gated_linear_unit=True,
        activation_func=F.silu,
        activation_func_tanh_clamp_scale=4.0,
        activation_func_tanh_clamp_scale_linear=25.0,
        add_bias_linear=False,
        gradient_accumulation_fusion=True,
        params_dtype=torch.bfloat16,
        bf16=True,
        attention_dropout=0,
        hidden_dropout=0,
    )
    specs = get_gpt_layer_with_transformer_engine_submodules(num_experts=8, moe_grouped_gemm=True)
    layer = MoELayer(config, get_submodules(specs.mlp), pg_collection=groups, layer_number=1)
    model = _RoutedMoK(layer.cuda())
    generator = torch.Generator(device="cuda").manual_seed(19 + groups.ep.rank())
    with torch.no_grad():
        for param in model.parameters():
            param.copy_(torch.randn(param.shape, generator=generator, device="cuda") / 16)
    return DistributedDataParallel(
        config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model, pg_collection=groups
    )


def _compare(actual, expected):
    """Bound both distributed accumulation error and the largest individual deviation."""
    a, b = actual.float(), expected.float()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    diff = a - b
    assert diff.norm() <= 0.005 * b.norm()
    assert diff.abs().max() <= 0.01 * b.abs().max()


@pytest.mark.launch_on_gb200
def test_mok_replicas_reduce_bf16_gradients_on_their_logical_owners(monkeypatch):
    """Exercise actual replica routing and weight publication during checkpoint replay."""
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in ((10, 0), (10, 3))
        or int(os.environ.get("WORLD_SIZE", 1)) != 4
    ):
        pytest.skip("requires four SM100/SM103 GPUs in one NVLink domain")
    functional = pytest.importorskip("mok.functional")
    from megatron.core.transformer.moe.mok_virtual_experts import MoKVirtualExperts

    counts = []
    plan = MoKVirtualExperts.plan

    def observe(module, routes):
        p = plan(module, routes)
        width = module.num_local_experts
        counts.append((p.virtual_experts.remainder(2 * width) >= width).sum())
        return p

    monkeypatch.setattr(MoKVirtualExperts, "plan", observe)
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    try:
        model_parallel_cuda_manual_seed(19)
        groups = ProcessGroupCollection.use_mpu_process_groups()
        owner, replica = _module(groups, False), _module(groups, True)
        generator = torch.Generator(device="cuda").manual_seed(181 + groups.ep.rank())
        x = torch.randn(1024, 256, generator=generator, device="cuda", dtype=torch.bfloat16)
        upstream = torch.randn(x.shape, generator=generator, device="cuda", dtype=x.dtype) / 8
        # Every rank sends its tokens to the two experts owned by EP rank zero.
        gates = torch.zeros(1024, 8, device="cuda")
        gates[:, 0], gates[:, 1] = 0.25, 0.75
        out = []
        for model in (owner, replica):
            model.zero_grad_buffer()
            hidden, weights = x.detach().requires_grad_(), gates.detach().requires_grad_()
            y = checkpoint(model, hidden, weights, gates.bool(), use_reentrant=True)
            y.backward(upstream)
            model.finish_grad_sync()
            out.append((y.detach(), hidden.grad, weights.grad))
        for actual, expected in zip(out[1], out[0], strict=True):
            _compare(actual, expected)
        for (n, a), (m, b) in zip(
            replica.module.named_parameters(), owner.module.named_parameters(), strict=True
        ):
            assert n == m
            _compare(a.main_grad, b.main_grad)
        # The original checkpoint forward and its grad-enabled replay both routed remote slots.
        assert len(counts) == 2
        physical = torch.stack(counts)
        torch.distributed.all_reduce(physical, group=groups.ep)
        assert (physical > 0).all()
    finally:
        functional.clear_workspace_cache()
        Utils.destroy_model_parallel()
