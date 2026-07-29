# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.activations import squared_relu
from megatron.core.fusions.fused_bias_geglu import quick_gelu
from megatron.core.transformer.moe import fused_a2a
from megatron.core.transformer.moe.fused_a2a import moonep_combine, moonep_dispatch


class _FakeMoonEPBuffer:
    """CPU implementation of MoonEP's saved-plan dispatch/combine contract."""

    def dispatch(
        self,
        hidden,
        route_weights=None,
        topk_experts=None,
        tokens_per_expert=None,
        plan=None,
        *,
        zero_copy=False,
        zero_copy_weights=None,
        hidden_buffer=None,
    ):
        del zero_copy, zero_copy_weights, hidden_buffer
        if plan is None:
            num_tokens, topk = topk_experts.shape
            flat_experts = topk_experts.reshape(-1).long()
            order = torch.argsort(flat_experts, stable=True)
            source_tokens = torch.arange(num_tokens).repeat_interleave(topk)[order]
            source_routes = torch.arange(topk).repeat(num_tokens)[order]
            plan = SimpleNamespace(
                source_tokens=source_tokens,
                source_routes=source_routes,
                num_tokens=num_tokens,
                topk=topk,
                experts_to_copy=torch.full((1, tokens_per_expert.numel()), -1, dtype=torch.int32),
            )
            counts = torch.cat(
                [tokens_per_expert, tokens_per_expert.new_zeros(tokens_per_expert.numel())]
            )
            cu_seqlens = counts.cumsum(0)
        else:
            cu_seqlens = None

        dispatched_hidden = hidden[plan.source_tokens]
        dispatched_weights = (
            None if route_weights is None else route_weights[plan.source_tokens, plan.source_routes]
        )
        return dispatched_hidden, dispatched_weights, cu_seqlens, plan

    def combine(
        self, *, plan, hidden_nvsh, route_weights_nvs=None, zero_copy=False, hidden_buffer=None
    ):
        del zero_copy, hidden_buffer
        hidden = hidden_nvsh.new_zeros((plan.num_tokens, hidden_nvsh.shape[-1]))
        hidden.index_add_(0, plan.source_tokens, hidden_nvsh)
        weights = None
        if route_weights_nvs is not None:
            weights = route_weights_nvs.new_zeros((plan.num_tokens, plan.topk))
            weights[plan.source_tokens, plan.source_routes] = route_weights_nvs
        return hidden, weights, None


class _FakeWeightBridge:
    def __init__(self, device=None):
        self.parameters = (
            torch.nn.Parameter(torch.ones((), device=device)),
            torch.nn.Parameter(torch.ones((), device=device)),
        )
        self.last_plan = None
        self.reduced_plans = []
        self.prefetched_plans = []
        self.buffer = None

    @property
    def source_parameters(self):
        return self.parameters

    @property
    def dummy_grads(self):
        return tuple(torch.zeros_like(parameter) for parameter in self.parameters)

    def reduce_grads(self, plan):
        self.reduced_plans.append(plan)

    def prefetch(self, plan):
        self.prefetched_plans.append(plan)

    def attach_buffer(self, buffer):
        self.buffer = buffer


class _FakeDispatchBufferPool:
    def __init__(self):
        self.acquired = []
        self.released = []

    def acquire(self):
        pair = (object(), object())
        self.acquired.append(pair)
        return pair

    def release(self, pair):
        self.released.append(pair)


def _run_fake_moonep(hidden, probs, indices, bridge, buffer, dispatch_buffer_pool=None):
    tokens_per_expert = torch.bincount(indices.reshape(-1), minlength=int(indices.max()) + 1).to(
        torch.int32
    )
    dispatched, dispatched_probs, runtime_counts = moonep_dispatch(
        hidden, probs, indices, tokens_per_expert, buffer, bridge, dispatch_buffer_pool
    )
    expert_output = dispatched * dispatched_probs.unsqueeze(-1)
    output = moonep_combine(expert_output, buffer, bridge.last_plan, bridge)
    return output, runtime_counts


def test_moonep_autograd_wrappers_preserve_hidden_and_probability_gradients():
    buffer = _FakeMoonEPBuffer()
    bridge = _FakeWeightBridge()
    indices = torch.tensor([[0, 2], [1, 2], [0, 1]], dtype=torch.int32)
    hidden = torch.randn(3, 4, requires_grad=True)
    probs = torch.randn(3, 2, requires_grad=True)
    ref_hidden = hidden.detach().clone().requires_grad_(True)
    ref_probs = probs.detach().clone().requires_grad_(True)

    output, runtime_counts = _run_fake_moonep(hidden, probs, indices, bridge, buffer)
    ref_output = (ref_hidden.unsqueeze(1) * ref_probs.unsqueeze(2)).sum(dim=1)
    grad = torch.randn_like(output)
    output.backward(grad)
    ref_output.backward(grad)

    torch.testing.assert_close(output, ref_output)
    torch.testing.assert_close(hidden.grad, ref_hidden.grad)
    torch.testing.assert_close(probs.grad, ref_probs.grad)
    assert runtime_counts.numel() == 6  # E+B, with B=E for the one-rank fake.
    assert runtime_counts.sum() == indices.numel()
    assert list(map(id, bridge.prefetched_plans)) == list(map(id, bridge.reduced_plans))
    assert len(bridge.reduced_plans) == 1
    assert all(parameter.grad.item() == 0 for parameter in bridge.parameters)


def test_moonep_saved_plans_are_restored_for_multiple_outstanding_forwards():
    buffer = _FakeMoonEPBuffer()
    bridge = _FakeWeightBridge()
    dispatch_buffer_pool = _FakeDispatchBufferPool()
    indices = torch.tensor([[0, 1], [1, 2]], dtype=torch.int32)
    hidden_1 = torch.randn(2, 4, requires_grad=True)
    hidden_2 = torch.randn(2, 4, requires_grad=True)
    probs_1 = torch.randn(2, 2, requires_grad=True)
    probs_2 = torch.randn(2, 2, requires_grad=True)

    output_1, _ = _run_fake_moonep(hidden_1, probs_1, indices, bridge, buffer, dispatch_buffer_pool)
    plan_1 = bridge.last_plan
    output_2, _ = _run_fake_moonep(
        hidden_2, probs_2, indices.flip(0), bridge, buffer, dispatch_buffer_pool
    )
    plan_2 = bridge.last_plan
    assert len(dispatch_buffer_pool.acquired) == 2
    assert dispatch_buffer_pool.released == []
    (output_1.sum() + output_2.sum()).backward()

    assert set(map(id, bridge.prefetched_plans)) == {id(plan_1), id(plan_2)}
    assert set(map(id, bridge.reduced_plans)) == {id(plan_1), id(plan_2)}
    assert set(map(id, dispatch_buffer_pool.released)) == set(
        map(id, dispatch_buffer_pool.acquired)
    )


def test_moonep_manager_preserves_static_dispatch_capacity():
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager.dispatched_probs = torch.randn(12)
    hidden = torch.randn(12, 4)

    expert_hidden, expert_probs = manager.get_permuted_hidden_states_by_experts(hidden)

    assert expert_hidden.data_ptr() == hidden.data_ptr()
    assert expert_hidden.shape == (12, 4)
    assert expert_probs.data_ptr() == manager.dispatched_probs.data_ptr()


def test_moonep_manager_exposes_shared_expert_zero_copy_buffers():
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    output_buffer = torch.empty(12, 4)
    dgrad_buffer = torch.empty(12, 4)
    manager._zero_copy_token_buffers = SimpleNamespace(
        forward=(object(), output_buffer), backward=(object(), dgrad_buffer)
    )

    actual_output, actual_dgrad = manager.get_expert_zero_copy_buffers()

    assert actual_output.data_ptr() == output_buffer.data_ptr()
    assert actual_dgrad.data_ptr() == dgrad_buffer.data_ptr()


def test_moonep_metadata_uses_fixed_gpu_histogram(monkeypatch):
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager

    manager = _MoonEPManager.__new__(_MoonEPManager)
    manager.num_experts = 4
    manager.router_topk = 2
    probs = torch.tensor([[4.0, 3.0, 2.0, 1.0], [1.0, 4.0, 3.0, 2.0], [4.0, 1.0, 3.0, 2.0]])
    routing_map = torch.zeros_like(probs, dtype=torch.bool)

    def unexpected_bincount(*_args, **_kwargs):
        raise AssertionError("MoonEP metadata must not call torch.bincount")

    monkeypatch.setattr(torch, "bincount", unexpected_bincount)
    manager.setup_metadata(routing_map, probs)

    torch.testing.assert_close(
        manager.tokens_per_expert, torch.tensor([2, 2, 2, 0], dtype=torch.int32)
    )


def test_moonep_finalize_is_idempotent(monkeypatch):
    class _Resource:
        def __init__(self):
            self.destroy_calls = 0

        def destroy(self):
            self.destroy_calls += 1

    buffer = _Resource()
    bridge = _Resource()
    dispatch_pool = _Resource()
    token_pool = _Resource()
    token_buffers = {"test": token_pool}
    monkeypatch.setattr(fused_a2a, "_moonep_buffers", weakref.WeakSet([buffer]))
    monkeypatch.setattr(fused_a2a, "_moonep_bridges", weakref.WeakSet([bridge]))
    monkeypatch.setattr(
        fused_a2a, "_moonep_dispatch_buffer_pools", weakref.WeakSet([dispatch_pool])
    )
    monkeypatch.setattr(fused_a2a, "_moonep_token_buffer_pools", token_buffers)

    fused_a2a.moonep_finalize()
    fused_a2a.moonep_finalize()

    assert buffer.destroy_calls == 1
    assert bridge.destroy_calls == 1
    assert dispatch_pool.destroy_calls == 1
    assert token_pool.destroy_calls == 1
    assert token_buffers == {}


@pytest.mark.skipif(not fused_a2a.HAVE_MOONEP, reason="MoonEP is not installed")
def test_moonep_availability_helper():
    assert fused_a2a.is_moonep_available()


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_MOONEP,
    reason="CUDA and MoonEP are required",
)
def test_moonep_four_rank_dispatch_probability_grad_and_redundant_counts():
    """Run with ``torch.distributed.run --nproc_per_node=4`` on an NVLink node."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("MoonEP distributed coverage requires a 4-rank torchrun launch")

    from megatron.core import parallel_state
    from megatron.core.transformer.moe.token_dispatcher import _MoonEPManager
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=4, expert_tensor_parallel_size=1
    )
    group = parallel_state.get_expert_tensor_and_model_parallel_group()
    config = SimpleNamespace(hidden_size=128, moe_flex_dispatcher_num_sms=None)
    manager = _MoonEPManager(
        group=group, num_local_experts=1, router_topk=2, num_experts=4, config=config
    )
    manager._bridge = _FakeWeightBridge(device="cuda")

    try:
        num_tokens = 16
        hidden = torch.randn(num_tokens, 128, device="cuda", dtype=torch.bfloat16)
        hidden.requires_grad_(True)
        # Every rank strongly favors experts 0 and 1. Non-owner ranks must use
        # MoonEP's redundant B slot for at least one of those experts.
        logits = torch.full((num_tokens, 4), -8.0, device="cuda")
        logits[:, 0] = 8.0
        logits[:, 1] = 7.0
        logits.requires_grad_(True)
        dense_probs = torch.softmax(logits, dim=-1)
        _, indices = torch.topk(dense_probs, 2, dim=-1)
        routing_map = torch.zeros_like(dense_probs, dtype=torch.bool)
        routing_map.scatter_(1, indices, True)

        manager.setup_metadata(routing_map, dense_probs)
        dispatched = manager.dispatch(hidden)
        runtime_counts = manager.get_number_of_tokens_per_expert()
        valid_hidden, valid_probs = manager.get_permuted_hidden_states_by_experts(dispatched)
        expert_output = (valid_hidden * valid_probs.unsqueeze(-1)).to(hidden.dtype)
        expert_output = manager.get_restored_hidden_states_by_experts(expert_output)
        output = manager.combine(expert_output)

        expected = hidden * manager.token_probs.sum(dim=-1, keepdim=True).to(hidden.dtype)
        torch.testing.assert_close(output, expected)
        output.float().sum().backward()
        assert hidden.grad is not None
        assert logits.grad is not None and torch.count_nonzero(logits.grad) > 0
        assert runtime_counts.numel() == 5  # E+B with E=4 and B=1.
        slot_tokens = runtime_counts[4:].sum().to(torch.int64)
        torch.distributed.all_reduce(slot_tokens, group=group)
        assert slot_tokens.item() > 0
    finally:
        fused_a2a.moonep_finalize()
        Utils.destroy_model_parallel()


def _set_main_grad(parameter):
    rowwise_data = getattr(parameter, "rowwise_data", parameter)
    parameter.main_grad = torch.zeros_like(rowwise_data).view(parameter.shape)
    parameter.grad_added_to_main_grad = False
    parameter.overwrite_main_grad = True


def _set_main_grads(layer):
    for linear in (layer.experts.linear_fc1, layer.experts.linear_fc2):
        _set_main_grad(linear.get_parameter("weight"))
    if layer.config.moe_latent_size is not None:
        _set_main_grad(layer.fc1_latent_proj.weight)
        _set_main_grad(layer.fc2_latent_proj.weight)


@pytest.mark.internal
@pytest.mark.skipif(
    not torch.cuda.is_available() or not fused_a2a.HAVE_MOONEP,
    reason="CUDA and MoonEP are required",
)
@pytest.mark.parametrize(
    (
        "activation_func",
        "gated_linear_unit",
        "weighted_squared_relu",
        "glu_interleave",
        "moe_latent_size",
    ),
    [
        (F.silu, True, False, None, None),
        (F.silu, True, False, 128, None),
        (quick_gelu, True, False, None, None),
        (squared_relu, False, True, None, None),
        (F.silu, True, False, None, 512),
    ],
)
def test_moonep_full_layer_parity_with_alltoall(
    monkeypatch,
    activation_func,
    gated_linear_unit,
    weighted_squared_relu,
    glu_interleave,
    moe_latent_size,
):
    """Compare full expert/router fwd+bwd and grouped main_grads on 4 NVLink GPUs."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("MoonEP distributed coverage requires a 4-rank torchrun launch")

    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.transformer.moe.moe_layer import MoELayer
    from megatron.core.transformer.spec_utils import get_submodules
    from megatron.core.transformer.transformer_config import TransformerConfig
    from tests.unit_tests.test_utilities import Utils

    monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_DISABLE_CUTEDSL_WGRAD_FUSED_GROUPED_MLP", "1")
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, expert_model_parallel_size=4, expert_tensor_parallel_size=1
    )

    common = {
        "num_layers": 1,
        "hidden_size": 1024,
        "ffn_hidden_size": 1024,
        "moe_ffn_hidden_size": 1024,
        "num_attention_heads": 8,
        "num_moe_experts": 4,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
        "moe_router_topk": 2,
        "moe_router_load_balancing_type": "none",
        "moe_router_dtype": "fp32",
        "moe_grouped_gemm": True,
        "moe_single_grouped_weight": True,
        "use_transformer_engine_op_fuser": True,
        "gradient_accumulation_fusion": True,
        "add_bias_linear": False,
        "bf16": True,
        "params_dtype": torch.bfloat16,
        "use_cpu_initialization": False,
        "activation_func": activation_func,
        "gated_linear_unit": gated_linear_unit,
        "use_fused_weighted_squared_relu": weighted_squared_relu,
        "moe_mlp_glu_interleave_size": glu_interleave,
        "moe_latent_size": moe_latent_size,
    }
    alltoall_config = TransformerConfig(**common, moe_token_dispatcher_type="alltoall")
    moonep_config = TransformerConfig(
        **common, moe_token_dispatcher_type="flex", moe_flex_dispatcher_backend="moonep"
    )
    mlp_spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=4, moe_grouped_gemm=True
    ).submodules.mlp
    submodules = get_submodules(mlp_spec)

    try:
        ref_layer = MoELayer(alltoall_config, submodules).cuda()
        moonep_layer = MoELayer(moonep_config, submodules).cuda()
        moonep_layer.load_state_dict(ref_layer.state_dict())
        assert moonep_layer.state_dict().keys() == ref_layer.state_dict().keys()
        _set_main_grads(ref_layer)
        _set_main_grads(moonep_layer)

        torch.manual_seed(1234)
        test_input = torch.randn(2, 4, 1024, device="cuda", dtype=torch.bfloat16)

        def run(layer):
            hidden = test_input.detach().clone().requires_grad_(True)
            output, _ = layer(hidden)
            output.float().sum().backward()
            values = [
                output.detach(),
                hidden.grad.detach(),
                layer.router.weight.grad.detach().clone(),
                layer.experts.linear_fc1.weight.main_grad.detach().clone(),
                layer.experts.linear_fc2.weight.main_grad.detach().clone(),
            ]
            if layer.config.moe_latent_size is not None:
                values.extend(
                    [
                        layer.fc1_latent_proj.weight.main_grad.detach().clone(),
                        layer.fc2_latent_proj.weight.main_grad.detach().clone(),
                    ]
                )
            return values

        ref_values = run(ref_layer)
        moonep_values = run(moonep_layer)
        value_names = ["output", "input grad", "router grad", "FC1 main_grad", "FC2 main_grad"]
        if moe_latent_size is not None:
            value_names.extend(["latent FC1 main_grad", "latent FC2 main_grad"])
        for value_name, actual, expected in zip(value_names, moonep_values, ref_values):
            torch.testing.assert_close(
                actual, expected, rtol=2e-2, atol=2e-2, msg=lambda msg: f"{value_name}: {msg}"
            )
    finally:
        fused_a2a.moonep_finalize()
        Utils.destroy_model_parallel()
