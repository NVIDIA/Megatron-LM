# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Scope of the delayed-wgrad post-accumulate hooks in the fine-grained EP-overlap schedule.

``TransformerLayerNode.backward_dw`` fires ``post_wgrad_grad_acc_hook`` for the parameters of
its own slot's wgrad callables. ``MoELayer.backward_dw`` computes only half of the MoE layer's
wgrads, so the mlp and pre-dispatch slots have to partition the delayed-wgrad parameters
between them; a slot that claims more than it computes reduces a gradient that does not exist
yet.
"""

from collections import Counter

import pytest
import torch

from megatron.core.models.common.model_chunk_schedule_plan import TransformerLayerSchedulePlan
from megatron.core.models.gpt.fine_grained_callables import build_transformer_layer_callables
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.pipeline_parallel.utils import get_comm_stream, get_comp_stream, set_streams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import DummyState
from tests.unit_tests.test_utilities import Utils

EXPERT_PARALLEL_SIZE = 2
NUM_EXPERTS = 4
HIDDEN_SIZE = 64
LATENT_SIZE = 32
SHARED_EXPERT_SIZE = 64

# Submodules whose ``backward_dw`` produces the wgrads of every parameter they own.
_WGRAD_LEAF_PATHS = (
    "self_attention",
    "mlp.experts",
    "mlp.shared_experts",
    "mlp.fc1_latent_proj",
    "mlp.fc2_latent_proj",
)


def _build_config(moe_latent_size=None, shared_expert_size=None):
    """Smallest config that reaches the delayed-wgrad fine-grained schedule."""
    kwargs = dict(
        num_layers=1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=4,
        ffn_hidden_size=HIDDEN_SIZE,
        moe_ffn_hidden_size=HIDDEN_SIZE,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=2,
        moe_grouped_gemm=True,
        moe_token_dispatcher_type="alltoall",
        expert_model_parallel_size=EXPERT_PARALLEL_SIZE,
        overlap_moe_expert_parallel_comm=True,
        delay_wgrad_compute=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        add_bias_linear=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    if moe_latent_size is not None:
        kwargs["moe_latent_size"] = moe_latent_size
    if shared_expert_size is not None:
        kwargs["moe_shared_expert_intermediate_size"] = shared_expert_size
    return TransformerConfig(**kwargs)


def _build_layer(config, num_experts):
    """Build a single TransformerLayer; ``num_experts=None`` gives a dense layer."""
    submodules = get_gpt_layer_with_transformer_engine_submodules(
        num_experts=num_experts, moe_grouped_gemm=config.moe_grouped_gemm
    )
    return TransformerLayer(config, submodules).cuda()


def _wgrad_leaf_modules(layer):
    """Map leaf-module name to module for the leaves present in this layer."""
    leaves = {}
    for path in _WGRAD_LEAF_PATHS:
        module = layer
        for attr in path.split("."):
            module = getattr(module, attr, None)
            if module is None:
                break
        if module is not None:
            leaves[path] = module
    return leaves


def _param_ids(modules):
    ids = set()
    for module in modules:
        ids.update(id(param) for param in module.parameters())
    return ids


def _claimed_param_ids(bwd_dw_callable):
    """Parameters a schedule slot collects post-wgrad hooks from, in collection order."""
    get_wgrad_params = getattr(bwd_dw_callable, "wgrad_parameters", bwd_dw_callable.parameters)
    return [id(param) for param in get_wgrad_params()]


class _WgradRecorder:
    """Stub out each leaf's ``backward_dw`` and its parameters' hooks onto one event log."""

    def __init__(self, leaves):
        self.leaves = leaves
        self.events = []
        for name, module in leaves.items():
            module.backward_dw = self._wgrad_recorder(name)

    def _wgrad_recorder(self, name):
        def record():
            self.events.append(("wgrad", name))

        return record

    def _hook_recorder(self, name):
        def record():
            self.events.append(("hook", name))

        return record

    def install_hooks(self):
        """Mark every leaf parameter as delayed-wgrad with a populated gradient.

        Collection skips parameters whose ``.grad`` is ``None``, so a freshly built layer
        would record no hooks at all and hide the bug.
        """
        for name, module in self.leaves.items():
            for param in module.parameters():
                param.grad = torch.zeros_like(param)
                param.post_wgrad_grad_acc_hook = self._hook_recorder(name)

    def recorded_wgrads(self):
        return {name for kind, name in self.events if kind == "wgrad"}

    def assert_hooks_follow_their_wgrad(self):
        computed = set()
        for kind, name in self.events:
            if kind == "wgrad":
                computed.add(name)
            elif name not in computed:
                pytest.fail(
                    f"post-wgrad hook for '{name}' fired before its wgrad; events: {self.events}"
                )

    def assert_every_hook_fired_once(self):
        fired = Counter(name for kind, name in self.events if kind == "hook")
        expected = Counter(
            {name: len(list(module.parameters())) for name, module in self.leaves.items()}
        )
        assert fired == expected, (
            "every delayed-wgrad parameter must have its gradient reduced exactly once, "
            f"got {dict(fired)} instead of {dict(expected)}"
        )


@pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
class TestDelayedWgradHookScope:
    """The two delayed-wgrad slots must claim exactly the parameters they compute."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=EXPERT_PARALLEL_SIZE,
        )
        model_parallel_cuda_manual_seed(123)
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_plan(self, layer):
        return TransformerLayerSchedulePlan(
            layer, torch.cuda.Event(), DummyState(), get_comp_stream, get_comm_stream, extra_args={}
        )

    @pytest.mark.parametrize("moe_latent_size", [None, LATENT_SIZE])
    @pytest.mark.parametrize("shared_expert_size", [None, SHARED_EXPERT_SIZE])
    def test_slots_partition_delayed_wgrad_parameters(self, moe_latent_size, shared_expert_size):
        """Each slot claims exactly the parameters its own ``backward_dw`` computes."""
        layer = _build_layer(_build_config(moe_latent_size, shared_expert_size), NUM_EXPERTS)
        leaves = _wgrad_leaf_modules(layer)
        recorder = _WgradRecorder(leaves)
        _, backward_dw = build_transformer_layer_callables(layer)

        claimed = {slot: _claimed_param_ids(c) for slot, c in backward_dw.items()}
        for slot, param_ids in claimed.items():
            assert len(param_ids) == len(set(param_ids)), f"'{slot}' claims a parameter twice"
        assert set(claimed["mlp"]).isdisjoint(
            claimed["pre_dispatch_computation"]
        ), "a parameter is claimed by both slots, so its gradient is reduced twice"

        computed = {}
        for slot, bwd_dw_callable in backward_dw.items():
            recorder.events.clear()
            bwd_dw_callable.backward_dw()
            computed[slot] = recorder.recorded_wgrads()

        assert computed["mlp"].isdisjoint(computed["pre_dispatch_computation"])
        assert computed["mlp"] | computed["pre_dispatch_computation"] == set(
            leaves
        ), "the two slots do not compute every delayed wgrad in the layer"
        for slot in backward_dw:
            assert set(claimed[slot]) == _param_ids(
                leaves[name] for name in computed[slot]
            ), f"'{slot}' claims parameters it does not compute, or vice versa"

    @pytest.mark.parametrize("moe_latent_size", [None, LATENT_SIZE])
    @pytest.mark.parametrize("shared_expert_size", [None, SHARED_EXPERT_SIZE])
    def test_no_hook_fires_before_its_wgrad(self, moe_latent_size, shared_expert_size):
        """Driving both slots in schedule order never reduces a gradient before it exists."""
        layer = _build_layer(_build_config(moe_latent_size, shared_expert_size), NUM_EXPERTS)
        recorder = _WgradRecorder(_wgrad_leaf_modules(layer))
        recorder.install_hooks()
        plan = self._build_plan(layer)

        # The order TransformerLayerSchedulePlan.run uses: mlp wgrad, then pre-dispatch wgrad.
        plan.mlp.backward_dw()
        plan.pre_dispatch_computation.backward_dw()
        torch.cuda.synchronize()

        recorder.assert_hooks_follow_their_wgrad()
        recorder.assert_every_hook_fired_once()

    def test_dense_mlp_callable_keeps_full_parameter_scope(self):
        """A callable without ``wgrad_parameters`` still has all its parameters collected."""
        layer = _build_layer(_build_config(), num_experts=None)
        assert not layer.is_moe_layer
        recorder = _WgradRecorder({"mlp": layer.mlp, "self_attention": layer.self_attention})
        recorder.install_hooks()
        _, backward_dw = build_transformer_layer_callables(layer)

        mlp_callable = backward_dw["mlp"]
        assert not hasattr(mlp_callable, "wgrad_parameters")
        assert _claimed_param_ids(mlp_callable) == [id(p) for p in layer.mlp.parameters()]

        plan = self._build_plan(layer)
        plan.mlp.backward_dw()
        plan.pre_dispatch_computation.backward_dw()
        torch.cuda.synchronize()

        recorder.assert_hooks_follow_their_wgrad()
        recorder.assert_every_hook_fired_once()
