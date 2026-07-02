# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt import moe_module_specs
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import shared_experts as shared_experts_module
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import FusedSharedExpertMLP, SharedExpertMLP
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _FakeTELinear(torch.nn.Module):
    def __init__(self, weight_shape=(4, 4)):
        super().__init__()
        self.weight = torch.empty(weight_shape)
        self.fuse_wgrad_accumulation = False
        self.wgrad_called = False
        self.reduce_hooks_called = False

    def backward_dw(self):
        self.wgrad_called = True

    def _trigger_wgrad_accumulation_and_reduce_hooks(self):
        self.reduce_hooks_called = True


class _FakeTEGroupedLinear(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs
        self.weight0 = None
        self._glu_interleave_size = None
        self.wgrad_called = False

    def backward_dw(self):
        self.wgrad_called = True


class _FakeTEScaledSwiGLU(torch.nn.Module):
    def __init__(self, glu_interleave_size):
        super().__init__()
        self.glu_interleave_size = glu_interleave_size


class _FakeTESequential(torch.nn.Module):
    def append(self, module):
        self.add_module(str(len(self._modules)), module)

    def forward(self, *args):
        self.args = args
        hidden_states = args[0]
        return torch.ones(
            hidden_states.size(0), 4, device=hidden_states.device, dtype=hidden_states.dtype
        )


class _FakeFP8Autocast:
    def __init__(self, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class _FakeMXFP8Recipe:
    pass


class _FakeNVFP4Recipe:
    pass


def _fake_te_module(linear_cls=_FakeTELinear):
    return SimpleNamespace(
        pytorch=SimpleNamespace(
            Linear=linear_cls,
            ops=SimpleNamespace(
                GroupedLinear=_FakeTEGroupedLinear,
                ScaledSwiGLU=_FakeTEScaledSwiGLU,
                Sequential=_FakeTESequential,
            ),
            fp8_autocast=_FakeFP8Autocast,
        ),
        common=SimpleNamespace(
            recipe=SimpleNamespace(
                MXFP8BlockScaling=_FakeMXFP8Recipe, NVFP4BlockScaling=_FakeNVFP4Recipe
            )
        ),
    )


def _patch_fake_shared_expert_te(monkeypatch, linear_cls=_FakeTELinear):
    fake_te = _fake_te_module(linear_cls)
    monkeypatch.setattr(shared_experts_module, "HAVE_TE", True)
    monkeypatch.setattr(shared_experts_module, "te", fake_te)
    monkeypatch.setattr(shared_experts_module, "is_te_min_version", lambda *args, **kwargs: True)
    monkeypatch.setattr(shared_experts_module, "get_pg_size", lambda group: 1)
    monkeypatch.setattr(
        shared_experts_module,
        "get_cuda_rng_tracker",
        lambda: SimpleNamespace(is_initialized=lambda: False),
    )
    return fake_te


def _fake_shared_expert(**config_kwargs):
    shared_expert = FusedSharedExpertMLP.__new__(FusedSharedExpertMLP)
    torch.nn.Module.__init__(shared_expert)
    config = SimpleNamespace(
        add_bias_linear=False,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_shared_expert_glu_interleave_size=32,
        delay_wgrad_compute=False,
        sequence_parallel=False,
    )
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    shared_expert.config = config
    shared_expert.linear_fc1 = _FakeTELinear((8, 4))
    shared_expert.linear_fc2 = _FakeTELinear((4, 4))
    shared_expert.tp_group = object()
    shared_expert._fused_grouped_swiglu_ops = None
    shared_expert._fused_grouped_swiglu_recipe = None
    return shared_expert


def test_shared_expert_builder_selects_implementation_from_config(monkeypatch):
    class FakeSharedExpert:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeFusedSharedExpert(FakeSharedExpert):
        pass

    monkeypatch.setattr(moe_module_specs, "SharedExpertMLP", FakeSharedExpert)
    monkeypatch.setattr(moe_module_specs, "FusedSharedExpertMLP", FakeFusedSharedExpert)
    submodules = object()

    shared = moe_module_specs._build_shared_experts(
        config=SimpleNamespace(use_grouped_gemm_for_shared_expert=False),
        pg_collection=None,
        gate=False,
        submodules=submodules,
        name="shared",
    )
    fused = moe_module_specs._build_shared_experts(
        config=SimpleNamespace(use_grouped_gemm_for_shared_expert=True),
        pg_collection=None,
        gate=False,
        submodules=submodules,
        name="shared",
    )

    assert isinstance(shared, FakeSharedExpert)
    assert not isinstance(shared, FakeFusedSharedExpert)
    assert isinstance(fused, FakeFusedSharedExpert)
    assert fused.kwargs["submodules"] is submodules


def test_validate_fused_grouped_swiglu_requires_te(monkeypatch):
    shared_expert = _fake_shared_expert()
    monkeypatch.setattr(shared_experts_module, "HAVE_TE", False)

    with pytest.raises(RuntimeError, match="requires Transformer Engine"):
        shared_expert._validate_fused_grouped_swiglu()


@pytest.mark.parametrize(
    ("config_kwargs", "bad_linear", "match"),
    [
        ({"add_bias_linear": True}, None, "add_bias_linear"),
        ({"activation_func": F.gelu}, None, "SwiGLU activation"),
        ({"gated_linear_unit": False}, None, "SwiGLU activation"),
        ({"moe_shared_expert_glu_interleave_size": None}, None, "glu_interleave_size"),
        ({}, "linear_fc1", "FC1"),
        ({}, "linear_fc2", "FC2"),
    ],
)
def test_validate_fused_grouped_swiglu_rejects_unsupported_configs(
    monkeypatch, config_kwargs, bad_linear, match
):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert(**config_kwargs)
    if bad_linear is not None:
        setattr(shared_expert, bad_linear, torch.nn.Linear(4, 4))

    with pytest.raises(ValueError, match=match):
        shared_expert._validate_fused_grouped_swiglu()


def test_make_fused_grouped_swiglu_ops_builds_grouped_pipeline(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert()
    shared_expert.linear_fc1.fuse_wgrad_accumulation = True

    ops = shared_expert._make_fused_grouped_swiglu_ops()

    fc1_op, activation_op, fc2_op = list(ops.children())
    assert isinstance(ops, _FakeTESequential)
    assert isinstance(fc1_op, _FakeTEGroupedLinear)
    assert fc1_op.kwargs["num_groups"] == 1
    assert fc1_op.kwargs["in_features"] == 4
    assert fc1_op.kwargs["out_features"] == 8
    assert fc1_op.kwargs["device"] == "meta"
    assert fc1_op.kwargs["bias"] is False
    assert fc1_op.kwargs["accumulate_into_main_grad"] is True
    assert fc1_op.weight0 is shared_expert.linear_fc1.weight
    assert fc1_op._glu_interleave_size == 32

    assert isinstance(activation_op, _FakeTEScaledSwiGLU)
    assert activation_op.glu_interleave_size == 32

    assert isinstance(fc2_op, _FakeTEGroupedLinear)
    assert fc2_op.kwargs["num_groups"] == 1
    assert fc2_op.kwargs["in_features"] == 4
    assert fc2_op.kwargs["out_features"] == 4
    assert fc2_op.kwargs["device"] == "meta"
    assert fc2_op.kwargs["bias"] is False
    assert fc2_op.kwargs["accumulate_into_main_grad"] is False
    assert fc2_op.weight0 is shared_expert.linear_fc2.weight


def test_fused_grouped_swiglu_ops_replay_linear_pre_forward_hooks(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert()
    ops = shared_expert._make_fused_grouped_swiglu_ops()
    calls = []

    shared_expert.linear_fc1.register_forward_pre_hook(
        lambda module, _args: calls.append(("fc1", module))
    )
    shared_expert.linear_fc2.register_forward_pre_hook(
        lambda module, _args: calls.append(("fc2", module))
    )

    hidden_states = torch.ones(2, 4)
    tokens_per_expert = torch.tensor([2])
    ops(hidden_states, tokens_per_expert, torch.ones(2), tokens_per_expert)

    assert calls == [("fc1", shared_expert.linear_fc1), ("fc2", shared_expert.linear_fc2)]


def test_fused_grouped_swiglu_ops_reject_input_modifying_hooks(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert()
    ops = shared_expert._make_fused_grouped_swiglu_ops()
    shared_expert.linear_fc1.register_forward_pre_hook(lambda _module, _args: torch.zeros(1))

    hidden_states = torch.ones(2, 4)
    tokens_per_expert = torch.tensor([2])
    with pytest.raises(RuntimeError, match="modifies inputs"):
        ops(hidden_states, tokens_per_expert, torch.ones(2), tokens_per_expert)


def test_fused_grouped_swiglu_no_comm_flattens_and_caches_fused_ops(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert()
    hidden_states = torch.randn(2, 3, 4)

    output = shared_expert._fused_grouped_swiglu_no_comm(hidden_states)

    (ops,) = shared_expert._fused_grouped_swiglu_ops
    hidden_states_2d, tokens_per_expert, scales, tokens_per_expert_again = ops.args
    assert output.shape == hidden_states.shape
    assert shared_expert._fused_grouped_swiglu_recipe.__class__ is _FakeMXFP8Recipe
    assert hidden_states_2d.shape == (6, 4)
    assert tokens_per_expert.tolist() == [6]
    assert tokens_per_expert_again is tokens_per_expert
    torch.testing.assert_close(scales, torch.ones(6))


def test_backward_dw_dispatches_fused_children_and_original_reduce_hooks(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert(delay_wgrad_compute=True)
    ops = shared_expert._make_fused_grouped_swiglu_ops()
    shared_expert._fused_grouped_swiglu_ops = (ops,)
    fc1_op, _, fc2_op = list(ops.children())
    call_order = []
    fc1_op.backward_dw = lambda: call_order.append("fc1")
    fc2_op.backward_dw = lambda: call_order.append("fc2")

    shared_expert.backward_dw()

    assert call_order == ["fc2", "fc1"]
    assert shared_expert.linear_fc1.reduce_hooks_called
    assert shared_expert.linear_fc2.reduce_hooks_called


class TestSharedExperts:
    def setup_method(self, method):
        self.config = TransformerConfig(
            num_layers=1,
            hidden_size=32,
            num_attention_heads=4,
            num_moe_experts=16,
            moe_shared_expert_intermediate_size=32,
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="alltoall",
            use_cpu_initialization=True,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            bias_activation_fusion=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=4,
            add_bias_linear=False,
        )

    def get_moe_layer(self, **kargs) -> MoELayer:
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=self.config.num_moe_experts, moe_grouped_gemm=False
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        new_config = dataclasses.replace(self.config, **kargs)
        if get_tensor_model_parallel_world_size() > 1:
            new_config.sequence_parallel = True
        moe_layer = MoELayer(new_config, submodules)
        moe_layer.cuda()
        return moe_layer

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("dispatcher_type", ["alltoall"])
    @pytest.mark.parametrize("tp_size, ep_size", [[1, 1], [4, 1], [1, 4], [2, 4]])
    def test_shared_expert_forward_backward(self, dispatcher_type: str, tp_size, ep_size):
        """
        Tests that the MoELayer with and without shared expert overlap produce
        identical outputs and gradients.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        # Create MoE layer with shared expert overlap enabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=True, moe_token_dispatcher_type=dispatcher_type
        ).to(dtype=torch.bfloat16)

        # Create MoE layer with shared expert overlap disabled.
        model_parallel_cuda_manual_seed(123)
        moe_layer_no_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=False, moe_token_dispatcher_type=dispatcher_type
        ).to(dtype=torch.bfloat16)
        moe_layer_no_overlap.load_state_dict(moe_layer_overlap.state_dict())

        # Sanity check that the weights are identical.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.equal(p_overlap, p_no_overlap)

        # Verify attributes of the MoE layers.
        num_weights_overlap = sum([p.numel() for p in moe_layer_overlap.parameters()])
        num_weights_no_overlap = sum([p.numel() for p in moe_layer_no_overlap.parameters()])
        assert num_weights_overlap == num_weights_no_overlap

        assert moe_layer_overlap.shared_experts is not None
        assert moe_layer_overlap.shared_experts.stream is not None
        assert moe_layer_overlap.token_dispatcher.shared_experts is not None

        assert moe_layer_no_overlap.shared_experts is not None
        assert moe_layer_no_overlap.token_dispatcher.shared_experts is None

        # Create a dummy input tensor.
        hidden_states = torch.randn(
            (32, 2, self.config.hidden_size),
            requires_grad=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
        hidden_states_no_overlap = hidden_states.clone().detach().requires_grad_(True)

        # Forward pass.
        output_overlap, _ = moe_layer_overlap(hidden_states)
        output_no_overlap, _ = moe_layer_no_overlap(hidden_states_no_overlap)
        torch.testing.assert_close(output_overlap, output_no_overlap)

        # Backward pass.
        output_overlap.mean().backward()
        output_no_overlap.mean().backward()

        # Check gradients.
        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.allclose(
                p_overlap.grad, p_no_overlap.grad
            ), f"max diff: {torch.max(torch.abs(p_overlap.grad - p_no_overlap.grad))}"
