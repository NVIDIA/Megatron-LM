# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.models.gpt import moe_module_specs
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
from megatron.core.parallel_state import get_tensor_model_parallel_world_size
from megatron.core.transformer.moe import shared_experts as shared_experts_module
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.shared_experts import FusedSharedExpertMLP, SharedExpertMLP
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
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


class _FakeTEScaledClampedQGeGLU(torch.nn.Module):
    def __init__(self, glu_interleave_size, *, alpha, limit, glu_linear_offset):
        super().__init__()
        self.glu_interleave_size = glu_interleave_size
        self.alpha = alpha
        self.limit = limit
        self.glu_linear_offset = glu_linear_offset


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
                ScaledClampedQGeGLU=_FakeTEScaledClampedQGeGLU,
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
        activation_func_clamp_value=None,
        moe_shared_expert_glu_interleave_size=32,
        delay_wgrad_compute=False,
        sequence_parallel=False,
        fp4=False,
        fp4_recipe="nvfp4",
        fp8=True,
        fp8_recipe="mxfp8",
    )
    for key, value in config_kwargs.items():
        setattr(config, key, value)
    shared_expert.config = config
    shared_expert.linear_fc1 = _FakeTELinear((8, 4))
    shared_expert.linear_fc2 = _FakeTELinear((4, 4))
    shared_expert.tp_group = object()
    shared_expert._fused_grouped_swiglu_ops = None
    shared_expert._fused_grouped_swiglu_recipe = None
    shared_expert._fused_grouped_swiglu_unit_scale = None
    shared_expert._fused_grouped_swiglu_tokens_per_expert = {}
    shared_expert._grouped_linear_kwargs = {}
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


@pytest.mark.parametrize("has_clamped_op", [True, False])
def test_validate_fused_grouped_swiglu_requires_clamped_te_support(monkeypatch, has_clamped_op):
    fake_te = _patch_fake_shared_expert_te(monkeypatch)
    if has_clamped_op:
        monkeypatch.setattr(
            shared_experts_module, "is_te_min_version", lambda version: version != "2.17.0.dev0"
        )
    else:
        del fake_te.pytorch.ops.ScaledClampedQGeGLU
    shared_expert = _fake_shared_expert(activation_func_clamp_value=7.0)

    with pytest.raises(RuntimeError, match="ScaledClampedQGeGLU"):
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
    assert activation_op._grouped_mlp_unit_activation_scale is True

    assert isinstance(fc2_op, _FakeTEGroupedLinear)
    assert fc2_op.kwargs["num_groups"] == 1
    assert fc2_op.kwargs["in_features"] == 4
    assert fc2_op.kwargs["out_features"] == 4
    assert fc2_op.kwargs["device"] == "meta"
    assert fc2_op.kwargs["bias"] is False
    assert fc2_op.kwargs["accumulate_into_main_grad"] is False
    assert fc2_op.weight0 is shared_expert.linear_fc2.weight


def test_make_fused_grouped_swiglu_ops_builds_clamped_activation(monkeypatch):
    _patch_fake_shared_expert_te(monkeypatch)
    shared_expert = _fake_shared_expert(activation_func_clamp_value=7.0)

    shared_expert._validate_fused_grouped_swiglu()
    ops = shared_expert._make_fused_grouped_swiglu_ops()

    activation_op = list(ops.children())[1]
    assert isinstance(activation_op, _FakeTEScaledClampedQGeGLU)
    assert activation_op.glu_interleave_size == 32
    assert activation_op.alpha == 1.0
    assert activation_op.limit == 7.0
    assert activation_op.glu_linear_offset == 0.0
    assert activation_op._grouped_mlp_unit_activation_scale is True


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
    hidden_states = torch.randn(128, 2, 4)

    output = shared_expert._fused_grouped_swiglu_no_comm(hidden_states)

    (ops,) = shared_expert._fused_grouped_swiglu_ops
    hidden_states_2d, tokens_per_expert, scales, tokens_per_expert_again = ops.args
    shared_expert._fused_grouped_swiglu_no_comm(torch.randn_like(hidden_states))
    _, cached_tokens_per_expert, cached_scales, _ = ops.args
    assert output.shape == hidden_states.shape
    assert shared_expert._fused_grouped_swiglu_recipe.__class__ is _FakeMXFP8Recipe
    assert hidden_states_2d.shape == (256, 4)
    assert tokens_per_expert.tolist() == [256]
    assert tokens_per_expert_again is tokens_per_expert
    torch.testing.assert_close(scales, torch.ones(1))
    assert cached_tokens_per_expert is tokens_per_expert
    assert cached_scales is scales


def test_fused_grouped_swiglu_rejects_unaligned_input():
    shared_expert = _fake_shared_expert()

    with pytest.raises(ValueError, match="divisible by 256, but got 128"):
        shared_expert._fused_grouped_swiglu_no_comm(torch.randn(64, 2, 4))

    assert shared_expert._fused_grouped_swiglu_ops is None


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
    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("overlap", [False, True])
    def test_gpu_split_alignment_enables_fusion_without_env(self, monkeypatch, overlap):
        """The shared expert selects fused forward/backward ops using explicit alignment."""
        import inspect

        import transformer_engine.pytorch as te

        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TERowParallelLinear,
        )
        from megatron.core.fp8_utils import get_fp8_context
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.mlp import MLPSubmodules

        grouped_linear = getattr(getattr(te, "ops", None), "GroupedLinear", None)
        if (
            grouped_linear is None
            or "split_alignment" not in inspect.signature(grouped_linear).parameters
        ):
            pytest.skip("Installed TE op fuser GroupedLinear lacks split_alignment")
        monkeypatch.delenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", raising=False)
        mxfp8_available, reason = te.is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            pytest.skip(reason)
        fused_cls = te.ops.fused.GroupedMLP_CuTeGEMMGLU
        if not fused_cls.is_supported():
            pytest.skip("Fused grouped MLP requires supported GPU and cuDNN frontend kernels")

        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        config = dataclasses.replace(
            self.config,
            hidden_size=128,
            ffn_hidden_size=256,
            moe_shared_expert_intermediate_size=256,
            use_grouped_gemm_for_shared_expert=True,
            moe_shared_expert_glu_interleave_size=32,
            moe_shared_expert_overlap=overlap,
            use_cpu_initialization=False,
            bias_activation_fusion=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            fp8="e4m3",
            fp8_recipe="mxfp8",
        )
        with get_fp8_context(config, is_init=True):
            shared_expert = FusedSharedExpertMLP(
                config,
                MLPSubmodules(linear_fc1=TEColumnParallelLinear, linear_fc2=TERowParallelLinear),
                gate=False,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            ).cuda()
        hidden = torch.randn(
            (128, 2, config.hidden_size), device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        if overlap:
            shared_expert.pre_forward_comm(hidden)
            shared_expert.linear_fc1_forward_and_act()
            shared_expert.linear_fc2_forward()
            shared_expert.post_forward_comm()
            output = shared_expert.get_output()
        else:
            output = shared_expert(hidden)
        output.backward(torch.randn_like(output))

        (sequential,) = shared_expert._fused_grouped_swiglu_ops
        assert sequential[0].split_alignment == 256
        assert sequential[2].split_alignment == 256
        (fuser,) = sequential._module_groups
        assert len(fuser._forward_ops) == 1
        assert len(fuser._backward_ops) == 1
        assert isinstance(fuser._forward_ops[0][0], fused_cls)
        assert fuser._backward_ops[0][0] is fuser._forward_ops[0][0]
        assert output.shape == hidden.shape
        assert torch.isfinite(output).all()
        for tensor in (hidden, *shared_expert.parameters()):
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()

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
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=True, moe_token_dispatcher_type=dispatcher_type
        ).to(dtype=torch.bfloat16)

        # Create MoE layer with shared expert overlap disabled.
        _set_random_seed(seed_=123, data_parallel_random_init=False)
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

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_shared_expert_glu_linear_offset_overlap_parity(self):
        """Nonzero GLU offsets produce identical overlap and synchronous results."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        shared_expert_kwargs = {
            "moe_token_dispatcher_type": "alltoall",
            "bias_activation_fusion": False,
            "activation_func_clamp_value": None,
            "glu_linear_offset": 0.5,
        }

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=True, **shared_expert_kwargs
        ).to(dtype=torch.bfloat16)

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_no_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=False, **shared_expert_kwargs
        ).to(dtype=torch.bfloat16)
        moe_layer_no_overlap.load_state_dict(moe_layer_overlap.state_dict())

        hidden_states = torch.randn(
            (32, 2, self.config.hidden_size),
            requires_grad=True,
            device="cuda",
            dtype=torch.bfloat16,
        )
        hidden_states_no_overlap = hidden_states.detach().clone().requires_grad_(True)

        shared_expert_overlap = moe_layer_overlap.shared_experts
        shared_expert_no_overlap = moe_layer_no_overlap.shared_experts
        assert shared_expert_overlap is not None
        assert shared_expert_no_overlap is not None

        # Isolate shared-expert parity from nondeterministic routed-token unpermutation.
        shared_expert_overlap.pre_forward_comm(hidden_states)
        shared_expert_overlap.linear_fc1_forward_and_act()
        shared_expert_overlap.linear_fc2_forward()
        shared_expert_overlap.post_forward_comm()
        output_overlap = shared_expert_overlap.get_output()
        output_no_overlap = shared_expert_no_overlap(hidden_states_no_overlap)
        torch.testing.assert_close(output_overlap, output_no_overlap)

        output_overlap.mean().backward()
        output_no_overlap.mean().backward()

        torch.testing.assert_close(hidden_states.grad, hidden_states_no_overlap.grad)
        for p_overlap, p_no_overlap in zip(
            shared_expert_overlap.parameters(), shared_expert_no_overlap.parameters()
        ):
            torch.testing.assert_close(p_overlap.grad, p_no_overlap.grad)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("bias_activation_fusion", [True, False])
    def test_shared_expert_clamped_swiglu(self, bias_activation_fusion):
        """Verify clamped SwiGLU parity for overlapped and synchronous shared experts."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
        clamp_value = 1.0

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=True,
            moe_token_dispatcher_type="alltoall",
            activation_func_clamp_value=clamp_value,
            bias_activation_fusion=bias_activation_fusion,
        ).to(dtype=torch.bfloat16)

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_no_overlap = self.get_moe_layer(
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="alltoall",
            activation_func_clamp_value=clamp_value,
            bias_activation_fusion=bias_activation_fusion,
        ).to(dtype=torch.bfloat16)
        moe_layer_no_overlap.load_state_dict(moe_layer_overlap.state_dict())

        hidden_states = (
            torch.randn((32, 2, self.config.hidden_size), device="cuda", dtype=torch.bfloat16) * 5.0
        ).requires_grad_(True)
        hidden_states_no_overlap = hidden_states.detach().clone().requires_grad_(True)

        output_overlap, _ = moe_layer_overlap(hidden_states)
        output_no_overlap, _ = moe_layer_no_overlap(hidden_states_no_overlap)

        cos_out = F.cosine_similarity(
            output_overlap.flatten().unsqueeze(0).float(),
            output_no_overlap.flatten().unsqueeze(0).float(),
        ).item()
        assert cos_out > 0.999, (
            f"shared-expert clamp output mismatch (fusion={bias_activation_fusion}): "
            f"cos sim = {cos_out:.6f}"
        )

        output_overlap.mean().backward()
        output_no_overlap.mean().backward()

        for p_overlap, p_no_overlap in zip(
            moe_layer_overlap.parameters(), moe_layer_no_overlap.parameters()
        ):
            assert torch.allclose(p_overlap.grad, p_no_overlap.grad), (
                f"shared-expert clamp mismatch (fusion={bias_activation_fusion}); "
                f"max diff: {torch.max(torch.abs(p_overlap.grad - p_no_overlap.grad))}"
            )

        _set_random_seed(seed_=123, data_parallel_random_init=False)
        moe_layer_unclamped = self.get_moe_layer(
            moe_shared_expert_overlap=False,
            moe_token_dispatcher_type="alltoall",
            activation_func_clamp_value=None,
            bias_activation_fusion=bias_activation_fusion,
        ).to(dtype=torch.bfloat16)
        moe_layer_unclamped.load_state_dict(moe_layer_overlap.state_dict())

        hidden_states_unclamped = hidden_states.detach().clone().requires_grad_(True)
        output_unclamped, _ = moe_layer_unclamped(hidden_states_unclamped)
        assert not torch.allclose(output_no_overlap, output_unclamped), (
            "Clamping had no observable effect on shared-expert output; "
            "activation_func_clamp_value may not be plumbed through."
        )
