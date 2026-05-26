# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TEFusedMLP,
    TEFusedMLPWithGroupedLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.models.gpt import gpt_layer_specs
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

_SKIP_REASON = "TEFusedMLPWithGroupedLinear requires Transformer Engine >= 2.14.0"
_SKIP = not HAVE_TE or not is_te_min_version("2.14.0")
_SKIP_NO_GROUPED_CLASS = TEFusedMLPWithGroupedLinear is None


def _make_submodules():
    return MLPSubmodules(linear_fc1=TELayerNormColumnParallelLinear, linear_fc2=TERowParallelLinear)


def _make_config(**overrides):
    defaults = dict(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        use_cpu_initialization=True,
    )
    defaults.update(overrides)
    return TransformerConfig(**defaults)


def _patch_fake_te_ops(monkeypatch):
    import megatron.core.extensions.transformer_engine as te_ext

    class FakeSequential(list):
        pass

    class FakeLayerNormLinear:
        pass

    class FakeLinear:
        pass

    class FakeNorm:

        def __init__(self, norm_shape, **kwargs):
            self.norm_shape = norm_shape
            self.kwargs = kwargs

    class FakeGroupedLinear:
        instances = []

        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.__class__.instances.append(self)

    class FakeScaledSwiGLU:

        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_ops = SimpleNamespace(
        Sequential=FakeSequential,
        LayerNorm=FakeNorm,
        RMSNorm=FakeNorm,
        GroupedLinear=FakeGroupedLinear,
        ScaledSwiGLU=FakeScaledSwiGLU,
    )
    monkeypatch.setattr(te_ext.te.pytorch, "LayerNormLinear", FakeLayerNormLinear, raising=False)
    monkeypatch.setattr(te_ext.te.pytorch, "Linear", FakeLinear, raising=False)
    monkeypatch.setattr(te_ext.te.pytorch, "ops", fake_ops, raising=False)
    monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        te_ext, "get_cuda_rng_tracker", lambda: SimpleNamespace(is_initialized=lambda: True)
    )
    return SimpleNamespace(
        te_ext=te_ext,
        Sequential=FakeSequential,
        LayerNormLinear=FakeLayerNormLinear,
        Linear=FakeLinear,
        GroupedLinear=FakeGroupedLinear,
        ScaledSwiGLU=FakeScaledSwiGLU,
    )


def _make_fake_grouped_mlp(fake_te, normalization="LayerNorm"):
    module = TEFusedMLPWithGroupedLinear.__new__(TEFusedMLPWithGroupedLinear)

    fc1 = fake_te.LayerNormLinear()
    fc1.normalization = normalization
    fc1.weight = torch.ones(8, 4)
    fc1.layer_norm_weight = torch.ones(4)
    fc1.layer_norm_bias = torch.zeros(4)
    fc1.eps = 1e-5
    fc1.zero_centered_gamma = False
    fc1.fuse_wgrad_accumulation = True

    fc2 = fake_te.Linear()
    fc2.weight = torch.ones(4, 4)
    fc2.fuse_wgrad_accumulation = False
    fc2.sequence_parallel = False

    module.linear_fc1 = fc1
    module.linear_fc2 = fc2

    def fake_register_hooks(fused_impl):
        module._hooked_fused_impl = fused_impl

    object.__setattr__(module, "_register_hooks_on_fused_impl", fake_register_hooks)
    return module


@pytest.mark.skipif(_SKIP_NO_GROUPED_CLASS, reason="TEFusedMLPWithGroupedLinear is unavailable")
class TestTEFusedMLPWithGroupedLinearControlFlow:

    def test_init_requires_te_214(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        def fake_base_init(self, config, submodules):
            self.config = config

        def fake_is_te_min_version(version, *args, **kwargs):
            return False if version == "2.14.0" else True

        monkeypatch.setattr(te_ext.TEFusedMLP, "__init__", fake_base_init)
        monkeypatch.setattr(te_ext, "is_te_min_version", fake_is_te_min_version)

        with pytest.raises(RuntimeError, match="Transformer Engine >= 2.14.0"):
            TEFusedMLPWithGroupedLinear(_make_config(), _make_submodules())

    @pytest.mark.parametrize(
        ("config_overrides", "match"),
        [
            ({"add_bias_linear": True}, "add_bias_linear"),
            ({"activation_func": F.gelu}, "SwiGLU activation"),
            ({"gated_linear_unit": False}, "SwiGLU activation"),
        ],
    )
    def test_init_validates_supported_dense_swiglu_config(
        self, monkeypatch, config_overrides, match
    ):
        import megatron.core.extensions.transformer_engine as te_ext

        def fake_base_init(self, config, submodules):
            self.config = config

        monkeypatch.setattr(te_ext.TEFusedMLP, "__init__", fake_base_init)
        monkeypatch.setattr(te_ext, "is_te_min_version", lambda *args, **kwargs: True)

        with pytest.raises(ValueError, match=match):
            TEFusedMLPWithGroupedLinear(_make_config(**config_overrides), _make_submodules())

    def test_make_fused_impl_falls_back_to_base_for_tensor_parallel(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        sentinel = object()
        module = TEFusedMLPWithGroupedLinear.__new__(TEFusedMLPWithGroupedLinear)

        monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(te_ext.TEFusedMLP, "_make_fused_impl", lambda self: sentinel)

        assert TEFusedMLPWithGroupedLinear._make_fused_impl(module) is sentinel

    @pytest.mark.parametrize("normalization", ["LayerNorm", "RMSNorm"])
    def test_make_fused_impl_builds_grouped_linear_pipeline(self, monkeypatch, normalization):
        fake_te = _patch_fake_te_ops(monkeypatch)
        module = _make_fake_grouped_mlp(fake_te, normalization=normalization)

        fused_impl = TEFusedMLPWithGroupedLinear._make_fused_impl(module)

        norm = module._norm_seq[0][0]
        fc1_op, fc2_op = fake_te.GroupedLinear.instances
        assert isinstance(fused_impl, fake_te.Sequential)
        assert module._hooked_fused_impl is fused_impl
        assert norm.norm_shape == 4
        assert norm.weight is module.linear_fc1.layer_norm_weight
        if normalization == "LayerNorm":
            assert norm.bias is module.linear_fc1.layer_norm_bias
        else:
            assert not hasattr(norm, "bias")

        assert fc1_op.kwargs["num_groups"] == 1
        assert fc1_op.kwargs["in_features"] == 4
        assert fc1_op.kwargs["out_features"] == 8
        assert fc1_op.kwargs["accumulate_into_main_grad"] is True
        assert fc1_op.weight0 is module.linear_fc1.weight
        assert fc1_op._glu_interleave_size == 32
        assert isinstance(fused_impl[1], fake_te.ScaledSwiGLU)
        assert fused_impl[1].kwargs == {"glu_interleave_size": 32}
        assert fc2_op.kwargs["num_groups"] == 1
        assert fc2_op.kwargs["in_features"] == 4
        assert fc2_op.kwargs["out_features"] == 4
        assert fc2_op.kwargs["accumulate_into_main_grad"] is False
        assert fc2_op.weight0 is module.linear_fc2.weight

    @pytest.mark.parametrize(("bad_attr", "match"), [("linear_fc1", "FC1"), ("linear_fc2", "FC2")])
    def test_make_fused_impl_validates_te_linear_types(self, monkeypatch, bad_attr, match):
        fake_te = _patch_fake_te_ops(monkeypatch)
        module = _make_fake_grouped_mlp(fake_te)
        setattr(module, bad_attr, object())

        with pytest.raises(ValueError, match=match):
            TEFusedMLPWithGroupedLinear._make_fused_impl(module)

    def test_make_fused_impl_rejects_unsupported_normalization(self, monkeypatch):
        fake_te = _patch_fake_te_ops(monkeypatch)
        module = _make_fake_grouped_mlp(fake_te, normalization="UnsupportedNorm")

        with pytest.raises(ValueError, match="Unsupported normalization"):
            TEFusedMLPWithGroupedLinear._make_fused_impl(module)

    def test_forward_falls_back_to_base_for_tensor_parallel(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        hidden_states = object()
        module = TEFusedMLPWithGroupedLinear.__new__(TEFusedMLPWithGroupedLinear)

        def fake_forward(self, hidden_states, **kwargs):
            return hidden_states, kwargs

        monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(te_ext.TEFusedMLP, "forward", fake_forward)

        assert TEFusedMLPWithGroupedLinear.forward(module, hidden_states, flag=True) == (
            hidden_states,
            {"flag": True},
        )

    @pytest.mark.parametrize(
        ("fp4_recipe", "recipe_name"), [("nvfp4", "NVFP4BlockScaling"), ("", "MXFP8BlockScaling")]
    )
    def test_forward_uses_recipe_and_drops_empty_bias(self, monkeypatch, fp4_recipe, recipe_name):
        import megatron.core.extensions.transformer_engine as te_ext

        recipe = object()
        recipe_calls = []
        fused_impl = lambda hidden_states, *args: hidden_states
        module = TEFusedMLPWithGroupedLinear.__new__(TEFusedMLPWithGroupedLinear)
        module._fused_impl = None
        module._norm_seq = (lambda hidden_states: hidden_states,)
        module.linear_fc2 = SimpleNamespace(te_return_bias=True, bias=torch.empty(0))
        object.__setattr__(module, "_make_fused_impl", lambda: fused_impl)

        def make_recipe(name):
            recipe_calls.append(name)
            return recipe

        monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(
            te_ext.te.common.recipe,
            "NVFP4BlockScaling",
            lambda: make_recipe("NVFP4BlockScaling"),
            raising=False,
        )
        monkeypatch.setattr(
            te_ext.te.common.recipe,
            "MXFP8BlockScaling",
            lambda: make_recipe("MXFP8BlockScaling"),
            raising=False,
        )
        monkeypatch.setattr(
            te_ext.te.pytorch,
            "quantized_model_init",
            lambda enabled, recipe: nullcontext(),
            raising=False,
        )
        monkeypatch.setattr(
            te_ext.te.pytorch, "autocast", lambda enabled, recipe: nullcontext(), raising=False
        )
        if fp4_recipe:
            monkeypatch.setenv("FP4_RECIPE", fp4_recipe)
        else:
            monkeypatch.delenv("FP4_RECIPE", raising=False)

        out, bias = TEFusedMLPWithGroupedLinear.forward(module, torch.ones(2, 3, 4))

        assert out.shape == (2, 3, 4)
        assert bias is None
        assert module._recipe is recipe
        assert recipe_calls == [recipe_name]

    def test_dense_grouped_spec_selected_only_with_te_op_fuser(self):
        grouped_spec = gpt_layer_specs.get_mlp_module_spec_for_backend(
            backend=gpt_layer_specs.TESpecProvider(),
            use_te_op_fuser=True,
            use_grouped_gemm_for_dense_mlp=True,
        )
        fused_spec = gpt_layer_specs.get_mlp_module_spec_for_backend(
            backend=gpt_layer_specs.TESpecProvider(),
            use_te_op_fuser=True,
            use_grouped_gemm_for_dense_mlp=False,
        )
        unfused_spec = gpt_layer_specs.get_mlp_module_spec_for_backend(
            backend=gpt_layer_specs.TESpecProvider(),
            use_te_op_fuser=False,
            use_grouped_gemm_for_dense_mlp=True,
        )

        assert grouped_spec.func.__self__ is TEFusedMLPWithGroupedLinear
        assert fused_spec.func.__self__ is TEFusedMLP
        assert unfused_spec.func.__self__ is MLP


@pytest.mark.skipif(_SKIP, reason=_SKIP_REASON)
class TestTEFusedMLPWithGroupedLinearSpec:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_instantiation(self):
        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        assert isinstance(mlp, TEFusedMLPWithGroupedLinear)

    def test_wrong_activation_raises(self):
        config = _make_config(activation_func=F.gelu, gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_gated_linear_unit_false_raises(self):
        config = _make_config(gated_linear_unit=False)
        with pytest.raises(ValueError, match="SwiGLU activation"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_add_bias_linear_raises(self):
        config = _make_config(add_bias_linear=True)
        with pytest.raises(ValueError, match="add_bias_linear"):
            TEFusedMLPWithGroupedLinear(config, _make_submodules())

    def test_tensor_parallel_falls_back_to_base_fused_mlp(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        calls = {}

        def fake_forward(self, hidden_states, **kwargs):
            calls["hidden_states"] = hidden_states
            calls["kwargs"] = kwargs
            return "output", "bias"

        monkeypatch.setattr(te_ext, "get_tensor_model_parallel_world_size", lambda: 2)
        monkeypatch.setattr(te_ext.TEFusedMLP, "forward", fake_forward)

        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        hidden_states = object()

        assert mlp.forward(hidden_states, test_kwarg=True) == ("output", "bias")
        assert calls == {"hidden_states": hidden_states, "kwargs": {"test_kwarg": True}}

    def test_norm_seq_not_registered_as_submodule(self):
        # _norm_seq must be stored in a tuple (not directly as nn.Module) to avoid
        # PyTorch registering it as a submodule, which would duplicate norm weights
        # in state_dict/parameters. Verify it starts as None and is never a bare Module.
        import torch.nn as nn

        config = _make_config()
        mlp = TEFusedMLPWithGroupedLinear(config, _make_submodules())
        assert mlp._norm_seq is None
        assert '_norm_seq' not in dict(mlp.named_children())

        # Simulate what _make_fused_impl does and confirm the tuple-wrap holds.
        import transformer_engine.pytorch.ops as te_ops

        fake_seq = te_ops.Sequential()
        mlp._norm_seq = (fake_seq,)
        assert not isinstance(mlp._norm_seq, nn.Module)
        assert '_norm_seq' not in dict(mlp.named_children())
