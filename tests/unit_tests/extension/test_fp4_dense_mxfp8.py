# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""MXFP8 non-expert linears in an FP4 model, configured through quant_recipe
(examples/quantization/fp4_experts_mxfp8_other_linears.yaml): the routed experts keep the FP4
recipe, the other linears run in MXFP8 and may keep MXFP8 parameters."""

import copy
import pathlib
from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TELayerNormColumnParallelLinear,
    TELinear,
    TEQuantizationParams,
    TEQuantizationRecipe,
    _get_fp8_model_init_for_quant_recipe,
)
from megatron.core.fp4_utils import get_fp4_context, is_grouped_nvfp4tensor, is_nvfp4tensor
from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import (
    DistributedOptimizer,
    _check_reuse_grad_buf_config,
)
from megatron.core.quantization.quant_config import MatchContext, RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

if HAVE_TE:
    from transformer_engine.pytorch.fp8 import FP8GlobalStateManager, fp8_model_init

_RECIPE_YAML = (
    pathlib.Path(__file__).resolve().parents[3]
    / "examples"
    / "quantization"
    / "fp4_experts_mxfp8_other_linears.yaml"
)

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 10


def _recipe_dict(fp8_param=True, overrides=()):
    """The example recipe as a dict; with fp8_param=False the MXFP8 linears keep BF16 params.
    overrides: (pattern, training_recipe) pairs matched before the example's matchers."""
    import yaml

    recipe = yaml.safe_load(_RECIPE_YAML.read_text())
    if not fp8_param:
        training = recipe["configs"]["mxfp8"]["training_recipe"]
        del training["fp8_param"], training["param_storage_follows_model_init"]
    matchers = {}
    for i, (pattern, training_recipe) in enumerate(overrides):
        key = f"override_{i}"
        recipe["configs"][key] = {
            "transformer_engine_config_type": "TEQuantizationParams",
            "training_recipe": training_recipe,
        }
        matchers[key] = {"type": "glob", "enabled": True, "pattern": pattern, "config": key}
    recipe["matchers"] = {**matchers, **recipe["matchers"]}
    return recipe


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestExampleRecipe:
    def test_loads_and_matches(self):
        recipe = RecipeConfig.from_yaml_file(str(_RECIPE_YAML))
        assert recipe.configs == RecipeConfig.from_config_dict(_recipe_dict()).configs
        cases = {
            "decoder.layers.1.mlp.experts.linear_fc1": "global_recipe",
            "mtp.layers.0.mtp_model_layer.mlp.experts.linear_fc2": "global_recipe",
            "decoder.layers.1.mlp.shared_experts.linear_fc1": "mxfp8",
            "decoder.layers.0.self_attention.linear_qkv": "mxfp8",
            "decoder.layers.0.mlp.linear_fc2": "mxfp8",
            "mtp.layers.0.eh_proj": "mxfp8",
        }
        for path, key in cases.items():
            config = recipe.match(MatchContext(module_path=path, layer_number=None))
            assert config.config_key == key, path
            params = TEQuantizationParams.parse_from_config(config)
            training = params.training_recipe
            if key == "mxfp8":
                assert training.fp8_quantization_recipe == Fp8Recipe.mxfp8, path
                assert training.fp8_param and training.param_storage_follows_model_init, path
            else:
                assert training.fp8_quantization_recipe is None, path
                assert training.fp4_quantization_recipe is None, path
                assert training.inherit_model_init_context, path
                assert not training.override_quantized_autocast, path


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestParamStorageFollowsModelInit:
    """param_storage_follows_model_init: quantized parameters only inside a model-init context
    that stores quantized parameters (so first_last_layers_bf16 layers keep BF16 weights)."""

    @staticmethod
    def _parse(**training):
        return TEQuantizationRecipe.parse_from_config(training)

    def test_parse(self):
        recipe = self._parse(
            fp8_quantization_recipe="mxfp8", fp8_param=True, param_storage_follows_model_init=True
        )
        assert recipe.param_storage_follows_model_init
        assert not self._parse(fp8_quantization_recipe="mxfp8").param_storage_follows_model_init

    def test_rejects_inherit_model_init_context(self):
        with pytest.raises(ValueError, match="inherit_model_init_context cannot be combined"):
            self._parse(inherit_model_init_context=True, param_storage_follows_model_init=True)

    def test_rejects_non_bool(self):
        with pytest.raises(ValueError, match="param_storage_follows_model_init must be a bool"):
            self._parse(fp8_quantization_recipe="mxfp8", param_storage_follows_model_init="yes")

    @pytest.mark.parametrize("follows", [False, True])
    @pytest.mark.parametrize("fp8_param", [False, True])
    @pytest.mark.parametrize("stores_quantized_weights", [False, True])
    def test_storage(self, follows, fp8_param, stores_quantized_weights):
        recipe = TEQuantizationRecipe(
            fp8_quantization_recipe=Fp8Recipe.mxfp8,
            fp8_param=fp8_param,
            param_storage_follows_model_init=follows,
        )
        with fp8_model_init(enabled=stores_quantized_weights):
            # Decided when the context is built (in the module constructor), as TELinear does.
            context = _get_fp8_model_init_for_quant_recipe(recipe)
        with context:
            stored = FP8GlobalStateManager.with_fp8_parameters()
        assert stored is (fp8_param and (stores_quantized_weights or not follows))


class TestReuseGradBufForFp4ParamGather:
    """The MXFP8 params of an FP4 model may reuse the grad buffer for their all-gather."""

    @staticmethod
    def _validate(**overrides):
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, reuse_grad_buf_for_mxfp8_param_ag=True, **overrides
        )
        model_config = TransformerConfig(num_layers=1, hidden_size=64, num_attention_heads=4)
        DistributedDataParallel._validate_config(model_config, ddp_config)
        return ddp_config

    @pytest.mark.parametrize("param_gather", ["fp8_param_gather", "fp4_param_gather"])
    def test_accepts_quantized_param_gather(self, param_gather):
        assert self._validate(**{param_gather: True}).reuse_grad_buf_for_mxfp8_param_ag

    def test_rejects_high_precision_params(self):
        with pytest.raises(AssertionError, match="Reuse grad buffer only"):
            self._validate()


class TestReuseGradBufWithPrecisionAwareOptimizer:
    """The grad-buffer param all-gather stages FP32 main params that the distributed optimizer
    does not keep when the precision-aware optimizer holds them."""

    @staticmethod
    def _optimizer_config(**overrides):
        kwargs = dict(
            optimizer="adam",
            use_distributed_optimizer=True,
            use_precision_aware_optimizer=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        kwargs.update(overrides)
        return OptimizerConfig(**kwargs)

    @staticmethod
    def _ddp_config(**overrides):
        kwargs = dict(
            use_distributed_optimizer=True,
            fp4_param_gather=True,
            reuse_grad_buf_for_mxfp8_param_ag=True,
        )
        kwargs.update(overrides)
        return DistributedDataParallelConfig(**kwargs)

    def _validate_run(self, optimizer_config, ddp_config):
        optimizer_config.validate_run(
            ddp_config=ddp_config, use_torch_fsdp2=False, ckpt_format="torch_dist"
        )

    @pytest.mark.parametrize("fp8_recipe", [None, "delayed"])
    def test_rejects_fp4_run(self, fp8_recipe):
        """FP4 runs have no FP8 recipe (or the default delayed one) for the optimizer."""
        config = self._optimizer_config(fp8_recipe=fp8_recipe)
        assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        with pytest.raises(ValueError, match="reuse_grad_buf_for_mxfp8_param_ag is not supported"):
            self._validate_run(config, self._ddp_config())

    def test_accepts_mxfp8_recipe_with_fp32_main_params(self):
        config = self._optimizer_config(fp8_recipe="mxfp8")
        assert not config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
        self._validate_run(config, self._ddp_config(fp4_param_gather=False, fp8_param_gather=True))

    def test_accepts_without_reuse(self):
        config = self._optimizer_config()
        self._validate_run(config, self._ddp_config(reuse_grad_buf_for_mxfp8_param_ag=False))

    def test_copy_to_param_buffer_raises(self):
        """Direct Megatron-Core users that skip the config validation get a clear error before
        any None main param is copied."""
        optimizer = SimpleNamespace(
            ddp_config=self._ddp_config(), config=self._optimizer_config(fp8_recipe=None)
        )
        with pytest.raises(NotImplementedError, match="precision-aware optimizer"):
            DistributedOptimizer._copy_main_params_to_param_buffer(optimizer)


@pytest.mark.internal
@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
@pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 and NVFP4 require Blackwell (SM >= 10)")
class TestFp4DenseMxfp8Model:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _model(num_moe_experts=None, **overrides):
        kwargs = dict(
            num_layers=2,
            hidden_size=128,
            ffn_hidden_size=256,
            num_attention_heads=4,
            use_cpu_initialization=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            add_bias_linear=False,
            fp4="e2m1",
            quant_recipe=RecipeConfig.from_config_dict(_recipe_dict()),
        )
        if num_moe_experts is not None:
            kwargs.update(
                num_moe_experts=num_moe_experts,
                moe_ffn_hidden_size=128,
                moe_router_topk=2,
                moe_grouped_gemm=True,
            )
        kwargs.update(overrides)
        config = TransformerConfig(**kwargs)
        return GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=num_moe_experts, moe_grouped_gemm=num_moe_experts is not None
            ),
            vocab_size=256,
            max_sequence_length=128,
        ).cuda()

    @staticmethod
    def _dense_linears(model):
        return {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, (TELinear, TELayerNormColumnParallelLinear))
            and ".experts." not in name
        }

    @pytest.mark.parametrize("dense_param", [False, True])
    def test_moe_model_params(self, dense_param):
        """Non-expert linears get the MXFP8 recipe (after GPTModel re-runs finish_init) and,
        with fp8_param, MXFP8 weights; routed experts keep NVFP4 weights. A matcher placed
        before the example's matchers overrides a module."""
        recipe = _recipe_dict(dense_param, overrides=[("*self_attention.linear_proj", {})])
        model = self._model(
            num_moe_experts=4, fp4_param=True, quant_recipe=RecipeConfig.from_config_dict(recipe)
        )
        dense = self._dense_linears(model)
        assert "decoder.layers.0.self_attention.linear_qkv" in dense
        for name, module in dense.items():
            recipe = module.te_quant_params.training_recipe
            if name.endswith("self_attention.linear_proj"):
                # The explicit BF16 override wins.
                assert recipe.fp8_quantization_recipe is None, name
                assert not is_mxfp8tensor(module.weight), name
            else:
                assert recipe.fp8_quantization_recipe == Fp8Recipe.mxfp8, name
                assert is_mxfp8tensor(module.weight) is dense_param, name

        expert_weights = [param for name, param in model.named_parameters() if ".experts." in name]
        assert expert_weights
        assert all(is_nvfp4tensor(p) or is_grouped_nvfp4tensor(p) for p in expert_weights)

    def test_first_last_bf16_layers_keep_bf16_weights(self):
        """With first_last_layers_bf16, the BF16 layers keep BF16 weights (and BF16 experts),
        while the dense linears of the FP4 layers keep MXFP8 weights."""
        model = self._model(
            num_moe_experts=4,
            num_layers=4,
            fp4_param=True,
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=1,
            num_layers_at_end_in_bf16=1,
        )
        bf16_layers = ("decoder.layers.0.", "decoder.layers.3.")
        dense = self._dense_linears(model)
        assert any(name.startswith(bf16_layers) for name in dense)
        for name, module in dense.items():
            assert name.startswith("decoder.layers."), name
            # GPTModel re-runs finish_init outside any model-init context; that must not change
            # the storage chosen at construction.
            recipe = module.te_quant_params.training_recipe
            assert recipe.fp8_quantization_recipe == Fp8Recipe.mxfp8, name
            if name.startswith(bf16_layers):
                assert not is_mxfp8tensor(module.weight), name
                assert module.weight.dtype == torch.bfloat16, name
            else:
                assert is_mxfp8tensor(module.weight), name
        for name, param in model.named_parameters():
            if ".experts." in name:
                is_nvfp4 = is_nvfp4tensor(param) or is_grouped_nvfp4tensor(param)
                assert is_nvfp4 is (not name.startswith(bf16_layers)), name

    def test_dense_model_runs_mxfp8(self):
        model = self._model()
        tokens = torch.randint(0, 256, (2, 128), device="cuda")
        position_ids = torch.arange(128, device="cuda").unsqueeze(0).expand(2, -1)
        with get_fp4_context(model.config):
            logits = model(tokens, position_ids, attention_mask=None)
        logits.float().sum().backward()
        assert torch.isfinite(logits).all()
        for name, module in self._dense_linears(model).items():
            assert module.fp8_meta["recipe"].mxfp8(), name
            assert module.weight.grad is not None or hasattr(module.weight, "main_grad"), name


class TestReuseGradBufConfigMismatch:
    """DDP reusing the grad buffer for the MXFP8 param all-gather needs the optimizer to stage its
    main params there too; a mismatch is rejected when the optimizer is built."""

    @staticmethod
    def _ddp_config(reuse):
        return DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            fp4_param_gather=True,
            reuse_grad_buf_for_mxfp8_param_ag=reuse,
        )

    def test_rejects_ddp_reuse_without_optimizer_reuse(self):
        with pytest.raises(ValueError, match="OptimizerConfig.reuse_grad_buf_for_mxfp8_param_ag"):
            _check_reuse_grad_buf_config(
                self._ddp_config(True), SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=False)
            )

    @pytest.mark.parametrize("reuse", [False, True])
    def test_accepts_matching_settings(self, reuse):
        _check_reuse_grad_buf_config(
            self._ddp_config(reuse), SimpleNamespace(reuse_grad_buf_for_mxfp8_param_ag=reuse)
        )


class TestDpZeroCheckpointLayout:
    """A dp_zero optimizer checkpoint stores one padded flat tensor per param buffer; one saved
    with a different set of buffers cannot be split into this run's buffers and is rejected."""

    # This run: NVFP4 experts, MXFP8 non-expert linears (both uint8) and BF16 params.
    RUN = [("param_torch:uint8", torch.uint8)] * 2 + [("param_torch:bfloat16", torch.bfloat16)]

    @classmethod
    def _optimizer(cls, run):
        return SimpleNamespace(
            buffers=[SimpleNamespace(param_dtype=dtype) for _, dtype in run],
            gbuf_ranges=[{key: None} for key, _ in run],
        )

    @staticmethod
    def _state(keys):
        state = {'buckets_coalesced': True}
        state.update({i: {key: {}} for i, key in enumerate(keys)})
        return state

    def test_same_layout_loads(self):
        state = self._state([key for key, _ in self.RUN])
        DistributedOptimizer.split_state_dict_if_needed(self._optimizer(self.RUN), state)
        assert set(state) == {'buckets_coalesced', 0, 1, 2}

    @pytest.mark.parametrize(
        "saved",
        [
            ["param_torch:bfloat16"],  # all-BF16 params
            ["param_torch:uint8", "param_torch:bfloat16"],  # NVFP4 experts, BF16 dense
        ],
        ids=["all-bf16", "nvfp4-experts-bf16-dense"],
    )
    def test_different_layout_is_rejected(self, saved):
        with pytest.raises(ValueError, match="different parameter-buffer layout"):
            DistributedOptimizer.split_state_dict_if_needed(
                self._optimizer(self.RUN), self._state(saved)
            )

    def test_no_quantized_buffers_is_unchanged(self):
        run = [("param_torch:bfloat16", torch.bfloat16)]
        DistributedOptimizer.split_state_dict_if_needed(
            self._optimizer(run), self._state(["param_torch:uint8", "param_torch:bfloat16"])
        )


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestFusedPathsApplyQuantRecipe:
    """Fused op paths bypass the linear wrappers that apply a quant_recipe override."""

    @staticmethod
    def _linear(override):
        return SimpleNamespace(te_quant_params=override, training=True, te_return_bias=False)

    @pytest.fixture
    def entered(self, monkeypatch):
        import megatron.core.extensions.transformer_engine as te_ext

        entered = []

        class Recorder:
            def __init__(self, recipe):
                self.recipe = recipe

            def __enter__(self):
                entered.append(self.recipe)

            def __exit__(self, *exc):
                entered.append(None)

        monkeypatch.setattr(
            te_ext, "_get_fp8_autocast_recipe_for_quant_params", lambda qparams, training: qparams
        )
        monkeypatch.setattr(te_ext, "_get_fp8_autocast_for_quant_recipe", Recorder)
        return entered

    def test_fused_mlp_runs_under_the_override(self, entered):
        from megatron.core.extensions.transformer_engine import TEFusedMLP

        def fused_impl(x):
            entered.append("impl")
            return x

        module = SimpleNamespace(
            linear_fc1=self._linear("mxfp8"),
            linear_fc2=self._linear("mxfp8"),
            _get_fused_impl=lambda: fused_impl,
        )
        x = torch.ones(2)
        out, bias = TEFusedMLP.forward(module, x)
        assert out is x and bias is None
        assert entered == ["mxfp8", "impl", None]

    def test_fused_mlp_without_override(self, entered):
        from megatron.core.extensions.transformer_engine import TEFusedMLP

        module = SimpleNamespace(
            linear_fc1=self._linear(None),
            linear_fc2=self._linear(None),
            _get_fused_impl=lambda: (lambda x: x),
        )
        TEFusedMLP.forward(module, torch.ones(2))
        assert entered == []

    def test_fused_mlp_rejects_different_overrides(self, entered):
        from megatron.core.extensions.transformer_engine import TEFusedMLP

        module = SimpleNamespace(
            linear_fc1=self._linear("mxfp8"),
            linear_fc2=self._linear(None),
            _get_fused_impl=lambda: (lambda x: x),
        )
        with pytest.raises(ValueError, match="same precision override"):
            TEFusedMLP.forward(module, torch.ones(2))

    def test_fused_shared_expert_rejects_override(self, monkeypatch):
        import megatron.core.transformer.moe.shared_experts as shared_experts

        monkeypatch.setattr(
            shared_experts,
            "_get_fp8_autocast_recipe_for_quant_params",
            lambda qparams, training: qparams,
        )
        module = SimpleNamespace(
            _fused_grouped_swiglu_recipe=None,
            linear_fc1=self._linear("mxfp8"),
            linear_fc2=self._linear("mxfp8"),
            config=SimpleNamespace(fp8=None, fp4="e2m1"),
        )
        with pytest.raises(ValueError, match="does not support a quant_recipe precision override"):
            shared_experts.FusedSharedExpertMLP._get_fused_grouped_swiglu_recipe(module)


@pytest.mark.launch_on_gb200
@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
@pytest.mark.skipif(not _IS_BLACKWELL, reason="MXFP8 and NVFP4 require Blackwell (SM >= 10)")
def test_mxfp8_param_shard_has_no_flat_view_under_delayed_recipe():
    """An FP4 run's fp8_recipe defaults to delayed; MXFP8 params from the quant_recipe must still
    skip the flat view (which would dequantize and keep the whole matrix)."""
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import DelayedScaling, MXFP8BlockScaling

    with fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
        mxfp8_weight = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16).weight
    with fp8_model_init(enabled=True, recipe=DelayedScaling()):
        fp8_weight = te.Linear(256, 256, bias=False, params_dtype=torch.bfloat16).weight
    bf16_weight = torch.nn.Parameter(torch.zeros(4, 4, device="cuda", dtype=torch.bfloat16))
    lacks_view = DistributedOptimizer._quantized_param_lacks_flat_view
    assert is_mxfp8tensor(mxfp8_weight)
    assert lacks_view(mxfp8_weight, "delayed")
    assert not lacks_view(fp8_weight, "delayed")
    assert lacks_view(fp8_weight, "tensorwise")
    assert not lacks_view(bf16_weight, "delayed")
