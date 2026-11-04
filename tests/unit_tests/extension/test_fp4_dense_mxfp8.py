# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for fp4_dense_mxfp8: MXFP8 non-expert linears in an FP4 model."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import (
    HAVE_TE,
    TELayerNormColumnParallelLinear,
    TELinear,
    TEQuantizationParams,
    TEQuantizationRecipe,
    _fp4_dense_mxfp8_quant_params,
)
from megatron.core.fp4_utils import get_fp4_context, is_grouped_nvfp4tensor, is_nvfp4tensor
from megatron.core.fp8_utils import is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

_IS_BLACKWELL = torch.cuda.is_available() and torch.cuda.get_device_properties(0).major >= 10


class TestFp4DenseMxfp8ConfigValidation:
    def _config(self, **overrides):
        kwargs = dict(num_layers=2, hidden_size=128, num_attention_heads=4, fp4="e2m1")
        kwargs.update(overrides)
        return TransformerConfig(**kwargs)

    def test_accepts_fp4(self):
        config = self._config(fp4_dense_mxfp8=True, fp4_param=True, fp4_dense_mxfp8_param=True)
        assert config.fp4_dense_mxfp8 and config.fp4_dense_mxfp8_param

    def test_requires_fp4(self):
        with pytest.raises(ValueError, match="fp4_dense_mxfp8 must be used together with fp4"):
            self._config(fp4=None, fp4_dense_mxfp8=True)

    @pytest.mark.parametrize(
        "overrides", [dict(fp4_param=True), dict(fp4_dense_mxfp8=True, fp4_param=False)]
    )
    def test_param_requires_dense_mxfp8_and_fp4_param(self, overrides):
        with pytest.raises(ValueError, match="fp4_dense_mxfp8_param requires"):
            self._config(fp4_dense_mxfp8_param=True, **overrides)


class TestReuseGradBufForFp4ParamGather:
    """The MXFP8 params of an FP4 model may reuse the grad buffer for their all-gather."""

    @pytest.mark.parametrize("param_gather", ["fp8_param_gather", "fp4_param_gather"])
    def test_accepts_quantized_param_gather(self, param_gather):
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            reuse_grad_buf_for_mxfp8_param_ag=True,
            **{param_gather: True},
        )
        assert config.reuse_grad_buf_for_mxfp8_param_ag

    def test_rejects_high_precision_params(self):
        with pytest.raises(AssertionError, match="Reuse grad buffer only"):
            DistributedDataParallelConfig(
                use_distributed_optimizer=True, reuse_grad_buf_for_mxfp8_param_ag=True
            )


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine not installed")
class TestFp4DenseMxfp8QuantParams:
    """The override applies to non-expert linears only, and an explicit override wins."""

    @staticmethod
    def _config(**overrides):
        kwargs = dict(fp4="e2m1", fp4_dense_mxfp8=True, fp4_dense_mxfp8_param=False)
        kwargs.update(overrides)
        return SimpleNamespace(**kwargs)

    @pytest.mark.parametrize("fp8_param", [False, True])
    def test_non_expert_linear_gets_mxfp8(self, fp8_param):
        params = _fp4_dense_mxfp8_quant_params(
            self._config(fp4_dense_mxfp8_param=fp8_param), is_expert=False, current=None
        )
        assert params.training_recipe.fp8_quantization_recipe == Fp8Recipe.mxfp8
        assert params.training_recipe.fp8_param is fp8_param
        assert params.evaluation_recipe is None

    def test_expert_linear_keeps_model_recipe(self):
        assert _fp4_dense_mxfp8_quant_params(self._config(), is_expert=True, current=None) is None

    def test_disabled(self):
        config = self._config(fp4_dense_mxfp8=False)
        assert _fp4_dense_mxfp8_quant_params(config, is_expert=False, current=None) is None

    def test_explicit_override_wins(self):
        explicit = TEQuantizationParams(
            training_recipe=TEQuantizationRecipe(), evaluation_recipe=None
        )
        params = _fp4_dense_mxfp8_quant_params(self._config(), is_expert=False, current=explicit)
        assert params is explicit


def _bf16_recipe(pattern):
    return RecipeConfig.from_config_dict(
        {
            "configs": {
                "bf16": {
                    "transformer_engine_config_type": "TEQuantizationParams",
                    "training_recipe": {},
                }
            },
            "matchers": {
                "bf16": {"type": "glob", "enabled": True, "pattern": pattern, "config": "bf16"}
            },
        }
    )


@pytest.mark.internal
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
            fp4_dense_mxfp8=True,
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
            and not module._is_expert_linear
        }

    @pytest.mark.parametrize("dense_param", [False, True])
    def test_moe_model_params(self, dense_param):
        """Non-expert linears get the MXFP8 override (after GPTModel re-runs finish_init) and,
        with fp4_dense_mxfp8_param, MXFP8 weights; routed experts keep NVFP4 weights."""
        model = self._model(
            num_moe_experts=4,
            fp4_param=True,
            fp4_dense_mxfp8_param=dense_param,
            quant_recipe=_bf16_recipe("*self_attention.linear_proj"),
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
