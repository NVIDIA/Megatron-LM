# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest
import torch

from examples.mimo.configs.mock import (
    get_mock_language_model_config,
    get_mock_projection_config,
    get_mock_vision_model_config,
)
from examples.mimo.model_providers import mock as mock_model_provider
from megatron.core.enums import Fp8Recipe
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine is required")
class TestMimoModelTEQuantizationConfig:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_vision_gemms_execute_bf16_under_mxfp8(self, monkeypatch):
        import transformer_engine.pytorch as te
        from transformer_engine.pytorch import is_mxfp8_available
        from transformer_engine.pytorch.quantization import FP8GlobalStateManager

        mxfp8_available, reason = is_mxfp8_available(return_reason=True)
        if not mxfp8_available:
            pytest.skip(f"MXFP8 is not available: {reason}")

        recipe = RecipeConfig.from_config_dict(
            {
                "configs": {
                    "bf16": {
                        "transformer_engine_config_type": "TEQuantizationParams",
                        "training_recipe": {},
                    }
                },
                "matchers": {
                    "vision_encoder": {
                        "config": "bf16",
                        "type": "glob",
                        "pattern": "*modality_submodules.*.encoders.*",
                        "enabled": True,
                    },
                    "vision_projection": {
                        "config": "bf16",
                        "type": "glob",
                        "pattern": "*modality_submodules.*projections.*",
                        "enabled": True,
                    },
                },
            }
        )

        def configure_mxfp8(config):
            config.bf16 = True
            config.params_dtype = torch.bfloat16
            config.fp8 = "e4m3"
            config.fp8_recipe = Fp8Recipe.mxfp8
            config.quant_recipe = recipe
            return config

        vision_config = configure_mxfp8(get_mock_vision_model_config())
        projection_config = configure_mxfp8(get_mock_projection_config())
        language_config = configure_mxfp8(get_mock_language_model_config())

        with (
            patch.object(
                mock_model_provider, "get_mock_vision_model_config", return_value=vision_config
            ),
            patch.object(
                mock_model_provider, "get_mock_projection_config", return_value=projection_config
            ),
            patch.object(
                mock_model_provider, "get_mock_language_model_config", return_value=language_config
            ),
        ):
            model = mock_model_provider.model_provider_mock_vlm_single_encoder().cuda()

        named_modules = dict(model.named_modules())
        module_names = {id(module): name for name, module in named_modules.items()}
        fp8_enabled_at_te_gemm = {}

        def record_fp8_state(original_forward):
            def wrapped(module, *args, **kwargs):
                name = module_names.get(id(module))
                if name is not None:
                    fp8_enabled_at_te_gemm.setdefault(name, []).append(
                        FP8GlobalStateManager.is_fp8_enabled()
                    )
                return original_forward(module, *args, **kwargs)

            return wrapped

        monkeypatch.setattr(te.Linear, "forward", record_fp8_state(te.Linear.forward))
        monkeypatch.setattr(
            te.LayerNormLinear, "forward", record_fp8_state(te.LayerNormLinear.forward)
        )

        language_name = "language_model.decoder.layers.0.mlp.linear_fc2"
        language_linear = named_modules[language_name]
        assert language_linear.te_quant_params is None

        language_input = torch.randn(
            32,
            1,
            language_linear.in_features,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        with get_fp8_context(language_config):
            language_output, _ = language_linear(language_input)

        images = torch.randn(1, 3, 224, 224, device="cuda", requires_grad=True)
        with get_fp8_context(vision_config):
            vision_output = model.modality_submodules["images"](
                encoder_inputs={"clip_encoder": {"x": images}}
            )

        (language_output.float().square().mean() + vision_output.float().square().mean()).backward()
        torch.cuda.synchronize()

        vision_gemm_states = {
            name: states
            for name, states in fp8_enabled_at_te_gemm.items()
            if name.startswith("modality_submodules.images.")
        }
        assert any(".encoders." in name for name in vision_gemm_states)
        assert any(".input_projections." in name for name in vision_gemm_states)
        assert all(
            not fp8_enabled for states in vision_gemm_states.values() for fp8_enabled in states
        ), f"MIMO vision GEMMs unexpectedly executed with FP8 enabled: {vision_gemm_states}"

        assert fp8_enabled_at_te_gemm[language_name]
        assert all(fp8_enabled_at_te_gemm[language_name])
        assert language_output.dtype == torch.bfloat16
        assert vision_output.dtype == torch.bfloat16
