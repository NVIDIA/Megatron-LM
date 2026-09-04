# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Module paths used to resolve a quant_recipe must be the same in both phases.

A quantization recipe is consulted twice for every linear: once while the module is
being constructed, which decides whether its parameters are created quantized, and
once afterwards through the model-level pass, which decides the forward autocast.
The first phase names modules from a string threaded down from the model root, the
second reads them from ``named_modules()``. When the two disagree, a recipe entry
silently applies to only one of them, so a layer can run BF16 over FP8 parameters.
"""

import pytest

from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Matches every linear, so the recipe is consulted for all of them.
_MATCH_EVERYTHING = {
    "matchers": {"all": {"type": "glob", "enabled": True, "pattern": "*", "config": "bf16"}},
    "configs": {
        "bf16": {"transformer_engine_config_type": "TEQuantizationParams", "training_recipe": {}}
    },
}

_PATCHED_MODULES = (
    "megatron.core.extensions.transformer_engine",
    "megatron.core.models.gpt.gpt_model",
)


def _record_module_paths(monkeypatch):
    """Capture every module_path the recipe is asked about, in both phases."""
    import megatron.core.quantization.utils as quant_utils

    seen = []
    original = quant_utils.get_quant_config_or_none

    def recording(module_path, recipe):
        seen.append(module_path)
        return original(module_path, recipe)

    for module in _PATCHED_MODULES:
        monkeypatch.setattr(module + ".get_quant_config_or_none", recording)
    return seen


@pytest.mark.skipif(not HAVE_TE, reason="Transformer Engine required for TE linear layers.")
class TestQuantizationModulePaths:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_gpt(self, with_mtp=False, **config_kwargs):
        config = TransformerConfig(
            num_layers=2,
            hidden_size=12,
            num_attention_heads=4,
            use_cpu_initialization=True,
            quant_recipe=RecipeConfig.from_config_dict(_MATCH_EVERYTHING),
            **config_kwargs,
        )
        layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
        mtp_block_spec = (
            get_gpt_mtp_block_spec(config=config, spec=layer_spec, use_transformer_engine=True)
            if with_mtp
            else None
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=64,
            mtp_block_spec=mtp_block_spec,
        )

    def test_construction_paths_exist_in_named_modules(self, monkeypatch):
        seen = _record_module_paths(monkeypatch)
        model = self._build_gpt()

        real_paths = {name for name, _ in model.named_modules()}
        unknown = sorted({path for path in seen if path is not None} - real_paths)
        assert not unknown, (
            "these module paths were used to resolve the recipe but do not exist in the "
            f"model: {unknown}"
        )

    def test_decoder_linears_are_named_during_construction(self, monkeypatch):
        seen = _record_module_paths(monkeypatch)
        self._build_gpt()

        # An unnamed module falls back to the model-wide fp8 settings, so the recipe
        # cannot keep its parameters in high precision.
        assert "decoder.layers.0.mlp.linear_fc1" in seen
        assert "decoder.layers.1.self_attention.linear_qkv" in seen

    def test_mtp_layers_are_named_during_construction(self, monkeypatch):
        seen = _record_module_paths(monkeypatch)
        self._build_gpt(with_mtp=True, mtp_num_layers=1, mtp_loss_scaling_factor=0.1)

        mtp_paths = [path for path in seen if path is not None and path.startswith("mtp.")]
        assert mtp_paths, "MTP block was built without any module names"
        # named_modules() indexes self.layers from 0; a 1-based name would miss depth 0.
        assert any(path.startswith("mtp.layers.0.") for path in mtp_paths), mtp_paths
