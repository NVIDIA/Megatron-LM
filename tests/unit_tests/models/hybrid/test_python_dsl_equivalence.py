# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""End-to-end equivalence tests between the legacy string DSL and the recipe DSL.

Building a HybridModel via ``hybrid_layer_pattern="M*-"`` and via a
``HybridModelConfig`` recipe must produce equivalent models: identical layer
types, identical parameter counts, and — when seeded the same way — identical
forward outputs.

Heterogeneous-within-type smoke tests confirm that the Python DSL's per-layer
config plumbing actually reaches the constructed layer instances.

These tests require CUDA + Transformer Engine (matching the rest of the
hybrid test suite) and run inside the Megatron-LM CI container.
"""

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support  # noqa: F401  # CI gate

pytest.importorskip("mamba_ssm", reason="HybridModel Mamba equivalence tests require MambaSSM")

from megatron.core.models.hybrid import (
    AttentionLayerConfig,
    CommonLayerConfig,
    CrossEntropyLayerConfig,
    EmbeddingLayerConfig,
    HybridModelConfig,
    MambaLayerConfig,
    MLPLayerConfig,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _make_common(**overrides) -> CommonLayerConfig:
    base = dict(hidden_size=256, use_cpu_initialization=True)
    base.update(overrides)
    return CommonLayerConfig(**base)


def _make_legacy_config(**overrides) -> TransformerConfig:
    base = dict(
        num_layers=3,
        hidden_size=256,
        num_attention_heads=4,
        use_cpu_initialization=True,
        is_hybrid_model=True,
    )
    base.update(overrides)
    return TransformerConfig(**base)


def _embedding(common: CommonLayerConfig) -> EmbeddingLayerConfig:
    return EmbeddingLayerConfig(common_config=common, vocab_size=100, max_sequence_length=4)


def _loss() -> CrossEntropyLayerConfig:
    return CrossEntropyLayerConfig()


def _build_model_from_recipe(recipe: HybridModelConfig) -> HybridModel:
    compiled = recipe.compile()
    return HybridModel(
        config=compiled.config,
        vocab_size=compiled.vocab_size,
        max_sequence_length=compiled.max_sequence_length,
        fp16_lm_cross_entropy=compiled.fp16_lm_cross_entropy,
        parallel_output=compiled.parallel_output,
        share_embeddings_and_output_weights=compiled.share_embeddings_and_output_weights,
        position_embedding_type=compiled.position_embedding_type,
        rotary_percent=compiled.rotary_percent,
        rotary_base=compiled.rotary_base,
        scatter_embedding_sequence_parallel=compiled.scatter_embedding_sequence_parallel,
        seq_len_interpolation_factor=compiled.seq_len_interpolation_factor,
        layer_type_list=compiled.layer_type_list,
        layer_config_list=compiled.layer_config_list,
    )


@pytest.mark.internal
class TestPythonDSLStringEquivalence:
    """Two models built from the same architecture — one via the string DSL, one
    via the recipe DSL — must be functionally identical."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_string_dsl_model(self) -> HybridModel:
        model_parallel_cuda_manual_seed(123)
        return HybridModel(
            config=_make_legacy_config(),
            hybrid_stack_spec=hybrid_stack_spec,
            vocab_size=100,
            max_sequence_length=4,
            hybrid_layer_pattern="M*-",
        )

    def _build_python_dsl_model(self) -> HybridModel:
        model_parallel_cuda_manual_seed(123)
        common = _make_common()
        recipe = HybridModelConfig(
            common_config=common,
            layer_pattern=[
                _embedding(common),
                MambaLayerConfig(common_config=common),
                AttentionLayerConfig(common_config=common, num_attention_heads=4),
                MLPLayerConfig(common_config=common),
                _loss(),
            ],
        )
        return _build_model_from_recipe(recipe)

    def test_layer_types_match(self):
        legacy = self._build_string_dsl_model()
        new = self._build_python_dsl_model()
        assert legacy.decoder.layer_type_list == new.decoder.layer_type_list
        assert len(legacy.decoder.layers) == len(new.decoder.layers)

    def test_parameter_count_matches(self):
        legacy_params = sum(p.numel() for p in self._build_string_dsl_model().parameters())
        new_params = sum(p.numel() for p in self._build_python_dsl_model().parameters())
        assert legacy_params == new_params

    def test_forward_outputs_match(self):
        legacy = self._build_string_dsl_model().cuda()
        new = self._build_python_dsl_model().cuda()

        seq_len = legacy.max_sequence_length
        bsz = 2
        data = list(range(seq_len))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((bsz, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((bsz, 1)).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=bool).cuda()

        legacy_out = legacy.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        new_out = new.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        torch.testing.assert_close(legacy_out, new_out, rtol=0, atol=0)


@pytest.mark.internal
class TestHeterogeneousWithinType:
    """Within-type heterogeneity (e.g. two distinct MambaLayerConfig instances)
    must produce layers whose constructed config reflects the per-layer
    overrides — not just the global common config."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_per_layer_mamba_overrides_reach_constructed_layer(self):
        common = _make_common()
        big = MambaLayerConfig(common_config=common, num_heads=8, head_dim=32)
        small = MambaLayerConfig(common_config=common, num_heads=4, head_dim=64)

        model_parallel_cuda_manual_seed(123)
        recipe = HybridModelConfig(
            common_config=common, layer_pattern=[_embedding(common), big, small, _loss()]
        )
        model = _build_model_from_recipe(recipe)

        # The constructed layers' configs must match the per-layer overrides,
        # not the (default) global common config.
        layer0_cfg = model.decoder.layer_config_list[0]
        layer1_cfg = model.decoder.layer_config_list[1]
        assert layer0_cfg.mamba_num_heads == 8
        assert layer0_cfg.mamba_head_dim == 32
        assert layer1_cfg.mamba_num_heads == 4
        assert layer1_cfg.mamba_head_dim == 64
        assert layer0_cfg is not layer1_cfg
