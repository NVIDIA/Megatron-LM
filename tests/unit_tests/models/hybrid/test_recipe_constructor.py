# Copyright (c) 2025-2026, NVIDIA CORPORATION. All rights reserved.

"""Equivalence tests for the two recipe-path entry points.

Phase 1 of the HybridModel recipe-API design adds direct
:class:`HybridModelConfig` support to :meth:`HybridModel.__init__`::

    HybridModel(config=recipe, ...)

The pre-existing :meth:`HybridModel.from_recipe` classmethod continues to
work — it is now a thin alias that delegates to the constructor. This
module verifies the two entry points produce equivalent models (same layer
types, same parameter counts, same forward signature) and that the
constructor rejects mutually-exclusive kwargs on the recipe path.

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
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.test_utilities import Utils


def _make_common(**overrides) -> CommonLayerConfig:
    # ``hidden_dropout=0.1`` matches the TransformerConfig default; the
    # CommonLayerConfig default (0.0) would silently diverge from any
    # legacy-CLI counterpart that left ``--hidden-dropout`` at its default.
    base = dict(hidden_size=256, use_cpu_initialization=True, hidden_dropout=0.1)
    base.update(overrides)
    return CommonLayerConfig(**base)


def _make_recipe() -> HybridModelConfig:
    common = _make_common()
    return HybridModelConfig(
        common_config=common,
        layer_pattern=[
            EmbeddingLayerConfig(common_config=common, vocab_size=100, max_sequence_length=4),
            MambaLayerConfig(common_config=common),
            AttentionLayerConfig(common_config=common, num_attention_heads=4),
            MLPLayerConfig(common_config=common),
            CrossEntropyLayerConfig(),
        ],
        # Match HybridModel's legacy default
        # (``share_embeddings_and_output_weights=False``).
        untie_embeddings_and_output_weights=True,
    )


@pytest.mark.internal
class TestRecipeConstructorEquivalence:
    """``HybridModel(config=recipe)`` and ``HybridModel.from_recipe(recipe)``
    must produce equivalent models.

    ``from_recipe`` is now a thin alias for the constructor; this guards
    against drift if either path is touched independently.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _build_via_constructor(self) -> HybridModel:
        model_parallel_cuda_manual_seed(123)
        return HybridModel(config=_make_recipe())

    def _build_via_from_recipe(self) -> HybridModel:
        model_parallel_cuda_manual_seed(123)
        return HybridModel.from_recipe(_make_recipe())

    def test_layer_types_match(self):
        ctor = self._build_via_constructor()
        factory = self._build_via_from_recipe()
        assert ctor.decoder.layer_type_list == factory.decoder.layer_type_list
        assert len(ctor.decoder.layers) == len(factory.decoder.layers)

    def test_parameter_count_matches(self):
        ctor_params = sum(p.numel() for p in self._build_via_constructor().parameters())
        factory_params = sum(p.numel() for p in self._build_via_from_recipe().parameters())
        assert ctor_params == factory_params

    def test_forward_signature_matches(self):
        """Forward outputs must agree on shape and dtype.

        Bit-equality is not asserted: ``from_recipe`` delegates to the
        constructor, so today the two paths consume the CUDA RNG
        identically — but if either picks up additional construction work
        (e.g. a future ``validate()`` call, or an extra log line that
        touches state), bit-equality could drift. Shape/dtype is the
        structural guarantee that matters here; deeper equivalence is
        already covered by ``test_python_dsl_equivalence``.
        """
        ctor = self._build_via_constructor().cuda()
        factory = self._build_via_from_recipe().cuda()

        seq_len = ctor.max_sequence_length
        bsz = 2
        data = list(range(seq_len))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((bsz, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((bsz, 1)).cuda()
        attention_mask = torch.ones((bsz, 1, seq_len, seq_len), dtype=bool).cuda()

        ctor_out = ctor.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        factory_out = factory.forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        assert ctor_out.shape == factory_out.shape
        assert ctor_out.dtype == factory_out.dtype


@pytest.mark.internal
class TestRecipeConstructorRejection:
    """The recipe-path entry point rejects kwargs that conflict with the
    recipe DSL. These guards keep recipes from being silently shadowed by
    a stale CLI flag in callers that mix entry points.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_rejects_string_pattern(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HybridModel(config=_make_recipe(), hybrid_layer_pattern="M*-")

    def test_rejects_legacy_attention_ratio(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HybridModel(config=_make_recipe(), hybrid_attention_ratio=0.5)

    def test_rejects_legacy_mlp_ratio(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HybridModel(config=_make_recipe(), hybrid_mlp_ratio=0.5)

    def test_rejects_legacy_override_pattern(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            HybridModel(config=_make_recipe(), hybrid_override_pattern="M*-")

    def test_rejects_explicit_recipe_lowering(self):
        recipe = _make_recipe()
        lowering = recipe._lower()
        with pytest.raises(ValueError, match="populated automatically"):
            HybridModel(config=recipe, _recipe_lowering=lowering)
