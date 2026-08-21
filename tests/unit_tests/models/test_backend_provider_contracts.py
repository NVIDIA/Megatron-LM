# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Contracts a provider has to keep that no other test was covering.

Each of these is a regression: a caller asking a provider for something, and getting an
answer that only worked before because of a module-level global, or that silently ignored
what the config asked for.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
from megatron.core.fusions.fused_layer_norm import HAVE_FUSED_LAYER_NORM, FusedLayerNorm
from megatron.core.models.backends import (
    BackendSpecProvider,
    InferenceSpecProvider,
    LocalSpecProvider,
    backend_slot,
    get_backend,
    get_backend_spec_provider,
    te_cross_entropy,
    unfused_cross_entropy,
)
from megatron.core.transformer.multi_token_prediction import get_mtp_layer_spec_for_backend
from megatron.core.transformer.torch_norm import WrappedTorchNorm
from tests.unit_tests.test_utilities import Utils


def _config(**overrides):
    settings = dict(
        transformer_impl="local",
        normalization="LayerNorm",
        cross_entropy_loss_fusion=False,
        cross_entropy_fusion_impl="native",
        cuda_graph_impl=None,
    )
    settings.update(overrides)
    return SimpleNamespace(**settings)


class TestNormalizationIsAsked:
    """A provider cannot guess the normalization; the caller has to say."""

    def test_rms_norm_never_gets_a_layer_norm_only_kernel(self):
        """Apex's fused kernel asserts on an RMSNorm config, so it must not be offered."""
        assert LocalSpecProvider().layer_norm(rms_norm=True) is WrappedTorchNorm

    @pytest.mark.skipif(not HAVE_FUSED_LAYER_NORM, reason="Apex is not installed")
    def test_layer_norm_still_gets_the_fused_kernel(self):
        assert LocalSpecProvider().layer_norm(rms_norm=False) is FusedLayerNorm

    def test_the_answer_does_not_depend_on_earlier_calls(self):
        """This is what the module-level LNImpl got wrong: it was mutated in place, so the
        answer depended on whether an RMSNorm model had been built earlier in the process."""
        backend = LocalSpecProvider()
        first = backend.layer_norm(rms_norm=False)
        backend.layer_norm(rms_norm=True)
        assert backend.layer_norm(rms_norm=False) is first


class TestMtpAsksForItsConfigsNorm:
    """MTP used to inherit its norm from whatever had mutated LNImpl first."""

    def test_local_rms_norm_mtp_does_not_get_a_layer_norm_only_kernel(self):
        spec = get_mtp_layer_spec_for_backend(
            mtp_model_layer_spec=None, backend=LocalSpecProvider(), rms_norm=True
        )
        for slot in ("enorm", "hnorm", "layer_norm"):
            assert getattr(spec.submodules, slot) is WrappedTorchNorm, slot

    def test_the_default_is_layer_norm(self):
        spec = get_mtp_layer_spec_for_backend(
            mtp_model_layer_spec=None, backend=LocalSpecProvider()
        )
        assert spec.submodules.enorm is LocalSpecProvider().layer_norm(rms_norm=False)


class TestCrossEntropyFollowsTheConfig:
    """Which kernel is a config decision, not a property of the chosen backend."""

    def test_no_fusion(self):
        target = get_backend_spec_provider(_config()).vocab_parallel_cross_entropy()
        assert target is unfused_cross_entropy

    def test_native_fusion(self):
        target = get_backend_spec_provider(
            _config(cross_entropy_loss_fusion=True)
        ).vocab_parallel_cross_entropy()
        assert target is fused_vocab_parallel_cross_entropy

    def test_an_explicit_te_request_is_not_silently_ignored(self):
        """megatron/training/arguments.py rejects this, but a config built directly can ask."""
        target = get_backend_spec_provider(
            _config(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="te")
        ).vocab_parallel_cross_entropy()
        assert target is not fused_vocab_parallel_cross_entropy
        assert getattr(target, "func", None) is te_cross_entropy

    def test_an_unknown_impl_is_refused_rather_than_quietly_downgraded(self):
        """Base raised UnboundLocalError here -- ugly, but not silent."""
        with pytest.raises(ValueError, match="unknown cross_entropy_fusion_impl"):
            get_backend_spec_provider(
                _config(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="TE")
            ).vocab_parallel_cross_entropy()

    def test_the_choice_does_not_depend_on_which_backend_supplies_the_model(self):
        settings = dict(cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="native")
        local = get_backend_spec_provider(_config(**settings)).vocab_parallel_cross_entropy()
        te = get_backend_spec_provider(
            _config(transformer_impl="transformer_engine", **settings)
        ).vocab_parallel_cross_entropy()
        assert local is te


class TestBertLmHeadNormDoesNotFollowTransformerImpl:
    """``--spec local`` picks local BERT layers without touching ``transformer_impl``.

    Reading ``transformer_impl`` for this norm would put a TE norm on a local encoder, and
    would silently replace the Apex/Torch norm this head has always used.
    """

    def _head(self, transformer_impl):
        from megatron.core.models.bert.bert_lm_head import BertLMHead
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            transformer_impl=transformer_impl,
            use_cpu_initialization=True,
        )
        return BertLMHead(hidden_size=16, config=config)

    @pytest.mark.parametrize("transformer_impl", ["local", "transformer_engine"])
    def test_the_head_norm_is_the_same_whatever_transformer_impl_says(self, transformer_impl):
        expected = LocalSpecProvider().layer_norm(rms_norm=False)
        assert type(self._head(transformer_impl).layer_norm) is expected

    def test_the_head_norm_is_never_the_transformer_engine_one(self):
        head = self._head("transformer_engine")
        assert type(head.layer_norm).__name__ != "TENorm"


class TestOlderProvidersKeepWorking:
    """BackendSpecProvider is a Protocol, so slots added to it reach third-party providers
    as missing attributes rather than as inherited defaults."""

    def test_a_provider_without_the_newer_slots_still_builds_a_model(self):
        class ProviderFromTheOlderContract:
            """Implements what the protocol asked for before mlp_module/moe_router existed."""

            def column_parallel_linear(self):
                return None

        backend = ProviderFromTheOlderContract()
        assert backend_slot(backend, "moe_router", lambda: None) is None
        assert backend_slot(backend, "mlp_module", lambda: "fallback", grouped=False) == "fallback"

    def test_a_provider_that_implements_them_is_still_asked(self):
        assert backend_slot(LocalSpecProvider(), "moe_router", lambda: "unused") is None

    def test_a_wrapper_that_inherits_the_protocol_does_not_answer_for_its_fallback(self):
        """The hazard Kitchen's wrapper would hit: subclassing the protocol picks up the new
        slots as inherited stubs, so a wrapper that delegates everything else would silently
        answer for the backend it wraps. backend_slot must not treat that as an answer."""

        class WrapperFromTheOlderContract(BackendSpecProvider):
            """Implements every method the protocol had before the three newer slots, and
            delegates each one -- as a wrapping provider does. It inherits the newer slots
            without implementing them, which is the hazard."""

            def __init__(self, fallback):
                self.fallback = fallback

            def column_parallel_linear(self):
                return self.fallback.column_parallel_linear()

            def row_parallel_linear(self):
                return self.fallback.row_parallel_linear()

            def fuse_layernorm_and_linear(self):
                return self.fallback.fuse_layernorm_and_linear()

            def column_parallel_layer_norm_linear(self):
                return self.fallback.column_parallel_layer_norm_linear()

            def layer_norm(self, rms_norm=False, for_qk=False, has_residual=False):
                return self.fallback.layer_norm(rms_norm, for_qk, has_residual)

            def core_attention(self):
                return self.fallback.core_attention()

            def grouped_mlp_modules(self, moe_use_grouped_gemm):
                return self.fallback.grouped_mlp_modules(moe_use_grouped_gemm)

            def activation_func(self):
                return self.fallback.activation_func()

        wrapper = WrapperFromTheOlderContract(LocalSpecProvider())
        sentinel = object()
        assert backend_slot(wrapper, "vocab_parallel_cross_entropy", lambda: sentinel) is sentinel
        assert backend_slot(wrapper, "mlp_module", lambda: sentinel, grouped=False) is sentinel

    def test_a_wrapper_still_forwards_the_slots_it_does_implement(self):
        """The fallback must keep answering for everything the wrapper delegates."""
        assert backend_slot(LocalSpecProvider(), "layer_norm", lambda: None, rms_norm=True) is (
            WrappedTorchNorm
        )


class TestLocalT5DoesNotNeedTransformerEngine:
    """A local block spec must be constructible without Transformer Engine installed."""

    def test_the_local_block_norm_comes_from_the_local_backend(self):
        from megatron.core.models.T5 import t5_spec

        block = t5_spec.get_t5_encoder_with_local_block_spec(1)
        assert block.layer_norm is LocalSpecProvider().layer_norm()


class TestDecoderBlockAsksForItsConfigsNorm:
    """The fix this PR is named for: a local RMSNorm run must not be handed Apex's
    LayerNorm-only kernel for the final norm."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _final_norm(self, normalization):
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
        from megatron.core.transformer.transformer_config import TransformerConfig

        config = TransformerConfig(
            num_layers=2,
            hidden_size=16,
            num_attention_heads=1,
            normalization=normalization,
            transformer_impl="local",
            use_cpu_initialization=True,
        )
        return get_gpt_decoder_block_spec(config, use_transformer_engine=False).layer_norm

    def test_rms_norm_gets_a_norm_that_can_serve_it(self):
        assert self._final_norm("RMSNorm") is WrappedTorchNorm

    def test_layer_norm_is_unchanged(self):
        assert self._final_norm("LayerNorm") is LocalSpecProvider().layer_norm(rms_norm=False)


class TestMissingOptionalPackagesAreRefused:
    """A stub or a None target must not be built into a model to fail somewhere else."""

    def test_kitchen_is_refused_when_it_is_only_the_public_stub(self):
        """The stub's KitchenSpecProvider is a MagicMock, so construction would "succeed"."""
        with patch("megatron.core.extensions.kitchen.HAVE_KITCHEN", False):
            with pytest.raises(ImportError, match="Kitchen is not installed"):
                get_backend("local", use_kitchen=True)

    def test_the_inference_backend_declares_that_it_needs_transformer_engine(self):
        assert InferenceSpecProvider.REQUIRES == "transformer_engine"
