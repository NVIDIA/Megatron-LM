"""Tests for ``LigerSpecProvider`` — the Liger-Kernel BackendSpecProvider."""

import pytest

pytest.importorskip("liger_kernel.megatron")

from liger_kernel.megatron import LigerMegatronRMSNorm

from megatron.core.extensions import liger_kernel_spec_provider as provider_module
from megatron.core.extensions.liger_kernel_spec_provider import LigerSpecProvider
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def test_layer_norm_slot_returns_liger_rmsnorm():
    """rms_norm=True must route to LigerMegatronRMSNorm."""
    provider = LigerSpecProvider()
    assert provider.layer_norm(rms_norm=True) is LigerMegatronRMSNorm


def test_layer_norm_slot_falls_back_for_non_rmsnorm():
    """rms_norm=False must fall through to the LocalSpecProvider's LayerNorm builder."""
    from megatron.core.models.backends import LocalSpecProvider

    provider = LigerSpecProvider()
    expected = LocalSpecProvider().layer_norm(rms_norm=False)
    assert provider.layer_norm(rms_norm=False) is expected


def test_missing_package_raises(monkeypatch):
    """Instantiation must raise ImportError with a clear message when liger-kernel is absent."""
    monkeypatch.setattr(provider_module, "HAVE_LIGER", False)
    monkeypatch.setattr(provider_module, "LigerMegatronRMSNorm", None)

    with pytest.raises(ImportError, match="Liger-Kernel is required"):
        LigerSpecProvider()


def test_use_liger_flag_wires_through_gpt_layer_specs():
    """get_gpt_layer_local_submodules(use_liger=True) must produce a spec whose
    norm slots resolve to LigerMegatronRMSNorm."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules

    submodules = get_gpt_layer_local_submodules(
        normalization="RMSNorm", use_liger=True
    )
    assert submodules.input_layernorm is LigerMegatronRMSNorm
    assert submodules.pre_mlp_layernorm is LigerMegatronRMSNorm


def test_use_liger_is_mutually_exclusive_with_use_kitchen():
    """use_liger=True and use_kitchen=True must not be combined."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules

    with pytest.raises(AssertionError, match="mutually exclusive"):
        get_gpt_layer_local_submodules(
            normalization="RMSNorm", use_kitchen=True, use_liger=True
        )


def test_block_spec_use_liger_is_mutually_exclusive_with_use_kitchen():
    """The block-spec builder must also reject use_kitchen + use_liger. The assert
    fires before any process-group access, so no distributed init is needed."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_layer_specs

    config = TransformerConfig(
        num_layers=2, hidden_size=128, num_attention_heads=8, normalization="RMSNorm"
    )
    # Set post-construction to bypass TransformerConfig's kitchen availability checks.
    config.use_kitchen = True

    with pytest.raises(AssertionError, match="mutually exclusive"):
        get_gpt_decoder_layer_specs(
            config, use_transformer_engine=False, normalization="RMSNorm", use_liger=True
        )


def test_decoder_block_spec_selects_liger_final_norm():
    """get_gpt_decoder_block_spec(use_liger=True) must select LigerMegatronRMSNorm as
    the block-level (final) norm, and fall back to the default when use_liger=False."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec

    Utils.initialize_model_parallel(1, 1)
    try:
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            normalization="RMSNorm",
            use_cpu_initialization=True,
        )
        liger_block = get_gpt_decoder_block_spec(
            config, use_transformer_engine=False, normalization="RMSNorm", use_liger=True
        )
        assert liger_block.layer_norm is LigerMegatronRMSNorm

        default_block = get_gpt_decoder_block_spec(
            config, use_transformer_engine=False, normalization="RMSNorm", use_liger=False
        )
        assert default_block.layer_norm is not LigerMegatronRMSNorm
    finally:
        Utils.destroy_model_parallel()


def test_constructed_model_final_norm_is_liger():
    """A GPTModel built from the Liger block spec must have both its per-layer norms
    and its ``decoder.final_layernorm`` resolve to LigerMegatronRMSNorm."""
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    Utils.initialize_model_parallel(1, 1)
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            normalization="RMSNorm",
            use_cpu_initialization=True,
        )
        block_spec = get_gpt_decoder_block_spec(
            config, use_transformer_engine=False, normalization="RMSNorm", use_liger=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=block_spec,
            vocab_size=128,
            max_sequence_length=16,
        )
        assert isinstance(model.decoder.final_layernorm, LigerMegatronRMSNorm)
        # Per-layer norms must be Liger too.
        assert isinstance(model.decoder.layers[0].input_layernorm, LigerMegatronRMSNorm)
    finally:
        Utils.destroy_model_parallel()
