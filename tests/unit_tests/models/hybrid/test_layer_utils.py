# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest

from megatron.core.models.hybrid.layers import utils as layer_utils
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig

_LAYER_CONFIG_TYPES = [
    (layer_utils.Symbols.MAMBA, MambaLayerConfig),
    (layer_utils.Symbols.GDN, GDNLayerConfig),
    (layer_utils.Symbols.ATTENTION, AttentionLayerConfig),
    (layer_utils.Symbols.DS_ATTENTION, DSALayerConfig),
    (layer_utils.Symbols.MLA, MLALayerConfig),
    (layer_utils.Symbols.MLP, MLPLayerConfig),
    (layer_utils.Symbols.MOE, MoELayerConfig),
]


def _make_transformer_config(tp_comm_overlap: bool = False) -> TransformerConfig:
    return TransformerConfig(
        num_layers=7, hidden_size=64, num_attention_heads=4, tp_comm_overlap=tp_comm_overlap
    )


@pytest.mark.internal
class TestSymbols:

    def test_name_sorted_valid_layer_symbols(self):
        assert layer_utils.Symbols.name_sorted_valid_layer_symbols() == [
            layer_utils.Symbols.ATTENTION,
            layer_utils.Symbols.DS_ATTENTION,
            layer_utils.Symbols.GDN,
            layer_utils.Symbols.MAMBA,
            layer_utils.Symbols.MLA,
            layer_utils.Symbols.MLP,
            layer_utils.Symbols.MOE,
        ]


@pytest.mark.internal
class TestCreateLayerConfig:

    @pytest.mark.parametrize(("layer_symbol", "config_type"), _LAYER_CONFIG_TYPES)
    def test_creates_independent_config_of_expected_type(self, layer_symbol, config_type):
        config = _make_transformer_config()
        config.test_mutable_value = {"items": []}

        layer_config = layer_utils.create_layer_config(config, layer_symbol)

        assert type(layer_config) is config_type
        assert layer_config is not config
        assert layer_config.__dict__.keys() == config.__dict__.keys()
        assert layer_config.num_layers == config.num_layers
        assert layer_config.hidden_size == config.hidden_size
        assert layer_config.num_attention_heads == config.num_attention_heads
        assert layer_config.test_mutable_value == config.test_mutable_value
        assert layer_config.test_mutable_value is not config.test_mutable_value

    def test_rejects_unknown_symbol(self):
        with pytest.raises(ValueError, match="Unexpected hybrid layer symbol: X"):
            layer_utils.create_layer_config(_make_transformer_config(), "X")


@pytest.mark.internal
class TestGetLayerSymbolFromConfig:

    @pytest.mark.parametrize(("expected_symbol", "config_type"), _LAYER_CONFIG_TYPES)
    def test_returns_symbol_for_each_config_type(self, expected_symbol, config_type):
        layer_config = config_type.from_config(_make_transformer_config())

        assert layer_utils.get_layer_symbol_from_config(layer_config) == expected_symbol

    def test_accepts_config_subclasses(self):
        class CustomMambaLayerConfig(MambaLayerConfig):
            pass

        layer_config = CustomMambaLayerConfig.from_config(_make_transformer_config())

        assert layer_utils.get_layer_symbol_from_config(layer_config) == layer_utils.Symbols.MAMBA

    def test_rejects_unknown_config_type(self):
        config = _make_transformer_config()

        with pytest.raises(
            ValueError, match="Unexpected hybrid layer config type: TransformerConfig"
        ):
            layer_utils.get_layer_symbol_from_config(config)

    def test_rejects_ambiguous_config_type(self):
        class AmbiguousLayerConfig(MambaLayerConfig, AttentionLayerConfig):
            pass

        layer_config = AmbiguousLayerConfig.from_config(_make_transformer_config())

        with pytest.raises(ValueError, match="Ambiguous hybrid layer config type"):
            layer_utils.get_layer_symbol_from_config(layer_config)


@pytest.mark.internal
class TestNormalizeTpCommOverlap:

    @pytest.mark.parametrize(
        ("segment", "has_mtp", "unsupported_features"),
        [
            (layer_utils.Symbols.MLA, False, "MLA"),
            (layer_utils.Symbols.DS_ATTENTION, False, "DSA"),
            ("", True, "MTP"),
            (layer_utils.Symbols.DS_ATTENTION + layer_utils.Symbols.MLA, True, "MLA/DSA/MTP"),
        ],
    )
    def test_disables_overlap_for_unsupported_features(
        self, segment, has_mtp, unsupported_features
    ):
        config = _make_transformer_config(tp_comm_overlap=True)
        expected_warning = (
            "TP communication overlap is not supported with hybrid "
            f"{unsupported_features} layers. Disabling tp_comm_overlap."
        )

        with pytest.warns(UserWarning) as warning_records:
            layer_utils.normalize_tp_comm_overlap(config, segment, has_mtp)

        assert config.tp_comm_overlap is False
        assert [str(warning.message) for warning in warning_records] == [expected_warning]

    @pytest.mark.parametrize(
        ("tp_comm_overlap", "segment", "has_mtp"),
        [
            (
                True,
                layer_utils.Symbols.MAMBA
                + layer_utils.Symbols.GDN
                + layer_utils.Symbols.ATTENTION
                + layer_utils.Symbols.MLP
                + layer_utils.Symbols.MOE,
                False,
            ),
            (False, layer_utils.Symbols.MLA, False),
            (False, "", True),
        ],
    )
    def test_leaves_supported_or_disabled_overlap_unchanged(
        self, tp_comm_overlap, segment, has_mtp
    ):
        config = _make_transformer_config(tp_comm_overlap=tp_comm_overlap)

        with warnings.catch_warnings(record=True) as warning_records:
            warnings.simplefilter("always")
            layer_utils.normalize_tp_comm_overlap(config, segment, has_mtp)

        assert config.tp_comm_overlap is tp_comm_overlap
        assert warning_records == []
