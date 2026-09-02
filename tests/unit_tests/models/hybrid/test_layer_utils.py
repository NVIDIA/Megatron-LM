# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

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

_EXPECTED_LAYER_CONFIG_TYPES = [
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

    @pytest.mark.parametrize(
        "layer_symbol", [layer_symbol for layer_symbol, _ in _EXPECTED_LAYER_CONFIG_TYPES]
    )
    def test_is_valid_symbol(self, layer_symbol):
        assert layer_utils.is_valid_symbol(layer_symbol)

    @pytest.mark.parametrize(
        "layer_symbol", [layer_utils.Symbols.PIPE, layer_utils.Symbols.MTP_SEPARATOR, "X"]
    )
    def test_is_not_valid_symbol(self, layer_symbol):
        assert not layer_utils.is_valid_symbol(layer_symbol)

    def test_is_valid_symbol_allows_pipe(self):
        assert layer_utils.is_valid_symbol(layer_utils.Symbols.PIPE, allow_pipe=True)

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

    def test_attention_layer_configs(self):
        assert layer_utils.Symbols.ATTENTION_LAYER_CONFIGS == {
            AttentionLayerConfig,
            DSALayerConfig,
            MLALayerConfig,
        }


@pytest.mark.internal
class TestCreateLayerConfig:

    @pytest.mark.parametrize(("layer_symbol", "config_type"), _EXPECTED_LAYER_CONFIG_TYPES)
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

    @pytest.mark.parametrize(("expected_symbol", "config_type"), _EXPECTED_LAYER_CONFIG_TYPES)
    def test_returns_symbol_for_each_config_type(self, expected_symbol, config_type):
        layer_config = config_type.from_config(_make_transformer_config())

        assert layer_utils.get_layer_symbol_from_config(layer_config) == expected_symbol

    def test_rejects_config_subclasses(self):
        class CustomMambaLayerConfig(MambaLayerConfig):
            pass

        layer_config = CustomMambaLayerConfig.from_config(_make_transformer_config())

        with pytest.raises(
            ValueError, match="Unexpected hybrid layer config type: CustomMambaLayerConfig"
        ):
            layer_utils.get_layer_symbol_from_config(layer_config)

    def test_rejects_unknown_config_type(self):
        config = _make_transformer_config()

        with pytest.raises(
            ValueError, match="Unexpected hybrid layer config type: TransformerConfig"
        ):
            layer_utils.get_layer_symbol_from_config(config)


@pytest.mark.internal
class TestValidateTpCommOverlap:

    @pytest.mark.parametrize(
        ("segment", "has_mtp", "unsupported_features"),
        [
            (layer_utils.Symbols.MLA, False, "MLA"),
            (layer_utils.Symbols.DS_ATTENTION, False, "DSA"),
            ("", True, "MTP"),
            (layer_utils.Symbols.DS_ATTENTION + layer_utils.Symbols.MLA, True, "MLA/DSA/MTP"),
        ],
    )
    def test_rejects_overlap_for_unsupported_features(self, segment, has_mtp, unsupported_features):
        config = _make_transformer_config(tp_comm_overlap=True)
        expected_error = (
            "TP communication overlap is not supported with hybrid "
            f"{unsupported_features} layers. Set tp_comm_overlap=False."
        )

        with pytest.raises(ValueError) as exc_info:
            layer_utils.validate_tp_comm_overlap(config, segment, has_mtp)

        assert config.tp_comm_overlap is True
        assert str(exc_info.value) == expected_error

    @pytest.mark.parametrize(
        ("config_type", "unsupported_feature"), [(MLALayerConfig, "MLA"), (DSALayerConfig, "DSA")]
    )
    def test_accepts_layer_configs(self, config_type, unsupported_feature):
        config = _make_transformer_config(tp_comm_overlap=True)
        layer_config = config_type.from_config(config)

        with pytest.raises(ValueError, match=f"hybrid {unsupported_feature} layers"):
            layer_utils.validate_tp_comm_overlap(config, [layer_config])

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

        layer_utils.validate_tp_comm_overlap(config, segment, has_mtp)

        assert config.tp_comm_overlap is tp_comm_overlap
