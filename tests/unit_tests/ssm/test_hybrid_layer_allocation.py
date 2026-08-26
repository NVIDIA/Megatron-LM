# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import functools
import operator
from unittest.mock import patch

import pytest

from megatron.core.models.hybrid.hybrid_layer_allocation import (
    ParsedHybridPattern,
    Symbols,
    get_hybrid_layer_counts,
    get_hybrid_total_layer_count,
    get_hybrid_total_pipeline_segment_count,
    get_layer_maps_from_layer_type_list,
    parse_hybrid_pattern,
    pattern_from_ratios,
    select_pipeline_segment,
    validate_segment_layers,
)
from megatron.core.ssm.gdn_layer_config import GDNLayerConfig
from megatron.core.ssm.mamba_layer_config import MambaLayerConfig
from megatron.core.ssm.mlp_layer_config import MLPLayerConfig
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.attention_layer_config import AttentionLayerConfig
from megatron.core.transformer.experimental_attention_variant.dsa_layer_config import DSALayerConfig
from megatron.core.transformer.mla_layer_config import MLALayerConfig
from megatron.core.transformer.moe.moe_layer_config import MoELayerConfig
from megatron.core.transformer.transformer_config import MLATransformerConfig

_EXPECTED_LAYER_CONFIG_CLASSES = {
    Symbols.MAMBA: MambaLayerConfig,
    Symbols.GDN: GDNLayerConfig,
    Symbols.ATTENTION: AttentionLayerConfig,
    Symbols.DS_ATTENTION: DSALayerConfig,
    Symbols.MLA: MLALayerConfig,
    Symbols.MLP: MLPLayerConfig,
    Symbols.MOE: MoELayerConfig,
}


def _make_transformer_config() -> TransformerConfig:
    return TransformerConfig(num_layers=7, hidden_size=64, num_attention_heads=4)


def _assert_layer_config_types(layer_config_list, pattern: str) -> None:
    assert [type(config) for config in layer_config_list] == [
        _EXPECTED_LAYER_CONFIG_CLASSES[layer_symbol] for layer_symbol in pattern
    ]


def _assert_config_contents_equal(actual, expected) -> None:
    assert vars(actual).keys() == vars(expected).keys()
    for field_name, expected_value in vars(expected).items():
        actual_value = getattr(actual, field_name)
        if isinstance(expected_value, functools.partial):
            assert isinstance(actual_value, functools.partial)
            assert actual_value.func is expected_value.func
            assert actual_value.args == expected_value.args
            assert actual_value.keywords == expected_value.keywords
        else:
            assert actual_value == expected_value


@pytest.mark.internal
class TestPatternFromRatios:

    def test_pure_mamba(self):
        result = pattern_from_ratios(8, attention_ratio=0.0, mlp_ratio=0.0)
        assert result == "MMMMMMMM"

    def test_attention_only(self):
        result = pattern_from_ratios(10, attention_ratio=0.3)
        assert result.count(Symbols.ATTENTION) == 3
        assert result.count(Symbols.MAMBA) == 7
        assert len(result) == 10

    def test_attention_and_mlp(self):
        result = pattern_from_ratios(10, attention_ratio=0.3, mlp_ratio=0.3)
        assert result.count(Symbols.ATTENTION) == 3
        assert result.count(Symbols.MLP) == 3
        assert result.count(Symbols.MAMBA) == 4
        assert len(result) == 10

    def test_attention_evenly_spaced(self):
        result = pattern_from_ratios(10, attention_ratio=0.5)
        assert result.count(Symbols.ATTENTION) == 5
        assert result.count(Symbols.MAMBA) == 5
        attn_positions = [i for i, ch in enumerate(result) if ch == Symbols.ATTENTION]
        gaps = [attn_positions[i + 1] - attn_positions[i] for i in range(len(attn_positions) - 1)]
        assert all(
            g in (1, 2, 3) for g in gaps
        ), f"Gaps between attention layers should be small, got {gaps}"

    def test_mlp_does_not_replace_attention(self):
        result = pattern_from_ratios(10, attention_ratio=0.3, mlp_ratio=0.3)
        attn_positions = [i for i, c in enumerate(result) if c == Symbols.ATTENTION]
        mlp_positions = [i for i, c in enumerate(result) if c == Symbols.MLP]
        assert not set(attn_positions) & set(mlp_positions)

    def test_single_layer(self):
        assert pattern_from_ratios(1, 0.0, 0.0) == "M"
        assert pattern_from_ratios(1, 1.0, 0.0) == "*"

    def test_returns_string(self):
        result = pattern_from_ratios(4, 0.5)
        assert isinstance(result, str)


@pytest.mark.internal
class TestValidateSegmentLayers:

    def setup_method(self):
        self.config = _make_transformer_config()

    def test_valid_patterns(self):
        """Test that valid segment patterns produce configs in the correct order."""
        for pattern in [
            "M*-M*-M*-",
            "MMMMMMMMM",
            "MM*-MM*-",
            "E",
            "",
            "GGG*GGG*",
            "GEGEGE*E",
            "MDMD",
            "M+M+",
        ]:
            result = validate_segment_layers(pattern, self.config)
            _assert_layer_config_types(result, pattern)

    def test_all_valid_pattern_characters(self):
        """Make sure all returned layers are valid."""
        for pattern in ["M*-M*-M*-", "MMMMMMMMM", "MM*-", "MEME"]:
            layer_config_list = validate_segment_layers(pattern, self.config)
            for layer_config in layer_config_list:
                assert type(layer_config) in _EXPECTED_LAYER_CONFIG_CLASSES.values()

    @pytest.mark.parametrize(
        ("layer_symbol", "config_class"), list(_EXPECTED_LAYER_CONFIG_CLASSES.items())
    )
    def test_all_symbols_map_to_layer_configs(self, layer_symbol, config_class):
        layer_config_list = validate_segment_layers(layer_symbol, self.config)

        assert len(layer_config_list) == 1
        assert type(layer_config_list[0]) is config_class
        assert isinstance(layer_config_list[0], TransformerConfig)
        assert layer_config_list[0] is not self.config
        _assert_config_contents_equal(layer_config_list[0], self.config)
        assert layer_config_list[0].hidden_size == self.config.hidden_size
        _assert_layer_config_types(layer_config_list, layer_symbol)

    def test_all_layer_symbols_have_an_expected_config_class(self):
        assert set(_EXPECTED_LAYER_CONFIG_CLASSES) == Symbols.VALID_LAYERS
        assert Symbols.PIPE not in _EXPECTED_LAYER_CONFIG_CLASSES
        assert Symbols.MTP_SEPARATOR not in _EXPECTED_LAYER_CONFIG_CLASSES

    def test_repeated_layers_receive_independent_config_copies(self):
        self.config.test_mutable_value = {"items": []}

        layer_config_list = validate_segment_layers("MMM", self.config)

        assert type(layer_config_list) is list
        assert len({id(config) for config in layer_config_list}) == 3
        assert all(config is not self.config for config in layer_config_list)
        assert all(config.test_mutable_value == {"items": []} for config in layer_config_list)
        assert len({id(config.test_mutable_value) for config in layer_config_list}) == 3

        layer_config_list[0].test_mutable_value["items"].append("changed")
        assert layer_config_list[1].test_mutable_value == {"items": []}
        assert self.config.test_mutable_value == {"items": []}

    def test_mla_configs_preserve_specialized_fields(self):
        config = MLATransformerConfig(
            num_layers=2,
            hidden_size=128,
            num_attention_heads=8,
            q_lora_rank=32,
            kv_lora_rank=16,
            qk_head_dim=32,
            qk_pos_emb_head_dim=16,
            v_head_dim=32,
            rope_type="rope",
        )

        dsa_config, mla_config = validate_segment_layers("D+", config)

        assert type(dsa_config) is DSALayerConfig
        assert type(mla_config) is MLALayerConfig
        assert isinstance(dsa_config, MLATransformerConfig)
        assert isinstance(mla_config, MLATransformerConfig)
        assert dsa_config.q_lora_rank == config.q_lora_rank
        assert mla_config.kv_lora_rank == config.kv_lora_rank
        assert dsa_config is not mla_config

    def test_invalid_symbols_cause_failure(self):
        """Test that invalid symbols raise ValueError."""
        with pytest.raises(ValueError):
            validate_segment_layers("M*X", self.config)
        with pytest.raises(ValueError):
            validate_segment_layers("M|M", self.config)  # pipe not valid in a segment
        with pytest.raises(ValueError):
            validate_segment_layers("M/M", self.config)  # MTP separator not valid in a segment
        with pytest.raises(ValueError):
            # Not allowed to have both standard Attention and MLA/DSA
            validate_segment_layers("MDM*-", self.config)
        with pytest.raises(ValueError):
            # Not allowed to have both standard Attention and MLA (same reason
            # as DSA: * uses the model-level rotary_pos_emb while + uses MLA's
            # own decoupled RoPE).
            validate_segment_layers("M+M*-", self.config)


@pytest.mark.internal
class TestGetHybridTotalLayerCount:

    def test_simple_patterns(self):
        assert get_hybrid_total_layer_count("M*M*") == 4
        assert get_hybrid_total_layer_count("MMMM") == 4
        assert get_hybrid_total_layer_count("M") == 1

    def test_with_pipe_separators(self):
        assert get_hybrid_total_layer_count("M-M-|M-M*-") == 9
        assert get_hybrid_total_layer_count("M-M-|M-M*-|M-M-|M-M*-") == 18
        assert get_hybrid_total_layer_count("||M") == 1
        assert get_hybrid_total_layer_count("M|M") == 2

    def test_with_mtp(self):
        assert get_hybrid_total_layer_count("M*M*/MM/MM") == 4
        assert get_hybrid_total_layer_count("M-M-|M-M*-/MM/MM") == 9

    def test_empty(self):
        assert get_hybrid_total_layer_count("") == 0


@pytest.mark.internal
class TestGetHybridTotalPipelineSegmentCount:

    def test_no_pipe(self):
        assert get_hybrid_total_pipeline_segment_count("M*M*") == 1

    def test_with_pipes(self):
        assert get_hybrid_total_pipeline_segment_count("M-M-|M-M*-") == 2
        assert get_hybrid_total_pipeline_segment_count("M|M|M|M") == 4
        assert get_hybrid_total_pipeline_segment_count("||M") == 3

    def test_with_mtp(self):
        assert get_hybrid_total_pipeline_segment_count("M-M-|M-M*-/MM/MM") == 2


@pytest.mark.internal
class TestParseHybridPattern:
    """Tests for parse_hybrid_pattern with unified pattern syntax."""

    def test_none_pattern(self):
        """Test that None pattern returns all None values."""
        result = parse_hybrid_pattern(None)
        assert result.main_pattern is None
        assert result.mtp_pattern is None
        assert result.mtp_num_depths == 0

    def test_main_pattern_only(self):
        """Test patterns without MTP (no / separator)."""
        test_cases = [
            ("M*M*", "M*M*"),
            ("MMMM", "MMMM"),
            ("*M*M", "*M*M"),
            ("MM-*", "MM-*"),
            ("E", "E"),
            ("GGG*GGG*", "GGG*GGG*"),
            ("GEGEGE*E", "GEGEGE*E"),
            ("MDMD", "MDMD"),
            ("DM", "DM"),
            ("M+M+", "M+M+"),
            ("+M", "+M"),
        ]
        for pattern, expected_main in test_cases:
            result = parse_hybrid_pattern(pattern)
            assert result.main_pattern == expected_main, f"Failed for pattern: {pattern}"
            assert result.mtp_pattern is None
            assert result.mtp_num_depths == 0

    def test_main_pattern_with_pipes(self):
        """Test patterns with pipe separators (no MTP)."""
        test_cases = [("M*|M*", "M*|M*"), ("M-M-|M-M*-", "M-M-|M-M*-"), ("M|M|M|M", "M|M|M|M")]
        for pattern, expected_main in test_cases:
            result = parse_hybrid_pattern(pattern)
            assert result.main_pattern == expected_main, f"Failed for pattern: {pattern}"
            assert result.mtp_pattern is None
            assert result.mtp_num_depths == 0

    def test_main_with_single_mtp_depth(self):
        """Test patterns with 1 MTP depth."""
        test_cases = [
            ("M*M*/MM", "M*M*", "MM", 1),
            ("MMMM/*M", "MMMM", "*M", 1),
            ("M/M", "M", "M", 1),
        ]
        for pattern, expected_main, expected_mtp, expected_depths in test_cases:
            result = parse_hybrid_pattern(pattern)
            assert result.main_pattern == expected_main, f"Failed for pattern: {pattern}"
            assert result.mtp_pattern == expected_mtp, f"Failed for pattern: {pattern}"
            assert result.mtp_num_depths == expected_depths, f"Failed for pattern: {pattern}"

    def test_main_with_multiple_mtp_depths(self):
        """Test patterns with multiple MTP depths."""
        test_cases = [
            ("M*M*/MM/MM", "M*M*", "MM", 2),
            ("M*M*/MM/MM/MM", "M*M*", "MM", 3),
            ("MMMM/*M/*M/*M", "MMMM", "*M", 3),
            ("M*/*/*/*", "M*", "*", 3),
            ("M/M/M/M/M", "M", "M", 4),
        ]
        for pattern, expected_main, expected_mtp, expected_depths in test_cases:
            result = parse_hybrid_pattern(pattern)
            assert result.main_pattern == expected_main, f"Failed for pattern: {pattern}"
            assert result.mtp_pattern == expected_mtp, f"Failed for pattern: {pattern}"
            assert result.mtp_num_depths == expected_depths, f"Failed for pattern: {pattern}"

    def test_pipe_with_mtp(self):
        """Test patterns with both pipe and MTP separators."""
        result = parse_hybrid_pattern("M-M-|M-M*-/MM/MM")
        assert result.main_pattern == "M-M-|M-M*-"
        assert result.mtp_pattern == "MM"
        assert result.mtp_num_depths == 2

    def test_mtp_patterns_must_be_identical(self):
        """Test that mismatched MTP patterns raise ValueError."""
        invalid_patterns = [
            "M*M*/MM/M*",  # MM != M*
            "M*M*/MM/MM/M",  # MM != M
            "MMMM/*M/M*",  # *M != M*
        ]
        for pattern in invalid_patterns:
            with pytest.raises(ValueError, match="All MTP patterns must be identical"):
                parse_hybrid_pattern(pattern)

    def test_invalid_symbols_in_main_pattern(self):
        """Test that invalid symbols in main pattern raise ValueError."""
        invalid_patterns = [
            "M*X*",  # X is not valid
            "MaMM",  # a is not valid
            "M*M*1",  # 1 is not valid
        ]
        for pattern in invalid_patterns:
            with pytest.raises(ValueError, match="not a valid layer symbol"):
                parse_hybrid_pattern(pattern)

    def test_invalid_symbols_in_mtp_pattern(self):
        """Test that invalid symbols in MTP pattern raise ValueError."""
        # Single MTP depth with invalid symbol - should raise "not a valid layer symbol"
        with pytest.raises(ValueError, match="not a valid layer symbol"):
            parse_hybrid_pattern("M*M*/MX")  # X is not valid

        # Multiple MTP depths with invalid symbol and matching patterns
        with pytest.raises(ValueError, match="not a valid layer symbol"):
            parse_hybrid_pattern("M*M*/Ma/Ma")  # a is not valid

        # Multiple MTP depths with invalid symbol but mismatched patterns
        # This raises "All MTP patterns must be identical" before checking symbols
        with pytest.raises(ValueError, match="All MTP patterns must be identical"):
            parse_hybrid_pattern("M*M*/MM/Ma")

    def test_pipe_not_allowed_in_mtp(self):
        """Test that pipe symbol in MTP pattern raises ValueError."""
        with pytest.raises(ValueError, match="not a valid layer symbol"):
            parse_hybrid_pattern("M*M*/M|M/M|M")

    def test_empty_main_pattern_with_mtp(self):
        """Test pattern that starts with / (empty main pattern)."""
        result = parse_hybrid_pattern("/MM/MM")
        assert result.main_pattern is None
        assert result.mtp_pattern == "MM"
        assert result.mtp_num_depths == 2

    def test_trailing_separator(self):
        """Test patterns with trailing separator."""
        # "M*M*/" means main="M*M*", one empty MTP pattern
        result = parse_hybrid_pattern("M*M*/")
        assert result.main_pattern == "M*M*"
        # Empty string after separator means no valid MTP pattern
        assert result.mtp_pattern is None
        assert result.mtp_num_depths == 0

    def test_complex_patterns(self):
        """Test more complex realistic patterns."""
        test_cases = [
            # Main decoder with attention, MTP with mamba only
            ("M*M*M*M*/MMM/MMM", "M*M*M*M*", "MMM", 2),
            # Main decoder with MLP, MTP with attention+mamba
            ("MM-MM-/*M/*M", "MM-MM-", "*M", 2),
            # All attention main, mamba MTP
            ("*****/M/M/M/M", "*****", "M", 4),
            # MoE in main pattern
            ("MEME/MM/MM", "MEME", "MM", 2),
            # GDN+MoE main pattern with GDN MTP
            ("GEGEGE*E/GG/GG", "GEGEGE*E", "GG", 2),
            # DSA in main pattern with MTP
            ("MDMD/MD/MD", "MDMD", "MD", 2),
            # MLA in main pattern with MTP
            ("M+M+/M+/M+", "M+M+", "M+", 2),
        ]
        for pattern, expected_main, expected_mtp, expected_depths in test_cases:
            result = parse_hybrid_pattern(pattern)
            assert result.main_pattern == expected_main, f"Failed for pattern: {pattern}"
            assert result.mtp_pattern == expected_mtp, f"Failed for pattern: {pattern}"
            assert result.mtp_num_depths == expected_depths, f"Failed for pattern: {pattern}"

    def test_dataclass_equality(self):
        """Test that ParsedHybridPattern supports equality comparison."""
        p1 = parse_hybrid_pattern("M*M*/MM/MM")
        p2 = ParsedHybridPattern(main_pattern="M*M*", mtp_pattern="MM", mtp_num_depths=2)
        assert p1 == p2


@pytest.mark.internal
class TestGetHybridLayerCounts:

    def test_simple_pattern(self):
        assert get_hybrid_layer_counts("M*M*") == {
            '*': 2,
            'D': 0,
            'G': 0,
            'M': 2,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_all_layer_types(self):
        # Not allowed to have both standard Attention and MLA/DSA, so we do separate asserts.
        assert get_hybrid_layer_counts("MG*-E") == {
            '*': 1,
            'D': 0,
            'G': 1,
            'M': 1,
            '+': 0,
            '-': 1,
            'E': 1,
        }
        assert get_hybrid_layer_counts("MGD-E") == {
            '*': 0,
            'D': 1,
            'G': 1,
            'M': 1,
            '+': 0,
            '-': 1,
            'E': 1,
        }
        assert get_hybrid_layer_counts("MG+-E") == {
            '*': 0,
            'D': 0,
            'G': 1,
            'M': 1,
            '+': 1,
            '-': 1,
            'E': 1,
        }

    def test_with_pipes(self):
        # Pipes should be skipped in counting
        assert get_hybrid_layer_counts("M*|M*") == {
            '*': 2,
            'D': 0,
            'G': 0,
            'M': 2,
            '+': 0,
            '-': 0,
            'E': 0,
        }
        assert get_hybrid_layer_counts("M-M-|M-M*-") == {
            '*': 1,
            'D': 0,
            'G': 0,
            'M': 4,
            '+': 0,
            '-': 4,
            'E': 0,
        }

    def test_with_mtp(self):
        # MTP pattern "MM" repeated 2 depths -> 4 extra mamba layers
        assert get_hybrid_layer_counts("M*M*/MM/MM") == {
            '*': 2,
            'D': 0,
            'G': 0,
            'M': 6,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_with_pipes_and_mtp(self):
        # Main: M-M-|M-M*- -> 1 attn, 4 mamba, 4 mlp
        # MTP: MM x 2 depths -> +4 mamba
        assert get_hybrid_layer_counts("M-M-|M-M*-/MM/MM") == {
            '*': 1,
            'D': 0,
            'G': 0,
            'M': 8,
            '+': 0,
            '-': 4,
            'E': 0,
        }

    def test_moe_pattern(self):
        assert get_hybrid_layer_counts("MEME") == {
            '*': 0,
            'D': 0,
            'G': 0,
            'M': 2,
            '+': 0,
            '-': 0,
            'E': 2,
        }

    def test_mtp_with_attention(self):
        # MTP pattern "*M" repeated 3 depths -> 3 attn + 3 mamba from MTP
        assert get_hybrid_layer_counts("MMMM/*M/*M/*M") == {
            '*': 3,
            'D': 0,
            'G': 0,
            'M': 7,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_gdn_pattern(self):
        assert get_hybrid_layer_counts("GMGM") == {
            '*': 0,
            'D': 0,
            'G': 2,
            'M': 2,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_gdn_hybrid_pattern(self):
        # GDN + Mamba + Attention
        assert get_hybrid_layer_counts("G*GM*") == {
            '*': 2,
            'D': 0,
            'G': 2,
            'M': 1,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_dsa_pattern(self):
        assert get_hybrid_layer_counts("DMDM") == {
            '*': 0,
            'D': 2,
            'G': 0,
            'M': 2,
            '+': 0,
            '-': 0,
            'E': 0,
        }

    def test_mla_pattern(self):
        assert get_hybrid_layer_counts("+M+M") == {
            '*': 0,
            'D': 0,
            'G': 0,
            'M': 2,
            '+': 2,
            '-': 0,
            'E': 0,
        }

    def test_empty_pattern(self):
        assert get_hybrid_layer_counts("") == {
            '*': 0,
            'D': 0,
            'G': 0,
            'M': 0,
            '+': 0,
            '-': 0,
            'E': 0,
        }


@pytest.mark.internal
class TestSelectPipelineSegment:
    """Tests for select_pipeline_segment with pp_group=None (single rank).

    When pp_group is None, pp_rank=0 and pp_size=1, so the segment index
    is simply the vp_stage value.
    """

    def setup_method(self):
        self.config = _make_transformer_config()

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_single_segment_no_vp(self, mock_log):
        """Single segment, no VPP."""
        layer_configs, offset = select_pipeline_segment(
            "M*M*", self.config, pp_group=None, vp_stage=None
        )
        _assert_layer_config_types(layer_configs, "M*M*")
        assert offset == 0

    @pytest.mark.parametrize("layer_symbol", [Symbols.MLA, Symbols.DS_ATTENTION])
    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_rejects_tp_overlap_for_selected_segment(self, mock_log, layer_symbol):
        self.config.tp_comm_overlap = True

        with pytest.raises(
            ValueError, match="TP communication overlap is not supported with hybrid"
        ):
            select_pipeline_segment(layer_symbol, self.config, pp_group=None, vp_stage=None)

        assert self.config.tp_comm_overlap is True

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_does_not_reject_tp_overlap_for_unselected_mla_segment(self, mock_log):
        self.config.tp_comm_overlap = True

        layer_configs, _ = select_pipeline_segment(
            f"{Symbols.MLA}|{Symbols.MAMBA}", self.config, pp_group=None, vp_stage=1
        )

        assert self.config.tp_comm_overlap is True
        assert all(layer_config.tp_comm_overlap is True for layer_config in layer_configs)

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_two_segments_vp0(self, mock_log):
        """Two segments, select first (vp_stage=0)."""
        layer_configs, offset = select_pipeline_segment(
            "M-M-|M-M*-", self.config, pp_group=None, vp_stage=0
        )
        _assert_layer_config_types(layer_configs, "M-M-")
        assert offset == 0

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_two_segments_vp1(self, mock_log):
        """Two segments, select second (vp_stage=1)."""
        layer_configs, offset = select_pipeline_segment(
            "M-M-|M-M*-", self.config, pp_group=None, vp_stage=1
        )
        _assert_layer_config_types(layer_configs, "M-M*-")
        assert offset == 4

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_four_segments(self, mock_log):
        """Four segments, verify each vp_stage selects correctly."""
        pattern = "MM|M*|M-|ME"
        expected = [("MM", 0), ("M*", 2), ("M-", 4), ("ME", 6)]
        for vp_stage, (expected_pattern, expected_offset) in enumerate(expected):
            layer_configs, offset = select_pipeline_segment(
                pattern, self.config, pp_group=None, vp_stage=vp_stage
            )
            _assert_layer_config_types(layer_configs, expected_pattern)
            assert offset == expected_offset, f"Failed for vp_stage={vp_stage}"

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_empty_segment(self, mock_log):
        """Empty segments are allowed for pipeline balancing."""
        layer_configs, offset = select_pipeline_segment(
            "||M*", self.config, pp_group=None, vp_stage=0
        )
        assert layer_configs == []
        assert offset == 0

        layer_configs, offset = select_pipeline_segment(
            "||M*", self.config, pp_group=None, vp_stage=2
        )
        _assert_layer_config_types(layer_configs, "M*")
        assert offset == 0

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_uneven_segments(self, mock_log):
        """Segments of different lengths."""
        pattern = "MMM|M|MMMMM"
        layer_configs, offset = select_pipeline_segment(
            pattern, self.config, pp_group=None, vp_stage=0
        )
        assert len(layer_configs) == 3
        assert offset == 0

        layer_configs, offset = select_pipeline_segment(
            pattern, self.config, pp_group=None, vp_stage=1
        )
        assert len(layer_configs) == 1
        assert offset == 3

        layer_configs, offset = select_pipeline_segment(
            pattern, self.config, pp_group=None, vp_stage=2
        )
        assert len(layer_configs) == 5
        assert offset == 4

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_empty_main_pattern(self, mock_log):
        """Empty main pattern produces one empty segment."""
        layer_configs, offset = select_pipeline_segment(
            "", self.config, pp_group=None, vp_stage=None
        )
        assert layer_configs == []
        assert offset == 0

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_invalid_segment_raises(self, mock_log):
        """Invalid layer symbols in a segment should raise ValueError."""
        with pytest.raises(ValueError):
            select_pipeline_segment("MX|M*", self.config, pp_group=None, vp_stage=0)

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_out_of_range_segment_raises(self, mock_log):
        """Segment index out of range should raise ValueError."""
        with pytest.raises(ValueError, match="out of range"):
            select_pipeline_segment("M*|M*", self.config, pp_group=None, vp_stage=5)

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_logging_is_called(self, mock_log):
        """Verify that log_on_each_pipeline_stage is called."""
        select_pipeline_segment("M*M*", self.config, pp_group=None, vp_stage=None)
        mock_log.assert_called_once()

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_logging_receives_explicit_groups(self, mock_log):
        tp_group = object()
        dp_cp_group = object()
        select_pipeline_segment(
            "M*M*",
            self.config,
            pp_group=None,
            vp_stage=None,
            tp_group=tp_group,
            dp_cp_group=dp_cp_group,
        )
        assert mock_log.call_args.kwargs["tp_group"] is tp_group
        assert mock_log.call_args.kwargs["dp_cp_group"] is dp_cp_group

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_mutual_exclusivity_pipes_with_first_stage(self, mock_log):
        """Pipe separators + first_stage_layers should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot specify"):
            select_pipeline_segment(
                "M*|M*", self.config, pp_group=None, vp_stage=0, first_stage_layers=1
            )

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_mutual_exclusivity_pipes_with_last_stage(self, mock_log):
        """Pipe separators + last_stage_layers should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot specify"):
            select_pipeline_segment(
                "M*|M*", self.config, pp_group=None, vp_stage=0, last_stage_layers=1
            )

    @patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_segment_count_not_divisible_by_pp_size(self, mock_log):
        """Segment count not divisible by pp_size should raise ValueError."""
        mock_group = object()
        with (
            patch('torch.distributed.get_rank', return_value=0),
            patch('torch.distributed.get_world_size', return_value=2),
        ):
            with pytest.raises(ValueError, match="evenly divisible"):
                select_pipeline_segment("M|M|M", self.config, pp_group=mock_group, vp_stage=None)


@pytest.mark.internal
class TestSelectPipelineSegmentLegacyFallback:
    """Tests for the no-pipes fallback path in select_pipeline_segment.

    These tests exercise the backwards-compatible auto-split logic that
    activates when the pattern has no pipe separators but pp_size > 1.
    """

    def setup_method(self):
        self.config = _make_transformer_config()

    def _call_for_rank(
        self,
        pattern,
        pp_rank,
        pp_size,
        vp_stage=None,
        first_stage_layers=None,
        last_stage_layers=None,
    ):
        """Call select_pipeline_segment with mocked PP group for a given rank."""
        mock_group = object()
        with (
            patch('torch.distributed.get_rank', return_value=pp_rank),
            patch('torch.distributed.get_world_size', return_value=pp_size),
            patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage'),
            patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_single_rank'),
        ):
            return select_pipeline_segment(
                pattern,
                self.config,
                pp_group=mock_group,
                vp_stage=vp_stage,
                first_stage_layers=first_stage_layers,
                last_stage_layers=last_stage_layers,
            )

    def test_even_split_2_ranks(self):
        """4 layers across 2 ranks -> 2 each."""
        layers0, off0 = self._call_for_rank("M*M-", pp_rank=0, pp_size=2)
        _assert_layer_config_types(layers0, "M*")
        assert off0 == 0

        layers1, off1 = self._call_for_rank("M*M-", pp_rank=1, pp_size=2)
        _assert_layer_config_types(layers1, "M-")
        assert off1 == 2

    def test_even_split_4_ranks(self):
        """8 layers across 4 ranks -> 2 each."""
        pattern = "M*M*M*M*"
        for rank in range(4):
            layers, offset = self._call_for_rank(pattern, pp_rank=rank, pp_size=4)
            assert len(layers) == 2
            assert offset == rank * 2

    def test_even_split_not_divisible_raises(self):
        """6 layers across 4 ranks with no uneven PP -> ValueError."""
        with pytest.raises(ValueError, match="evenly divisible"):
            self._call_for_rank("M*M*M*", pp_rank=0, pp_size=4)

    def test_uneven_pp_first_stage(self):
        """6 layers, pp_size=4, first_stage=3 -> first gets 3, others get 1."""
        pattern = "M*M*M*"
        layers0, off0 = self._call_for_rank(pattern, pp_rank=0, pp_size=4, first_stage_layers=3)
        assert len(layers0) == 3
        assert off0 == 0

        layers1, off1 = self._call_for_rank(pattern, pp_rank=1, pp_size=4, first_stage_layers=3)
        assert len(layers1) == 1
        assert off1 == 3

        layers3, off3 = self._call_for_rank(pattern, pp_rank=3, pp_size=4, first_stage_layers=3)
        assert len(layers3) == 1
        assert off3 == 5

    def test_uneven_pp_last_stage(self):
        """6 layers, pp_size=4, last_stage=3 -> last gets 3, others get 1."""
        pattern = "M*M*M*"
        layers0, off0 = self._call_for_rank(pattern, pp_rank=0, pp_size=4, last_stage_layers=3)
        assert len(layers0) == 1
        assert off0 == 0

        layers3, off3 = self._call_for_rank(pattern, pp_rank=3, pp_size=4, last_stage_layers=3)
        assert len(layers3) == 3
        assert off3 == 3

    def test_uneven_pp_first_and_last(self):
        """8 layers, pp_size=4, first=1, last=1 -> first 1, middle 3 each, last 1."""
        pattern = "M*M*M*M*"
        layers0, off0 = self._call_for_rank(
            pattern, pp_rank=0, pp_size=4, first_stage_layers=1, last_stage_layers=1
        )
        assert len(layers0) == 1
        assert off0 == 0

        layers1, off1 = self._call_for_rank(
            pattern, pp_rank=1, pp_size=4, first_stage_layers=1, last_stage_layers=1
        )
        assert len(layers1) == 3
        assert off1 == 1

        layers2, off2 = self._call_for_rank(
            pattern, pp_rank=2, pp_size=4, first_stage_layers=1, last_stage_layers=1
        )
        assert len(layers2) == 3
        assert off2 == 4

        layers3, off3 = self._call_for_rank(
            pattern, pp_rank=3, pp_size=4, first_stage_layers=1, last_stage_layers=1
        )
        assert len(layers3) == 1
        assert off3 == 7

    def test_uneven_pp_middle_not_divisible_raises(self):
        """Middle layers not divisible by middle stages -> ValueError."""
        with pytest.raises(ValueError, match="Middle layers"):
            self._call_for_rank("M*M*M*M*M", pp_rank=0, pp_size=4, first_stage_layers=2)

    def test_vpp_with_no_pipes_raises(self):
        """VPP (vp_stage != None) without pipe separators -> ValueError."""
        with pytest.raises(ValueError, match="Virtual pipeline parallelism"):
            self._call_for_rank("M*M*", pp_rank=0, pp_size=2, vp_stage=0)

    def test_deprecation_warning_logged(self):
        """Legacy path should log a deprecation warning via log_single_rank."""
        mock_group = object()
        with (
            patch('torch.distributed.get_rank', return_value=0),
            patch('torch.distributed.get_world_size', return_value=2),
            patch('megatron.core.models.hybrid.hybrid_layer_allocation.log_on_each_pipeline_stage'),
            patch(
                'megatron.core.models.hybrid.hybrid_layer_allocation.log_single_rank'
            ) as mock_warn,
        ):
            select_pipeline_segment("M*M*", self.config, pp_group=mock_group, vp_stage=None)
            mock_warn.assert_called_once()
            call_args = mock_warn.call_args
            assert "DEPRECATION" in call_args[0][2]

    def test_all_ranks_cover_full_pattern(self):
        """All ranks together should reconstruct the original layer list."""
        pattern = "M*M*M*"
        pp_size = 3
        all_layers = []
        for rank in range(pp_size):
            layers, offset = self._call_for_rank(pattern, pp_rank=rank, pp_size=pp_size)
            assert offset == len(all_layers)
            all_layers.extend(layers)
        _assert_layer_config_types(all_layers, "M*M*M*")


@pytest.mark.internal
class TestGetLayerMapsFromLayerTypeList:
    """Tests for get_layer_maps_from_layer_type_list."""

    def test_standard_layer_types(self):
        """Standard symbols each produce a single-entry map at local index 0."""
        maps = get_layer_maps_from_layer_type_list(
            [Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE]
        )
        # We always get all symbols, not only those contained in the pattern.
        assert len(maps) == len(Symbols.VALID_LAYERS)
        attention_map, mamba_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE
        )(maps)
        assert attention_map == {0: 0}
        assert mamba_map == {1: 0}
        assert mlp_map == {2: 0}
        assert moe_map == {3: 0}

    def test_dsa(self):
        """DSA layers have their own local cache indices."""
        maps = get_layer_maps_from_layer_type_list(
            [Symbols.DS_ATTENTION, Symbols.MAMBA, Symbols.DS_ATTENTION, Symbols.MAMBA]
        )
        attention_map, dsa_map, mamba_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE
        )(maps)
        assert attention_map == {}
        assert dsa_map == {0: 0, 2: 1}
        assert mamba_map == {1: 0, 3: 1}
        assert mlp_map == {}
        assert moe_map == {}

    def test_mixed_attention_and_dsa(self):
        """Attention and DSA layers maintain separate local indices."""
        maps = get_layer_maps_from_layer_type_list(
            [Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.MAMBA, Symbols.MLP]
        )
        attention_map, dsa_map, mamba_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION, Symbols.DS_ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE
        )(maps)
        assert attention_map == {0: 0}
        assert dsa_map == {1: 0}
        assert mamba_map == {2: 0}
        assert mlp_map == {3: 0}
        assert moe_map == {}

    def test_all_mamba(self):
        """All-Mamba patterns leave the other maps empty."""
        maps = get_layer_maps_from_layer_type_list([Symbols.MAMBA] * 3)
        attention_map, mamba_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION, Symbols.MAMBA, Symbols.MLP, Symbols.MOE
        )(maps)
        assert attention_map == {}
        assert mamba_map == {0: 0, 1: 1, 2: 2}
        assert mlp_map == {}
        assert moe_map == {}

    def test_mla(self):
        """MLA layers have their own local cache indices."""
        maps = get_layer_maps_from_layer_type_list(
            [Symbols.MLA, Symbols.MAMBA, Symbols.MLA, Symbols.MAMBA]
        )
        attention_map, dsa_map, mamba_map, mla_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION,
            Symbols.DS_ATTENTION,
            Symbols.MAMBA,
            Symbols.MLA,
            Symbols.MLP,
            Symbols.MOE,
        )(maps)
        assert attention_map == {}
        assert dsa_map == {}
        assert mla_map == {0: 0, 2: 1}
        assert mamba_map == {1: 0, 3: 1}
        assert mlp_map == {}
        assert moe_map == {}

    def test_mixed_dsa_and_mla(self):
        """DSA and MLA layers maintain separate local indices."""
        maps = get_layer_maps_from_layer_type_list(
            [Symbols.DS_ATTENTION, Symbols.MLA, Symbols.MAMBA, Symbols.MLP]
        )
        attention_map, dsa_map, mamba_map, mla_map, mlp_map, moe_map = operator.itemgetter(
            Symbols.ATTENTION,
            Symbols.DS_ATTENTION,
            Symbols.MAMBA,
            Symbols.MLA,
            Symbols.MLP,
            Symbols.MOE,
        )(maps)
        assert attention_map == {}
        assert dsa_map == {0: 0}
        assert mla_map == {1: 0}
        assert mamba_map == {2: 0}
        assert mlp_map == {3: 0}
        assert moe_map == {}
