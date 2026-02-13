# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from unittest.mock import patch

import pytest

from megatron.core.ssm.mamba_hybrid_layer_allocation import (
    ParsedHybridPattern,
    Symbols,
    get_hybrid_layer_counts,
    get_hybrid_total_layer_count,
    get_hybrid_total_pipeline_segment_count,
    parse_hybrid_pattern,
    select_pipeline_segment,
    validate_segment_layers,
)


@pytest.mark.internal
class TestValidateSegmentLayers:

    def test_valid_patterns(self):
        """Test that valid segment patterns produce the correct layer type lists."""
        test_cases = [
            ("M*-M*-M*-", ['M', '*', '-', 'M', '*', '-', 'M', '*', '-']),
            ("MMMMMMMMM", ['M'] * 9),
            ("MM*-MM*-", ['M', 'M', '*', '-', 'M', 'M', '*', '-']),
            ("E", ['E']),
            ("", []),
        ]
        for pattern, expected in test_cases:
            result = validate_segment_layers(pattern)
            assert result == expected, f"Failed for pattern: {pattern}"

    def test_all_valid_symbols(self):
        """Make sure all returned layers are valid."""
        for pattern in ["M*-M*-M*-", "MMMMMMMMM", "MM*-", "MEME"]:
            layer_types = validate_segment_layers(pattern)
            for layer_type in layer_types:
                assert layer_type in Symbols.VALID_LAYERS

    def test_invalid_symbols_cause_failure(self):
        """Test that invalid symbols raise ValueError."""
        with pytest.raises(ValueError):
            validate_segment_layers("M*X")
        with pytest.raises(ValueError):
            validate_segment_layers("M|M")  # pipe not valid in a segment
        with pytest.raises(ValueError):
            validate_segment_layers("M/M")  # MTP separator not valid in a segment


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
        assert get_hybrid_layer_counts("M*M*") == (2, 2, 0, 0)

    def test_all_layer_types(self):
        assert get_hybrid_layer_counts("M*-E") == (1, 1, 1, 1)

    def test_with_pipes(self):
        # Pipes should be skipped in counting
        assert get_hybrid_layer_counts("M*|M*") == (2, 2, 0, 0)
        assert get_hybrid_layer_counts("M-M-|M-M*-") == (1, 4, 4, 0)

    def test_with_mtp(self):
        # MTP pattern "MM" repeated 2 depths -> 4 extra mamba layers
        assert get_hybrid_layer_counts("M*M*/MM/MM") == (2, 6, 0, 0)

    def test_with_pipes_and_mtp(self):
        # Main: M-M-|M-M*- -> 1 attn, 4 mamba, 4 mlp
        # MTP: MM x 2 depths -> +4 mamba
        assert get_hybrid_layer_counts("M-M-|M-M*-/MM/MM") == (1, 8, 4, 0)

    def test_moe_pattern(self):
        assert get_hybrid_layer_counts("MEME") == (0, 2, 0, 2)

    def test_mtp_with_attention(self):
        # MTP pattern "*M" repeated 3 depths -> 3 attn + 3 mamba from MTP
        assert get_hybrid_layer_counts("MMMM/*M/*M/*M") == (3, 7, 0, 0)

    def test_empty_pattern(self):
        assert get_hybrid_layer_counts("") == (0, 0, 0, 0)


@pytest.mark.internal
class TestSelectPipelineSegment:
    """Tests for select_pipeline_segment with pp_group=None (single rank).

    When pp_group is None, pp_rank=0 and pp_size=1, so the segment index
    is simply the vp_stage value.
    """

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_single_segment_no_vp(self, mock_log):
        """Single segment, no VPP."""
        layer_types, offset = select_pipeline_segment("M*M*", pp_group=None, vp_stage=None)
        assert layer_types == ['M', '*', 'M', '*']
        assert offset == 0

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_two_segments_vp0(self, mock_log):
        """Two segments, select first (vp_stage=0)."""
        layer_types, offset = select_pipeline_segment("M-M-|M-M*-", pp_group=None, vp_stage=0)
        assert layer_types == ['M', '-', 'M', '-']
        assert offset == 0

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_two_segments_vp1(self, mock_log):
        """Two segments, select second (vp_stage=1)."""
        layer_types, offset = select_pipeline_segment("M-M-|M-M*-", pp_group=None, vp_stage=1)
        assert layer_types == ['M', '-', 'M', '*', '-']
        assert offset == 4

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_four_segments(self, mock_log):
        """Four segments, verify each vp_stage selects correctly."""
        pattern = "MM|M*|M-|ME"
        expected = [(['M', 'M'], 0), (['M', '*'], 2), (['M', '-'], 4), (['M', 'E'], 6)]
        for vp_stage, (expected_layers, expected_offset) in enumerate(expected):
            layer_types, offset = select_pipeline_segment(pattern, pp_group=None, vp_stage=vp_stage)
            assert layer_types == expected_layers, f"Failed for vp_stage={vp_stage}"
            assert offset == expected_offset, f"Failed for vp_stage={vp_stage}"

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_empty_segment(self, mock_log):
        """Empty segments are allowed for pipeline balancing."""
        layer_types, offset = select_pipeline_segment("||M*", pp_group=None, vp_stage=0)
        assert layer_types == []
        assert offset == 0

        layer_types, offset = select_pipeline_segment("||M*", pp_group=None, vp_stage=2)
        assert layer_types == ['M', '*']
        assert offset == 0

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_uneven_segments(self, mock_log):
        """Segments of different lengths."""
        pattern = "MMM|M|MMMMM"
        layer_types, offset = select_pipeline_segment(pattern, pp_group=None, vp_stage=0)
        assert len(layer_types) == 3
        assert offset == 0

        layer_types, offset = select_pipeline_segment(pattern, pp_group=None, vp_stage=1)
        assert len(layer_types) == 1
        assert offset == 3

        layer_types, offset = select_pipeline_segment(pattern, pp_group=None, vp_stage=2)
        assert len(layer_types) == 5
        assert offset == 4

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_empty_main_pattern(self, mock_log):
        """Empty main pattern produces one empty segment."""
        layer_types, offset = select_pipeline_segment("", pp_group=None, vp_stage=None)
        assert layer_types == []
        assert offset == 0

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_invalid_segment_raises(self, mock_log):
        """Invalid layer symbols in a segment should raise ValueError."""
        with pytest.raises(ValueError):
            select_pipeline_segment("MX|M*", pp_group=None, vp_stage=0)

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_out_of_range_segment_raises(self, mock_log):
        """Segment index out of range should raise IndexError."""
        with pytest.raises(IndexError):
            select_pipeline_segment("M*|M*", pp_group=None, vp_stage=5)

    @patch('megatron.core.ssm.mamba_hybrid_layer_allocation.log_on_each_pipeline_stage')
    def test_logging_is_called(self, mock_log):
        """Verify that log_on_each_pipeline_stage is called."""
        select_pipeline_segment("M*M*", pp_group=None, vp_stage=None)
        mock_log.assert_called_once()
