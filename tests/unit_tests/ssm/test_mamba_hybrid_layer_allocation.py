# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import re

import pytest
import torch

from megatron.core.ssm.mamba_hybrid_layer_allocation import (
    ParsedHybridPattern,
    Symbols,
    allocate_layers,
    parse_hybrid_pattern,
)


@pytest.mark.internal
class TestMambaHybridLayerAllocation:

    def test_hybrid_layer_allocation(self):
        # The format for the test cases is:
        # (layers_count, attention_ratio, mlp_ratio, override_pattern).
        test_cases = [
            (9, 0.0, 0.0, "M*-M*-M*-"),
            (9, 0.0, 0.0, "MMMMMMMMM"),
            (30, 0.0, 0.0, None),
            (8, 0.25, 0.25, "MM*-MM*-"),
            (8, 0.5, 0.25, "M**-M**-"),
            (48, 0.5, 0.2, None),
        ]
        for test in test_cases:
            (layers_count, attention_ratio, mlp_ratio, override_pattern) = test

            layer_types = allocate_layers(*test)

            # Check that return value is in the right format.
            assert isinstance(layer_types, list)
            assert layers_count == len(layer_types)

            # Make sure all the layers are valid.
            for layer_type in layer_types:
                assert layer_type in Symbols.VALID

            # Make sure each layer is as requested by override_pattern.
            if override_pattern is not None:
                assert len(override_pattern) == len(layer_types)
                for index, layer_type in enumerate(layer_types):
                    assert override_pattern[index] == layer_types[index]
            else:
                # Make sure the count of each type of layer is correct.
                counts = {layer_type: 0 for layer_type in Symbols.VALID}  # Initialize all to zero.
                for layer_type in layer_types:
                    assert layer_type in counts
                    counts[layer_type] += 1
                # Check the ratios.
                remainder = 1.0 - attention_ratio - mlp_ratio
                assert remainder >= 0
                assert int(attention_ratio * layers_count + 0.5) == counts[Symbols.ATTENTION]
                assert int(mlp_ratio * layers_count + 0.5) == counts[Symbols.MLP]
                assert int(remainder * layers_count + 0.5) == counts[Symbols.MAMBA]

            # Make sure the ratios are as requested.
            # This code is not working yet because capsys seems broken in Megatron.
            # captured = capsys.readouterr()  # Remove this output from the capture buffer.
            # out = captured.out  # Get stdout.
            # if attention_ratio != 0 or mlp_ratio != 0:
            #     assert (
            #             match := re.search(r'Actual attention ratio: (1\.0|0\.[0-9]+)\.', out)
            #     ) and math.isclose(match.group(1), attention_ratio)
            #     assert (
            #             match := re.search(r'Actual mlp ratio: (1\.0|0\.[0-9]+)\.', out)
            #     ) and math.isclose(match.group(1), mlp_ratio)

    @pytest.mark.xfail(raises=ValueError)
    def test_wrong_length_override_pattern(self):
        # This override_pattern is too short.
        layer_types = allocate_layers(9, 0.0, 0.0, "M*-M*-")

    @pytest.mark.xfail(raises=ValueError)
    def test_wrong_number_of_layer_types_in_override_pattern(self):
        # This override_pattern has too many mlps and not enough attention
        layer_types = allocate_layers(8, 0.5, 0.25, "M*--M**-")


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
