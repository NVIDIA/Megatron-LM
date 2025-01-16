# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import math
import re

import pytest
import torch

from megatron.core.ssm.mamba_hybrid_layer_allocation import Symbols, allocate_layers


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
