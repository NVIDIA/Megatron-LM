# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest
import torch

from megatron.core.transformer.moe.moe_utils import pad_routing_map


def test_pad_routing_map_aligns_when_enough_zeros():
    routing_map = torch.zeros((100, 2), dtype=torch.bool)
    routing_map[:30, 0] = True
    routing_map[:45, 1] = True

    padded = pad_routing_map(routing_map, pad_multiple=32)

    tokens_per_expert = padded.sum(dim=0)
    assert int(tokens_per_expert[0]) % 32 == 0
    assert int(tokens_per_expert[1]) % 32 == 0


def test_pad_routing_map_warns_when_alignment_is_not_reachable():
    # Expert 0 has 90 tokens but only 10 zero entries, so it cannot be padded
    # up to the next multiple of 128. The function should report this instead
    # of silently returning an unaligned routing map.
    routing_map = torch.zeros((100, 2), dtype=torch.bool)
    routing_map[:90, 0] = True

    with pytest.warns(UserWarning, match="unaligned"):
        padded = pad_routing_map(routing_map, pad_multiple=128)

    tokens_per_expert = padded.sum(dim=0)
    assert int(tokens_per_expert[0]) % 128 != 0


def test_pad_routing_map_no_warning_when_already_aligned():
    routing_map = torch.zeros((64, 2), dtype=torch.bool)
    routing_map[:32, 0] = True
    routing_map[:16, 1] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        padded = pad_routing_map(routing_map, pad_multiple=32)

    tokens_per_expert = padded.sum(dim=0)
    assert int(tokens_per_expert[0]) % 32 == 0
    assert int(tokens_per_expert[1]) % 32 == 0
