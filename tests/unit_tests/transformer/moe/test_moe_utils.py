# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import logging

import torch

from megatron.core.transformer.moe.moe_utils import (
    pad_routing_map,
    warn_if_tokens_per_expert_unaligned,
)


def test_pad_routing_map_aligns_when_enough_zeros():
    routing_map = torch.zeros((100, 2), dtype=torch.bool)
    routing_map[:30, 0] = True
    routing_map[:45, 1] = True

    padded = pad_routing_map(routing_map, pad_multiple=32)

    tokens_per_expert = padded.sum(dim=0)
    assert int(tokens_per_expert[0]) % 32 == 0
    assert int(tokens_per_expert[1]) % 32 == 0


def test_pad_routing_map_flips_every_zero_it_can():
    # Expert 0 has to grow from 90 to 128 tokens but only has 10 zeros left to flip, so
    # the padding stops at 100. pad_routing_map stays quiet here, the caller reports it.
    routing_map = torch.zeros((100, 2), dtype=torch.bool)
    routing_map[:90, 0] = True

    padded = pad_routing_map(routing_map, pad_multiple=128)

    assert int(padded.sum(dim=0)[0]) == 100


def test_warn_if_tokens_per_expert_unaligned_names_the_offending_experts(caplog):
    tokens_per_expert = torch.tensor([90, 0, 128])

    with caplog.at_level(logging.WARNING):
        warn_if_tokens_per_expert_unaligned(tokens_per_expert, pad_multiple=128)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "expert(s) [0]" in message
    assert "90" in message
    assert "not a multiple of 128" in message


def test_warn_if_tokens_per_expert_unaligned_is_quiet_when_aligned(caplog):
    tokens_per_expert = torch.tensor([0, 128, 256])

    with caplog.at_level(logging.WARNING):
        warn_if_tokens_per_expert_unaligned(tokens_per_expert, pad_multiple=128)

    assert not caplog.records


def test_warn_if_tokens_per_expert_unaligned_skips_unrequested_alignment(caplog):
    with caplog.at_level(logging.WARNING):
        warn_if_tokens_per_expert_unaligned(torch.tensor([7]), pad_multiple=0)

    assert not caplog.records
