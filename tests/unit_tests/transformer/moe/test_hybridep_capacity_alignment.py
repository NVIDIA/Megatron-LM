# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from megatron.core.transformer.moe import fused_a2a


@pytest.mark.parametrize(
    "dispatch,combine,expected", [(64, 64, 64), (64, 128, 128), (96, 128, 384)]
)
def test_alignment_honors_both_kernel_chunks(monkeypatch, dispatch, combine, expected):
    monkeypatch.setenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", str(dispatch))
    monkeypatch.setenv("NUM_OF_TOKENS_PER_CHUNK_COMBINE_API", str(combine))
    assert fused_a2a._hybrid_ep_token_alignment() == expected


@pytest.mark.parametrize("chunk", ["0", "-1", "invalid"])
def test_invalid_kernel_chunk_is_rejected(monkeypatch, chunk):
    monkeypatch.setenv("NUM_OF_TOKENS_PER_CHUNK_DISPATCH_API", chunk)
    with pytest.raises(ValueError):
        fused_a2a._hybrid_ep_token_alignment()


def test_growing_reservation_is_aligned_and_does_not_shrink(monkeypatch):
    monkeypatch.setattr(fused_a2a, "HYBRIDEP_TOKEN_ALIGNMENT", 128)
    config = SimpleNamespace(max_num_of_tokens_per_rank=256)
    buffer = SimpleNamespace(configurer=SimpleNamespace(buffer_config=config))
    with patch.object(fused_a2a.torch.cuda, "empty_cache") as release:
        fused_a2a._reserve_hybrid_ep_capacity(buffer, 2576)
        assert config.max_num_of_tokens_per_rank == 2688
        release.assert_called_once_with()
        fused_a2a._reserve_hybrid_ep_capacity(buffer, 256)
        assert config.max_num_of_tokens_per_rank == 2688
        release.assert_called_once_with()
