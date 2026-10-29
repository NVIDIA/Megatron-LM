# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay packed DSA indexer rotary kernels and gradients across runtime CP sizes."""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]


@pytest.mark.parametrize("fused_rope", [False, True])
@pytest.mark.parametrize("last_rank", [False, True])
def test_dsa_indexer_runtime_cp_rope_replays(fused_rope, last_rank):
    """Replay packed indexer RoPE and its gradients across runtime CP1/2/4/1."""
    from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
    from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexer
    from tests.unit_tests.context_parallel.runtime.replay_test_utils import _zigzag_shard

    seeded()
    # RoPE only reads group geometry; no communication is needed to cover every CP rank.
    static_group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    indexer = SimpleNamespace(
        pg_collection=SimpleNamespace(cp=static_group),
        use_rope=True,
        qk_pos_emb_head_dim=32,
        index_head_dim=64,
        config=SimpleNamespace(
            apply_rope_fusion=fused_rope,
            rotary_interleaved=False,
            dsa_indexer_rope_interleaved=False,
        ),
    )
    hidden = torch.randn(320, 1, 2, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    bounds = torch.tensor([0, 128, 320], device="cuda", dtype=torch.int32)
    freqs = RotaryEmbedding(32, 1.0, cp_group=static_group)(192, packed_seq=True)
    reference = None
    for size in (1, 2, 4, 1):
        rank = size - 1 if last_rank else 0
        group = SimpleNamespace(size=lambda size=size: size, rank=lambda rank=rank: rank)
        local = _zigzag_shard(hidden, (128, 192), size, rank).detach().requires_grad_()

        def run(value):
            return DSAIndexer._apply_rope(
                indexer, value, freqs, 1.0, cu_seqlens=bounds, cp_group=group
            )

        result = assert_replays_bit_exact(
            run, (local,), replays=3, contention=True, what=f"DSA indexer RoPE[CP{size}]"
        )
        assert indexer.pg_collection.cp is static_group
        if reference is None:
            reference = result
        for expected, actual in zip(reference, result):
            assert expected.keys() == actual.keys()
            for name, tensor in expected.items():
                assert bytes_equal(_zigzag_shard(tensor, (128, 192), size, rank), actual[name])
