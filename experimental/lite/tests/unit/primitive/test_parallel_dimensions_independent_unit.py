# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from megatron.lite.primitive.parallel.cp import split_packed_for_cp
from megatron.lite.primitive.parallel.pipeline import _num_microbatches_from_config
from megatron.lite.primitive.parallel.pp import build_pipeline_chunk_layout
from megatron.lite.primitive.parallel.sp import (
    gather_for_non_sp_head,
    gather_from_sequence_parallel,
    scatter_to_sequence_parallel,
)
from megatron.lite.primitive.parallel.state import ParallelState

pytestmark = pytest.mark.mlite


def test_tp_vocab_embedding_and_output_single_rank_contract(transformer_engine_import_stub):
    transformer_engine_import_stub()
    from megatron.lite.primitive.parallel.linear import (
        VocabParallelEmbedding,
        VocabParallelOutput,
        pad_vocab_for_tp,
    )

    ps = ParallelState(tp_size=1, tp_rank=0)

    assert pad_vocab_for_tp(151936, 2) == 151936
    assert pad_vocab_for_tp(129, 2) == 256

    embedding = VocabParallelEmbedding(5, 3, ps, deterministic=True)
    with torch.no_grad():
        values = torch.arange(embedding.local_vocab * 3, dtype=torch.float32).view(
            embedding.local_vocab, 3
        )
        embedding.embedding.weight.copy_(values)

    input_ids = torch.tensor([[0, 4, 1]])
    output = embedding(input_ids)

    assert output.shape == (3, 1, 3)
    torch.testing.assert_close(output[:, 0], values[input_ids[0]])

    head = VocabParallelOutput(5, 3, ps)
    logits = head(torch.ones(2, 1, 3, dtype=torch.bfloat16))
    gathered = head.gather(logits)

    assert logits.shape == (2, 1, 128)
    assert gathered.shape == (2, 1, 5)


def test_sp_helpers_are_identity_for_single_rank_tp():
    ps = ParallelState(tp_size=1, tp_rank=0)
    x = torch.randn(4, 2, 3, requires_grad=True)

    assert scatter_to_sequence_parallel(x, ps) is x
    assert gather_from_sequence_parallel(x, ps) is x
    assert gather_for_non_sp_head(x, ps) is x


def test_cp_packed_split_handles_each_sample_independently():
    input_ids = torch.arange(16)
    position_ids = torch.arange(100, 116)
    cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int32)

    rank0 = split_packed_for_cp(
        input_ids, position_ids, cu_seqlens, max_seqlen=8, cp_rank=0, cp_size=2
    )
    rank1 = split_packed_for_cp(
        input_ids, position_ids, cu_seqlens, max_seqlen=8, cp_rank=1, cp_size=2
    )

    torch.testing.assert_close(rank0[0], torch.tensor([0, 1, 6, 7, 8, 9, 14, 15]))
    torch.testing.assert_close(rank0[1], torch.tensor([100, 101, 106, 107, 108, 109, 114, 115]))
    torch.testing.assert_close(rank0[2], torch.tensor([0, 4, 8], dtype=torch.int32))
    assert rank0[3] == 4

    torch.testing.assert_close(rank1[0], torch.tensor([2, 3, 4, 5, 10, 11, 12, 13]))
    torch.testing.assert_close(rank1[1], torch.tensor([102, 103, 104, 105, 110, 111, 112, 113]))
    torch.testing.assert_close(rank1[2], torch.tensor([0, 4, 8], dtype=torch.int32))
    assert rank1[3] == 4


def test_pp_layout_auto_balances_non_divisible_layer_counts():
    # Non-divisible counts no longer raise "not divisible": mcore's layout balances
    # them. 7/pp2 with embedding/loss accounting -> [4, 3]. (Needs megatron.core.)
    pytest.importorskip("megatron.core.transformer.pipeline_parallel_layer_layout")
    rank0 = ParallelState(pp_size=2, pp_rank=0, pp_is_first=True, pp_is_last=False)
    rank1 = ParallelState(pp_size=2, pp_rank=1, pp_is_first=False, pp_is_last=True)

    assert build_pipeline_chunk_layout(7, rank0).layer_indices == [0, 1, 2, 3]
    assert build_pipeline_chunk_layout(7, rank1).layer_indices == [4, 5, 6]


def test_dp_dimension_controls_dense_microbatch_contract():
    ps = ParallelState(dp_size=4, dp_rank=2, dp_cp_size=8, dp_cp_rank=5)
    config = SimpleNamespace(gbs=32, mbs=2, num_microbatches=None)

    assert _num_microbatches_from_config(config, ps) == 4

    config.num_microbatches = 7
    assert _num_microbatches_from_config(config, ps) == 7
    assert ps.dp_rank == 2
    assert ps.dp_cp_size == 8
    assert ps.dp_cp_rank == 5


def test_ep_token_dispatcher_local_roundtrip_is_independent_of_deepep():
    from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

    ps = ParallelState(ep_size=1, ep_rank=0)
    dispatcher = TokenDispatcher(num_experts=3, hidden_size=2, ps=ps, use_deepep=False)
    hidden = torch.tensor([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]], requires_grad=True)
    topk_indices = torch.tensor([[0], [2], [1], [2]])
    topk_scores = torch.ones(4, 1)

    dispatched, tokens_per_expert, dispatched_probs = dispatcher.dispatch(
        hidden, topk_scores, topk_indices
    )
    combined = dispatcher.combine(dispatched)

    torch.testing.assert_close(tokens_per_expert, torch.tensor([1, 1, 2]))
    torch.testing.assert_close(dispatched_probs, torch.ones(4))
    torch.testing.assert_close(combined, hidden)

    combined.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.ones_like(hidden))
