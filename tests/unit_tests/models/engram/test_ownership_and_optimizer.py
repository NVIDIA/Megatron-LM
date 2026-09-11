# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.engram.distributed_embedding import (
    EPShardedEmbeddingTable,
    EPShardedMultiTableEmbedding,
    get_contiguous_row_range,
)
from megatron.core.optimizer import OptimizerConfig, ParamKey, get_engram_config_overrides
from megatron.training.utils import get_pipeline_prefetched_tokens, prepare_tokens_for_pipeline

from ._test_utils import make_module_config


def test_uneven_row_ownership_exactly_covers_global_table():
    ranges = [get_contiguous_row_range(11, rank, 4) for rank in range(4)]
    assert ranges == [(0, 3), (3, 6), (6, 9), (9, 11)]
    assert [row for start, end in ranges for row in range(start, end)] == list(range(11))


def test_multi_table_checkpoint_keeps_global_logical_shape():
    embedding = EPShardedMultiTableEmbedding(
        config=make_module_config(dtype=torch.float32),
        table_sizes=(11, 13),
        embedding_dim=3,
        init_method=lambda _: None,
    )
    state = embedding.sharded_state_dict()
    assert state["tables.0.weight"].global_shape == (11, 3)
    assert state["tables.1.weight"].global_shape == (13, 3)


def test_deterministic_embedding_backward_accumulates_repeated_rows():
    table = EPShardedEmbeddingTable(
        config=make_module_config(dtype=torch.float32, deterministic_mode=True),
        global_num_embeddings=7,
        embedding_dim=3,
        init_method=torch.nn.init.zeros_,
    )
    row_ids = torch.tensor([2, 1, 2, 5, 2, 1], dtype=torch.int64)
    output_grad = torch.arange(18, dtype=torch.float32).view(6, 3)

    table(row_ids).backward(output_grad)

    expected = torch.zeros_like(table.weight)
    for row_id, row_grad in zip(row_ids, output_grad):
        expected[row_id] += row_grad
    torch.testing.assert_close(table.weight.grad, expected, rtol=0, atol=0)


def test_engram_optimizer_override_is_sparse_only():
    config = OptimizerConfig(lr=2.0e-4, min_lr=1.0e-5, optimizer="muon")
    overrides = get_engram_config_overrides(config, lr_multiplier=5.0, weight_decay=0.0)
    assert len(overrides) == 1
    key, override = next(iter(overrides.items()))
    assert isinstance(key, ParamKey)
    table = torch.nn.Parameter(torch.ones(2, 2))
    table.is_engram_embedding = True
    dense = torch.nn.Parameter(torch.ones(2, 2))
    assert key.matches(table, "embedding.weight")
    assert not key.matches(dense, "value_projection.weight")
    assert override == {
        "max_lr": pytest.approx(1.0e-3),
        "min_lr": pytest.approx(5.0e-5),
        "start_wd": 0.0,
        "end_wd": 0.0,
        "wd_mult": 1.0,
    }


def test_pp_token_prefetch_replays_source_batches_in_order(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    batches = [
        {"tokens": torch.arange(8).view(2, 4), "id": 0},
        {"tokens": torch.arange(8, 16).view(2, 4), "id": 1},
    ]
    iterator = prepare_tokens_for_pipeline(iter(batches), 2, 2, 4, object(), object())

    assert next(iterator) is batches[0]
    first_tokens = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(first_tokens, batches[0]["tokens"].to(first_tokens.device))
    assert next(iterator) is batches[1]
    second_tokens = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(second_tokens, batches[1]["tokens"].to(second_tokens.device))
    with pytest.raises(RuntimeError, match="queue is exhausted"):
        get_pipeline_prefetched_tokens(iterator)

    # A single rank derives its tokens from the batch the scheduler just consumed, so a
    # malformed shape is rejected when the tokens are requested rather than at prefetch.
    malformed = prepare_tokens_for_pipeline(
        iter([{"tokens": torch.arange(8).view(2, 4)}]), 1, 1, 8, object(), object()
    )
    next(malformed)
    with pytest.raises(ValueError, match="expected token shape"):
        get_pipeline_prefetched_tokens(malformed)


def test_pp_token_prefetch_handles_variable_length_packed_rows(monkeypatch):
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 1)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    # Packed (THD) rows shorter than the sequence capacity come back zero-padded to the
    # capacity, matching pad_sequence_for_thd's padding of the model-side tokens.
    short = torch.arange(1, 6).view(1, 5)
    full = torch.arange(1, 9).view(1, 8)
    iterator = prepare_tokens_for_pipeline(
        iter([{"tokens": short}, {"tokens": full}]), 2, 1, 8, object(), object()
    )
    next(iterator)
    first = get_pipeline_prefetched_tokens(iterator)
    # Short rows come back zero-padded to the capacity, matching pad_sequence_for_thd.
    assert first.shape == (1, 8)
    torch.testing.assert_close(first, torch.nn.functional.pad(short, (0, 3)).to(first.device))
    next(iterator)
    second = get_pipeline_prefetched_tokens(iterator)
    assert second.shape == (1, 8)
    torch.testing.assert_close(second, full.to(second.device))

    too_long = prepare_tokens_for_pipeline(
        iter([{"tokens": torch.arange(9).view(1, 9)}]), 1, 1, 8, object(), object()
    )
    next(too_long)
    with pytest.raises(ValueError, match="expected token shape"):
        get_pipeline_prefetched_tokens(too_long)


def test_pp_token_prefetch_stacks_and_pads_every_microbatch_when_distributing(monkeypatch):
    """With peers to distribute to, the source prefetches and pads the whole global batch."""
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_size", lambda _: 2)
    monkeypatch.setattr("megatron.training.utils.common_utils.get_pg_rank", lambda _: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.distributed, "get_global_rank", lambda group, rank: rank)

    short = torch.arange(1, 6).view(1, 5)
    full = torch.arange(1, 9).view(1, 8)
    batches = [{"tokens": short, "id": 0}, {"tokens": full, "id": 1}]
    iterator = prepare_tokens_for_pipeline(iter(batches), 2, 1, 8, object(), object())

    # The source consumed its iterator during prefetch and replays those same batches.
    assert next(iterator) is batches[0]
    first = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(first, torch.nn.functional.pad(short, (0, 3)).to(first.device))
    assert next(iterator) is batches[1]
    second = get_pipeline_prefetched_tokens(iterator)
    torch.testing.assert_close(second, full.to(second.device))
    with pytest.raises(RuntimeError, match="queue is exhausted"):
        get_pipeline_prefetched_tokens(iterator)

    # A source that runs out mid-prefetch fails through the availability flag, so the peers
    # raise with it instead of hanging in the token broadcast.
    with pytest.raises(RuntimeError, match="exhausted during token prefetch"):
        prepare_tokens_for_pipeline(iter([{"tokens": full}]), 2, 1, 8, object(), object())
