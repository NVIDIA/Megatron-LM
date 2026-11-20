# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""pretrain_hybrid.get_batch hands every pipeline stage the prefetched Engram token row.

Middle pipeline stages build no dataset unless rows are packed, so they must take the row
without advancing their data iterator and split it over context parallelism exactly like the
hidden states. Each case runs the real get_batch in spawned Gloo processes on CPU; only the
arguments, the run config, the parallel-state getters and, where noted, the prefetch wrapper
are stubbed. The cases spawn their own one- and two-process worlds instead of using the
torchrun world, so the CP and PP groups have the sizes each case needs whatever the CI world
size.
"""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import pretrain_hybrid
from megatron.training.utils import get_pipeline_prefetched_tokens, prepare_tokens_for_pipeline
from megatron.training.utils.common_utils import PipelineTokenPrefetchIterator

SEQ_LENGTH = 8
MICRO_BATCH_SIZE = 2


class _MustNotAdvance:
    """Data iterator of a rank that get_batch must not read from."""

    def __iter__(self):
        return self

    def __next__(self):
        raise AssertionError("get_batch advanced a data iterator it must not read")


def _microbatch(index, packed=False):
    """One microbatch as the data loader yields it; a fresh dict, since get_batch edits it."""
    tokens = torch.arange(MICRO_BATCH_SIZE * SEQ_LENGTH).view(MICRO_BATCH_SIZE, SEQ_LENGTH)
    tokens = tokens + 100 * index
    batch = {
        "tokens": tokens,
        "labels": tokens + 1,
        "loss_mask": torch.ones(MICRO_BATCH_SIZE, SEQ_LENGTH),
        "position_ids": torch.arange(SEQ_LENGTH).repeat(MICRO_BATCH_SIZE, 1),
    }
    if packed:
        # Two documents per row, each row ending at the sequence capacity.
        batch["cu_seqlens"] = torch.tensor(
            [[0, 3, SEQ_LENGTH], [0, 5, SEQ_LENGTH]], dtype=torch.int32
        )
        batch["max_seqlen"] = torch.tensor([5, 5], dtype=torch.int32)
    return batch


def _zigzag_shard(row, cp_size, cp_rank):
    """The sequence chunks of ``row`` that the zigzag CP layout assigns to ``cp_rank``."""
    chunks = row.chunk(2 * cp_size, dim=-1)
    return torch.cat([chunks[cp_rank], chunks[2 * cp_size - 1 - cp_rank]], dim=-1)


def _own_group():
    """A one-rank group of this process; every rank creates every group in the same order."""
    groups = [dist.new_group([rank]) for rank in range(dist.get_world_size())]
    return groups[dist.get_rank()]


def _stub_get_batch(*, cp_size, pp_size, cp_group, first_or_last, pp_first, pp_last, sft=False):
    """Point get_batch at this process: TP = 1 on a singleton group, CP over ``cp_group``.

    Returns the transformer config get_batch reads, whose ``engram_enabled`` the cases toggle.
    """
    args = SimpleNamespace(
        sft=sft,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        micro_batch_size=MICRO_BATCH_SIZE,
        seq_length=SEQ_LENGTH,
    )
    # get_batch reads the parallel layout and the Engram switch from the run config.
    config = SimpleNamespace(
        sequence_packing_scheduler=None,
        dynamic_context_parallel=False,
        context_parallel_size=cp_size,
        pipeline_model_parallel_size=pp_size,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        linear_cp_layout="zigzag",
        attention_cp_layout="zigzag",
        sequence_parallel=False,
        tensor_model_parallel_size=1,
        engram_enabled=True,
    )
    run_config = SimpleNamespace(model=SimpleNamespace(transformer=config))
    tp_group = _own_group()
    pretrain_hybrid.get_args = lambda: args
    pretrain_hybrid.get_run_config = lambda: run_config
    pretrain_hybrid.mtp_on_this_rank_func = lambda **_kwargs: False
    pretrain_hybrid.is_first_or_last_pipeline_stage = first_or_last
    pretrain_hybrid.get_context_parallel_group = lambda: cp_group
    mpu = pretrain_hybrid.mpu
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_src_rank = dist.get_rank
    mpu.get_tensor_model_parallel_group = lambda: tp_group
    # Only handed to the MTP placement, which is stubbed above.
    mpu.get_pipeline_model_parallel_group = lambda: tp_group
    mpu.is_pipeline_first_stage = lambda *_args, **_kwargs: pp_first
    mpu.is_pipeline_last_stage = lambda *_args, **_kwargs: pp_last
    return config, tp_group


def _assert_same_batch(actual, expected, skip=()):
    assert actual.keys() == expected.keys()
    for key in actual.keys() - set(skip):
        if expected[key] is None:
            assert actual[key] is None, key
        else:
            assert torch.equal(actual[key], expected[key]), key


def _middle_stage_case():
    cp_size, cp_rank = dist.get_world_size(), dist.get_rank()
    config, _ = _stub_get_batch(
        cp_size=cp_size,
        pp_size=3,
        cp_group=dist.group.WORLD,
        first_or_last=lambda _vp_stage: False,
        pp_first=False,
        pp_last=False,
    )
    row = _microbatch(0)["tokens"]
    # The stage owns no dataset; the prefetch broadcast left it the row and nothing else.
    wrapper = PipelineTokenPrefetchIterator(_MustNotAdvance(), [], [row])

    cp_batch = pretrain_hybrid.get_batch(wrapper)

    batch = cp_batch.get_batch()
    assert torch.equal(batch["tokens"], _zigzag_shard(row, cp_size, cp_rank))
    assert all(batch[key] is None for key in pretrain_hybrid.BATCH_KEYS if key != "tokens")
    assert cp_batch.get_packed_seq_params() is None
    with pytest.raises(RuntimeError, match="exhausted"):
        get_pipeline_prefetched_tokens(wrapper)

    # Without Engram the middle stage still returns an empty batch.
    config.engram_enabled = False
    batch = pretrain_hybrid.get_batch(_MustNotAdvance()).get_batch()
    assert all(value is None for value in batch.values())


def _first_and_last_stages_case():
    cp_size, cp_rank = dist.get_world_size(), dist.get_rank()
    row = _microbatch(0)["tokens"]
    for stage in ("first", "last", "only"):
        pp_size = 1 if stage == "only" else 2
        config, tp_group = _stub_get_batch(
            cp_size=cp_size,
            pp_size=pp_size,
            cp_group=dist.group.WORLD,
            first_or_last=lambda _vp_stage: True,
            pp_first=stage != "last",
            pp_last=stage != "first",
        )
        config.engram_enabled = False
        expected = pretrain_hybrid.get_batch(iter([_microbatch(0)])).get_batch()

        config.engram_enabled = True
        if stage == "first":
            # The first stage's TP source drained its iterator during the prefetch.
            wrapper = PipelineTokenPrefetchIterator(_MustNotAdvance(), [_microbatch(0)], [row])
        elif stage == "last":
            wrapper = PipelineTokenPrefetchIterator(iter([_microbatch(0)]), [], [row])
        else:
            # A single stage at TP = 1 has nothing to distribute and derives the row on demand.
            wrapper = prepare_tokens_for_pipeline(
                iter([_microbatch(0)]), 1, MICRO_BATCH_SIZE, SEQ_LENGTH, tp_group, tp_group
            )
        batch = pretrain_hybrid.get_batch(wrapper).get_batch()

        # The last stage reads no tokens of its own; Engram layers there take the row.
        _assert_same_batch(batch, expected, skip=("tokens",) if stage == "last" else ())
        assert torch.equal(batch["tokens"], _zigzag_shard(row, cp_size, cp_rank)), stage


def _vpp_middle_chunk_case():
    pp_rank, num_microbatches = dist.get_rank(), 2
    # PP = 2 with two model chunks per rank: chunk c on PP rank r is virtual stage 2c + r, so
    # (chunk 0, rank 0) is the first stage, (chunk 1, rank 1) the last, and the others middle.
    edge_chunk = 0 if pp_rank == 0 else 1

    def first_or_last(vp_stage):
        return vp_stage == edge_chunk

    _, tp_group = _stub_get_batch(
        cp_size=1,
        pp_size=2,
        cp_group=_own_group(),
        first_or_last=first_or_last,
        pp_first=pp_rank == 0,
        pp_last=pp_rank == 1,
    )
    data = iter([_microbatch(index) for index in range(num_microbatches)])
    chunk_iterators = [data, None] if pp_rank == 0 else [None, data]
    wrappers = prepare_tokens_for_pipeline(
        chunk_iterators, num_microbatches, MICRO_BATCH_SIZE, SEQ_LENGTH, dist.group.WORLD, tp_group
    )

    # Every chunk keeps its own cursor, so chunk 0 may run ahead of chunk 1 as in the
    # interleaved schedule.
    for chunk, wrapper in enumerate(wrappers):
        for index in range(num_microbatches):
            batch = pretrain_hybrid.get_batch(wrapper, vp_stage=chunk).get_batch()
            assert torch.equal(batch["tokens"], _microbatch(index)["tokens"]), (chunk, index)
            if chunk != edge_chunk:
                assert all(value is None for key, value in batch.items() if key != "tokens")
        with pytest.raises(RuntimeError, match="exhausted"):
            get_pipeline_prefetched_tokens(wrapper)


def _packed_middle_stage_case():
    config, _ = _stub_get_batch(
        cp_size=1,
        pp_size=3,
        cp_group=_own_group(),
        first_or_last=lambda _vp_stage: False,
        pp_first=False,
        pp_last=False,
        sft=True,
    )
    config.engram_enabled = False
    expected = pretrain_hybrid.get_batch(iter([_microbatch(0, packed=True)])).get_batch()

    config.engram_enabled = True
    row = _microbatch(0)["tokens"]
    # Packed middle stages build a dataset for the document boundaries and read it.
    wrapper = PipelineTokenPrefetchIterator(iter([_microbatch(0, packed=True)]), [], [row])
    cp_batch = pretrain_hybrid.get_batch(wrapper)

    batch = cp_batch.get_batch()
    assert expected["tokens"] is None
    _assert_same_batch(batch, expected, skip=("tokens",))
    # flatten_batch_for_packed_sequences merges the microbatch into one packed row.
    assert torch.equal(batch["tokens"], row.reshape(1, -1))
    assert torch.equal(batch["cu_seqlens"], torch.tensor([[0, 3, 8, 13, 16]], dtype=torch.int32))
    assert cp_batch.get_packed_seq_params() is not None


def _run(rank, world_size, store_path, case):
    # get_batch places tensors with Tensor.cuda() and torch.cuda.current_device(); keep them on
    # the host, where Gloo runs.
    torch.Tensor.cuda = lambda self, *_args, **_kwargs: self
    torch.cuda.current_device = lambda: torch.device("cpu")
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=2),
    )
    try:
        case()
    finally:
        dist.destroy_process_group()


def _spawn(tmp_path, world_size, case):
    mp.spawn(_run, args=(world_size, str(tmp_path / "store"), case), nprocs=world_size, join=True)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_middle_stage_splits_the_prefetched_row_over_cp(tmp_path, cp_size):
    _spawn(tmp_path, cp_size, _middle_stage_case)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_first_and_last_stages_are_unchanged(tmp_path, cp_size):
    _spawn(tmp_path, cp_size, _first_and_last_stages_case)


def test_vpp_middle_chunks_take_the_prefetched_row(tmp_path):
    _spawn(tmp_path, 2, _vpp_middle_chunk_case)


def test_packed_middle_stage_keeps_its_tokens(tmp_path):
    _spawn(tmp_path, 1, _packed_middle_stage_case)
