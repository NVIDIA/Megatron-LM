# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Static THD batches for full-iteration CUDA graphs.

Covers the pre-graph batch transform through both data-read wrappers, the SFT transform of
pretrain_hybrid against the eager static-THD padding, the real-token FLOPs stats under graph
replay, and the fixed-length TP broadcast of the static batch.
"""

import itertools
import os
from types import SimpleNamespace

import pytest
import torch

import megatron.training.training as training
from megatron.core import parallel_state
from megatron.core.context_parallel import ContextParallelBatch
from megatron.core.full_cuda_graph import (
    FullCudaGraphWrapper,
    StaticBufferLoader,
    set_pre_graph_batch_transform,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.utils import get_batch_on_this_tp_rank
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

TARGET, MAX_SEQS, NUM_MICROBATCHES = 16, 4, 2


@pytest.fixture(autouse=True)
def _clean_state():
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    set_pre_graph_batch_transform(None)
    yield
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}
    set_pre_graph_batch_transform(None)


def _raw_batch(doc_lens, offset=0):
    """An SFT-like packed row: variable length, cu_seqlens with a batch dim."""
    length = sum(doc_lens)
    cu = torch.tensor([0, *itertools.accumulate(doc_lens)], dtype=torch.int32)
    return {
        "tokens": (torch.arange(length, dtype=torch.int64) + 1 + offset).view(1, -1),
        "labels": (torch.arange(length, dtype=torch.int64) + 2 + offset).view(1, -1),
        "loss_mask": torch.ones(1, length),
        "position_ids": torch.cat([torch.arange(n) for n in doc_lens]).view(1, -1),
        "cu_seqlens": cu.view(1, -1),
        "max_seqlen": torch.tensor([max(doc_lens)], dtype=torch.int32),
    }


class _CountingPad:
    """A minimal static transform: pad tokens to TARGET, mark the batch, count real calls."""

    def __init__(self):
        self.calls = 0

    def __call__(self, batch):
        if batch.get("static_thd", False):
            return batch
        self.calls += 1
        tokens = batch["tokens"].cuda()
        padded = torch.nn.functional.pad(tokens, (0, TARGET - tokens.shape[-1]))
        return {"tokens": padded, "static_thd": True}


def _full_iteration_read(iterator):
    wrapper = FullCudaGraphWrapper(lambda **kwargs: None)
    (replay,) = wrapper.data_read(iterator, [object()], True, NUM_MICROBATCHES)
    return list(replay)


def test_transform_runs_before_static_buffers_and_replays_new_packs():
    """Variable-length packs land in fixed static buffers whose addresses never change."""
    transform = _CountingPad()
    set_pre_graph_batch_transform(transform)
    pointers = None
    for step, doc_lens in enumerate(([3, 5], [7, 2, 4], [11])):
        raws = [_raw_batch(doc_lens, offset=10 * step + b) for b in range(NUM_MICROBATCHES)]
        seen = _full_iteration_read(iter(raws))
        step_pointers = []
        for batch, raw in zip(seen, raws):
            length = raw["tokens"].shape[-1]
            assert batch["tokens"].shape == (1, TARGET)
            torch.testing.assert_close(batch["tokens"][:, :length].cpu(), raw["tokens"])
            assert torch.all(batch["tokens"][:, length:] == 0)
            step_pointers.append(batch["tokens"].data_ptr())
        pointers = pointers or step_pointers
        assert step_pointers == pointers
    assert transform.calls == 3 * NUM_MICROBATCHES


@pytest.fixture
def hybrid(monkeypatch):
    """pretrain_hybrid with the static THD contract (TARGET tokens, MAX_SEQS sequences)."""
    Utils.initialize_model_parallel(1, 1)
    import pretrain_hybrid

    config = SimpleNamespace(
        pad_packed_seq_alignment="max",
        max_seqlen_per_dp_cp_rank=TARGET,
        thd_max_packed_sequences=MAX_SEQS,
        cuda_graph_impl="full_iteration",
        thd_tail_padding_policy=None,
        linear_cp_layout="zigzag",
    )
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: SimpleNamespace(seq_length=TARGET))
    monkeypatch.setattr(pretrain_hybrid, "_STATIC_THD_CONFIG", config)
    yield pretrain_hybrid, config
    Utils.destroy_model_parallel()


def _eager_static(pretrain_hybrid, config, raw):
    """What forward_step does for an eager static-THD run (MBS 1)."""
    cu = raw["cu_seqlens"].squeeze(0).cuda()
    params = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        max_seqlen_q=int(raw["max_seqlen"].max()),
        max_seqlen_kv=int(raw["max_seqlen"].max()),
        total_tokens=int(cu[-1]),
        tokens_per_sample=TARGET,
    )
    batch = {key: raw[key].cuda() for key in ("tokens", "labels", "loss_mask", "position_ids")}
    padded = pretrain_hybrid._pad_packed_batch_to_static_thd(
        config,
        None,
        ContextParallelBatch.from_single_layout(config.linear_cp_layout, batch, params),
    )
    out = padded.get_batch()
    return (
        out["tokens"],
        out["labels"],
        out["loss_mask"],
        out["position_ids"],
        padded.get_packed_seq_params(),
        out["padding_mask"],
    )


@pytest.mark.parametrize("doc_lens", [[3, 5], [4, 4, 8], [16], [1, 2, 3]])
def test_sft_transform_matches_eager_static_padding(hybrid, doc_lens):
    pretrain_hybrid, config = hybrid
    raw = _raw_batch(doc_lens)
    static = pretrain_hybrid._static_thd_graph_batch(raw)
    tokens, labels, loss_mask, position_ids, params, padding_mask = _eager_static(
        pretrain_hybrid, config, raw
    )
    for key, ref in (
        ("tokens", tokens),
        ("labels", labels),
        ("loss_mask", loss_mask),
        ("position_ids", position_ids),
        ("padding_mask", padding_mask),
    ):
        assert torch.equal(static[key], ref), key
    assert static["cu_seqlens"].shape == (1, MAX_SEQS + 1)
    assert torch.equal(static["cu_seqlens"].squeeze(0), params.cu_seqlens_q)
    # get_batch rebuilds the params from config values only.
    rebuilt = pretrain_hybrid._static_thd_packed_seq_params(
        static["cu_seqlens"].squeeze(0), static["padding_mask"], config
    )
    for field in (
        "max_seqlen_q",
        "max_seqlen_kv",
        "total_tokens",
        "pad_between_seqs",
        "cu_seqlens_q_padded",
        "cu_seqlens_kv_padded",
        "qkv_format",
    ):
        assert getattr(rebuilt, field) == getattr(params, field), field
    assert torch.equal(rebuilt.padding_mask, params.padding_mask)
    # Idempotent: the wrapped FullCudaGraphWrapper drains the same batch again.
    assert pretrain_hybrid._static_thd_graph_batch(static) is static


def test_sft_transform_raises_on_too_many_sequences(hybrid):
    pretrain_hybrid, _ = hybrid
    with pytest.raises(ValueError, match="thd-max-packed-sequences"):
        # Four documents plus the padding tail need five sequences.
        pretrain_hybrid._static_thd_graph_batch(_raw_batch([2, 2, 2, 2]))


def _stats(update):
    training._seqlen_stats_in_iteration = None
    training._seqlen_stats_active = False
    training._seqlen_stats_in_graph = False
    update()
    return training.consume_seqlen_stats_in_iteration()


@pytest.mark.parametrize("doc_lens", [[3, 5], [4, 4, 8], [16]])
def test_static_seqlen_stats_count_only_real_tokens(hybrid, doc_lens):
    pretrain_hybrid, config = hybrid
    raw = _raw_batch(doc_lens)
    static = pretrain_hybrid._static_thd_graph_batch(raw)
    real = _stats(lambda: training.update_seqlen_stats_from_cu_seqlens(raw["cu_seqlens"][0].cuda()))
    padded = _stats(
        lambda: training.update_seqlen_stats_from_static_thd(
            static["cu_seqlens"].squeeze(0), static["padding_mask"]
        )
    )
    assert real == padded


def test_static_seqlen_stats_survive_graph_replay(hybrid):
    """The captured update keeps feeding consume() on replays (no Python re-run)."""
    pretrain_hybrid, _ = hybrid
    packs = [[3, 5], [4, 4, 8], [1, 2, 3]]
    statics = [pretrain_hybrid._static_thd_graph_batch(_raw_batch(p)) for p in packs]
    cu = statics[0]["cu_seqlens"].squeeze(0).clone()
    mask = statics[0]["padding_mask"].clone()

    def update():
        training.update_seqlen_stats_from_static_thd(cu, mask)

    training._seqlen_stats_in_iteration = None
    training._seqlen_stats_in_graph = False
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        update()  # warmup allocates the accumulator outside the capture
    torch.cuda.current_stream().wait_stream(stream)
    training.consume_seqlen_stats_in_iteration()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        update()
    training.consume_seqlen_stats_in_iteration()  # discard the capture's (non-)update
    for doc_lens, static in zip(packs, statics):
        cu.copy_(static["cu_seqlens"].squeeze(0))
        mask.copy_(static["padding_mask"])
        graph.replay()
        total, squared = training.consume_seqlen_stats_in_iteration()
        world = torch.distributed.get_world_size()
        assert total == sum(doc_lens) * world
        assert squared == sum(n * n for n in doc_lens) * world
    training._seqlen_stats_in_graph = False


@pytest.mark.skipif(
    int(os.environ.get("WORLD_SIZE", "1")) < 2, reason="requires at least two ranks"
)
def test_fixed_length_tp_broadcast_is_capturable():
    """TP > 1: the static batch broadcasts with a fixed cu_seqlens length, inside a graph."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=2)
    try:
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        source = {
            "tokens": torch.zeros(1, TARGET, dtype=torch.int64, device="cuda"),
            "labels": torch.zeros(1, TARGET, dtype=torch.int64, device="cuda"),
            "loss_mask": torch.zeros(1, TARGET, device="cuda"),
            "position_ids": torch.zeros(1, TARGET, dtype=torch.int64, device="cuda"),
            "padding_mask": torch.zeros(1, TARGET, dtype=torch.bool, device="cuda"),
            "cu_seqlens": torch.zeros(1, MAX_SEQS + 1, dtype=torch.int32, device="cuda"),
            "max_seqlen": torch.full((1,), TARGET, dtype=torch.int32, device="cuda"),
            "attention_mask": None,
            "cu_seqlens_padded": None,
            "local_cp_size": None,
            "hybrid_cp_group": None,
        }

        def fill(seed):
            source["tokens"].copy_(torch.arange(TARGET, device="cuda").view(1, -1) + seed)
            source["padding_mask"].copy_(torch.arange(TARGET, device="cuda").view(1, -1) >= seed)
            source["cu_seqlens"].copy_(
                torch.tensor([0, 1, seed, seed + 1, TARGET], dtype=torch.int32, device="cuda")
            )

        def broadcast():
            return get_batch_on_this_tp_rank(
                dict(source) if tp_rank == 0 else {},
                has_cu_seqlens=True,
                is_hybrid_cp=False,
                create_attention_mask_in_dataloader=False,
                broadcast_src_rank=parallel_state.get_tensor_model_parallel_src_rank(),
                broadcast_group=parallel_state.get_tensor_model_parallel_group(),
                cp_size=1,
                tp_rank=tp_rank,
                micro_batch_size=1,
                seq_length=TARGET,
                mtp_on_this_rank=False,
                static_cu_seqlens_entries=MAX_SEQS + 1,
                has_padding_mask=True,
            )

        fill(3)
        eager = broadcast()
        assert torch.equal(eager["cu_seqlens"], source["cu_seqlens"])
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            broadcast()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, stream=stream):
                out = broadcast()
        torch.cuda.current_stream().wait_stream(stream)
        for seed in (5, 9):
            fill(seed)
            graph.replay()
            torch.cuda.synchronize()
            for key in ("tokens", "padding_mask", "cu_seqlens"):
                assert torch.equal(out[key], source[key]), key
    finally:
        Utils.destroy_model_parallel()


def test_get_batch_keeps_static_tokens_for_engram_with_two_rows_per_pack(hybrid, monkeypatch):
    """micro_batch_size 2: the static batch is one flattened pack, not [2, seq_length] rows.

    The Engram token prefetch derives [micro_batch_size, seq_length] tokens per microbatch;
    get_batch must keep the static batch's own flattened tokens instead.
    """
    from megatron.training.utils.common_utils import (
        PipelineTokenPrefetchIterator,
        get_pipeline_prefetched_tokens,
    )

    pretrain_hybrid, config = hybrid
    row = TARGET // 2
    args = SimpleNamespace(
        seq_length=row,
        micro_batch_size=2,
        max_seqlen_per_dp_cp_rank=TARGET,
        thd_max_packed_sequences=MAX_SEQS,
        pad_packed_seq_alignment="max",
        cuda_graph_impl="full_iteration",
        sft=True,
        dataloader_inter_document_masking=False,
        create_attention_mask_in_dataloader=False,
        dynamic_context_parallel=False,
        sequence_packing_scheduler=None,
        context_parallel_size=1,
        pipeline_model_parallel_size=1,
        engram_enabled=True,
        engram_verify_training=False,
    )
    config.pipeline_model_parallel_layout = None
    config.mtp_num_layers = None
    monkeypatch.setattr(pretrain_hybrid, "get_args", lambda: args)
    monkeypatch.setattr(pretrain_hybrid, "core_transformer_config_from_args", lambda _: config)

    # Two rows of one full-length document each, as the SFT loader yields them at MBS 2.
    raw = {
        "tokens": torch.arange(2 * row, dtype=torch.int64).view(2, row) + 1,
        "labels": torch.arange(2 * row, dtype=torch.int64).view(2, row) + 2,
        "loss_mask": torch.ones(2, row),
        "position_ids": torch.arange(row).repeat(2, 1),
        "cu_seqlens": torch.tensor([[0, row], [0, row]], dtype=torch.int32),
        "max_seqlen": torch.tensor([[row], [row]], dtype=torch.int32),
    }
    static = pretrain_hybrid._static_thd_graph_batch(raw)
    assert static["tokens"].shape == (1, TARGET)
    # What the full-iteration data read hands the schedule on a single-rank model.
    iterator = PipelineTokenPrefetchIterator(None, [static], [], token_spec=(2, row, "cuda"))
    tokens = pretrain_hybrid.get_batch(iterator).get_batch()["tokens"]
    assert torch.equal(tokens, static["tokens"])
    # The prefetch path is the one that cannot describe the flattened pack.
    prefetch = PipelineTokenPrefetchIterator(None, [static], [], token_spec=(2, row, "cuda"))
    next(prefetch)
    with pytest.raises(ValueError, match="Engram expected token shape"):
        get_pipeline_prefetched_tokens(prefetch)
