# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay the packed layout, indexer metadata and CP collective operations."""

import pytest
import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    csa_indexer_loss_kernels,
    packed_layout,
    packed_sparse_attention,
)
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_cp_compressor_layout_replays():
    seeded()
    cu = torch.tensor([0, 37, 37, 215, 512], device="cuda", dtype=torch.int32)
    hidden = torch.randn(256, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundary = torch.randn(16, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, halo):
        return cp_utils.prepare_cp_compressor_input(x, halo, cu, 256, 2, 4)

    assert_replays_bit_exact(run, (hidden, boundary), contention=True, what="CSA packed layout")
    layout = run(hidden, boundary)

    def indices(comp_cu, row_map):
        return packed_layout.build_attention_indices(
            cu,
            256,
            256,
            16,
            16,
            4,
            64,
            cu_seqlens_compressed=comp_cu,
            seq_to_rank_row=row_map,
            compressed_rows=layout[1].shape[0] * 2,
            output_alignment=128,
        )

    assert_replays_bit_exact(indices, (layout[-2], layout[-1]), backward=False, contention=True)


def test_packed_indexer_metadata_and_loss_replay():
    seeded()
    cu_q = torch.tensor([0, 65, 65, 256], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 32, 32, 96], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([63, 0, 65], device="cuda", dtype=torch.int32)
    expected_lengths = packed_layout.build_seq_lens(cu_q, cu_k, 256, 4, offsets)
    for start, end in ((0, 33), (33, 101), (101, 256)):
        sliced_cu, sliced_offsets = packed_layout.slice_query_layout(cu_q, offsets, start, end)
        torch.testing.assert_close(
            packed_layout.build_seq_lens(sliced_cu, cu_k, end - start, 4, sliced_offsets),
            expected_lengths[start:end],
            rtol=0,
            atol=0,
        )
    for length in (128 * 1024, 256 * 1024):
        rows = packed_layout.query_chunk_rows(length // 8, length // 4, live_buffers=4)
        assert rows * (length // 4) * 4 * 4 <= packed_layout._INDEXER_WORKSPACE_BYTES
    scores = torch.randn(256, 64, device="cuda")
    candidates = torch.randint(-1, 70, (256, 32), device="cuda", dtype=torch.int32)

    def select(score, ids):
        lengths = packed_layout.build_seq_lens(cu_q, cu_k, 256, 4, offsets)
        return packed_layout.sanitize_topk(ids, score, lengths, output_width=128)

    assert_replays_bit_exact(select, (scores, candidates), backward=False, contention=True)
    indices, _ = select(scores, candidates)
    predict = torch.softmax(torch.randn_like(indices, dtype=torch.float32), dim=-1)
    target = torch.softmax(torch.randn_like(predict), dim=-1)

    def loss(t, p, ids):
        return csa_indexer_loss_kernels.sparse_kl_loss(t, p, ids, 0.2, True, 256)

    assert_replays_bit_exact(loss, (target, predict, indices), backward=False, contention=True)
    # The packed adapter routes KL through the same registered compiled reduction.
    torch.testing.assert_close(
        packed_sparse_attention._kl_loss_from_target_predict(
            target, predict, indices, 0.2, True, 256
        ),
        loss(target, predict, indices),
    )

    def dense_loss(t, score):
        return packed_sparse_attention._kl_loss_from_dense_scores(
            t, t.sum(-1), score, torch.logsumexp(score, dim=-1), 0.2, True
        )

    assert_replays_bit_exact(
        dense_loss, (target, torch.randn_like(target)), backward=False, contention=True
    )


def test_async_context_parallel_collectives_replay():
    """Launch/wait preserves autograd and bit-exact NCCL replay under the suite's Ring policy."""
    import torch.distributed as dist

    from megatron.core.tensor_parallel.mappings import (
        async_gather_from_sequence_parallel_region,
        async_reduce_scatter_along_first_dim,
    )
    from tests.unit_tests.test_utilities import Utils

    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    try:
        seeded()
        group = dist.group.WORLD
        value = torch.randn(128, 64, device="cuda", requires_grad=True)

        def run(x):
            gathered = async_gather_from_sequence_parallel_region(x, group=group).wait()
            return gathered.square()

        assert_replays_bit_exact(
            run, (value,), contention=True, what="async all-gather/reduce-scatter"
        )
        global_grad = torch.randn(group.size() * 128, 64, device="cuda")

        def reduce(x):
            return async_reduce_scatter_along_first_dim(x, group=group).wait()

        assert_replays_bit_exact(reduce, (global_grad,), backward=False, contention=True)
    finally:
        Utils.destroy_model_parallel()
