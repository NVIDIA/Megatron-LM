# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay the packed layout, indexer metadata and CP collective operations."""

import dataclasses
from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    cp_utils,
    csa_indexer_loss_kernels,
    packed_layout,
    thd_utils,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_replays_bit_exact,
    bytes_equal,
    seeded,
)

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


class _SingleRankGroup:
    def size(self):
        return 1

    def rank(self):
        return 0


@pytest.mark.parametrize("device_checks", [False, True], ids=["checks_off", "checks_on"])
def test_thd_metadata_device_checks_only_validate(monkeypatch, device_checks):
    """MCORE_CSA2_DEVICE_CHECKS only adds device asserts: the layouts are bit-identical with
    the checks on or off, off launches no check, and CPU metadata is validated either way."""
    seeded()
    cu = torch.tensor([0, 37, 37, 215, 300], device="cuda", dtype=torch.int32)
    padded = torch.tensor([0, 40, 40, 256, 512], device="cuda", dtype=torch.int32)
    config = SimpleNamespace(dsa_kernel_backend="cudnn", deterministic_mode=False)

    def packed():
        # Separate Q and KV tensors, so aliasing does not skip the Q/KV equality checks.
        return PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu.clone(),
            cu_seqlens_kv=cu.clone(),
            cu_seqlens_q_padded=padded.clone(),
            cu_seqlens_kv_padded=padded.clone(),
            max_seqlen_q=256,
            max_seqlen_kv=256,
        )

    def build():
        layout = thd_utils.build_csa2_thd_layout(packed(), 512)
        layout.validate_compatible(packed(), 512)
        cp_utils.validate_packed_inputs(packed(), config, _SingleRankGroup())
        return layout, layout.for_compression(2)

    monkeypatch.setattr(thd_utils, "DEVICE_CHECKS", True)
    reference = build()
    torch.cuda.synchronize()

    launched = []
    assert_async = torch._assert_async

    def counting_assert_async(*args, **kwargs):
        launched.append(args)
        return assert_async(*args, **kwargs)

    monkeypatch.setattr(torch, "_assert_async", counting_assert_async)
    monkeypatch.setattr(thd_utils, "DEVICE_CHECKS", device_checks)
    actual = build()
    torch.cuda.synchronize()

    assert bool(launched) == device_checks
    for expected, value in zip(reference, actual):
        for field in dataclasses.fields(expected):
            got, want = getattr(value, field.name), getattr(expected, field.name)
            if isinstance(want, torch.Tensor):
                assert bytes_equal(got, want), field.name
            else:
                assert got == want, field.name

    bad = torch.tensor([1, 37, 37, 215, 300], dtype=torch.int32)
    with pytest.raises(ValueError, match="must start at zero"):
        thd_utils.build_csa2_thd_layout(
            PackedSeqParams(
                qkv_format="thd",
                cu_seqlens_q=bad,
                cu_seqlens_kv=bad.clone(),
                max_seqlen_q=256,
                max_seqlen_kv=256,
            ),
            512,
        )


def test_packed_indexer_metadata_and_loss_replay():
    seeded()
    cu_q = torch.tensor([0, 65, 65, 256], device="cuda", dtype=torch.int32)
    cu_k = torch.tensor([0, 32, 32, 96], device="cuda", dtype=torch.int32)
    offsets = torch.tensor([63, 0, 65], device="cuda", dtype=torch.int32)
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


@pytest.mark.parametrize("ratio", [1, 2])
def test_cp_projected_compaction_replay_and_live_graph_maps(ratio):
    """Unique-source backward remains bit exact when captured padding maps change."""
    from megatron.core.ops.attention.csa.compressor_projection import (
        compact_cp_projection,
        cp_projection_maps,
    )

    seeded()
    sources = torch.arange(32, device="cuda").reshape(-1, ratio)
    valid = torch.ones(sources.shape[0], dtype=torch.bool, device="cuda")
    local = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    halo = torch.randn(ratio, 1, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    def run(x, boundary):
        mapping, inverse = cp_projection_maps(sources, valid, ratio, ratio, x.shape[0])
        return compact_cp_projection(x, boundary, mapping, inverse)

    assert_replays_bit_exact(
        run, (local, halo), contention=True, what="CSA2 projected CP compaction"
    )
    probe = torch.randn(32, 1, 128, device="cuda", dtype=torch.bfloat16)

    def forward_backward():
        output = run(local, halo)
        return output, *torch.autograd.grad(output, (local, halo), probe)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        forward_backward()
        forward_backward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = forward_backward()
    for shift in (0, 1):
        sources.copy_(torch.arange(32, device="cuda").roll(shift).reshape_as(sources))
        valid.fill_(True)
        valid[shift::3] = False
        expected = forward_backward()
        graph.replay()
        torch.cuda.synchronize()
        for actual, wanted in zip(captured, expected):
            torch.testing.assert_close(actual, wanted, rtol=0, atol=0)
