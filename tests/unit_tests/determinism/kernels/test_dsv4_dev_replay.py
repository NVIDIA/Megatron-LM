# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay the retained dev CSA metadata/compaction kernels and fused compressor forward.

The external compressor dAPE backward uses atomics and is not claimed bit-exact;
the existing strict-determinism dispatch guard is checked explicitly instead.
"""

from types import SimpleNamespace

import pytest
import torch

from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("compatibility_export", [False, True])
def test_dev_compressor_input_compaction_replays(compatibility_export):
    """Replay the real CuTe gather and unique-store backward through both public exports."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        cp_layout_kernels,
        thd_layout_kernels,
    )

    if not thd_layout_kernels._CUTE_AVAILABLE:
        pytest.skip("CuTe DSL is unavailable")
    seeded()
    cu = torch.tensor([0, 96, 256], dtype=torch.int32, device="cuda")
    hidden = torch.randn(128, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    boundary = torch.randn(8, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    module = cp_layout_kernels if compatibility_export else thd_layout_kernels
    assert_replays_bit_exact(
        lambda x, halo: module.CompressorInputCompact.apply(x, halo, cu, 128, 4, 8, 40, 2),
        (hidden, boundary),
        contention=True,
        what="dev CSA compressor compaction",
    )


def test_dev_indexer_lengths_and_sanitization_replay():
    """Run the actual Triton row-bound and top-k sanitization kernels, including padding."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        thd_indexer_kernels,
    )

    if not thd_indexer_kernels._TRITON_AVAILABLE:
        pytest.skip("Triton is unavailable")
    seeded()
    cu_q = torch.tensor([0, 5, 5, 9], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device="cuda")
    scores = torch.randn(12, 5, device="cuda")
    candidates = torch.arange(7, device="cuda", dtype=torch.int32).expand(12, -1).contiguous()

    def run(q, k, logits):
        lengths = thd_indexer_kernels.build_seq_lens(q, k, 12, 4)
        indices, widths = thd_indexer_kernels.sanitize_topk(candidates, logits, lengths)
        return lengths, indices, widths

    assert_replays_bit_exact(
        run, (cu_q, cu_k, scores), backward=False, contention=True, what="dev CSA top-k metadata"
    )


def test_dev_flat_sparse_indices_replay():
    """Replay the compact route bookkeeping used before external sparse attention dispatch."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        fused_sparse_attention,
    )

    indices = torch.tensor(
        [[2, -1, 7, -1, 3], [-1, -1, -1, -1, -1], [4, 1, 8, 2, 5]], dtype=torch.int32, device="cuda"
    )
    assert_replays_bit_exact(
        fused_sparse_attention._compact_flat_topk_idxs,
        (indices,),
        backward=False,
        contention=True,
        what="dev CSA compact flat indices",
    )


def test_dev_fused_compressor_forward_and_determinism_guard():
    """Replay actual frontend forward; retain the explicit guard against atomic dAPE backward."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import fused_compressor

    if not fused_compressor.fused_compressor_available():
        pytest.skip("CSA compressor frontend requires supported Blackwell hardware")
    seeded()
    kv = torch.randn(256, 1, 256, device="cuda", dtype=torch.bfloat16)
    score = torch.randn_like(kv)
    ape = torch.randn(4, 256, device="cuda", dtype=torch.float32)
    cu = torch.tensor([0, 128, 256], device="cuda", dtype=torch.int32)
    cu_comp = torch.tensor([0, 32, 64], device="cuda", dtype=torch.int32)

    def run(kv, score, ape):
        result = fused_compressor.maybe_compress_thd_fused(
            kv, score, ape, cu, cu_comp, 64, ratio=4, head_dim=128, coff=2
        )
        assert result is not None
        return result

    prior = torch.are_deterministic_algorithms_enabled()
    prior_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        assert_replays_bit_exact(
            run, (kv, score, ape), backward=False, contention=True, what="CSA compressor forward"
        )
        torch.use_deterministic_algorithms(True)
        assert (
            fused_compressor.maybe_compress_thd_fused(
                kv, score, ape, cu, cu_comp, 64, ratio=4, head_dim=128, coff=2
            )
            is None
        )
    finally:
        torch.use_deterministic_algorithms(prior, warn_only=prior_warn)


def test_dev_seq_zigzag_halo_exchange_replays():
    """Per-sequence zigzag CP halo: gather forward and the masked index_add_ scatter backward.

    A single-rank group serves its own halo (block (i, 1) reads the tail of block (i, 0)), which
    runs the same gather and scatter as the multi-rank exchange without point-to-point messages.
    Three of the five sequences are empty: their tail indices repeat row 0 (clamped) and rows
    56..63 of the first real sequence, so index_add_ runs with duplicate destinations, and the
    gradient arriving in their slots must not reach those rows.
    """
    from megatron.core.transformer.experimental_attention_variant.csa_utils import cp_seq_zigzag

    seeded()
    group = SimpleNamespace(size=lambda: 1, rank=lambda: 0)
    cu = torch.tensor([0, 0, 64, 64, 64, 160], dtype=torch.int32, device="cuda")
    d_window = 8
    geom = cp_seq_zigzag.make_geometry(group, cu, 160, d_window)
    cp_seq_zigzag.validate_geometry(geom, host_check=True)
    tail_rows = geom.tail_indices(0, d_window).tolist()
    assert len(set(tail_rows)) < len(tail_rows)  # the scatter really sees duplicate rows
    hidden = torch.randn(160, 1, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    slot_grad = torch.randn(geom.halo_rows, 1, 32, device="cuda", dtype=torch.bfloat16)
    _, grads = assert_replays_bit_exact(
        lambda x: cp_seq_zigzag.exchange_cp_boundary_hidden_seq_zigzag(x, geom, group),
        (hidden,),
        grad_outputs={"out": slot_grad},
        contention=True,
        what="DSv4 per-sequence zigzag halo exchange",
    )
    (hidden_grad,) = grads.values()
    slots = slot_grad.view(geom.n_blocks, d_window, 1, 32)
    expected = torch.zeros_like(hidden_grad)
    expected[24:32] = slots[3]  # block (1, 1) reads the tail of block (1, 0): rows 24..31
    expected[104:112] = slots[9]  # block (4, 1) reads the tail of block (4, 0): rows 104..111
    assert torch.equal(hidden_grad, expected)


def test_dev_seq_zigzag_block_layout_kernels_replay():
    """Replay the block-table compressor layout and attention-index kernels (integer routing)."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        cp_seq_zigzag,
        thd_layout_kernels,
    )

    if not thd_layout_kernels._CUTE_AVAILABLE:
        pytest.skip("CuTe DSL is unavailable")
    cp_size, d_window, ratio, total = 4, 16, 4, 640
    l_local = total // cp_size
    cu = torch.tensor([0, 256, 256, total], dtype=torch.int32, device="cuda")

    def run(cu_seqlens):
        outputs = []
        for rank in (0, cp_size - 1):
            group = SimpleNamespace(size=lambda: cp_size, rank=lambda rank=rank: rank)
            geom = cp_seq_zigzag.make_geometry(group, cu_seqlens, l_local, d_window)
            c_cap = cp_seq_zigzag.compact_group_capacity_blocks(l_local, geom.n_seq, ratio)
            layout = cp_seq_zigzag.build_cp_compressor_layout_seq_zigzag(
                cu_seqlens, geom, ratio, c_cap
            )
            indices = cp_seq_zigzag.build_attention_indices_seq_zigzag(
                cu_seqlens,
                geom,
                d_window,
                ratio,
                total // ratio,
                None,
                cu_seqlens_compressed=layout.cu_seqlens_compressed,
                seq_to_rank_row=layout.seq_to_rank_row,
                compressed_rows=c_cap * cp_size,
                output_alignment=1,
            )
            outputs.append(
                (
                    layout.compact_to_source,
                    layout.source_to_compact,
                    layout.group_ids,
                    layout.position_ids,
                    layout.local_cu_seqlens,
                    layout.local_cu_seqlens_compressed,
                    layout.seq_to_rank_row,
                    indices[0],
                    indices[1],
                )
            )
        return outputs

    assert_replays_bit_exact(
        run,
        (cu,),
        backward=False,
        contention=True,
        what="DSv4 per-sequence zigzag block layout kernels",
    )
