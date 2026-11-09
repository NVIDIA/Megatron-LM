# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay dev CSA layout kernels, aligned HCA backward, and compressor forward.

The external compressor dAPE backward uses atomics and is not claimed bit-exact;
the existing strict-determinism dispatch guard is checked explicitly instead.
"""

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


@pytest.mark.parametrize("sequence_length,cp_size", [(16384, 4), (32768, 8), (65536, 16)])
@pytest.mark.parametrize("reconstruct", [False, True])
def test_dev_aligned_hca_backward_replays(sequence_length, cp_size, reconstruct):
    """Replay HCA autograd with prepared packing and scheduling contention."""
    from megatron.core.transformer.experimental_attention_variant.csa_utils import (
        aligned_hca,
        fused_sparse_attention,
        thd_layout_kernels,
    )
    from tests.unit_tests.transformer.experimental_attention_variant.test_aligned_hca import (
        _metadata,
        _select,
    )

    if torch.cuda.get_device_capability() not in ((10, 3), (10, 7)):
        pytest.skip("Aligned HCA requires GB300 or Rubin")
    cudnn = pytest.importorskip("cudnn")
    api = getattr(cudnn, "AlignedHCABackward", None)
    tokens, rank = sequence_length // cp_size, cp_size - 1
    if api is None or not api.supports_configuration(tokens, cp_size, torch.device("cuda")):
        pytest.skip("Supporting cuDNN Frontend is unavailable")
    pytest.importorskip("flash_mla")
    seeded()
    _, _, params, group = _metadata(rank, sequence_length, cp_size)
    aligned_hca.prepare_aligned_hca_metadata(params)
    cu = params.cu_seqlens_q_padded
    compressed = sequence_length // 128 + cp_size
    comp = thd_layout_kernels.build_cp_compressor_layout(cu, rank * tokens, tokens, cp_size, 128)
    indices, lengths, _, _ = thd_layout_kernels.build_attention_indices(
        cu,
        rank * tokens,
        tokens,
        128,
        128,
        128,
        sequence_length // 128,
        cu_seqlens_compressed=comp.cu_seqlens_compressed,
        seq_to_rank_row=comp.seq_to_rank_row,
        compressed_rows=compressed,
        output_alignment=64,
    )
    q = torch.randn(tokens, 128, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    kv = torch.randn(
        tokens + 128 + compressed, 512, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    sink = torch.randn(128, device="cuda", requires_grad=True)

    def run(q, kv, sink):
        selected = _select(q, kv, params, group)
        assert selected == rank
        parts = (
            tuple(part.detach() for part in kv.split((128, tokens, compressed)))
            if reconstruct
            else None
        )
        output = fused_sparse_attention.csa_sparse_attn(
            q,
            kv,
            sink,
            indices,
            512**-0.5,
            topk_length=lengths,
            is_thd=True,
            kv_reconstruction_parts=parts,
            deterministic=True,
            hca_cp_rank=selected,
            hca_cp_size=cp_size,
        )
        context = output.grad_fn.next_functions[0][0]
        assert context.saved_tensors[-3] is None
        assert context.topk_length is None
        return output

    _, grads = assert_replays_bit_exact(
        run, (q, kv, sink), contention=True, what="aligned HCA backward"
    )
    assert len(grads) == 3


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
