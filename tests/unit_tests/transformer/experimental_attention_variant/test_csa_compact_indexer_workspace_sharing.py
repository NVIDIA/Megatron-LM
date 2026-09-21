# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
"""Cross-layer sharing of the compact DSA indexer workspace under CUDA graphs: real kernels.
(The sharing contract itself is tested next to the other compact-workspace tests in
``test_attention_variant_csa.py::TestCompressedSparseAttentionThd``.)"""

import pytest
import torch


class TestCompactIndexerWorkspaceSharingKernels:
    """Real compact-indexer kernels: two layers dispatched through one shared workspace must produce
    exactly what each produces through its own workspace (stale content from the other layer must
    not leak: quantize overwrites the staging buffers, the top-k kernel reads only them)."""

    @pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
    def test_interleaved_layers_through_one_workspace_match_private_workspaces(self, precision):
        import inspect

        from megatron.core.transformer.experimental_attention_variant.csa_utils import (
            fused_sparse_attention as dk,
        )
        from tests.unit_tests.transformer.experimental_attention_variant import (
            test_csa_fused_sparse_attention as fsa_tests,
        )

        fsa_tests._skip_if_real_kernels_unavailable()
        if torch.cuda.get_device_capability()[0] < 10:
            pytest.skip("compact THD indexer forward + Top-K requires SM100+")
        from cudnn import DSA

        if not hasattr(DSA, 'compress_topk_cand_buffer_size_thd'):
            pytest.skip("installed cuDNN Frontend lacks the compact THD workspace helper")
        compact_wrapper = getattr(DSA, "indexer_forward_top_k_wrapper", None)
        mxfp8_parameters = {"q_scale", "cu_seqlens_q_scale_padded", "cu_seqlens_k_scale_padded"}
        if precision == "mxfp8" and mxfp8_parameters - set(
            inspect.signature(compact_wrapper).parameters if callable(compact_wrapper) else ()
        ):
            pytest.skip("installed cuDNN Frontend lacks compact MXFP8 indexer support")

        ratio, topk, idx_nh, idx_hd = 4, 16, 64, 128
        q_lens, k_lens = [64, 96], [16, 24]
        max_seqlen_q, max_seqlen_k = 96, 24
        cu_q = fsa_tests._make_cu_seqlens(q_lens, device='cuda')
        cu_k = fsa_tests._make_cu_seqlens(k_lens, device='cuda')
        total_q, total_k = sum(q_lens), sum(k_lens)
        sm_scale = idx_hd**-0.5
        torch.manual_seed(0)
        layers = []
        for _ in range(2):  # two "layers" with the same geometry and different values
            q = torch.randn(total_q, idx_nh, idx_hd, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(total_k, idx_hd, dtype=torch.bfloat16, device='cuda')
            w = torch.randn(total_q, idx_nh, dtype=torch.bfloat16, device='cuda').float() * sm_scale
            layers.append((q, k, w.to(torch.bfloat16)))

        def prepare(q, k):
            return dk.prepare_thd_compact_indexer_workspace(
                q,
                k,
                topk=topk,
                ratio=ratio,
                cu_seqlens_q=cu_q,
                cu_seqlens_k=cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                return_softmax=True,
                precision=precision,
            )

        def run(q, k, w, workspace):
            indices, lengths, _, softmax = dk._indexer_topk_core(
                q,
                k,
                w,
                topk=topk,
                ratio=ratio,
                cu_seqlens_q=cu_q,
                cu_seqlens_kv=cu_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_k,
                use_compact=True,
                return_softmax=True,
                compact_workspace=workspace,
                precision=precision,
            )
            torch.cuda.synchronize()
            return indices.clone(), lengths.clone(), softmax.clone()

        private = [prepare(q, k) for q, k, _ in layers]
        shared = prepare(*layers[0][:2])  # the geometry is the same for both layers
        assert shared is not None and all(ws is not None for ws in private)
        # cuDNN Frontend: eager calls perform value validation and JIT compilation first.
        for _ in range(3):
            run(*layers[0], private[0])
        reference = [run(*layers[i], private[i]) for i in (0, 1)]
        # Interleave the two layers through the one shared workspace: every dispatch finds the
        # other layer's stale quantized q/k and scales in the buffers.
        for i in (0, 1, 0, 1, 1, 0):
            indices, lengths, softmax = run(*layers[i], shared)
            assert torch.equal(indices, reference[i][0])
            assert torch.equal(lengths, reference[i][1])
            assert torch.equal(softmax, reference[i][2])
