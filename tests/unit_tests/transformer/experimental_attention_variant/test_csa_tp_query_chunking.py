"""Tests for the feature-gated TP query chunking adapter."""

from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from unittest.mock import patch

import torch

from megatron.core.transformer.experimental_attention_variant.csa_utils import (
    fused_sparse_attention as dk,
)


class TestTPQueryChunking(unittest.TestCase):

    def test_flash_mla_forward_chunks_rows_and_compacts_output(self):
        """Chunking must preserve row order while padding one chunk at a time."""
        calls = []

        def flash_stub(q, kv, indices, scale, **kwargs):
            calls.append((tuple(q.shape), tuple(indices.shape)))
            out = q + 1
            lse = torch.zeros(q.size(0), q.size(1), dtype=torch.float32)
            return out, torch.empty(0), lse

        with (
            patch.dict(os.environ, {'DSV4_FLASH_MLA_QUERY_CHUNK_SIZE': '3'}),
            patch.object(dk, '_get_head_padding', return_value=8),
            patch.object(dk, '_get_topk_alignment', return_value=4),
            patch.object(
                dk.torch.cuda.nvtx, 'range', side_effect=lambda _name: nullcontext()
            ),
            patch.object(dk, '_flash_mla_sparse_fwd', flash_stub),
        ):
            q = torch.arange(7 * 4 * 2, dtype=torch.float32).reshape(7, 4, 2)
            kv = torch.zeros(5, 2)
            topk = torch.zeros(7, 4, dtype=torch.int32)
            topk_length = torch.full((7,), 4, dtype=torch.int32)

            out, lse, lse_indexer = dk._csa_fwd_flash_mla(
                q,
                kv,
                topk,
                softmax_scale=1.0,
                attn_sink=torch.zeros(4),
                topk_length=topk_length,
            )

        self.assertIsNone(lse_indexer)
        self.assertEqual(
            calls,
            [((3, 8, 2), (3, 1, 4)), ((3, 8, 2), (3, 1, 4)), ((1, 8, 2), (1, 1, 4))],
        )
        self.assertTrue(torch.equal(out, q + 1))
        self.assertEqual(out.shape, q.shape)
        self.assertEqual(lse.shape, (7, 4))

    def test_dsa_backward_chunks_and_accumulates_kv_gradient(self):
        """Chunked DSA backward must concatenate dq and sum dkv/d_sink."""
        calls = []

        class FakeDSA:

            def sparse_attention_backward_wrapper(
                self, q, kv, out, d_out, lse, sink, indices, **kwargs
            ):
                rows = q.size(0)
                calls.append(tuple(q.shape))
                return {
                    'dq': torch.full_like(q, rows),
                    'dkv': torch.full_like(kv, rows),
                    'd_sink': torch.full_like(sink, rows),
                }

        with (
            patch.dict(os.environ, {'DSV4_FLASH_MLA_QUERY_CHUNK_SIZE': '3'}),
            patch.object(dk, '_get_head_padding', return_value=8),
            patch.object(dk, '_DSA', FakeDSA()),
        ):
            q = torch.zeros(7, 4, 2)
            kv = torch.zeros(5, 2)
            out = torch.zeros_like(q)
            d_out = torch.zeros_like(q)
            lse = torch.zeros(7, 4)
            sink = torch.zeros(4)
            topk = torch.zeros(7, 4, dtype=torch.int32)
            topk_length = torch.full((7,), 4, dtype=torch.int32)

            dq, dkv, d_sink = dk._sparse_attention_backward(
                q,
                kv,
                out,
                d_out,
                lse,
                sink,
                topk,
                softmax_scale=1.0,
                topk_length=topk_length,
            )

        self.assertEqual(calls, [(3, 8, 2), (3, 8, 2), (1, 8, 2)])
        self.assertTrue(torch.equal(dq[:3], torch.full_like(dq[:3], 3)))
        self.assertTrue(torch.equal(dq[3:6], torch.full_like(dq[3:6], 3)))
        self.assertTrue(torch.equal(dq[6:], torch.ones_like(dq[6:])))
        self.assertTrue(torch.equal(dkv, torch.full_like(dkv, 7)))
        self.assertTrue(torch.equal(d_sink, torch.full_like(d_sink, 7)))


if __name__ == '__main__':
    unittest.main()
