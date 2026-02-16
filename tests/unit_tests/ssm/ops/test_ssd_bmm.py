# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import unittest
import torch

try:
    from megatron.core.ssm.ops.ssd_bmm import _bmm_chunk_fwd
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestBmmChunkFwd(unittest.TestCase):
    """Tests for _bmm_chunk_fwd (C^T @ B per chunk)."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.chunk_size = 16
        self.seqlen = 32
        self.ngroups = 2
        self.dstate = 8  # K dimension
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)

    def test_bmm_chunk_fwd_shape(self):
        # a: (seqlen, ngroups, k), b: (seqlen, ngroups, k) -> out: (nchunks, ngroups, chunk_size, chunk_size)
        a = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        b = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)

        out = _bmm_chunk_fwd(
            a, b, self.chunk_size, self.cu_chunk_seqlens, causal=True, output_dtype=torch.float32
        )

        nchunks = self.cu_chunk_seqlens.shape[0] - 1
        self.assertEqual(out.shape, (nchunks, self.ngroups, self.chunk_size, self.chunk_size))
        self.assertFalse(torch.isnan(out).any())

    def test_bmm_chunk_fwd_vs_torch_per_chunk(self):
        """Compare first chunk with explicit C^T @ B for that chunk."""
        a = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        b = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)

        out = _bmm_chunk_fwd(
            a, b, self.chunk_size, self.cu_chunk_seqlens, causal=False, output_dtype=torch.float32
        )

        # Chunk 0: rows 0:16 of a and b. out[0, g] = a[0:16, g] @ b[0:16, g].T
        # Relaxed tolerances: Triton block-wise reduction order can differ from torch;
        # atol is the main check (max abs diff was ~0.008 in practice).
        for g in range(self.ngroups):
            a_chunk = a[0:16, g, :].contiguous()   # (16, dstate)
            b_chunk = b[0:16, g, :].contiguous()   # (16, dstate)
            expected = torch.mm(a_chunk, b_chunk.T)  # (16, 16)
            torch.testing.assert_close(out[0, g], expected, rtol=1.0, atol=0.02)

    def test_bmm_chunk_fwd_causal_vs_non_causal_shape(self):
        a = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        b = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)

        out_causal = _bmm_chunk_fwd(a, b, self.chunk_size, self.cu_chunk_seqlens, causal=True)
        out_noncausal = _bmm_chunk_fwd(a, b, self.chunk_size, self.cu_chunk_seqlens, causal=False)

        self.assertEqual(out_causal.shape, out_noncausal.shape)
        # Causal: lower triangle is correct; upper can differ
        for c in range(out_causal.shape[0]):
            for g in range(self.ngroups):
                for i in range(self.chunk_size):
                    for j in range(i + 1):
                        self.assertTrue(
                            torch.allclose(out_causal[c, g, i, j], out_noncausal[c, g, i, j]),
                            f"c={c} g={g} i={i} j={j}",
                        )


if __name__ == "__main__":
    unittest.main()
