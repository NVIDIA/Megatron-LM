# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import unittest
import torch

try:
    from megatron.core.ssm.ops.ssd_chunk_scan import _chunk_scan_fwd
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestChunkScanFwd(unittest.TestCase):
    """Tests for _chunk_scan_fwd."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.seqlen = 32
        self.nheads = 4
        self.headdim = 16
        self.ngroups = 2
        self.dstate = 8
        self.chunk_size = 16
        self.nchunks = 2
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        self.seq_idx = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

    def test_chunk_scan_fwd_shape_and_inplace_out(self):
        cb = torch.randn(
            self.nchunks, self.ngroups, self.chunk_size, self.chunk_size,
            device=self.device, dtype=torch.float32,
        )
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        C = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        states = torch.randn(self.nchunks, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.zeros(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        _chunk_scan_fwd(
            cb, x, dt, dA_cumsum, C, states,
            self.cu_chunk_seqlens, out, self.seq_idx,
            D=None, z=None, initial_states=None,
        )

        self.assertEqual(out.shape, (self.seqlen, self.nheads, self.headdim))
        self.assertFalse(torch.isnan(out).any())
        # Output should be non-zero (scan writes to out)
        self.assertGreater(out.abs().max().item(), 0.0)

    def test_chunk_scan_fwd_with_D(self):
        cb = torch.randn(
            self.nchunks, self.ngroups, self.chunk_size, self.chunk_size,
            device=self.device, dtype=torch.float32,
        )
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        C = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        states = torch.randn(self.nchunks, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.zeros(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        D = torch.ones(self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        _chunk_scan_fwd(
            cb, x, dt, dA_cumsum, C, states,
            self.cu_chunk_seqlens, out, self.seq_idx,
            D=D, z=None, initial_states=None,
        )

        self.assertFalse(torch.isnan(out).any())

    def test_chunk_scan_fwd_with_z(self):
        cb = torch.randn(
            self.nchunks, self.ngroups, self.chunk_size, self.chunk_size,
            device=self.device, dtype=torch.float32,
        )
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        z = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32)
        C = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        states = torch.randn(self.nchunks, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.zeros(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        _chunk_scan_fwd(
            cb, x, dt, dA_cumsum, C, states,
            self.cu_chunk_seqlens, out, self.seq_idx,
            D=None, z=z, initial_states=None,
        )

        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
