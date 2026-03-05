# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import unittest
import torch

try:
    from megatron.core.ssm.ops.ssd_chunk_state import (
        _chunk_cumsum_fwd,
        _chunk_state_fwd,
        chunk_state_varlen,
    )
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestChunkCumsumFwd(unittest.TestCase):
    """Tests for _chunk_cumsum_fwd."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.seqlen = 32
        self.nheads = 4
        self.chunk_size = 16
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)

    def test_chunk_cumsum_fwd_shape(self):
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt, A, self.chunk_size, self.cu_chunk_seqlens
        )

        nchunks = self.cu_chunk_seqlens.shape[0] - 1
        self.assertEqual(dA_cumsum.shape, (self.nheads, nchunks, self.chunk_size))
        self.assertEqual(dt_out.shape, (self.nheads, nchunks, self.chunk_size))
        self.assertFalse(torch.isnan(dA_cumsum).any())
        self.assertFalse(torch.isnan(dt_out).any())

    def test_chunk_cumsum_fwd_cumsum_per_chunk(self):
        """dA_cumsum should be cumsum of dt * A along the chunk dimension."""
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt, A, self.chunk_size, self.cu_chunk_seqlens,
            dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")),
        )

        nchunks = self.cu_chunk_seqlens.shape[0] - 1
        for c in range(nchunks):
            start = self.cu_chunk_seqlens[c].item()
            end = self.cu_chunk_seqlens[c + 1].item()
            chunk_len = end - start
            for h in range(self.nheads):
                dA_chunk = (dt_out[h, c, :chunk_len] * A[h]).cpu()
                expected_cumsum = torch.cumsum(dA_chunk, dim=0)
                torch.testing.assert_close(
                    dA_cumsum[h, c, :chunk_len].cpu(), expected_cumsum, rtol=1e-4, atol=1e-4
                )

    def test_chunk_cumsum_fwd_with_dt_bias(self):
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        dt_bias = torch.randn(self.nheads, device=self.device, dtype=torch.float32)

        dA_cumsum, dt_out = _chunk_cumsum_fwd(
            dt, A, self.chunk_size, self.cu_chunk_seqlens, dt_bias=dt_bias
        )

        nchunks = self.cu_chunk_seqlens.shape[0] - 1
        self.assertEqual(dA_cumsum.shape, (self.nheads, nchunks, self.chunk_size))
        self.assertFalse(torch.isnan(dA_cumsum).any())


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestChunkStateFwd(unittest.TestCase):
    """Tests for _chunk_state_fwd."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.seqlen = 32
        self.nheads = 4
        self.headdim = 16
        self.ngroups = 2
        self.dstate = 8
        self.chunk_size = 16
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)

    def test_chunk_state_fwd_shape(self):
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        B = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)

        states = _chunk_state_fwd(B, x, dt, dA_cumsum, self.cu_chunk_seqlens)

        nchunks = self.cu_chunk_seqlens.shape[0] - 1
        self.assertEqual(states.shape, (nchunks, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(states).any())


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestChunkStateVarlen(unittest.TestCase):
    """Tests for chunk_state_varlen."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.seqlen = 32
        self.nheads = 4
        self.headdim = 16
        self.ngroups = 2
        self.dstate = 8
        self.chunk_size = 16
        self.batch = 2
        self.cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        self.last_chunk_indices = torch.tensor([0, 1], dtype=torch.int64, device=self.device)

    def test_chunk_state_varlen_shape(self):
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        B = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        chunk_states = torch.randn(2, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)

        states = chunk_state_varlen(
            B, x, dt, dA_cumsum, self.cu_seqlens, chunk_states,
            last_chunk_indices=self.last_chunk_indices,
            cu_chunk_seqlens=self.cu_chunk_seqlens,
        )

        self.assertEqual(states.shape, (self.batch, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(states).any())

    def test_chunk_state_varlen_with_initial_states(self):
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        B = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(self.nheads, 2, self.chunk_size, device=self.device, dtype=torch.float32)
        chunk_states = torch.randn(2, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)
        initial_states = torch.randn(self.batch, self.nheads, self.headdim, self.dstate, device=self.device, dtype=torch.float32)

        states = chunk_state_varlen(
            B, x, dt, dA_cumsum, self.cu_seqlens, chunk_states,
            initial_states=initial_states,
            last_chunk_indices=self.last_chunk_indices,
            cu_chunk_seqlens=self.cu_chunk_seqlens,
        )

        self.assertEqual(states.shape, (self.batch, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(states).any())


if __name__ == "__main__":
    unittest.main()
