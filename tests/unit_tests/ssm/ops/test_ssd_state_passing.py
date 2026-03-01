# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import unittest
import torch

try:
    from megatron.core.ssm.ops.ssd_state_passing import _state_passing_fwd
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestStatePassingFwd(unittest.TestCase):
    """Tests for _state_passing_fwd: recurrence out = exp(dA_cs_last) * prev + new_states."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.nchunks = 4
        self.nheads = 2
        self.chunk_size = 16
        self.dim = self.chunk_size * 8  # headdim * dstate flattened
        self.cu_chunk_seqlens = torch.tensor(
            [0, 16, 32, 48, 64], dtype=torch.int32, device=self.device
        )

    def test_state_passing_fwd_shape(self):
        states = torch.randn(
            self.nchunks, self.nheads, self.dim, device=self.device, dtype=torch.float32
        )
        dA_cumsum = torch.randn(
            self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32
        )
        seq_idx = torch.zeros(self.nchunks, dtype=torch.int32, device=self.device)

        out = _state_passing_fwd(
            states, dA_cumsum, self.cu_chunk_seqlens, seq_idx, initial_states=None
        )

        self.assertEqual(out.shape, (self.nchunks, self.nheads, self.dim))
        self.assertFalse(torch.isnan(out).any())

    def test_state_passing_fwd_with_initial_states(self):
        states = torch.randn(
            self.nchunks, self.nheads, self.dim, device=self.device, dtype=torch.float32
        )
        dA_cumsum = torch.randn(
            self.nheads, self.nchunks, self.chunk_size, device=self.device, dtype=torch.float32
        )
        seq_idx = torch.tensor([0, 0, 1, 1], dtype=torch.int32, device=self.device)
        initial_states = torch.randn(2, self.nheads, self.dim, device=self.device, dtype=torch.float32)

        out = _state_passing_fwd(
            states,
            dA_cumsum,
            self.cu_chunk_seqlens,
            seq_idx,
            initial_states=initial_states,
        )

        self.assertEqual(out.shape, (self.nchunks, self.nheads, self.dim))
        self.assertFalse(torch.isnan(out).any())

    def test_state_passing_fwd_recurrence_single_head_single_dim(self):
        """Sanity: single head, small dim, check recurrence manually for first elements."""
        dim = 4
        nchunks = 2
        nheads = 1
        chunk_size = 2
        cu_chunk_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)

        states = torch.randn(nchunks, nheads, dim, device=self.device, dtype=torch.float32)
        dA_cumsum = torch.randn(nheads, nchunks, chunk_size, device=self.device, dtype=torch.float32)

        out = _state_passing_fwd(states, dA_cumsum, cu_chunk_seqlens, seq_idx)

        # Chunk 0: out[0] = exp(dA_cumsum[0,-1]) * 0 + states[0] = states[0] (no initial state)
        # So out[0] should equal states[0]
        torch.testing.assert_close(out[0], states[0], rtol=1e-4, atol=1e-4)
        self.assertEqual(out.shape, (nchunks, nheads, dim))


if __name__ == "__main__":
    unittest.main()
