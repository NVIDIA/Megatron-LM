# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import unittest

import torch

try:
    from megatron.core.ssm.ops.ssd_combined import is_int_pow_2, mamba_chunk_scan_combined_varlen

    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestIsIntPow2(unittest.TestCase):
    """Tests for is_int_pow_2 utility."""

    def test_powers_of_two(self):
        for exp in range(12):
            n = 2**exp
            self.assertTrue(is_int_pow_2(n), f"2^{exp}={n} should be power of 2")

    def test_non_powers_of_two(self):
        for n in [0, 3, 5, 6, 7, 9, 10, 12, 15, 18]:
            self.assertFalse(is_int_pow_2(n), f"{n} should not be power of 2")

    def test_negative_and_float(self):
        self.assertFalse(is_int_pow_2(-1))
        self.assertFalse(is_int_pow_2(-4))
        self.assertFalse(is_int_pow_2(2.0))
        self.assertFalse(is_int_pow_2(0))


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestMambaChunkScanCombinedVarlen(unittest.TestCase):
    """Tests for mamba_chunk_scan_combined_varlen."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.chunk_size = 16
        self.seqlen = 32
        self.nheads = 4
        self.headdim = 16
        self.ngroups = 2
        self.dstate = 8
        self.batch = 2
        # 2 chunks of 16 each
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        # last chunk index per sequence: seq0 ends in chunk 0, seq1 ends in chunk 1
        self.last_chunk_indices = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        # seq_idx: which sequence each chunk belongs to (nchunks,)
        self.seq_idx = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

    def test_mamba_chunk_scan_combined_varlen_shape_and_no_nan(self):
        x = torch.randn(
            self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32
        )
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(
            self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32
        )
        C = torch.randn(
            self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32
        )
        out = torch.empty(
            self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32
        )

        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=self.cu_chunk_seqlens,
            last_chunk_indices=self.last_chunk_indices,
            seq_idx=self.seq_idx,
            out=out,
        )

        self.assertEqual(varlen_states.shape, (self.batch, self.nheads, self.headdim, self.dstate))
        self.assertEqual(out.shape, (self.seqlen, self.nheads, self.headdim))
        self.assertFalse(torch.isnan(out).any(), "output should have no NaN")
        self.assertFalse(torch.isnan(varlen_states).any(), "varlen_states should have no NaN")

    def test_mamba_chunk_scan_combined_varlen_with_D_and_dt_bias(self):
        x = torch.randn(
            self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32
        )
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(
            self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32
        )
        C = torch.randn(
            self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32
        )
        D = torch.ones(self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt_bias = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        out = torch.empty(
            self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32
        )

        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=self.cu_chunk_seqlens,
            last_chunk_indices=self.last_chunk_indices,
            seq_idx=self.seq_idx,
            out=out,
            D=D,
            dt_bias=dt_bias,
        )

        self.assertEqual(varlen_states.shape, (self.batch, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(out).any())

    def test_mamba_chunk_scan_combined_varlen_single_sequence(self):
        """Single sequence of 32 tokens, split into 2 chunks of 16."""
        cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        last_chunk_indices = torch.tensor([1], dtype=torch.int64, device=self.device)
        seq_idx = torch.tensor([0, 0], dtype=torch.int32, device=self.device)

        x = torch.randn(32, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(32, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(32, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        C = torch.randn(32, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.empty(32, self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
        )

        self.assertEqual(varlen_states.shape, (1, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(out).any())


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestIntermediateStateExtraction(unittest.TestCase):
    """Tests for intermediate_chunk_indices parameter."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")
        self.chunk_size = 16
        self.nheads = 4
        self.headdim = 16
        self.ngroups = 2
        self.dstate = 8

    def _make_inputs(self, seqlen):
        """Create random inputs for a single sequence of given length."""
        x = torch.randn(seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        C = torch.randn(seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.empty(
            seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32
        )
        return x, dt, A, B, C, out

    def test_intermediate_states_shape_and_no_nan(self):
        """1 sequence, 4 chunks. Request intermediates at chunks [0, 1, 2]."""
        seqlen = 64  # 4 chunks of 16
        nchunks = seqlen // self.chunk_size
        x, dt, A, B, C, out = self._make_inputs(seqlen)
        cu_chunk_seqlens = torch.arange(
            0, seqlen + 1, self.chunk_size, dtype=torch.int32, device=self.device
        )
        last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int64, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)
        intermediate_chunk_indices = torch.tensor([0, 1, 2], dtype=torch.int64, device=self.device)

        result = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            intermediate_chunk_indices=intermediate_chunk_indices,
        )

        self.assertIsInstance(result, tuple)
        final_states, intermediate_states = result
        self.assertEqual(final_states.shape, (1, self.nheads, self.headdim, self.dstate))
        self.assertEqual(intermediate_states.shape, (3, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(final_states).any())
        self.assertFalse(torch.isnan(intermediate_states).any())

    def test_intermediate_states_match_full_states(self):
        """Intermediate states should match corresponding entries from full states."""
        seqlen = 64  # 4 chunks
        nchunks = seqlen // self.chunk_size
        x, dt, A, B, C, out = self._make_inputs(seqlen)
        cu_chunk_seqlens = torch.arange(
            0, seqlen + 1, self.chunk_size, dtype=torch.int32, device=self.device
        )
        last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int64, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)

        # Run with return_intermediate_states=True to get all states
        out1 = torch.empty_like(out)
        all_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out1,
            return_intermediate_states=True,
        )

        # Run with intermediate_chunk_indices
        indices = [0, 1, 2]
        intermediate_chunk_indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        out2 = torch.empty_like(out)
        final_states, intermediate_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out2,
            intermediate_chunk_indices=intermediate_chunk_indices,
        )

        # Intermediate states should match the corresponding all_states entries
        for i, chunk_idx in enumerate(indices):
            torch.testing.assert_close(
                intermediate_states[i],
                all_states[chunk_idx],
                msg=f"intermediate state at index {i} (chunk {chunk_idx}) does not match",
            )

        # Final state should match last chunk
        torch.testing.assert_close(final_states[0], all_states[nchunks - 1])

    def test_intermediate_states_multi_sequence(self):
        """2 packed sequences, verify intermediate extraction across sequence boundaries."""
        seq1_len = 32  # 2 chunks
        seq2_len = 48  # 3 chunks
        total_len = seq1_len + seq2_len
        x, dt, A, B, C, out = self._make_inputs(total_len)

        # cu_chunk_seqlens: seq1 has chunks at [0, 16, 32], seq2 at [32, 48, 64, 80]
        boundaries = list(range(0, seq1_len + 1, self.chunk_size)) + list(
            range(seq1_len + self.chunk_size, total_len + 1, self.chunk_size)
        )
        cu_chunk_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=self.device)
        nchunks = len(boundaries) - 1  # 5 chunks total
        # Last chunk for seq1 is chunk 1, for seq2 is chunk 4
        last_chunk_indices = torch.tensor([1, 4], dtype=torch.int64, device=self.device)
        # seq_idx: [0, 0, 1, 1, 1]
        seq_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=self.device)

        # Request chunk 0 from seq1 and chunks 2, 3 from seq2
        intermediate_chunk_indices = torch.tensor([0, 2, 3], dtype=torch.int64, device=self.device)

        # Also get full states for comparison
        out_full = torch.empty_like(out)
        all_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out_full,
            return_intermediate_states=True,
        )

        out2 = torch.empty_like(out)
        final_states, intermediate_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out2,
            intermediate_chunk_indices=intermediate_chunk_indices,
        )

        self.assertEqual(final_states.shape, (2, self.nheads, self.headdim, self.dstate))
        self.assertEqual(intermediate_states.shape, (3, self.nheads, self.headdim, self.dstate))

        # Verify intermediate states match full states
        for i, chunk_idx in enumerate([0, 2, 3]):
            torch.testing.assert_close(intermediate_states[i], all_states[chunk_idx])

    def test_no_intermediate_returns_tensor(self):
        """Without intermediate_chunk_indices, result should be a plain tensor."""
        seqlen = 32
        nchunks = seqlen // self.chunk_size
        x, dt, A, B, C, out = self._make_inputs(seqlen)
        cu_chunk_seqlens = torch.arange(
            0, seqlen + 1, self.chunk_size, dtype=torch.int32, device=self.device
        )
        last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int64, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)

        result = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
        )

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, self.nheads, self.headdim, self.dstate))


if __name__ == "__main__":
    unittest.main()
