# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import unittest

import torch

try:
    from megatron.core.ssm.ops.intermediate_extraction import (
        scatter_intermediate_conv,
        scatter_intermediate_ssm,
    )
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
    """Tests for the ``return_raw_states=True`` contract of the chunk scan.

    Intermediate extraction was refactored: the kernel no longer gathers
    requested chunks internally (old ``intermediate_chunk_indices`` /
    ``return_intermediate_states`` kwargs). Instead the scan returns the full
    ``(nchunks, ...)`` raw state tensor via ``return_raw_states=True``, and the
    caller extracts from it with the fused ``scatter_intermediate_ssm`` kernel
    (covered directly in TestScatterIntermediateKernels below).
    """

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

    def test_raw_states_shape_and_no_nan(self):
        """1 sequence, 4 chunks. return_raw_states yields (final, all-chunk) states."""
        seqlen = 64  # 4 chunks of 16
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
            return_raw_states=True,
        )

        self.assertIsInstance(result, tuple)
        final_states, raw_states = result
        self.assertEqual(final_states.shape, (1, self.nheads, self.headdim, self.dstate))
        # raw_states is the full per-chunk boundary state tensor.
        self.assertEqual(raw_states.shape, (nchunks, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(final_states).any())
        self.assertFalse(torch.isnan(raw_states).any())
        # The final state is exactly the last chunk's raw state.
        torch.testing.assert_close(final_states[0], raw_states[nchunks - 1])

    def test_scatter_from_scan_raw_states(self):
        """End-to-end: extract requested chunks from scan raw_states via the fused
        scatter kernel and compare against the reference gather."""
        seqlen = 64  # 4 chunks
        nchunks = seqlen // self.chunk_size
        x, dt, A, B, C, out = self._make_inputs(seqlen)
        cu_chunk_seqlens = torch.arange(
            0, seqlen + 1, self.chunk_size, dtype=torch.int32, device=self.device
        )
        last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int64, device=self.device)
        seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=self.device)

        final_states, raw_states = mamba_chunk_scan_combined_varlen(
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
            return_raw_states=True,
        )
        raw_states = raw_states.contiguous()

        indices = [0, 1, 2]
        chunk_indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        real_count_gpu = torch.tensor([len(indices)], dtype=torch.int32, device=self.device)
        scratch = torch.empty(
            len(indices),
            self.nheads,
            self.headdim,
            self.dstate,
            device=self.device,
            dtype=raw_states.dtype,
        )
        scatter_intermediate_ssm(raw_states, chunk_indices, real_count_gpu, scratch)

        torch.testing.assert_close(scratch, raw_states[chunk_indices])

    def test_scatter_from_scan_raw_states_multi_sequence(self):
        """2 packed sequences: extract chunks that straddle a sequence boundary."""
        seq1_len = 32  # 2 chunks
        seq2_len = 48  # 3 chunks
        total_len = seq1_len + seq2_len
        x, dt, A, B, C, out = self._make_inputs(total_len)

        boundaries = list(range(0, seq1_len + 1, self.chunk_size)) + list(
            range(seq1_len + self.chunk_size, total_len + 1, self.chunk_size)
        )
        cu_chunk_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=self.device)
        nchunks = len(boundaries) - 1  # 5 chunks total
        last_chunk_indices = torch.tensor([1, 4], dtype=torch.int64, device=self.device)
        seq_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device=self.device)

        final_states, raw_states = mamba_chunk_scan_combined_varlen(
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
            return_raw_states=True,
        )
        raw_states = raw_states.contiguous()
        self.assertEqual(final_states.shape, (2, self.nheads, self.headdim, self.dstate))
        self.assertEqual(raw_states.shape, (nchunks, self.nheads, self.headdim, self.dstate))

        # Request chunk 0 from seq1 and chunks 2, 3 from seq2.
        indices = [0, 2, 3]
        chunk_indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        real_count_gpu = torch.tensor([len(indices)], dtype=torch.int32, device=self.device)
        scratch = torch.empty(
            len(indices),
            self.nheads,
            self.headdim,
            self.dstate,
            device=self.device,
            dtype=raw_states.dtype,
        )
        scatter_intermediate_ssm(raw_states, chunk_indices, real_count_gpu, scratch)

        torch.testing.assert_close(scratch, raw_states[chunk_indices])

    def test_no_raw_states_returns_tensor(self):
        """Without return_raw_states, result should be a plain final-state tensor."""
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


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestScatterIntermediateKernels(unittest.TestCase):
    """Direct tests for the fused gather+scatter extraction kernels.

    These are the sole correctness coverage for scatter_intermediate_ssm /
    scatter_intermediate_conv: equivalence vs. a reference gather, real_count
    gating (padded slots left untouched), and the sub-d_conv clamp.
    """

    def setUp(self):
        torch.manual_seed(0)
        self.device = torch.device("cuda")
        self.nheads = 4
        self.headdim = 16
        self.dstate = 8

    def _ssm_states(self, num_chunks):
        return torch.randn(num_chunks, self.nheads, self.headdim, self.dstate, device=self.device)

    @staticmethod
    def _ref_conv(src, abs_positions, real_count, d_conv):
        """Reference for scatter_intermediate_conv: for each meaningful slot, gather
        the window [pos - d_conv, pos) (clamped into [0, seq_len-1]) and store it
        transposed as out[slot, c, j]."""
        _, seq_len, conv_dim = src.shape
        max_count = abs_positions.shape[0]
        out = torch.zeros(max_count, conv_dim, d_conv, device=src.device, dtype=src.dtype)
        for slot in range(real_count):
            pos = int(abs_positions[slot].item())
            for j in range(d_conv):
                p = max(0, min(pos - d_conv + j, seq_len - 1))
                out[slot, :, j] = src[0, p, :]
        return out

    def test_scatter_ssm_matches_reference(self):
        """Fused gather matches the dense states[chunk_indices] reference."""
        states = self._ssm_states(num_chunks=6)
        indices = [4, 0, 2]
        chunk_indices = torch.tensor(indices, dtype=torch.int64, device=self.device)
        real_count_gpu = torch.tensor([len(indices)], dtype=torch.int32, device=self.device)
        out = torch.empty(len(indices), self.nheads, self.headdim, self.dstate, device=self.device)

        scatter_intermediate_ssm(states, chunk_indices, real_count_gpu, out)

        torch.testing.assert_close(out, states[chunk_indices])

    def test_scatter_ssm_real_count_gating(self):
        """Slots >= real_count are never written (padded scratch left untouched)."""
        states = self._ssm_states(num_chunks=6)
        max_count, real_count = 5, 3
        # Trailing indices are valid but must NOT be gathered (gated out).
        chunk_indices = torch.tensor([4, 0, 2, 1, 5], dtype=torch.int64, device=self.device)
        real_count_gpu = torch.tensor([real_count], dtype=torch.int32, device=self.device)
        sentinel = 12345.0
        out = torch.full(
            (max_count, self.nheads, self.headdim, self.dstate), sentinel, device=self.device
        )

        scatter_intermediate_ssm(states, chunk_indices, real_count_gpu, out)

        # First real_count slots gathered...
        torch.testing.assert_close(out[:real_count], states[chunk_indices[:real_count]])
        # ...trailing slots left at the sentinel (no HBM write).
        self.assertTrue(torch.all(out[real_count:] == sentinel))

    def test_scatter_conv_matches_reference(self):
        """Fused conv-window gather matches the transposed reference (positions in range)."""
        d_conv = 4
        seq_len, conv_dim = 32, 12
        src = torch.randn(1, seq_len, conv_dim, device=self.device)
        abs_positions = torch.tensor([10, 20, 5], dtype=torch.int32, device=self.device)
        real_count_gpu = torch.tensor([3], dtype=torch.int32, device=self.device)
        out = torch.empty(3, conv_dim, d_conv, device=self.device)

        scatter_intermediate_conv(src, abs_positions, real_count_gpu, out, d_conv)

        ref = self._ref_conv(src, abs_positions, real_count=3, d_conv=d_conv)
        torch.testing.assert_close(out, ref)

    def test_scatter_conv_sub_dconv_clamp(self):
        """A window whose start falls below token 0 clamps into range (reads token 0)."""
        d_conv = 4
        seq_len, conv_dim = 32, 12
        src = torch.randn(1, seq_len, conv_dim, device=self.device)
        # slot 0: pos=2 < d_conv -> window [-2, -1, 0, 1] clamps to [0, 0, 0, 1].
        abs_positions = torch.tensor([2, 20], dtype=torch.int32, device=self.device)
        real_count_gpu = torch.tensor([2], dtype=torch.int32, device=self.device)
        out = torch.empty(2, conv_dim, d_conv, device=self.device)

        scatter_intermediate_conv(src, abs_positions, real_count_gpu, out, d_conv)

        ref = self._ref_conv(src, abs_positions, real_count=2, d_conv=d_conv)
        torch.testing.assert_close(out, ref)
        # Explicitly: the three out-of-range positions all clamp to token 0.
        torch.testing.assert_close(out[0, :, 0], src[0, 0, :])
        torch.testing.assert_close(out[0, :, 1], src[0, 0, :])
        torch.testing.assert_close(out[0, :, 2], src[0, 0, :])
        torch.testing.assert_close(out[0, :, 3], src[0, 1, :])

    def test_scatter_conv_real_count_gating(self):
        """Slots >= real_count are never written by the conv kernel either."""
        d_conv = 4
        seq_len, conv_dim = 32, 12
        src = torch.randn(1, seq_len, conv_dim, device=self.device)
        max_count, real_count = 4, 2
        abs_positions = torch.tensor([10, 20, 15, 25], dtype=torch.int32, device=self.device)
        real_count_gpu = torch.tensor([real_count], dtype=torch.int32, device=self.device)
        sentinel = -999.0
        out = torch.full((max_count, conv_dim, d_conv), sentinel, device=self.device)

        scatter_intermediate_conv(src, abs_positions, real_count_gpu, out, d_conv)

        ref = self._ref_conv(src, abs_positions, real_count=real_count, d_conv=d_conv)
        torch.testing.assert_close(out[:real_count], ref[:real_count])
        self.assertTrue(torch.all(out[real_count:] == sentinel))


if __name__ == "__main__":
    unittest.main()
