# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import unittest
import torch

try:
    from megatron.core.ssm.ops.ssd_combined import (
        is_int_pow_2,
        mamba_chunk_scan_combined_varlen,
    )
    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestIsIntPow2(unittest.TestCase):
    """Tests for is_int_pow_2 utility."""

    def test_powers_of_two(self):
        for exp in range(12):
            n = 2 ** exp
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
        # cu_seqlens: [0, 16, 32] -> two sequences of length 16 each
        self.cu_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        # 2 chunks of 16 each
        self.cu_chunk_seqlens = torch.tensor([0, 16, 32], dtype=torch.int32, device=self.device)
        # last chunk index per sequence: seq0 ends in chunk 0, seq1 ends in chunk 1
        self.last_chunk_indices = torch.tensor([0, 1], dtype=torch.int64, device=self.device)
        # seq_idx: which sequence each chunk belongs to (nchunks,)
        self.seq_idx = torch.tensor([0, 1], dtype=torch.int32, device=self.device)

    def test_mamba_chunk_scan_combined_varlen_shape_and_no_nan(self):
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        C = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        out = torch.empty(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_seqlens=self.cu_seqlens,
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
        x = torch.randn(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt = torch.randn(self.seqlen, self.nheads, device=self.device, dtype=torch.float32)
        A = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        B = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        C = torch.randn(self.seqlen, self.ngroups, self.dstate, device=self.device, dtype=torch.float32)
        D = torch.ones(self.nheads, self.headdim, device=self.device, dtype=torch.float32)
        dt_bias = torch.randn(self.nheads, device=self.device, dtype=torch.float32)
        out = torch.empty(self.seqlen, self.nheads, self.headdim, device=self.device, dtype=torch.float32)

        varlen_states = mamba_chunk_scan_combined_varlen(
            x=x,
            dt=dt,
            A=A,
            B=B,
            C=C,
            chunk_size=self.chunk_size,
            cu_seqlens=self.cu_seqlens,
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
        """Single sequence: cu_seqlens [0, 32], one sequence of 32."""
        cu_seqlens = torch.tensor([0, 32], dtype=torch.int32, device=self.device)
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
            cu_seqlens=cu_seqlens,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
        )

        self.assertEqual(varlen_states.shape, (1, self.nheads, self.headdim, self.dstate))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()
