# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Numerical correctness tests for the batch-invariant SSM decode kernel.

`bik_decode_buffered_scan` claims that its single-token output is bitwise
identical to running `mamba_chunk_scan_combined` over the full
(prefill + decode_token) sequence — that's what makes BIK RL rollout
log-probs match the training-side recompute. These tests verify that claim
directly on plain tensors, across the configurations the BIK decode path
needs to handle.
"""

import unittest

import torch

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from megatron.core.ssm.ops.bik_decode import (
        bik_decode_buffered_scan,
        make_bik_decode_buffers,
        seed_bik_decode_buffers,
    )

    HAVE_BIK_DECODE = True
except (ImportError, Exception):
    HAVE_BIK_DECODE = False


def _full_scan(x, dt, A, B, C, D, dt_bias, chunk_size, initial_states=None):
    """Reference: a single `mamba_chunk_scan_combined` over the whole sequence."""
    y, final = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        D=D, z=None, dt_bias=dt_bias, dt_softplus=True,
        initial_states=initial_states, return_final_states=True,
    )
    return y, final


@unittest.skipIf(not HAVE_BIK_DECODE, "mamba_ssm / bik_decode unavailable")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestBikDecodeBufferedScan(unittest.TestCase):
    """Verify the BIK decode scan matches a full-sequence scan bitwise."""

    def setUp(self):
        torch.manual_seed(0)
        # use_deterministic_algorithms is what BIK actually flips on; the
        # mamba_chunk_scan_combined kernel is deterministic under that flag.
        torch.use_deterministic_algorithms(False)
        self.device = torch.device("cuda")
        self.dtype = torch.bfloat16
        # Small but non-trivial mamba dims.
        self.nh = 8
        self.headdim = 32
        self.ngroups = 1
        self.dstate = 16
        self.chunk_size = 32
        self.A = -torch.exp(
            torch.randn(self.nh, device=self.device, dtype=torch.float32).abs()
        )
        self.D = torch.randn(self.nh, device=self.device, dtype=torch.float32) * 0.1
        self.dt_bias = (
            torch.randn(self.nh, device=self.device, dtype=torch.float32) * 0.01
        )

    def _make_seq(self, total_len):
        """Generate a (1, total_len, ...) random mamba input sequence."""
        nh, p, ng, n = self.nh, self.headdim, self.ngroups, self.dstate
        return (
            torch.randn(1, total_len, nh, p, device=self.device, dtype=self.dtype) * 0.1,
            torch.randn(1, total_len, nh, device=self.device, dtype=self.dtype).abs() * 0.1,
            torch.randn(1, total_len, ng, n, device=self.device, dtype=self.dtype) * 0.1,
            torch.randn(1, total_len, ng, n, device=self.device, dtype=self.dtype) * 0.1,
        )

    def _make_bufs(self, max_batch):
        return make_bik_decode_buffers(
            max_batch, self.chunk_size,
            self.nh, self.headdim, self.ngroups, self.dstate,
            self.device, self.dtype, self.dtype, self.dtype, self.dtype, self.dtype,
        )

    def _seed_from_prefill(self, bufs, x, dt, B, C, prefill_len, slot, max_batch):
        """Run the prefill through the reference scan, store its ssm_state at
        the slot, and seed the BIK buffer with the partial-chunk tail."""
        ssm_state = torch.zeros(
            max_batch, self.nh, self.headdim, self.dstate,
            device=self.device, dtype=self.dtype,
        )
        if prefill_len >= self.chunk_size:
            # Prefill on the largest chunk-aligned prefix; the tail goes in the buffer.
            aligned = (prefill_len // self.chunk_size) * self.chunk_size
            _, final = _full_scan(
                x[:, :aligned], dt[:, :aligned], self.A,
                B[:, :aligned], C[:, :aligned], self.D, self.dt_bias, self.chunk_size,
                initial_states=None,
            )
            ssm_state[slot] = final[0].to(self.dtype)

        cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        # seed_bik_decode_buffers expects the squeeze-batch layout
        # (the flat (total, nh, p) form used inside the mixer's prefill path).
        # Here total == prefill_len since we have 1 sequence.
        seed_bik_decode_buffers(
            bufs,
            x[0, :prefill_len],
            dt[0, :prefill_len],
            B[0, :prefill_len],
            C[0, :prefill_len],
            torch.zeros_like(x[0, :prefill_len]),  # z unused
            cu, batch_indices,
        )
        return ssm_state

    def _decode_one_step(self, bufs, x, dt, B, C, pos, slot, ssm_state):
        """Call bik_decode_buffered_scan for the single token at index `pos`."""
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        return bik_decode_buffered_scan(
            bufs,
            x[:, pos : pos + 1],
            dt[:, pos : pos + 1],
            B[:, pos : pos + 1],
            C[:, pos : pos + 1],
            None,
            self.A, self.D, self.dt_bias,
            batch_indices,
            ssm_state,
        )

    def _assert_bitwise(self, a, b, msg):
        # bf16 outputs — bitwise-equal is the actual BIK claim.
        diff = (a.float() - b.float()).abs().max().item()
        self.assertEqual(diff, 0.0, f"{msg}: max_abs_diff={diff:.3e}")

    def test_single_decode_matches_full_scan(self):
        """Default case: prefill > chunk_size, single decode token, partial tail."""
        max_batch, slot = 4, 1
        for prefill_len in [33, 50, 95, 128]:
            with self.subTest(prefill_len=prefill_len):
                total = prefill_len + 1
                x, dt, B, C = self._make_seq(total)
                # Reference: full scan over the whole (prefill + 1) sequence.
                y_full, _ = _full_scan(
                    x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
                )
                # BIK: seed from prefill, then one decode step.
                bufs = self._make_bufs(max_batch)
                ssm_state = self._seed_from_prefill(
                    bufs, x, dt, B, C, prefill_len, slot, max_batch,
                )
                y_bik = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state,
                )
                self._assert_bitwise(
                    y_bik[0, 0], y_full[0, prefill_len],
                    f"prefill_len={prefill_len}",
                )

    def test_short_prefill_zero_state_path(self):
        """prefill_len < chunk_size: state_is_zero path, no boundary state."""
        max_batch, slot = 2, 0
        for prefill_len in [1, 7, 16, 31]:
            with self.subTest(prefill_len=prefill_len):
                total = prefill_len + 1
                x, dt, B, C = self._make_seq(total)
                y_full, _ = _full_scan(
                    x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
                )
                bufs = self._make_bufs(max_batch)
                ssm_state = self._seed_from_prefill(
                    bufs, x, dt, B, C, prefill_len, slot, max_batch,
                )
                # Sanity: state_is_zero should be True here.
                self.assertTrue(bool(bufs.state_is_zero[slot].item()))
                y_bik = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state,
                )
                self._assert_bitwise(
                    y_bik[0, 0], y_full[0, prefill_len],
                    f"prefill_len={prefill_len}",
                )

    def test_multi_step_decode_across_chunk_boundary(self):
        """Step decode several times so the per-slot buffer fills, crosses
        a chunk boundary, and resets. Each step must match the full scan."""
        max_batch, slot = 2, 0
        prefill_len = 20  # < chunk_size, so first decode step will keep growing buf
        n_decode = self.chunk_size + 5  # enough to cross at least one boundary
        total = prefill_len + n_decode
        x, dt, B, C = self._make_seq(total)
        y_full, _ = _full_scan(
            x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
        )

        bufs = self._make_bufs(max_batch)
        ssm_state = self._seed_from_prefill(
            bufs, x, dt, B, C, prefill_len, slot, max_batch,
        )

        for k in range(n_decode):
            pos = prefill_len + k
            y_bik = self._decode_one_step(bufs, x, dt, B, C, pos, slot, ssm_state)
            self._assert_bitwise(
                y_bik[0, 0], y_full[0, pos],
                f"step k={k} (pos={pos}, buf_count_before={bufs.count[slot].item()})",
            )

    def test_multi_slot_independent_streams(self):
        """Two slots with different prefill lengths decoded in the same call —
        each slot's output must match its own full scan."""
        max_batch = 4
        slots = [0, 2]
        prefill_lens = [25, 70]  # one short (zero-state), one long (boundary state)
        x_per_slot, dt_per_slot, B_per_slot, C_per_slot = [], [], [], []
        y_refs = []
        for plen in prefill_lens:
            x, dt, B, C = self._make_seq(plen + 1)
            x_per_slot.append(x); dt_per_slot.append(dt)
            B_per_slot.append(B); C_per_slot.append(C)
            y_full, _ = _full_scan(
                x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
            )
            y_refs.append(y_full[0, plen])

        bufs = self._make_bufs(max_batch)
        # Per-slot seeding (each slot's prefill done independently).
        ssm_state = torch.zeros(
            max_batch, self.nh, self.headdim, self.dstate,
            device=self.device, dtype=self.dtype,
        )
        for slot, plen, x, dt, B, C in zip(
            slots, prefill_lens, x_per_slot, dt_per_slot, B_per_slot, C_per_slot,
        ):
            partial = self._seed_from_prefill(
                bufs, x, dt, B, C, plen, slot, max_batch,
            )
            ssm_state[slot] = partial[slot]

        # Both slots step at once.
        x_step = torch.cat(
            [x_per_slot[i][:, prefill_lens[i] : prefill_lens[i] + 1] for i in range(2)],
            dim=0,
        )
        dt_step = torch.cat(
            [dt_per_slot[i][:, prefill_lens[i] : prefill_lens[i] + 1] for i in range(2)],
            dim=0,
        )
        B_step = torch.cat(
            [B_per_slot[i][:, prefill_lens[i] : prefill_lens[i] + 1] for i in range(2)],
            dim=0,
        )
        C_step = torch.cat(
            [C_per_slot[i][:, prefill_lens[i] : prefill_lens[i] + 1] for i in range(2)],
            dim=0,
        )
        batch_indices = torch.tensor(slots, dtype=torch.int32, device=self.device)
        y_bik = bik_decode_buffered_scan(
            bufs, x_step, dt_step, B_step, C_step, None,
            self.A, self.D, self.dt_bias, batch_indices, ssm_state,
        )
        for i, plen in enumerate(prefill_lens):
            self._assert_bitwise(
                y_bik[i, 0], y_refs[i],
                f"multi-slot slot={slots[i]} prefill_len={plen}",
            )

    def test_deterministic_across_calls(self):
        """Same inputs → bitwise-identical output across repeated invocations."""
        max_batch, slot = 2, 0
        prefill_len = 50
        total = prefill_len + 1
        x, dt, B, C = self._make_seq(total)

        outs = []
        for _ in range(3):
            bufs = self._make_bufs(max_batch)
            ssm_state = self._seed_from_prefill(
                bufs, x, dt, B, C, prefill_len, slot, max_batch,
            )
            outs.append(
                self._decode_one_step(bufs, x, dt, B, C, prefill_len, slot, ssm_state)
            )
        for i in range(1, len(outs)):
            self.assertTrue(torch.equal(outs[0], outs[i]),
                            f"determinism: run {i} differs from run 0")


if __name__ == "__main__":
    unittest.main()
