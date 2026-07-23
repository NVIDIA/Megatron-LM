# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Tests for batch-invariant Mamba decode."""

import unittest

import torch

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from megatron.core.ssm.ops.batch_invariant_decode import (
        BatchInvariantDecodeBuffers,
        batch_invariant_decode_buffered_scan,
    )
    from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_combined_varlen

    HAVE_BATCH_INVARIANT_DECODE = True
except ImportError:
    HAVE_BATCH_INVARIANT_DECODE = False


def _full_scan(x, dt, A, B, C, D, dt_bias, chunk_size, initial_states=None):
    """Reference: a single `mamba_chunk_scan_combined` over the whole sequence."""
    y, final = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        D=D, z=None, dt_bias=dt_bias, dt_softplus=True,
        initial_states=initial_states, return_final_states=True,
    )
    return y, final


@unittest.skipIf(not HAVE_BATCH_INVARIANT_DECODE, "mamba_ssm / batch_invariant_decode unavailable")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestBatchInvariantDecodeBufferedScan(unittest.TestCase):
    """Verify the batch-invariant decode scan matches a full-sequence scan bitwise."""

    @classmethod
    def setUpClass(cls):
        # Pin the Mamba autotuners exactly like enable_batch_invariant_mode
        # does in production: without pinning, autotune timing noise can pick
        # different tile configs per process and flake the bitwise asserts.
        from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
            _pin_mamba_autotuners,
        )

        _pin_mamba_autotuners()

    def setUp(self):
        torch.manual_seed(0)
        # No global flags: batch_invariant_decode_buffered_scan is a pure tensor-ops function
        # and batch-invariant mode by design does not require
        # torch.use_deterministic_algorithms.
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
        return BatchInvariantDecodeBuffers.allocate(
            max_batch, self.chunk_size,
            self.nh, self.headdim, self.ngroups, self.dstate,
            self.device, self.dtype,
        )

    def _make_ssm_state(self, max_batch):
        """Production BIK state cache: FP32 carry across Mamba chunks."""
        return torch.zeros(
            max_batch,
            self.nh,
            self.headdim,
            self.dstate,
            device=self.device,
            dtype=torch.float32,
        )

    def _seed_from_prefill(self, bufs, x, dt, B, C, prefill_len, slot, max_batch):
        """Run the prefill through the reference scan, store its ssm_state at
        the slot, and seed the batch-invariant buffer with the partial-chunk tail."""
        # Production batch-invariant prefill keeps ssm_state at a full Mamba chunk
        # boundary. Short prefills therefore keep the zero initial boundary;
        # longer prefills store the largest chunk-aligned prefix state.
        ssm_state = self._make_ssm_state(max_batch)
        if prefill_len >= self.chunk_size:
            # Prefill on the largest chunk-aligned prefix; the tail goes in the buffer.
            aligned = (prefill_len // self.chunk_size) * self.chunk_size
            _, final = _full_scan(
                x[:, :aligned], dt[:, :aligned], self.A,
                B[:, :aligned], C[:, :aligned], self.D, self.dt_bias, self.chunk_size,
                initial_states=None,
            )
            ssm_state[slot] = final[0]

        cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        # Buffer seeding expects the flat layout used by the mixer's prefill path.
        # Here total == prefill_len since we have 1 sequence.
        bufs.seed(
            x[0, :prefill_len],
            dt[0, :prefill_len],
            B[0, :prefill_len],
            C[0, :prefill_len],
            cu, batch_indices,
        )
        return ssm_state

    def _decode_one_step(self, bufs, x, dt, B, C, pos, slot, ssm_state):
        """Call batch_invariant_decode_buffered_scan for the single token at index `pos`."""
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        return batch_invariant_decode_buffered_scan(
            bufs,
            x[:, pos : pos + 1],
            dt[:, pos : pos + 1],
            B[:, pos : pos + 1],
            C[:, pos : pos + 1],
            self.A, self.D, self.dt_bias,
            batch_indices,
            ssm_state,
        )

    def _varlen_boundary_state_from_prefill(
        self, x, dt, B, C, prefill_len, initial_states=None
    ):
        """Production-shaped varlen prefill returning the last full chunk boundary state."""
        chunk_boundaries = [0]
        pos = self.chunk_size
        while pos < prefill_len:
            chunk_boundaries.append(pos)
            pos += self.chunk_size
        chunk_boundaries.append(prefill_len)

        cu_chunk_seqlens = torch.tensor(
            chunk_boundaries, dtype=torch.int32, device=self.device
        )
        last_chunk_indices = torch.tensor(
            [len(chunk_boundaries) - 2], dtype=torch.int32, device=self.device
        )
        tail_len = prefill_len % self.chunk_size
        has_boundary = prefill_len >= self.chunk_size
        boundary_idx = last_chunk_indices.to(torch.long)
        if tail_len != 0:
            boundary_idx = boundary_idx - 1
        boundary_idx = boundary_idx.clamp(min=0)

        out = torch.zeros_like(x[0, :prefill_len])
        seq_idx = torch.zeros(
            len(chunk_boundaries) - 1, dtype=torch.int32, device=self.device
        )
        chunk_states = mamba_chunk_scan_combined_varlen(
            x=x[0, :prefill_len],
            dt=dt[0, :prefill_len],
            A=self.A,
            B=B[0, :prefill_len],
            C=C[0, :prefill_len],
            chunk_size=self.chunk_size,
            cu_chunk_seqlens=cu_chunk_seqlens,
            last_chunk_indices=last_chunk_indices,
            seq_idx=seq_idx,
            out=out,
            D=self.D,
            z=None,
            dt_bias=self.dt_bias,
            initial_states=initial_states,
            return_intermediate_states=True,
            dt_softplus=True,
            dt_limit=(0.0, float("inf")),
            state_dtype=torch.float32,
        )
        final_state = chunk_states[last_chunk_indices]
        boundary_state = chunk_states[boundary_idx]
        if not has_boundary:
            boundary_state = (
                torch.zeros_like(boundary_state)
                if initial_states is None
                else initial_states
            )
        return final_state, boundary_state

    def _assert_bitwise(self, a, b, msg):
        # bf16 outputs — bitwise-equal is the actual batch-invariant claim.
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
                # batch-invariant: seed from prefill, then one decode step.
                bufs = self._make_bufs(max_batch)
                ssm_state = self._seed_from_prefill(
                    bufs, x, dt, B, C, prefill_len, slot, max_batch,
                )
                y_batch_invariant = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state,
                )
                self._assert_bitwise(
                    y_batch_invariant[0, 0], y_full[0, prefill_len],
                    f"prefill_len={prefill_len}",
                )

    def test_rejects_bf16_state_cache(self):
        """A rounded state cache cannot preserve carry across multiple chunks."""
        x, dt, B, C = self._make_seq(1)
        bufs = self._make_bufs(max_batch=2)
        ssm_state = torch.zeros(
            2,
            self.nh,
            self.headdim,
            self.dstate,
            device=self.device,
            dtype=torch.bfloat16,
        )
        with self.assertRaisesRegex(AssertionError, "requires an FP32 SSM state cache"):
            self._decode_one_step(bufs, x, dt, B, C, pos=0, slot=0, ssm_state=ssm_state)

    def test_inference_config_uses_fp32_state_cache(self):
        """BIK model-derived inference config cannot select a rounded state dtype."""
        from types import SimpleNamespace
        from unittest.mock import patch

        from megatron.core.inference.config import MambaInferenceStateConfig
        from megatron.core.models.hybrid.hybrid_layer_allocation import Symbols

        model = SimpleNamespace(
            config=SimpleNamespace(
                batch_invariant_mode=True,
                params_dtype=torch.bfloat16,
            )
        )
        decoder = SimpleNamespace(
            layer_type_list=[Symbols.MAMBA],
            layers=[SimpleNamespace(mixer=SimpleNamespace(chunk_size=self.chunk_size))],
            mamba_state_shapes_per_request=lambda: ((4, 8), (8, 32, 16)),
        )
        with patch(
            "megatron.core.inference.config.get_attr_wrapped_model",
            return_value=decoder,
        ):
            config = MambaInferenceStateConfig.from_model(model)
            self.assertEqual(config.ssm_states_dtype, torch.float32)
            with self.assertRaisesRegex(ValueError, "requires FP32 Mamba SSM states"):
                MambaInferenceStateConfig.from_model(
                    model,
                    ssm_states_dtype=torch.bfloat16,
                )
            model.config.batch_invariant_mode = False
            config = MambaInferenceStateConfig.from_model(model)
            self.assertEqual(config.ssm_states_dtype, torch.bfloat16)

    def test_dynamic_prefill_uses_boundary_state_not_prompt_end_state(self):
        """Production prefill returns the prompt-end state too, but batch-invariant decode
        must keep the cache at the last full chunk boundary and put the tail in
        the replay buffer."""
        max_batch, slot = 4, 1
        for prefill_len in [31, 33, 50, 95, 128]:
            with self.subTest(prefill_len=prefill_len):
                total = prefill_len + 1
                x, dt, B, C = self._make_seq(total)
                y_full, _ = _full_scan(
                    x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
                )

                _, boundary_state = self._varlen_boundary_state_from_prefill(
                    x, dt, B, C, prefill_len
                )
                ssm_state = torch.randn_like(self._make_ssm_state(max_batch))
                ssm_state[slot] = boundary_state[0]

                bufs = self._make_bufs(max_batch)
                cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
                batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
                bufs.seed(
                    x[0, :prefill_len],
                    dt[0, :prefill_len],
                    B[0, :prefill_len],
                    C[0, :prefill_len],
                    cu,
                    batch_indices,
                )
                y_batch_invariant = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state
                )
                self._assert_bitwise(
                    y_batch_invariant[0, 0], y_full[0, prefill_len],
                    f"dynamic prefill boundary state prefill_len={prefill_len}",
                )

    def test_chunked_prefill_handoff_matches_full_scan(self):
        """Splitting prefill at a Mamba boundary preserves exact decode output."""
        max_batch, slot = 2, 0
        first_chunk_len = 2 * self.chunk_size

        for final_chunk_len in [20, self.chunk_size + 13]:
            with self.subTest(final_chunk_len=final_chunk_len):
                prefill_len = first_chunk_len + final_chunk_len
                x, dt, B, C = self._make_seq(prefill_len + 1)
                y_full, _ = _full_scan(
                    x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
                )

                _, first_boundary = self._varlen_boundary_state_from_prefill(
                    x[:, :first_chunk_len],
                    dt[:, :first_chunk_len],
                    B[:, :first_chunk_len],
                    C[:, :first_chunk_len],
                    first_chunk_len,
                )
                _, final_boundary = self._varlen_boundary_state_from_prefill(
                    x[:, first_chunk_len:prefill_len],
                    dt[:, first_chunk_len:prefill_len],
                    B[:, first_chunk_len:prefill_len],
                    C[:, first_chunk_len:prefill_len],
                    final_chunk_len,
                    initial_states=first_boundary,
                )

                ssm_state = self._make_ssm_state(max_batch)
                ssm_state[slot] = final_boundary[0]
                bufs = self._make_bufs(max_batch)
                cu = torch.tensor(
                    [0, final_chunk_len], dtype=torch.int32, device=self.device
                )
                bufs.seed(
                    x[0, first_chunk_len:prefill_len],
                    dt[0, first_chunk_len:prefill_len],
                    B[0, first_chunk_len:prefill_len],
                    C[0, first_chunk_len:prefill_len],
                    cu,
                    torch.tensor([slot], dtype=torch.int32, device=self.device),
                )
                y_batch_invariant = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state
                )
                self._assert_bitwise(
                    y_batch_invariant[0, 0],
                    y_full[0, prefill_len],
                    f"chunked prefill final_chunk_len={final_chunk_len}",
                )

    def test_seed_ignores_nonfinite_physical_padding_rows(self):
        """Dynamic prefill can carry padded physical token rows after the real
        prefix. Seed must duplicate a valid per-sequence tail token into unused
        replay-buffer rows; otherwise masked future rows can still poison the
        row-gated Triton dot as 0 * NaN."""
        max_batch, slot = 4, 0
        prefill_len = self.chunk_size + 1
        total = prefill_len + 1
        x, dt, B, C = self._make_seq(total)
        y_full, _ = _full_scan(
            x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
        )

        _, boundary_state = self._varlen_boundary_state_from_prefill(x, dt, B, C, prefill_len)
        ssm_state = self._make_ssm_state(max_batch)
        ssm_state[slot] = boundary_state[0]

        nan_x = torch.full_like(x[0, :1], float("nan"))
        nan_dt = torch.full_like(dt[0, :1], float("nan"))
        nan_B = torch.full_like(B[0, :1], float("nan"))
        nan_C = torch.full_like(C[0, :1], float("nan"))

        bufs = self._make_bufs(max_batch)
        cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        bufs.seed(
            torch.cat([x[0, :prefill_len], nan_x], dim=0),
            torch.cat([dt[0, :prefill_len], nan_dt], dim=0),
            torch.cat([B[0, :prefill_len], nan_B], dim=0),
            torch.cat([C[0, :prefill_len], nan_C], dim=0),
            cu,
            batch_indices,
        )
        self.assertTrue(torch.isfinite(bufs.x[slot]).all())
        self.assertTrue(torch.isfinite(bufs.dt[slot]).all())
        self.assertTrue(torch.isfinite(bufs.B[slot]).all())
        self.assertTrue(torch.isfinite(bufs.C[slot]).all())

        y_batch_invariant = self._decode_one_step(
            bufs, x, dt, B, C, prefill_len, slot, ssm_state
        )
        self._assert_bitwise(
            y_batch_invariant[0, 0], y_full[0, prefill_len],
            "nonfinite physical padding rows",
        )

    def test_short_prefill_uses_zero_boundary_state(self):
        """prefill_len < chunk_size: decode replays from the zero boundary."""
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
                y_batch_invariant = self._decode_one_step(
                    bufs, x, dt, B, C, prefill_len, slot, ssm_state,
                )
                self._assert_bitwise(
                    y_batch_invariant[0, 0], y_full[0, prefill_len],
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
            y_batch_invariant = self._decode_one_step(bufs, x, dt, B, C, pos, slot, ssm_state)
            self._assert_bitwise(
                y_batch_invariant[0, 0], y_full[0, pos],
                f"step k={k} (pos={pos}, num_buffered_before={bufs.num_buffered[slot].item()})",
            )

    def test_multi_slot_independent_streams(self):
        """Two slots with different prefill lengths decoded in the same call —
        each slot's output must match its own full scan."""
        max_batch = 4
        slots = [0, 2]
        prefill_lens = [25, 70]  # one short, one long with a boundary state
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
        ssm_state = self._make_ssm_state(max_batch)
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
        y_batch_invariant = batch_invariant_decode_buffered_scan(
            bufs, x_step, dt_step, B_step, C_step,
            self.A, self.D, self.dt_bias, batch_indices, ssm_state,
        )
        for i, plen in enumerate(prefill_lens):
            self._assert_bitwise(
                y_batch_invariant[i, 0], y_refs[i],
                f"multi-slot slot={slots[i]} prefill_len={plen}",
            )

    def test_inactive_padding_entries(self):
        """batch_indices mixing -1 padding entries with active slot 0 (the CUDA-
        graph padding pattern). Padding entries must not perturb slot 0's buffer
        or output — they are redirected to the buffers' trash row."""
        max_batch, slot = 2, 0
        prefill_len = 50
        n_decode = 8
        total = prefill_len + n_decode
        x, dt, B, C = self._make_seq(total)
        y_full, _ = _full_scan(
            x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
        )

        bufs = self._make_bufs(max_batch)
        ssm_state = self._seed_from_prefill(
            bufs, x, dt, B, C, prefill_len, slot, max_batch,
        )
        batch_indices = torch.tensor([slot, -1, -1], dtype=torch.int32, device=self.device)
        for k in range(n_decode):
            pos = prefill_len + k
            # Entry 0 carries the real token; padding entries carry garbage.
            def pad3(t):
                junk = torch.randn(
                    2, *t.shape[1:], device=t.device, dtype=t.dtype
                )
                return torch.cat([t, junk], dim=0)

            y_batch_invariant = batch_invariant_decode_buffered_scan(
                bufs,
                pad3(x[:, pos : pos + 1]),
                pad3(dt[:, pos : pos + 1]),
                pad3(B[:, pos : pos + 1]),
                pad3(C[:, pos : pos + 1]),
                self.A, self.D, self.dt_bias,
                batch_indices,
                ssm_state,
            )
            self._assert_bitwise(
                y_batch_invariant[0, 0], y_full[0, pos], f"padded step k={k}"
            )
            # Padding entries must return zeros.
            self.assertEqual(y_batch_invariant[1:].abs().max().item(), 0.0)

    def test_cuda_graph_replay_matches_full_scan(self):
        """A captured decode step advances persistent state exactly across replays."""
        max_batch, slot = 2, 0
        prefill_len = 20
        x, dt, B, C = self._make_seq(prefill_len + 2)
        y_full, _ = _full_scan(
            x, dt, self.A, B, C, self.D, self.dt_bias, self.chunk_size,
        )

        # Compile Triton before capture without touching the graph's buffers.
        warmup_bufs = self._make_bufs(max_batch)
        warmup_state = self._seed_from_prefill(
            warmup_bufs, x, dt, B, C, prefill_len, slot, max_batch,
        )
        self._decode_one_step(
            warmup_bufs, x, dt, B, C, prefill_len, slot, warmup_state
        )

        bufs = self._make_bufs(max_batch)
        ssm_state = self._seed_from_prefill(
            bufs, x, dt, B, C, prefill_len, slot, max_batch,
        )
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        static_x = x[:, prefill_len : prefill_len + 1].clone()
        static_dt = dt[:, prefill_len : prefill_len + 1].clone()
        static_B = B[:, prefill_len : prefill_len + 1].clone()
        static_C = C[:, prefill_len : prefill_len + 1].clone()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = batch_invariant_decode_buffered_scan(
                bufs,
                static_x,
                static_dt,
                static_B,
                static_C,
                self.A,
                self.D,
                self.dt_bias,
                batch_indices,
                ssm_state,
            )

        # Capture executes once, so restore the replay cursor before the first replay.
        cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
        bufs.seed(
            x[0, :prefill_len],
            dt[0, :prefill_len],
            B[0, :prefill_len],
            C[0, :prefill_len],
            cu,
            batch_indices,
        )
        graph.replay()
        self._assert_bitwise(
            graph_output[0, 0], y_full[0, prefill_len], "CUDA graph replay step 0"
        )

        static_x.copy_(x[:, prefill_len + 1 : prefill_len + 2])
        static_dt.copy_(dt[:, prefill_len + 1 : prefill_len + 2])
        static_B.copy_(B[:, prefill_len + 1 : prefill_len + 2])
        static_C.copy_(C[:, prefill_len + 1 : prefill_len + 2])
        graph.replay()
        self._assert_bitwise(
            graph_output[0, 0], y_full[0, prefill_len + 1], "CUDA graph replay step 1"
        )

    def test_crossing_with_dominant_carried_state(self):
        """Repeated boundary crossings where the carried state dominates the output
        (weak decay: A ~ -0.01 → exp(dA_cs) ≈ 1). Guards the pipeline
        ordering and FP32 state-passing carry. With strong decay, either
        corruption can round away in BF16 and hide."""
        max_batch, slot = 2, 0
        prefill_len = 20
        # Cross twice: the second transition detects an accidental BF16
        # store/reload of state passing's FP32 carry.
        n_decode = 2 * self.chunk_size + 5
        total = prefill_len + n_decode
        x, dt, B, C = self._make_seq(total)

        weak_A = self.A * 0.01
        y_full, _ = mamba_chunk_scan_combined(
            x, dt, weak_A, B, C, self.chunk_size,
            D=self.D, z=None, dt_bias=self.dt_bias, dt_softplus=True,
            initial_states=None, return_final_states=True,
        )

        bufs = self._make_bufs(max_batch)
        ssm_state = self._make_ssm_state(max_batch)
        cu = torch.tensor([0, prefill_len], dtype=torch.int32, device=self.device)
        batch_indices = torch.tensor([slot], dtype=torch.int32, device=self.device)
        bufs.seed(
            x[0, :prefill_len], dt[0, :prefill_len],
            B[0, :prefill_len], C[0, :prefill_len], cu, batch_indices,
        )
        for k in range(n_decode):
            pos = prefill_len + k
            y_batch_invariant = batch_invariant_decode_buffered_scan(
                bufs,
                x[:, pos : pos + 1], dt[:, pos : pos + 1],
                B[:, pos : pos + 1], C[:, pos : pos + 1],
                weak_A, self.D, self.dt_bias,
                batch_indices, ssm_state,
            )
            self._assert_bitwise(
                y_batch_invariant[0, 0], y_full[0, pos], f"weak-decay step k={k}"
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
