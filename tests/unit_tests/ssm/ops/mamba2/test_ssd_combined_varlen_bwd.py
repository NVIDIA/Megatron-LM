# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import unittest

import torch
import torch.nn.functional as F

try:
    from megatron.core.ssm.ops.mamba2.ssd_combined import mamba_chunk_scan_combined_varlen_fn

    HAVE_SSD_OPS = True
except (ImportError, Exception):
    HAVE_SSD_OPS = False


def _chunk_metadata(lens, chunk_size, device):
    """Varlen chunk metadata: chunks restart at every sequence start."""
    boundaries, last_chunk, chunk_seq = [0], [], []
    for i, n in enumerate(lens):
        start, end = boundaries[-1], boundaries[-1] + n
        pos = start + chunk_size
        while pos < end:
            boundaries.append(pos)
            chunk_seq.append(i)
            pos += chunk_size
        boundaries.append(end)
        chunk_seq.append(i)
        last_chunk.append(len(boundaries) - 2)
    as_int32 = lambda v: torch.tensor(v, dtype=torch.int32, device=device)
    return as_int32(boundaries), as_int32(last_chunk), as_int32(chunk_seq)


def _reference(x, dt, A, B, C, D, z, dt_bias, lens):
    """fp64 token-by-token scan with a state reset at every sequence start."""
    nheads, ngroups = x.shape[1], B.shape[1]
    d = F.softplus(dt + dt_bias)
    Bh = B.repeat_interleave(nheads // ngroups, 1)
    Ch = C.repeat_interleave(nheads // ngroups, 1)
    D_term = D if D.dim() == 2 else D[:, None]
    ys, start = [], 0
    for n in lens:
        h = x.new_zeros(nheads, x.shape[2], B.shape[2])
        for t in range(start, start + n):
            h = h * torch.exp(d[t] * A)[:, None, None]
            h = h + (d[t][:, None, None] * x[t][:, :, None]) * Bh[t][:, None, :]
            ys.append((h * Ch[t][:, None, :]).sum(-1) + D_term * x[t])
        start += n
    y = torch.stack(ys)
    return y if z is None else y * F.silu(z)


@unittest.skipIf(not HAVE_SSD_OPS, "SSD ops (Triton 3+) not available")
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required for SSD ops")
class TestMambaChunkScanCombinedVarlenFn(unittest.TestCase):
    """Forward and backward of the differentiable varlen SSD against an fp64 reference."""

    def setUp(self):
        self.device = torch.device("cuda")
        self.nheads, self.headdim, self.ngroups, self.dstate = 4, 16, 2, 16

    def _inputs(self, seqlen, dtype, d_has_hdim, with_z, seed):
        g = torch.Generator(device=self.device).manual_seed(seed)
        H, P, G, N = self.nheads, self.headdim, self.ngroups, self.dstate
        rand = lambda *s, scale=1.0: torch.randn(*s, device=self.device, generator=g) * scale
        x = rand(seqlen, H, P).to(dtype)
        dt = rand(seqlen, H).to(dtype)
        B = rand(seqlen, G, N, scale=0.3).to(dtype)
        C = rand(seqlen, G, N, scale=0.3).to(dtype)
        z = rand(seqlen, H, P).to(dtype) if with_z else None
        A = -torch.rand(H, device=self.device, generator=g) * 2 - 0.1
        D = rand(H, P) if d_has_hdim else rand(H)
        dt_bias = rand(H, scale=0.5)
        return [x, dt, A, B, C, D, z, dt_bias]

    def _run(self, inputs, lens, chunk_size, dout, state_dtype=None):
        x, dt, A, B, C, D, z, dt_bias = [
            None if t is None else t.detach().clone().requires_grad_() for t in inputs
        ]
        cu_chunk_seqlens, last_chunk, chunk_seq = _chunk_metadata(lens, chunk_size, self.device)
        out = mamba_chunk_scan_combined_varlen_fn(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            cu_chunk_seqlens,
            last_chunk,
            chunk_seq,
            D=D,
            z=z,
            dt_bias=dt_bias,
            dt_softplus=True,
            state_dtype=state_dtype,
        )
        out.backward(dout)
        leaves = [x, dt, A, B, C, D, z, dt_bias]
        return out.detach(), [None if t is None else t.grad for t in leaves]

    def _check_against_reference(
        self, chunk_size, lens, dtype, d_has_hdim, with_z, tol, state_dtype=None
    ):
        inputs = self._inputs(sum(lens), dtype, d_has_hdim, with_z, seed=chunk_size + len(lens))
        g = torch.Generator(device=self.device).manual_seed(1)
        dout = torch.randn(
            sum(lens), self.nheads, self.headdim, device=self.device, generator=g
        ).to(dtype)
        out, grads = self._run(inputs, lens, chunk_size, dout, state_dtype)

        ref_leaves = [None if t is None else t.detach().double().requires_grad_() for t in inputs]
        ref_out = _reference(*ref_leaves, lens)
        ref_out.backward(dout.double())
        names = ["out", "dx", "ddt", "dA", "dB", "dC", "dD", "dz", "ddt_bias"]
        ref_grads = [None if t is None else t.grad for t in ref_leaves]
        for name, got, ref in zip(names, [out] + grads, [ref_out.detach()] + ref_grads):
            if ref is None:
                self.assertIsNone(got, name)
                continue
            if ref.norm() == 0:
                # e.g. dA of single-token sequences: no state is carried to a next token.
                self.assertEqual(got.abs().max().item(), 0, name)
                continue
            rel = ((got.double() - ref).norm() / ref.norm()).item()
            self.assertLess(rel, tol, f"{name}: relative error {rel:.2e}")

    def test_matches_reference(self):
        cases = [
            # chunk_size, sequence lengths (single tokens, exact chunk multiples and +-1 around them)
            (16, [1, 15, 16, 17, 40, 5, 33]),
            (128, [1, 127, 128, 129, 300]),
            # A single chunk: Triton specializes nchunks == 1 to a constant.
            (128, [100]),
            (128, [128]),
            (16, [1]),
        ]
        for chunk_size, lens in cases:
            for d_has_hdim, with_z in ((False, True), (True, True), (False, False)):
                with self.subTest(chunk_size=chunk_size, d_has_hdim=d_has_hdim, with_z=with_z):
                    self._check_against_reference(
                        chunk_size, lens, torch.bfloat16, d_has_hdim, with_z, tol=2e-2
                    )

    def test_non_power_of_two_headdim(self):
        # Pruned models (e.g. Minitron) use head dims that are multiples of 8, not powers of 2.
        self.headdim = 56
        self._check_against_reference(
            128, [1, 127, 128, 129, 300], torch.bfloat16, False, True, tol=2e-2
        )

    def test_fp32_states(self):
        """state_dtype keeps the inter-chunk states and their gradients in fp32."""
        for with_z in (True, False):
            with self.subTest(with_z=with_z):
                self._check_against_reference(
                    128,
                    [1, 127, 128, 129, 300],
                    torch.bfloat16,
                    False,
                    with_z,
                    tol=2e-2,
                    state_dtype=torch.float32,
                )

    def test_packing_invariance(self):
        """A sequence's output and its x/dt/z gradients do not depend on its packed neighbours."""
        chunk_size, target_len = 16, 50
        target = self._inputs(target_len, torch.bfloat16, False, True, seed=7)
        g = torch.Generator(device=self.device).manual_seed(8)
        dout_target = torch.randn(
            target_len, self.nheads, self.headdim, device=self.device, generator=g
        ).to(torch.bfloat16)

        def pack(neighbour_lens_before, neighbour_lens_after, seed):
            before = self._inputs(sum(neighbour_lens_before), torch.bfloat16, False, True, seed)
            after = self._inputs(sum(neighbour_lens_after), torch.bfloat16, False, True, seed + 1)
            per_token = lambda i: torch.cat([before[i], target[i], after[i]])
            inputs = [per_token(i) if i in (0, 1, 3, 4, 6) else target[i] for i in range(8)]
            lens = neighbour_lens_before + [target_len] + neighbour_lens_after
            start = sum(neighbour_lens_before)
            dout = torch.zeros(
                sum(lens), self.nheads, self.headdim, device=self.device, dtype=torch.bfloat16
            )
            dout[start : start + target_len] = dout_target
            out, grads = self._run(inputs, lens, chunk_size, dout)
            sl = slice(start, start + target_len)
            return out[sl], [grads[0][sl], grads[1][sl], grads[6][sl]]

        out_a, grads_a = pack([7, 30], [9], seed=10)
        out_b, grads_b = pack([45, 3, 12], [20], seed=20)
        self.assertTrue(torch.equal(out_a, out_b))
        for name, a, b in zip(["dx", "ddt", "dz"], grads_a, grads_b):
            self.assertTrue(torch.equal(a, b), name)


if __name__ == "__main__":
    unittest.main()
