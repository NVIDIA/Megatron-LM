# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Runtime correctness gates for canonical factorized GDP (requires CUDA)."""

import unittest

import torch

from megatron.core.ssm.ops.gdp.batch_invariant import (
    GDPCache,
    _run,
    append,
    append_packed,
    prefill,
    train,
)
from tests.unit_tests.ssm.gdp_batch_invariant_reference import oracle


def inputs(b=2, t=23, h=24, d=128, w=64, r=3):
    torch.manual_seed(17)

    def normalized(shape):
        return torch.nn.functional.normalize(torch.randn(shape, device="cuda"), dim=-1).bfloat16()

    return (
        normalized((b, t, h, d)),
        normalized((b, t, r, h, d)),
        torch.randn(b, t, r, h, w, device="cuda", dtype=torch.bfloat16),
        -torch.rand(b, t, h, device="cuda") * 0.1,
        torch.rand(b, t, r, h, device="cuda"),
    )


def exact(test, a, b):
    test.assertEqual(a.dtype, b.dtype)
    test.assertEqual(a.shape, b.shape)
    test.assertTrue(
        torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)),
        f"max difference {(a.float() - b.float()).abs().max().item()}",
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestGDPBatchInvariantRecurrence(unittest.TestCase):
    def setUp(self):
        self.previous_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False

    def tearDown(self):
        torch.backends.cuda.matmul.allow_tf32 = self.previous_tf32

    def test_long_uneven_chunks(self):
        # Short comparisons miss rare BF16 rounding ties in the triangular
        # factors and right-hand sides; recurrence amplifies these differences.
        for c in (16, 32):
            xs = inputs(8, 4096)
            initial = torch.randn(8, 24, 128, 64, device="cuda") * 0.1
            expected, reference_cache = prefill(*xs, block_size=c, initial_state=initial)
            cache = GDPCache.allocate(8, 24, 128, 64, block_size=c)
            cache.state.copy_(initial)
            parts = []
            start = 0
            for width in (1, 5, 17, 7, 128, 29, 200, 3709):
                parts.append(append(*(x[:, start : start + width].contiguous() for x in xs), cache))
                start += width
            exact(self, torch.cat(parts, 1), expected)
            for actual, reference in zip(cache.tensors(), reference_cache.tensors()):
                exact(self, actual, reference)

    def test_partitions_training_and_batch(self):
        for c in (16, 32):
            xs = inputs()
            initial = torch.randn(2, 24, 128, 64, device="cuda") * 0.1
            out, cache = prefill(*xs, block_size=c, initial_state=initial)
            trained, state = train(*xs, block_size=c, initial_state=initial)
            exact(self, out, trained)
            exact(self, state, cache.state)
            for widths in ([1] * 23, [5, 1, 4, 1, 5, 7], [7, 11, 5]):
                target = GDPCache.allocate(2, 24, 128, 64, block_size=c)
                target.state.copy_(initial)
                outputs = []
                start = 0
                for width in widths:
                    outputs.append(
                        append(*(x[:, start : start + width].contiguous() for x in xs), target)
                    )
                    start += width
                exact(self, out, torch.cat(outputs, 1))
                for x, y in zip(cache.tensors(), target.tensors()):
                    exact(self, x, y)
            # Permute requests and run alone; compare bytes, including persistent state.
            for batch in (torch.tensor([1, 0], device="cuda"), torch.tensor([1], device="cuda")):
                perm, pc = prefill(
                    *(x[batch].contiguous() for x in xs), block_size=c, initial_state=initial[batch]
                )
                exact(self, perm, out[batch])
                for x, y in zip(cache.tensors(), pc.tensors()):
                    exact(self, x[batch], y)

    def test_long_prefix_across_launch_sizes(self):
        # Cross the small/large-batch backward launch threshold, and cover
        # enough state commits for a one-ULP factor difference to propagate.
        for c in (16, 32):
            xs = inputs(8, 513)
            initial = torch.randn(8, 24, 128, 64, device="cuda") * 0.1
            expected, cache = prefill(*xs, block_size=c, initial_state=initial)
            actual, state = train(*xs, block_size=c, initial_state=initial)
            exact(self, actual, expected)
            exact(self, state, cache.state)
            alone, alone_state = train(
                *(x[:1].contiguous() for x in xs), block_size=c, initial_state=initial[:1]
            )
            exact(self, alone, expected[:1])
            exact(self, alone_state, cache.state[:1])

    def test_boundaries_and_single_update(self):
        for c in (16, 32):
            for r in (1, 3):
                for t in (1, 5, 6, 10, 11, 16, 17, 32, 33):
                    xs = inputs(1, t, 2, 32, 32, r)
                    expected, cache = prefill(*xs, block_size=c)
                    target = GDPCache.allocate(1, 2, 32, 32, block_size=c)
                    result = torch.cat(
                        [
                            append(*(x[:, j : j + 1].contiguous() for x in xs), target)
                            for j in range(t)
                        ],
                        1,
                    )
                    exact(self, expected, result)
                    for a, b in zip(cache.tensors(), target.tensors()):
                        exact(self, a, b)

    def test_slots_reset_and_graph(self):
        xs = inputs(3, 1, 2, 32, 32)
        cache = GDPCache.allocate(4, 2, 32, 32)
        slots = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
        # Warm compiler and allocator outside capture.
        append(*xs, cache, slots=slots)
        cache.reset(torch.arange(4, device="cuda"))
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = append(*xs, cache, slots=slots)
        cache.reset(torch.arange(4, device="cuda"))
        graph.replay()
        torch.cuda.synchronize()
        expected, ec = prefill(*(x[[0, 2]].contiguous() for x in xs))
        exact(self, result[[0, 2]], expected)
        exact(self, result[1], torch.zeros_like(result[1]))
        for a, b in zip(cache.tensors(), ec.tensors()):
            exact(self, a[[2, 0]], b)
        cache.reset(torch.tensor([2], device="cuda"))
        for a in cache.tensors():
            exact(self, a[2], torch.zeros_like(a[2]))
        graph.replay()
        torch.cuda.synchronize()
        exact(self, result[0], expected[0])

    def test_backward(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        for b, h, c, t in (
            (1, 2, 16, 3),
            (1, 2, 16, 13),
            (1, 2, 16, 16),
            (1, 2, 32, 23),
            (8, 24, 16, 23),
            (8, 24, 32, 23),
        ):
            xs = tuple(x.requires_grad_() for x in inputs(b, t, h, 128, 64))
            self._check_backward_oracle(xs, c)

    def _check_backward_oracle(self, xs, c):
        b, t, h, d = xs[0].shape
        w = xs[2].shape[-1]
        initial = (torch.randn(b, h, d, w, device="cuda") * 0.1).requires_grad_()
        out, state = train(*xs, initial_state=initial, block_size=c)
        do = torch.randn_like(out)
        ds = torch.randn_like(state)
        actual = torch.autograd.grad((out, state), (*xs, initial), (do, ds))
        cache = GDPCache.allocate(b, h, d, w, block_size=c)
        cache.state.copy_(initial.detach())
        with torch.no_grad():
            _, tape = _run(
                *xs,
                cache,
                torch.arange(b, device="cuda", dtype=torch.int32),
                scale=d**-0.5,
                tape=True,
                prepared=True,
            )
        ox = tuple(x.detach().float().requires_grad_() for x in xs)
        si = initial.detach().clone().requires_grad_()
        oo, ss = oracle(*ox, si, c=c, tape=tape)
        expected = torch.autograd.grad((oo, ss), (*ox, si), (do.float(), ds))
        for name, a, e in zip(("q", "k", "v", "g", "beta", "initial"), actual, expected):
            e = e.to(a.dtype).float()
            a = a.float()
            self.assertTrue(torch.isfinite(a).all(), name)
            relative = (a - e).norm() / e.norm().clamp_min(1e-12)
            self.assertLess(relative.item(), 3e-3, (c, t, name, relative.item()))
            torch.testing.assert_close(a, e, rtol=0.02, atol=3e-4, msg=lambda m: name + " " + m)

    def test_backward_multiple_value_tiles(self):
        for c in (16, 32):
            xs = tuple(x.requires_grad_() for x in inputs(1, 23, 2, 128, 128))
            self._check_backward_oracle(xs, c)

    def test_backward_correlated_keys(self):
        # Powers of the full triangular matrix suffer severe cancellation for
        # correlated keys. Blockwise solves must pass the same oracle tolerance.
        torch.backends.cuda.matmul.allow_tf32 = False
        for c in (16, 32):
            for r in (1, 2, 3):
                for correlation in (0.5, 0.9, 1.0):
                    for beta_value in (0.5, 1.0):
                        with self.subTest(c=c, r=r, correlation=correlation, beta=beta_value):
                            xs = list(inputs(1, 23, 2, 128, 64, r))
                            torch.manual_seed(43)
                            common = torch.nn.functional.normalize(
                                torch.randn(1, 1, 1, 2, 128, device="cuda"), dim=-1
                            )
                            xs[1] = torch.nn.functional.normalize(
                                xs[1].float() * (1 - correlation) + common * correlation, dim=-1
                            ).bfloat16()
                            xs[3].zero_()
                            xs[4].fill_(beta_value)
                            self._check_backward_oracle(tuple(x.requires_grad_() for x in xs), c)

    def test_accuracy_and_large_decay(self):
        for c in (16, 32):
            xs = inputs(1, 129, 2, 128, 64)
            q, k, v, g, beta = (x.double() for x in xs)
            state = torch.zeros(1, 2, 128, 64, device="cuda", dtype=torch.float64)
            outputs = []
            for t in range(q.shape[1]):
                state = state * g[:, t].exp()[..., None, None]
                for r in range(3):
                    key = k[:, t, r]
                    residual = (v[:, t, r] - (state * key[..., None]).sum(-2)) * beta[
                        :, t, r, ..., None
                    ]
                    state = state + key[..., None] * residual[..., None, :]
                outputs.append((state * q[:, t, ..., None]).sum(-2) * 128**-0.5)
            reference = torch.stack(outputs, 1)
            out, _ = prefill(*xs, block_size=c)
            relative = (out.double() - reference).norm() / reference.norm()
            self.assertLess(relative.item(), 0.01)
            # Large negative gates must not overflow masked future decay ratios.
            ys = list(inputs(1, 23, 2, 32, 32))
            ys[3].fill_(-1000)
            ys = tuple(x.requires_grad_() for x in ys)
            out, state = train(*ys, block_size=c)
            grads = torch.autograd.grad(out.sum() + state.sum(), ys)
            for x in (out, state, *grads):
                self.assertTrue(torch.isfinite(x).all())

    def test_rollback_and_heterogeneous_cursors(self):
        xs = inputs(2, 37, 2, 32, 32)
        cache = GDPCache.allocate(3, 2, 32, 32)
        slots = torch.tensor([2, 0], device="cuda", dtype=torch.int32)
        append(*(x[:, :5].contiguous() for x in xs), cache, slots=slots)
        saved = cache.snapshot(slots.long())
        a = append(*(x[:, 5:19].contiguous() for x in xs), cache, slots=slots)
        cache.restore(slots.long(), saved)
        b = append(*(x[:, 5:19].contiguous() for x in xs), cache, slots=slots)
        exact(self, a, b)
        # Advance one request independently, then put unequal cursors in one batch.
        append(*(x[:1, 19:20].contiguous() for x in xs), cache, slots=slots[:1])
        mixed = tuple(torch.cat([x[:1, 20:21], x[1:, 19:20]], 0).contiguous() for x in xs)
        output = append(*mixed, cache, slots=slots)
        for lane, end in ((0, 21), (1, 20)):
            expected, ec = prefill(*(x[lane : lane + 1, :end].contiguous() for x in xs))
            exact(self, output[lane : lane + 1], expected[:, -1:])
            for x, y in zip(cache.tensors(), ec.tensors()):
                exact(self, x[slots[lane].long()].unsqueeze(0), y)

    def test_packed_append_and_graph_metadata(self):
        xs = inputs(1, 32, 2, 32, 32)
        slots = torch.tensor([2, -1, 0], device="cuda", dtype=torch.int32)
        cu = torch.tensor([0, 5, 5, 32], device="cuda", dtype=torch.int32)
        cache = GDPCache.allocate(3, 2, 32, 32)
        append_packed(*xs, cache, cu, slots)
        cache.reset(torch.arange(3, device="cuda"))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = append_packed(*xs, cache, cu, slots)
        for offsets, active in (([0, 5, 5, 32], [2, -1, 0]), ([0, 11, 19, 32], [2, -1, 0])):
            cache.reset(torch.arange(3, device="cuda"))
            cu.copy_(torch.tensor(offsets, device="cuda", dtype=torch.int32))
            graph.replay()
            torch.cuda.synchronize()
            for i, slot in enumerate(active):
                a, b = offsets[i : i + 2]
                if a == b:
                    continue
                if slot < 0:
                    exact(self, output[:, a:b], torch.zeros_like(output[:, a:b]))
                    continue
                expected, ec = prefill(*(x[:, a:b].contiguous() for x in xs))
                exact(self, output[:, a:b], expected)
                for x, y in zip(cache.tensors(), ec.tensors()):
                    exact(self, x[slot : slot + 1], y)
        # Resume unequal-length prefixes in the same packed launch.
        continued = append_packed(*xs, cache, cu, slots)
        for i, slot in ((0, 2), (2, 0)):
            a, b = offsets[i : i + 2]
            repeated = tuple(torch.cat([x[:, a:b], x[:, a:b]], 1).contiguous() for x in xs)
            expected, ec = prefill(*repeated)
            exact(self, continued[:, a:b], expected[:, b - a :])
            for x, y in zip(cache.tensors(), ec.tensors()):
                exact(self, x[slot : slot + 1], y)
