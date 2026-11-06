# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for speculative (rejection) sampling verification of MTP drafts.

Covers the chain speculative sampling primitive (torch reference and FlashInfer), and the
controller glue (`MTPControllerMixin._sample_mtp_draft` / `_mtp_rejection_plan` /
`_sample_speculative_verify_rows` / `_apply_mtp_rejection_sampling`) driven
through the real exact-match verifier and `prepare_next_forward_pass` kernels.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core.inference.disaggregation.mtp_draft_handoff import (
    decode_mtp_draft_q,
    encode_mtp_draft_q,
)
from megatron.core.inference.sampling.speculative import chain_speculative_sampling_torch
from megatron.core.inference.sampling.torch_sampling import TorchSampling
from megatron.core.inference.text_generation_controllers.mtp_controller_mixin import (
    MTPControllerMixin,
)
from megatron.core.inference.text_generation_controllers.mtp_utils_triton import (
    prepare_next_forward_pass,
    verify_speculative_tokens,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)

try:
    import flashinfer

    HAVE_FLASHINFER_CHAIN = hasattr(flashinfer.sampling, "chain_speculative_sampling")
except ImportError:
    HAVE_FLASHINFER_CHAIN = False

DEVICE = "cuda"


def _tv(a: torch.Tensor, b: torch.Tensor) -> float:
    return 0.5 * (a.double() - b.double()).abs().sum().item()


def _hist(tokens: torch.Tensor, vocab: int) -> torch.Tensor:
    return torch.bincount(tokens.long(), minlength=vocab).double() / max(tokens.numel(), 1)


def _tv_tol(vocab: int, n: int) -> float:
    # E[TV] of an n-sample empirical histogram is <= ~0.4 sqrt(V / n); allow ~4x that.
    return 1.6 * math.sqrt(vocab / max(n, 1))


def _random_dist(num: int, vocab: int, scale: float, gen: torch.Generator) -> torch.Tensor:
    logits = torch.randn(num, vocab, generator=gen, device=DEVICE) * scale
    return torch.softmax(logits, dim=-1)


def _chain_impls():
    impls = [("torch", lambda d, x, t, g: chain_speculative_sampling_torch(d, x, t, generator=g))]
    if HAVE_FLASHINFER_CHAIN:

        def _fi(d, x, t, g):
            out, _, emitted = flashinfer.sampling.chain_speculative_sampling(
                d.float().contiguous(),
                x.to(torch.int32).contiguous(),
                t.float().contiguous(),
                deterministic=True,
                generator=g,
            )
            return out.long(), emitted.long()

        impls.append(("flashinfer", _fi))
    return impls


CHAIN_IMPLS = _chain_impls()


def _emitted_token_stats(out: torch.Tensor, num_acc: torch.Tensor, vocab: int, depth: int):
    """Histogram of the token emitted at chain position `j`, over rows that reach position j."""
    reached = num_acc >= depth
    return _hist(out[reached, depth], vocab), int(reached.sum().item())


class TestChainSpeculativeSampling:
    """The chain sampler emits target-distributed tokens with sum_x min(p, q) acceptance."""

    VOCAB = 8
    TRIALS = 200_000

    def _setup(self, num_drafts, q_mode, seed):
        gen = torch.Generator(device=DEVICE).manual_seed(seed)
        p = _random_dist(num_drafts + 1, self.VOCAB, 1.5, gen)  # [D+1, V]
        if q_mode == "close":
            q = torch.softmax(
                torch.log(p[:num_drafts]) + 0.3 * torch.randn_like(p[:num_drafts]), -1
            )
        elif q_mode == "different":
            # q concentrated where p is small.
            q = torch.softmax(-3.0 * torch.log(p[:num_drafts]), dim=-1)
        else:  # "disjoint-ish": q puts almost all mass on p's least likely token.
            q = torch.full_like(p[:num_drafts], 1e-4)
            q[torch.arange(num_drafts), p[:num_drafts].argmin(-1)] = 1.0
            q = q / q.sum(-1, keepdim=True)
        draft_probs = q.unsqueeze(0).expand(self.TRIALS, -1, -1).contiguous()
        target_probs = p.unsqueeze(0).expand(self.TRIALS, -1, -1).contiguous()
        drafts = torch.multinomial(
            q.repeat(self.TRIALS, 1), 1, replacement=True, generator=gen
        ).view(self.TRIALS, num_drafts)
        return p, q, draft_probs, drafts, target_probs, gen

    @pytest.mark.parametrize("impl_name,impl", CHAIN_IMPLS)
    @pytest.mark.parametrize("num_drafts", [1, 2, 3])
    @pytest.mark.parametrize("q_mode", ["close", "different", "disjoint"])
    def test_emitted_distribution_matches_target(self, impl_name, impl, num_drafts, q_mode):
        p, q, draft_probs, drafts, target_probs, gen = self._setup(num_drafts, q_mode, seed=1)
        out, num_acc = impl(draft_probs, drafts, target_probs, gen)

        assert out.shape == (self.TRIALS, num_drafts + 1)
        assert ((num_acc >= 0) & (num_acc <= num_drafts)).all()
        # Contract: accepted drafts, then a real token at num_acc, then -1 padding.
        pos = torch.arange(num_drafts + 1, device=DEVICE).unsqueeze(0)
        padded = torch.cat([drafts, torch.full_like(drafts[:, :1], -1)], dim=1)
        acc_rows = pos < num_acc.unsqueeze(1)
        assert torch.equal(out[acc_rows], padded[acc_rows])
        emitted = out.gather(1, num_acc.unsqueeze(1)).squeeze(1)
        assert ((emitted >= 0) & (emitted < self.VOCAB)).all()
        assert (out[pos > num_acc.unsqueeze(1)] == -1).all()

        # Conditional on reaching chain position j, the emitted token is distributed as p_{j+1}.
        for depth in range(num_drafts + 1):
            hist, n = _emitted_token_stats(out, num_acc, self.VOCAB, depth)
            if n < 2000:
                continue
            assert _tv(hist, p[depth]) < _tv_tol(self.VOCAB, n), (impl_name, depth, n)

    @pytest.mark.parametrize("impl_name,impl", CHAIN_IMPLS)
    @pytest.mark.parametrize("q_mode", ["close", "different", "disjoint"])
    def test_single_draft_acceptance_matches_analytic(self, impl_name, impl, q_mode):
        p, q, draft_probs, drafts, target_probs, gen = self._setup(1, q_mode, seed=2)
        _, num_acc = impl(draft_probs, drafts, target_probs, gen)
        rate = num_acc.double().mean().item()
        analytic = torch.minimum(p[0], q[0]).sum().item()
        exact_match = (p[0] * q[0]).sum().item()
        sigma = math.sqrt(max(analytic * (1 - analytic), 1e-6) / self.TRIALS)
        assert abs(rate - analytic) < 5 * sigma + 1e-4, (impl_name, rate, analytic)
        assert analytic >= exact_match - 1e-9
        assert rate >= exact_match - 5 * sigma

    @pytest.mark.parametrize("impl_name,impl", CHAIN_IMPLS)
    def test_chain_acceptance_matches_analytic(self, impl_name, impl):
        # Position-independent p/q: P(num_acc >= k) = prod_{i<=k} sum_x min(p_i, q_i).
        p, q, draft_probs, drafts, target_probs, gen = self._setup(3, "close", seed=3)
        _, num_acc = impl(draft_probs, drafts, target_probs, gen)
        alpha = torch.minimum(p[:3], q).sum(-1)
        for k in range(1, 4):
            expected = alpha[:k].prod().item()
            rate = (num_acc >= k).double().mean().item()
            sigma = math.sqrt(expected * (1 - expected) / self.TRIALS)
            assert abs(rate - expected) < 5 * sigma + 1e-4, (impl_name, k, rate, expected)

    @pytest.mark.parametrize("impl_name,impl", CHAIN_IMPLS)
    def test_one_hot_matches_exact_match(self, impl_name, impl):
        # Greedy p and q (one-hot): accept iff the draft is the target argmax, else emit argmax p.
        num_drafts, vocab, batch = 3, 16, 64
        gen = torch.Generator(device=DEVICE).manual_seed(4)
        target_arg = torch.randint(0, vocab, (batch, num_drafts + 1), device=DEVICE, generator=gen)
        drafts = torch.where(
            torch.rand(batch, num_drafts, device=DEVICE, generator=gen) < 0.7,
            target_arg[:, :num_drafts],
            torch.randint(0, vocab, (batch, num_drafts), device=DEVICE, generator=gen),
        )
        target_probs = torch.nn.functional.one_hot(target_arg, vocab).float()
        draft_probs = torch.nn.functional.one_hot(drafts, vocab).float()
        out, num_acc = impl(draft_probs, drafts, target_probs, gen)

        match = (drafts == target_arg[:, :num_drafts]).long().cumprod(1)
        expected_acc = match.sum(1)
        assert torch.equal(num_acc, expected_acc)
        emitted = out.gather(1, num_acc.unsqueeze(1)).squeeze(1)
        assert torch.equal(emitted, target_arg.gather(1, expected_acc.unsqueeze(1)).squeeze(1))

    @pytest.mark.skipif(not HAVE_FLASHINFER_CHAIN, reason="flashinfer chain kernel unavailable")
    @pytest.mark.parametrize("num_drafts", [1, 3])
    def test_torch_and_flashinfer_agree(self, num_drafts):
        (_, torch_impl), (_, fi_impl) = CHAIN_IMPLS[0], CHAIN_IMPLS[1]
        p, q, draft_probs, drafts, target_probs, gen = self._setup(num_drafts, "different", 5)
        out_t, acc_t = torch_impl(draft_probs, drafts, target_probs, gen)
        out_f, acc_f = fi_impl(draft_probs, drafts, target_probs, gen)
        n = self.TRIALS
        assert _tv(_hist(acc_t, num_drafts + 1), _hist(acc_f, num_drafts + 1)) < 2 * _tv_tol(
            num_drafts + 1, n
        )
        for depth in range(num_drafts + 1):
            h_t, n_t = _emitted_token_stats(out_t, acc_t, self.VOCAB, depth)
            h_f, n_f = _emitted_token_stats(out_f, acc_f, self.VOCAB, depth)
            if min(n_t, n_f) < 2000:
                continue
            assert _tv(h_t, h_f) < 2 * _tv_tol(self.VOCAB, min(n_t, n_f))


# ---------------------------------------------------------------------------
# Controller glue
# ---------------------------------------------------------------------------


class _Ctx:
    """Minimal DynamicInferenceContext stand-in for the sampling / rejection paths."""

    def __init__(self, temperature, top_k, top_p, request_id_base=100):
        n = len(temperature)
        self.total_request_count = n
        self.paused_request_count = 0
        self.request_ids = torch.arange(request_id_base, request_id_base + n, dtype=torch.int32)
        self.active_request_metadata = {
            "temperature": torch.tensor(temperature, dtype=torch.float32),
            "top_k": torch.tensor(top_k, dtype=torch.int32),
            "top_p": torch.tensor(top_p, dtype=torch.float32),
        }
        self.gpu_view = SimpleNamespace(
            **{k: v.to(DEVICE) for k, v in self.active_request_metadata.items()}
        )

    def active_sampling_filter_flags(self, active_request_count=None):
        n = self.total_request_count if active_request_count is None else active_request_count
        top_k = self.active_request_metadata["top_k"][:n]
        top_p = self.active_request_metadata["top_p"][:n]
        return bool((top_k <= 0).all()), bool(((top_p <= 0) | (top_p >= 1)).all())


class _Controller(MTPControllerMixin):
    _sample_from_logits_2d = TextGenerationController._sample_from_logits_2d

    def __init__(self, ctx, sampling, num_spec, vocab, mode):
        self.inference_wrapped_model = SimpleNamespace(inference_context=ctx)
        self.num_speculative_tokens = num_spec
        self.num_mtp_depths = num_spec
        self.mtp_verification = mode
        self._sampling = sampling
        n = ctx.total_request_count
        self._sampled_mtp_tokens_cuda = torch.empty(num_spec, n, dtype=torch.long, device=DEVICE)
        self._mtp_draft_probs = (
            torch.empty(num_spec, n, vocab, dtype=torch.float32, device=DEVICE)
            if mode == "rejection"
            else None
        )
        self._mtp_draft_row_of_request = None
        self._imported_mtp_draft_q = {}


def _make_sampling(backend, seed):
    rng = torch.Generator(device=DEVICE).manual_seed(seed)
    if backend == "torch":
        return TorchSampling(rng, VOCAB)
    from megatron.core.inference.sampling.flashinfer_sampling import FlashInferSampling

    return FlashInferSampling(VOCAB, rng)


VOCAB = 16
BACKENDS = ["torch"] + (["flashinfer"] if HAVE_FLASHINFER_CHAIN else [])


def _reference_probs(logits, temperature, top_k, top_p, backend="flashinfer"):
    # TorchSampling temperature-scales in the logits dtype; FlashInfer scales in fp32.
    logits = logits if backend == "torch" else logits.float()
    filtered = TorchSampling.filter_logits(logits, temperature, top_k, top_p)
    return torch.softmax(filtered, dim=-1, dtype=torch.float32)


def _run_step(ctrl, ctx, draft_logits, target_logits, base_tokens, num_prefill=0):
    """Draft every depth, then verify one step. Returns (emitted lists, accepted counts, ...)."""
    num_spec = ctrl.num_speculative_tokens
    n = ctx.total_request_count
    num_decode = n - num_prefill
    for depth in range(num_spec):
        tokens = ctrl._sample_mtp_draft(depth, draft_logits[depth], n)
        ctrl._sampled_mtp_tokens_cuda[depth, :n] = tokens
    ctrl._record_mtp_draft_rows(ctx, n)

    stride = num_spec + 1
    decode_inputs = torch.cat(
        [base_tokens[:num_decode, None], ctrl._sampled_mtp_tokens_cuda[:, :num_decode].T], dim=1
    ).reshape(-1)
    input_tokens = torch.cat([decode_inputs, base_tokens[num_decode:]])
    plan = ctrl._mtp_rejection_plan(num_decode) if ctrl.mtp_verification == "rejection" else None
    output = ctrl._sample_speculative_verify_rows(
        target_logits, None, num_decode, num_prefill, plan
    )
    before = output.clone()
    if plan is not None:
        ctrl._apply_mtp_rejection_sampling(output, target_logits, None, input_tokens, plan)
    last_one, mask, input_tokens = verify_speculative_tokens(
        input_tokens, output, num_decode, num_prefill, num_spec
    )
    sampled = torch.empty(n, dtype=torch.long, device=DEVICE)
    last_seq = torch.empty(n, dtype=torch.long, device=DEVICE)
    accepted_tokens = torch.full((n, num_spec), -1, dtype=torch.long, device=DEVICE)
    accepted_counts = torch.zeros(n, dtype=torch.long, device=DEVICE)
    required_idx = torch.arange(output.numel(), device=DEVICE)
    prepare_next_forward_pass(
        num_decode,
        output,
        required_idx,
        last_one,
        mask,
        input_tokens,
        sampled,
        last_seq,
        accepted_tokens,
        accepted_counts,
        num_spec,
    )
    return SimpleNamespace(
        plan=plan,
        before=before,
        output=output,
        input_tokens=input_tokens,
        last_one=last_one,
        mask=mask,
        sampled=sampled,
        accepted_tokens=accepted_tokens,
        accepted_counts=accepted_counts,
        stride=stride,
    )


def _logits(num_rows, scale, gen, dtype=torch.bfloat16):
    return (torch.randn(num_rows, VOCAB, device=DEVICE, generator=gen) * scale).to(dtype)


class TestControllerRejectionSampling:

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize(
        "params", [(1.0, 0, 0.0), (0.7, 0, 0.9), (1.3, 5, 0.0)], ids=["t1", "t0.7-p0.9", "t1.3-k5"]
    )
    def test_end_to_end_distribution_and_acceptance(self, backend, params):
        """Draft -> rejection verify -> exact-match kernels: emitted tokens follow filtered p."""
        temperature, top_k, top_p = params
        num_spec, n, rounds = 2, 8192, 6
        gen = torch.Generator(device=DEVICE).manual_seed(11)
        # Same p/q for every request; q is deliberately different from p.
        target_row = _logits(num_spec + 1, 2.0, gen)
        draft_row = _logits(num_spec, 2.0, gen)
        p = _reference_probs(target_row, temperature, top_k, top_p)
        q = _reference_probs(draft_row, temperature, top_k, top_p)
        ctx = _Ctx([temperature] * n, [top_k] * n, [top_p] * n)
        ctrl = _Controller(ctx, _make_sampling(backend, 12), num_spec, VOCAB, "rejection")
        target_logits = target_row.repeat(n, 1)
        draft_logits = draft_row.unsqueeze(1).expand(num_spec, n, VOCAB).contiguous()
        base = torch.zeros(n, dtype=torch.long, device=DEVICE)

        first, second, acc = [], [], []
        for _ in range(rounds):
            r = _run_step(ctrl, ctx, draft_logits, target_logits, base)
            a = r.accepted_counts
            drafts = r.accepted_tokens.clamp(min=0)
            emitted0 = torch.where(a >= 1, drafts[:, 0], r.sampled)
            first.append(emitted0)
            second.append(torch.where(a >= 2, drafts[:, 1], r.sampled)[a >= 1])
            acc.append(a)
            # Downstream contract: last_one_indices / mask / counts are consistent.
            rows = torch.arange(n, device=DEVICE)
            assert torch.equal(r.last_one, rows * r.stride + a)
            assert torch.equal(r.mask.view(n, r.stride).sum(1) - 1, a)
            assert torch.equal(r.sampled, r.output[r.last_one])
        first, second, acc = torch.cat(first), torch.cat(second), torch.cat(acc)

        assert _tv(_hist(first, VOCAB), p[0]) < _tv_tol(VOCAB, first.numel())
        assert _tv(_hist(second, VOCAB), p[1]) < _tv_tol(VOCAB, second.numel())
        alpha = torch.minimum(p[0], q[0]).sum().item()
        rate = (acc >= 1).double().mean().item()
        sigma = math.sqrt(alpha * (1 - alpha) / acc.numel())
        assert abs(rate - alpha) < 5 * sigma + 1e-3, (rate, alpha)
        assert rate >= (p[0] * q[0]).sum().item()

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_greedy_rows_identical_to_exact_match(self, backend):
        num_spec, n = 3, 64
        gen = torch.Generator(device=DEVICE).manual_seed(21)
        target_logits = _logits(n * (num_spec + 1), 3.0, gen)
        draft_logits = _logits(num_spec * n, 3.0, gen).view(num_spec, n, VOCAB)
        # Make the drafts often agree with the target argmax.
        draft_logits[:, : n // 2] = target_logits.view(n, num_spec + 1, VOCAB)[
            : n // 2, :num_spec
        ].transpose(0, 1)
        base = torch.randint(0, VOCAB, (n,), device=DEVICE, generator=gen)
        results = {}
        for mode in ("exact_match", "rejection"):
            ctx = _Ctx([1.0] * n, [1] * n, [0.0] * n)
            ctrl = _Controller(ctx, _make_sampling(backend, 22), num_spec, VOCAB, mode)
            results[mode] = _run_step(ctrl, ctx, draft_logits, target_logits, base)
        a, b = results["exact_match"], results["rejection"]
        assert torch.equal(a.output, b.output)
        assert torch.equal(a.accepted_counts, b.accepted_counts)
        assert torch.equal(a.accepted_tokens, b.accepted_tokens)
        assert torch.equal(a.sampled, b.sampled)
        assert a.accepted_counts.max() > 0

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_mixed_batch_with_prefill(self, backend):
        """Greedy + sampled rows with per-row params, plus a trailing prefill row."""
        num_spec = 3
        temps = [1.0, 0.8, 1.0, 1.2, 0.6, 1.0, 1.0]
        top_ks = [1, 0, 0, 4, 0, 1, 0]
        top_ps = [0.0, 0.9, 0.0, 0.0, 0.95, 0.0, 0.0]
        n, num_prefill = len(temps), 1
        num_decode = n - num_prefill
        gen = torch.Generator(device=DEVICE).manual_seed(31)
        target_logits = _logits(num_decode * (num_spec + 1) + num_prefill, 2.0, gen)
        draft_logits = _logits(num_spec * n, 2.0, gen).view(num_spec, n, VOCAB)
        base = torch.randint(0, VOCAB, (n,), device=DEVICE, generator=gen)
        ctx = _Ctx(temps, top_ks, top_ps)
        ctrl = _Controller(ctx, _make_sampling(backend, 32), num_spec, VOCAB, "rejection")

        chain_calls = []
        real_chain = ctrl._sampling.chain_speculative_sampling

        def spy(draft_probs, draft_token_ids, target_probs):
            out = real_chain(draft_probs, draft_token_ids, target_probs)
            chain_calls.append(((draft_probs, draft_token_ids, target_probs), out))
            return out

        ctrl._sampling.chain_speculative_sampling = spy
        r = _run_step(ctrl, ctx, draft_logits, target_logits, base, num_prefill=num_prefill)

        # Greedy decode rows (0, 5) and the prefill row keep the independent target sample
        # (argmax for greedy); only the sampled rows are left to the chain sampler.
        stride = num_spec + 1
        sampled_rows = [row for row in range(num_decode) if top_ks[row] != 1]
        assert r.plan.rows == sampled_rows
        for row in (0, 5):
            sl = slice(row * stride, (row + 1) * stride)
            assert torch.equal(r.output[sl], r.before[sl])
            assert torch.equal(r.output[sl], target_logits[sl].argmax(-1))
        assert torch.equal(r.output[num_decode * stride :], r.before[num_decode * stride :])
        assert (r.before[num_decode * stride :] >= 0).all()
        for row in sampled_rows:
            assert (r.before[row * stride : (row + 1) * stride] == -1).all()

        # Sampled decode rows follow the chain sampler exactly.
        assert len(chain_calls) == 1
        (draft_probs, draft_tokens, target_probs), (chain_tokens, num_acc) = chain_calls[0]
        assert draft_tokens.shape == (len(sampled_rows), num_spec)
        for i, row in enumerate(sampled_rows):
            a = int(num_acc[i])
            assert int(r.accepted_counts[row]) == a
            assert int(r.last_one[row]) == row * stride + a
            assert int(r.sampled[row]) == int(chain_tokens[i, a])
            assert torch.equal(r.accepted_tokens[row, :a], draft_tokens[i, :a])
            assert (r.accepted_tokens[row, a:] == -1).all()
            # Filtered draft / target distributions use each row's own parameters.
            ref_q = _reference_probs(
                draft_logits[:, row], temps[row], top_ks[row], top_ps[row], backend
            )
            torch.testing.assert_close(draft_probs[i], ref_q, atol=1e-5, rtol=1e-4)
            ref_p = _reference_probs(
                target_logits[row * stride : row * stride + num_spec + 1],
                temps[row],
                top_ks[row],
                top_ps[row],
                backend,
            )
            torch.testing.assert_close(target_probs[i], ref_p, atol=1e-5, rtol=1e-4)
            # Drafts lie in the support of their filtered q.
            assert (draft_probs[i].gather(1, draft_tokens[i].unsqueeze(1)) > 0).all()
        # The prefill row: always accepted, sampled token is its own target sample.
        assert int(r.last_one[num_decode]) == num_decode * stride
        assert int(r.accepted_counts[num_decode]) == 0

    def test_stale_or_replaced_drafts_use_exact_match_semantics(self):
        """Drafts not from the last draft loop keep exact match; replaced drafts get a point q."""
        num_spec, n = 2, 4
        gen = torch.Generator(device=DEVICE).manual_seed(41)
        target_logits = _logits(n * (num_spec + 1), 1.0, gen)
        draft_logits = _logits(num_spec * n, 1.0, gen).view(num_spec, n, VOCAB)
        ctx = _Ctx([1.0] * n, [0] * n, [0.0] * n)
        ctrl = _Controller(ctx, _make_sampling("torch", 42), num_spec, VOCAB, "rejection")
        for depth in range(num_spec):
            ctrl._sampled_mtp_tokens_cuda[depth] = ctrl._sample_mtp_draft(
                depth, draft_logits[depth], n
            )
        ctrl._record_mtp_draft_rows(ctx, n)
        # Request 3 resumes from a pause and was not in the last draft loop.
        del ctrl._mtp_draft_row_of_request[int(ctx.request_ids[3])]
        # Request 0's verified drafts differ from what was drafted (e.g. replaced upstream).
        inputs = torch.cat(
            [torch.zeros(n, 1, dtype=torch.long, device=DEVICE), ctrl._sampled_mtp_tokens_cuda.T],
            dim=1,
        )
        inputs[0, 1:] = (inputs[0, 1:] + 1) % VOCAB

        captured = {}
        real_chain = ctrl._sampling.chain_speculative_sampling

        def spy(d, x, t):
            captured["draft_probs"], captured["drafts"] = d.clone(), x
            return real_chain(d, x, t)

        ctrl._sampling.chain_speculative_sampling = spy
        plan = ctrl._mtp_rejection_plan(n)
        assert plan.rows == [0, 1, 2]
        # The map is consumed: a second verification without a new draft loop is exact match.
        assert ctrl._mtp_draft_row_of_request is None
        assert ctrl._mtp_rejection_plan(n) is None

        output = ctrl._sample_speculative_verify_rows(target_logits, None, n, 0, plan)
        stride = num_spec + 1
        assert (output[3 * stride :] >= 0).all()  # exact-match request sampled normally
        ctrl._apply_mtp_rejection_sampling(output, target_logits, None, inputs.reshape(-1), plan)
        point = torch.nn.functional.one_hot(inputs[0, 1:], VOCAB).float()
        torch.testing.assert_close(captured["draft_probs"][0], point)
        stored_q = ctrl._mtp_draft_probs[:, 1]
        torch.testing.assert_close(captured["draft_probs"][1], stored_q)

    def test_reordered_requests_use_their_own_drafts(self):
        """Verification looks draft rows up by request id, not by row position."""
        num_spec, n = 2, 3
        gen = torch.Generator(device=DEVICE).manual_seed(51)
        target_logits = _logits(n * (num_spec + 1), 1.0, gen)
        draft_logits = _logits(num_spec * n, 1.0, gen).view(num_spec, n, VOCAB)
        ctx = _Ctx([1.0] * n, [0] * n, [0.0] * n)
        ctrl = _Controller(ctx, _make_sampling("torch", 52), num_spec, VOCAB, "rejection")
        for depth in range(num_spec):
            ctrl._sampled_mtp_tokens_cuda[depth] = ctrl._sample_mtp_draft(
                depth, draft_logits[depth], n
            )
        ctrl._record_mtp_draft_rows(ctx, n)
        perm = [2, 0, 1]
        ctx.request_ids = ctx.request_ids[perm].clone()
        drafts = ctrl._sampled_mtp_tokens_cuda.T[perm]
        inputs = torch.cat([torch.zeros(n, 1, dtype=torch.long, device=DEVICE), drafts], dim=1)

        captured = {}
        real_chain = ctrl._sampling.chain_speculative_sampling

        def spy(d, x, t):
            captured["draft_probs"], captured["drafts"] = d.clone(), x
            return real_chain(d, x, t)

        ctrl._sampling.chain_speculative_sampling = spy
        plan = ctrl._mtp_rejection_plan(n)
        assert plan.draft_rows == perm
        output = ctrl._sample_speculative_verify_rows(target_logits, None, n, 0, plan)
        ctrl._apply_mtp_rejection_sampling(output, target_logits, None, inputs.reshape(-1), plan)
        assert torch.equal(captured["drafts"], drafts)
        for i, src in enumerate(perm):
            ref_q = _reference_probs(draft_logits[:, src], 1.0, 0, 0.0)
            torch.testing.assert_close(captured["draft_probs"][i], ref_q, atol=1e-5, rtol=1e-4)

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_imported_draft_q_matches_collocated_verification(self, backend):
        """Disaggregated handoff: q exported by the prefill controller, encoded, decoded and
        imported by the decode controller verifies the first drafts exactly like the
        collocated path (same plan coverage, emitted distribution and acceptance)."""
        temperature, top_k, top_p = 1.0, 0, 0.9
        num_spec, n, rounds = 2, 2048, 6
        stride = num_spec + 1
        gen = torch.Generator(device=DEVICE).manual_seed(61)
        target_row = _logits(num_spec + 1, 2.0, gen)
        draft_row = _logits(num_spec, 2.0, gen)
        p = _reference_probs(target_row, temperature, top_k, top_p)
        q = _reference_probs(draft_row, temperature, top_k, top_p)
        target_logits = target_row.repeat(n, 1)
        draft_logits = draft_row.unsqueeze(1).expand(num_spec, n, VOCAB).contiguous()
        base = torch.zeros(n, 1, dtype=torch.long, device=DEVICE)
        rows = torch.arange(n, device=DEVICE)

        stats = {}
        for path in ("collocated", "imported"):
            prefill_ctx = _Ctx([temperature] * n, [top_k] * n, [top_p] * n)
            prefill = _Controller(
                prefill_ctx, _make_sampling(backend, 62), num_spec, VOCAB, "rejection"
            )
            decode_ctx = _Ctx([temperature] * n, [top_k] * n, [top_p] * n)
            decode = _Controller(
                decode_ctx, _make_sampling(backend, 63), num_spec, VOCAB, "rejection"
            )
            first, acc = [], []
            for _ in range(rounds):
                for depth in range(num_spec):
                    prefill._sampled_mtp_tokens_cuda[depth] = prefill._sample_mtp_draft(
                        depth, draft_logits[depth], n
                    )
                prefill._record_mtp_draft_rows(prefill_ctx, n)
                drafts = prefill._sampled_mtp_tokens_cuda.T.clone()
                verifier = prefill
                if path == "imported":
                    verifier = decode
                    for rid in prefill_ctx.request_ids.tolist():
                        exported_q, exported_drafts = prefill.export_mtp_draft_q(rid)
                        imported_q = decode_mtp_draft_q(
                            encode_mtp_draft_q(exported_q, exported_drafts),
                            num_depths=num_spec,
                            vocab=VOCAB,
                            drafts=exported_drafts,
                        )
                        assert torch.equal(imported_q, exported_q)
                        decode.import_mtp_draft_q(rid, imported_q, exported_drafts)
                plan = verifier._mtp_rejection_plan(n)
                assert plan.rows == list(range(n))
                assert len(plan.imported) == (n if path == "imported" else 0)
                output = verifier._sample_speculative_verify_rows(target_logits, None, n, 0, plan)
                inputs = torch.cat([base, drafts], dim=1).reshape(-1)
                verifier._apply_mtp_rejection_sampling(output, target_logits, None, inputs, plan)
                assert not verifier._imported_mtp_draft_q
                last_one, mask, _ = verify_speculative_tokens(inputs, output, n, 0, num_spec)
                a = mask.view(n, stride).sum(1) - 1
                assert torch.equal(last_one, rows * stride + a)
                first.append(torch.where(a >= 1, drafts[:, 0], output[last_one]))
                acc.append(a)
            stats[path] = (torch.cat(first), torch.cat(acc))

        alpha = torch.minimum(p[0], q[0]).sum().item()
        for path, (first, acc) in stats.items():
            assert _tv(_hist(first, VOCAB), p[0]) < _tv_tol(VOCAB, first.numel()), path
            rate = (acc >= 1).double().mean().item()
            sigma = math.sqrt(alpha * (1 - alpha) / acc.numel())
            assert abs(rate - alpha) < 5 * sigma + 1e-3, (path, rate, alpha)
        rates = {path: (acc >= 1).double().mean().item() for path, (_, acc) in stats.items()}
        sigma = math.sqrt(2 * alpha * (1 - alpha) / (n * rounds))
        assert abs(rates["imported"] - rates["collocated"]) < 5 * sigma + 1e-3, rates
