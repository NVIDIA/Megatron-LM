# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for :class:`MaskDiffusionProcess`.

Tests target behaviours whose failure would silently corrupt training or
inference — for example, corruption that fails to mask any tokens, loss
applied at non-masked positions, or sampling that overwrites committed tokens.
CPU-only; no GPU required.
"""

import pytest
import torch

from megatron.diffusion.two_tower.diffusion_process import DiffusionProcess, MaskDiffusionProcess


class TestMaskDiffusionProcess:
    """Core corruption, loss, and sampling tests for :class:`MaskDiffusionProcess`."""

    def setup_method(self):
        self.vocab_size = 100
        self.mask_token_id = 99
        self.process = MaskDiffusionProcess(
            mask_token_id=self.mask_token_id, vocab_size=self.vocab_size
        )

    def test_corrupt_suffix_masks_tokens(self):
        """If corruption doesn't mask anything, training loss is zero and denoiser learns nothing."""
        clean = torch.randint(0, self.mask_token_id, (4, 64))
        noisy, aux = self.process.corrupt_suffix(clean, eps=0.5)
        has_masks = (noisy == self.mask_token_id).any()
        assert has_masks, "Expected some tokens to be masked"
        assert "masked_indices" in aux
        assert "t" in aux

    def test_corrupt_suffix_respects_loss_mask(self):
        """If masking ignores loss_mask, context tokens get corrupted silently."""
        clean = torch.randint(0, self.mask_token_id, (2, 16))
        loss_mask = torch.zeros(2, 16, dtype=torch.bool)
        loss_mask[:, 8:] = True
        noisy, aux = self.process.corrupt_suffix(clean, loss_mask=loss_mask, eps=0.99)
        prefix_unchanged = torch.equal(noisy[:, :8], clean[:, :8])
        assert prefix_unchanged, "Prefix should be unchanged when loss_mask is False"

    def test_training_loss_positive_with_masks(self):
        """Scalar loss must be positive when masked positions exist."""
        B, S, V = 2, 16, self.vocab_size
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        masked_indices = torch.zeros(B, S, dtype=torch.bool)
        masked_indices[:, ::2] = True
        aux = {"masked_indices": masked_indices}
        loss = self.process.training_loss(logits, labels, aux)
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert loss > 0, "Loss should be positive when there are masked positions"

    def test_training_loss_zero_without_masks(self):
        """Scalar loss must be zero when no positions are masked."""
        B, S, V = 2, 16, self.vocab_size
        logits = torch.randn(B, S, V)
        labels = torch.randint(0, V, (B, S))
        aux = {"masked_indices": torch.zeros(B, S, dtype=torch.bool)}
        loss = self.process.training_loss(logits, labels, aux)
        assert loss.dim() == 0, "training_loss should return a scalar"
        assert loss == 0, "Loss should be zero when no positions are masked"

    def test_sample_block_preserves_unmasked(self):
        """If sampling overwrites already-decoded tokens, inference produces garbage."""
        B, L = 1, 8
        init_ids = torch.full((B, L), self.mask_token_id, dtype=torch.long)
        init_ids[:, :4] = torch.tensor([[1, 2, 3, 4]])

        def run_denoiser(x, *args):
            logits = torch.randn(B, L, self.vocab_size)
            logits[..., self.mask_token_id] = -1e12
            return logits

        result = self.process.sample_block(
            run_denoiser, init_ids=init_ids, num_steps=20, temperature=0.0
        )
        assert torch.equal(result[:, :4], init_ids[:, :4])


class TestNoiseSchedule:
    """Boundary values and monotonicity of the noise schedule ``alpha_t(t)``."""

    def setup_method(self):
        self.process = MaskDiffusionProcess(mask_token_id=99, vocab_size=100)

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "exponential"])
    def test_boundary_t0(self, schedule):
        t = torch.tensor(0.0)
        alpha_t, _ = self.process._noise_schedule(t, schedule)
        assert abs(alpha_t.item() - 1.0) < 1e-5

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "exponential"])
    def test_boundary_t1(self, schedule):
        t = torch.tensor(1.0)
        alpha_t, _ = self.process._noise_schedule(t, schedule)
        assert abs(alpha_t.item()) < 1e-5

    @pytest.mark.parametrize("schedule", ["linear", "cosine", "exponential"])
    def test_monotonically_decreasing(self, schedule):
        ts = torch.linspace(0.0, 1.0, 50)
        alphas = torch.stack([self.process._noise_schedule(t, schedule)[0] for t in ts])
        diffs = alphas[1:] - alphas[:-1]
        assert (diffs <= 1e-6).all(), f"{schedule}: alpha_t not monotonically decreasing"


class TestComputePosterior:
    """Verify the analytic reverse posterior ``q(x_s | x_t, x_theta)``."""

    def setup_method(self):
        self.mask_id = 99
        self.V = 100
        self.process = MaskDiffusionProcess(mask_token_id=self.mask_id, vocab_size=self.V)

    def test_mask_token_probability(self):
        """Mask token posterior should be (1-alpha_s)/(1-alpha_t)."""
        alpha_t = torch.tensor(0.3).view(1, 1, 1)
        alpha_s = torch.tensor(0.5).view(1, 1, 1)
        x_theta = torch.rand(1, 4, self.V)
        x_theta = x_theta / x_theta.sum(dim=-1, keepdim=True)
        xt = torch.full((1, 4), self.mask_id, dtype=torch.long)
        q_xs = self.process._compute_posterior(x_theta, xt, alpha_t, alpha_s)
        expected_mask_prob = (1.0 - 0.5) / (1.0 - 0.3)
        assert torch.allclose(
            q_xs[..., self.mask_id], torch.tensor(expected_mask_prob).expand(1, 4), atol=1e-5
        )


class TestSampleCategorical:
    """Verify :meth:`_sample_categorical` sampling and greedy modes."""

    def setup_method(self):
        self.process = MaskDiffusionProcess(mask_token_id=99, vocab_size=100)

    def test_greedy_returns_argmax(self):
        """If greedy doesn't return argmax, deterministic inference is silently wrong."""
        B, L, V = 2, 4, 10
        logits = torch.randn(B, L, V)
        sampled, probs = self.process._sample_categorical(logits, temperature=0)
        assert torch.equal(sampled, logits.argmax(dim=-1))


class TestSampleBlockBatched:
    """Batched ``sample_block`` regression tests (B > 1)."""

    def setup_method(self):
        self.vocab_size = 50
        self.mask_token_id = 49
        self.process = MaskDiffusionProcess(
            mask_token_id=self.mask_token_id, vocab_size=self.vocab_size
        )

    def test_batched_predict_and_noise(self):
        """Regression: num_noise_indices.item() crashed on B>1 tensors."""
        B, L = 4, 16
        init_ids = torch.full((B, L), self.mask_token_id, dtype=torch.long)

        def run_denoiser(x, *args):
            logits = torch.randn(B, L, self.vocab_size)
            logits[..., self.mask_token_id] = -1e12
            return logits

        result = self.process.sample_block(
            run_denoiser, init_ids=init_ids, num_steps=20, temperature=1.0
        )
        assert result.shape == (B, L)
        assert (result != self.mask_token_id).all()


class TestSampleBlockPosterior:
    """Verify the ``posterior`` sampling strategy produces valid output."""

    def setup_method(self):
        self.vocab_size = 50
        self.mask_token_id = 49
        self.process = MaskDiffusionProcess(
            mask_token_id=self.mask_token_id, vocab_size=self.vocab_size
        )

    def test_posterior_removes_all_masks(self):
        """Posterior sampling must unmask every position by the final step."""
        B, L = 2, 16
        init_ids = torch.full((B, L), self.mask_token_id, dtype=torch.long)

        def run_denoiser(x, *args):
            logits = torch.randn(B, L, self.vocab_size)
            logits[..., self.mask_token_id] = -1e12
            return logits

        result = self.process.sample_block(
            run_denoiser,
            init_ids=init_ids,
            num_steps=20,
            sampling_strategy="posterior",
            temperature=1.0,
        )
        assert result.shape == (B, L)
        assert (result != self.mask_token_id).all(), "Posterior sampling left mask tokens"


class TestSampleBlockConfidenceUnmasking:
    """Verify the ``confidence_unmasking`` sampling strategy produces valid output."""

    def setup_method(self):
        self.vocab_size = 50
        self.mask_token_id = 49
        self.process = MaskDiffusionProcess(
            mask_token_id=self.mask_token_id, vocab_size=self.vocab_size
        )

    def test_confidence_unmasking_removes_all_masks(self):
        """Confidence unmasking must unmask every position by the final step."""
        B, L = 2, 16
        init_ids = torch.full((B, L), self.mask_token_id, dtype=torch.long)

        def run_denoiser(x, *args):
            logits = torch.randn(B, L, self.vocab_size)
            logits[..., self.mask_token_id] = -1e12
            return logits

        result = self.process.sample_block(
            run_denoiser,
            init_ids=init_ids,
            num_steps=20,
            sampling_strategy="confidence_unmasking",
            temperature=1.0,
        )
        assert result.shape == (B, L)
        assert (result != self.mask_token_id).all(), "Confidence unmasking left mask tokens"


class TestSampleBlockBatchInvariance:
    """Per-row independence of ``sample_block`` across batch sizes.

    Each row's denoising trajectory must depend only on that row's tokens,
    not on any sibling row. These tests use a row-pure fake denoiser whose
    logits for row ``i`` depend only on ``(xt[i], t[i])`` so that B=1 and
    B>1 are mathematically identical on each row whenever ``sample_block``
    itself is free of cross-row reductions (``min`` / ``max`` / ``mean``
    over the batch dim) or cross-row stalls.

    Failures here mean ``sample_block`` is coupling rows in the batch and
    breaking independent batched inference.
    """

    def setup_method(self):
        self.vocab_size = 32
        self.mask_token_id = 31
        self.L = 12
        self.process = MaskDiffusionProcess(
            mask_token_id=self.mask_token_id, vocab_size=self.vocab_size
        )

    def _row_pure_denoiser(self):
        """Build a denoiser whose logits[i] depend only on (xt[i], t[i]).

        Row-by-row seeding from a hash of the row contents and (optional)
        timestep guarantees: logits for row i in B=1 equal logits for row i
        in B>1. Any divergence between B=1 and B>1 is then attributable to
        ``sample_block`` itself, not to the fake model.
        """
        L, V, mask_id = self.L, self.vocab_size, self.mask_token_id

        def run_denoiser(xt, t_arg=None, *args):
            B = xt.shape[0]
            if t_arg is not None:
                assert isinstance(
                    t_arg, torch.Tensor
                ), f"sample_block contract: t must be a Tensor, got {type(t_arg)}"
                assert t_arg.dim() == 1 and t_arg.shape[0] == B, (
                    f"sample_block contract: t must have shape (B,)=({B},), "
                    f"got dim={t_arg.dim()} shape={tuple(t_arg.shape)}. "
                    "A 0-d t breaks time-conditioned denoisers."
                )
            out = torch.empty(B, L, V, dtype=torch.float32)
            for i in range(B):
                row = tuple(int(x) for x in xt[i].cpu().tolist())
                if t_arg is None:
                    t_val = -1.0
                else:
                    t_val = float(t_arg[i].item())
                seed = abs(hash((row, round(t_val, 6)))) % (2**31)
                g = torch.Generator(device="cpu")
                g.manual_seed(seed)
                row_logits = torch.randn(L, V, generator=g)
                row_logits[..., mask_id] = -1e12
                out[i] = row_logits
            return out

        return run_denoiser

    def _make_asymmetric_init_ids(self, B):
        """Construct init_ids whose rows have different mask counts.

        Row 0: only 2 masks (finishes in a few steps).
        Row 1: half masked.
        Other rows: fully masked.

        This is the regime where the old ``num_masked.min() == 0`` early-exit
        would have stalled batched inference relative to serial.
        """
        ids = torch.full((B, self.L), self.mask_token_id, dtype=torch.long)
        if B >= 1:
            for j in range(self.L - 2):
                ids[0, j] = (j * 7 + 3) % self.mask_token_id
        if B >= 2:
            for j in range(self.L // 2):
                ids[1, j] = (j * 11 + 5) % self.mask_token_id
        return ids

    def _run(self, init_ids_subset, **kwargs):
        return self.process.sample_block(
            self._row_pure_denoiser(),
            init_ids=init_ids_subset,
            use_model_output_cache=False,
            temperature=0.0,
            **kwargs,
        )

    def _assert_serial_vs_batched(self, B, **kwargs):
        full = self._make_asymmetric_init_ids(B)
        batched = self._run(full, **kwargs)
        for i in range(B):
            single = self._run(full[i : i + 1], **kwargs)
            assert torch.equal(single[0], batched[i]), (
                f"Row {i} differs between B=1 and B={B} for {kwargs}: "
                f"single={single[0].tolist()} batched={batched[i].tolist()}"
            )

    def test_predict_and_noise_adaptive_batch_invariance(self):
        """``predict_and_noise`` + ``adaptive_unmasking``: B=1 == B=k row-wise.

        Exercises per-row ``tokens_to_commit`` and ``num_noise_indices`` and
        the rank-based re-masking selection.
        """
        self._assert_serial_vs_batched(
            B=3,
            num_steps=12,
            sampling_strategy="predict_and_noise",
            adaptive_unmasking=True,
            confidence_based_noising=True,
        )

    def test_predict_and_noise_non_adaptive_batch_invariance(self):
        """``predict_and_noise`` (non-adaptive) with confidence noising: B=1 == B=k row-wise."""
        self._assert_serial_vs_batched(
            B=3,
            num_steps=12,
            sampling_strategy="predict_and_noise",
            adaptive_unmasking=False,
            confidence_based_noising=True,
        )

    def test_confidence_unmasking_batch_invariance(self):
        """``confidence_unmasking``: B=1 == B=k row-wise even when rows finish at different steps.

        This is the path where the old ``num_masked.min() == 0`` early-exit
        would have stalled the batch and caused B=k to diverge from B=1.
        """
        self._assert_serial_vs_batched(
            B=3, num_steps=12, sampling_strategy="confidence_unmasking", confidence_threshold=0.5
        )

    def test_posterior_batch_invariance(self):
        """``posterior``: B=1 == B=k row-wise."""
        self._assert_serial_vs_batched(B=3, num_steps=12, sampling_strategy="posterior")

    def test_permutation_equivariance_predict_and_noise(self):
        """``sample_block`` is permutation-equivariant in the batch dimension.

        Strongest statement of per-row independence: shuffling input rows,
        running, and unshuffling the output must reproduce the original
        result bit-for-bit. Any future change that reintroduces a batch-wide
        reduction on a per-row quantity will break this.
        """
        B = 4
        full = self._make_asymmetric_init_ids(B)
        kwargs = dict(
            num_steps=12,
            sampling_strategy="predict_and_noise",
            adaptive_unmasking=True,
            confidence_based_noising=True,
        )
        out_orig = self._run(full, **kwargs)
        perm = torch.tensor([3, 0, 2, 1])
        inv = torch.argsort(perm)
        out_shuffled = self._run(full[perm], **kwargs)
        assert torch.equal(out_orig, out_shuffled[inv]), (
            "sample_block is not permutation-equivariant in the batch dim: "
            "rows are coupled to each other."
        )

    def test_permutation_equivariance_confidence_unmasking(self):
        """``confidence_unmasking`` is permutation-equivariant in the batch dimension."""
        B = 4
        full = self._make_asymmetric_init_ids(B)
        kwargs = dict(
            num_steps=12, sampling_strategy="confidence_unmasking", confidence_threshold=0.5
        )
        out_orig = self._run(full, **kwargs)
        perm = torch.tensor([2, 3, 0, 1])
        inv = torch.argsort(perm)
        out_shuffled = self._run(full[perm], **kwargs)
        assert torch.equal(
            out_orig, out_shuffled[inv]
        ), "confidence_unmasking is not permutation-equivariant in the batch dim."
