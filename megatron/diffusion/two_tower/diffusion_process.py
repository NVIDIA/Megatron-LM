# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Callable, Dict, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffusionProcess(ABC, nn.Module):
    """Abstract base class for discrete diffusion processes.

    Encapsulates three concerns that vary across diffusion strategies:

    1. **Forward corruption** (:meth:`corrupt_suffix`) — stochastically degrade
       clean token sequences.
    2. **Training loss** (:meth:`training_loss`) — compute per-token losses
       given model logits and corruption metadata.
    3. **Reverse sampling** (:meth:`sample_block`) — iteratively denoise a
       masked/noisy block of tokens to produce clean output.

    The class is independent of model architecture.  At sampling time the
    caller supplies a ``run_denoiser(x_t) -> logits`` closure so that any
    model can be plugged in.

    Args:
        mask_token_id (int): Token ID reserved for the absorbing mask state.
        vocab_size (int): Full vocabulary size (including mask token).
    """

    def __init__(self, mask_token_id: int, vocab_size: int):
        super().__init__()
        self.mask_token_id = mask_token_id
        self.vocab_size = vocab_size
        self._cross_entropy_fn: Optional[
            Callable[[torch.FloatTensor, torch.LongTensor], torch.Tensor]
        ] = None

    def set_cross_entropy_fn(
        self, fn: Callable[[torch.FloatTensor, torch.LongTensor], torch.Tensor]
    ) -> None:
        """Register a custom cross-entropy function for tensor-parallel vocabularies.

        When set, :meth:`_compute_cross_entropy` delegates to *fn* instead of
        the default ``F.cross_entropy``.  This is required when the output
        logits are sharded across TP ranks.

        Args:
            fn (Callable[[FloatTensor, LongTensor], Tensor]): Function mapping
                ``(logits [B, S, V_local], labels [B, S])`` to per-token losses
                ``(B, S)``.
        """
        self._cross_entropy_fn = fn

    def _compute_cross_entropy(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, reduction: str = "none"
    ) -> torch.Tensor:
        """Compute cross-entropy loss, delegating to the TP-aware function if set.

        Args:
            logits (FloatTensor): Model output logits ``(B, S, V)``.
            labels (LongTensor): Ground-truth token IDs ``(B, S)``.
            reduction (str): ``"none"`` for per-token ``(B, S)`` or ``"mean"``
                for a scalar.

        Returns:
            Tensor: Per-token losses ``(B, S)`` or scalar, depending on *reduction*.
        """
        if self._cross_entropy_fn is not None:
            per_token_loss = self._cross_entropy_fn(logits, labels)
        else:
            B, S, V = logits.shape
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, V), labels.reshape(-1), reduction="none"
            ).reshape(B, S)

        if reduction == "mean":
            return per_token_loss.mean()
        return per_token_loss

    @abstractmethod
    def corrupt_suffix(
        self,
        clean_suffix: torch.LongTensor,
        *,
        eps: float = 1e-3,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Apply forward diffusion (corruption) to a clean token sequence.

        Args:
            clean_suffix (LongTensor): Clean token IDs ``(B, S)``.
            eps (float): Minimum noise level to avoid ``t = 0`` (no corruption).
            loss_mask (Optional[Tensor]): Boolean mask ``(B, S)`` restricting
                which positions may be corrupted.
            labels (Optional[LongTensor]): Unused by most implementations;
                reserved for subclass-specific label manipulation.

        Returns:
            Tuple[LongTensor, Dict[str, Tensor]]: ``(noisy_suffix, aux)`` where
                *aux* carries per-sample metadata needed by :meth:`training_loss`
                (e.g. ``masked_indices``, ``t``).
        """
        ...

    @abstractmethod
    def training_loss(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, aux: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute the training loss for Megatron's ``loss_func``.

        Implementations may return either a **scalar** (already reduced) or
        unreduced ``(B, S)`` per-token losses.  ``MaskDiffusionProcess``
        returns a scalar mean over masked positions; Megatron's ``loss_func``
        broadcasts this correctly (``scalar * loss_mask.sum() / num_tokens``
        recovers the original scalar).

        Args:
            logits (FloatTensor): Model predictions ``(B, S, V)``.
            labels (LongTensor): Target token IDs ``(B, S)``.
            aux (Dict[str, Tensor]): Metadata from :meth:`corrupt_suffix`.

        Returns:
            Tensor: Scalar or per-token losses ``(B, S)``.
        """
        ...

    @abstractmethod
    def sample_block(
        self,
        run_denoiser: Callable[[torch.LongTensor], torch.FloatTensor],
        *,
        init_ids: torch.LongTensor,
        num_steps: int = 10,
        **kwargs,
    ) -> torch.LongTensor:
        """Iteratively denoise *init_ids* via the reverse diffusion process.

        Args:
            run_denoiser (Callable): Closure ``(token_ids) -> logits`` that
                runs the denoiser model on the current noisy sequence.
            init_ids (LongTensor): Initial (typically fully masked) token IDs
                ``(B, S)``.
            num_steps (int): Number of reverse-diffusion steps.

        Returns:
            LongTensor: Denoised token IDs ``(B, S)``.
        """
        ...


# ---------------------------------------------------------------------------
# Mask Diffusion
# ---------------------------------------------------------------------------


class MaskDiffusionProcess(DiffusionProcess):
    """Discrete diffusion with an absorbing mask state (MDLM-style).

    Forward process:
        At a randomly sampled timestep *t* in ``[eps, 1]``, each token is
        independently replaced with ``mask_token_id`` with probability
        ``1 - alpha_t``, where ``alpha_t = 1 - t`` (linear schedule).

    Training:
        Cross-entropy loss computed only at masked positions.  Unmasked
        positions contribute zero loss so that the model learns to predict
        corrupted tokens without being rewarded for copying visible ones.

    Reverse sampling:
        Iteratively denoises from ``t = 1`` (fully masked) toward ``t ~ 0``
        (clean) using one of three strategies: posterior sampling,
        predict-and-noise, or confidence-based unmasking.

    Args:
        mask_token_id (int): Token ID used for the absorbing mask state.
        vocab_size (int): Full vocabulary size.
        neg_infinity (float): Large negative value for logit masking.
    """

    def __init__(self, mask_token_id: int, vocab_size: int, neg_infinity: float = -1e12):
        super().__init__(mask_token_id, vocab_size)
        self.neg_infinity = neg_infinity
        self.mode = "mask_diffusion"

    # ----- Forward diffusion (corruption) ---------------------------------

    def corrupt_suffix(
        self,
        clean_suffix: torch.LongTensor,
        *,
        eps: float = 1e-3,
        loss_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.LongTensor, Dict[str, torch.Tensor]]:
        """Apply absorbing-state corruption to *clean_suffix*.

        Samples a per-example timestep ``t ~ U[eps, 1]`` and masks each token
        with probability ``1 - alpha_t``.  Positions where ``loss_mask`` is
        ``False`` are never corrupted.

        Args:
            clean_suffix (LongTensor): Clean token IDs ``(B, S)``.
            eps (float): Lower bound for *t*, avoiding zero corruption.
            loss_mask (Optional[Tensor]): Boolean ``(B, S)`` — only ``True``
                positions may be masked.
            labels (Optional[LongTensor]): Unused.

        Returns:
            Tuple[LongTensor, Dict[str, Tensor]]: ``(noisy_suffix, aux)`` where
                *aux* contains ``masked_indices``, ``p_mask``, ``alpha_t``,
                and ``t``.
        """
        b, l = clean_suffix.shape
        device = clean_suffix.device

        t = torch.rand(b, device=device)
        t = (1.0 - eps) * t + eps
        alpha_t = 1.0 - t
        alpha_t = alpha_t[:, None].expand(-1, l)

        masked_indices = torch.rand((b, l), device=device) < (1.0 - alpha_t)
        if loss_mask is not None:
            masked_indices = masked_indices & loss_mask

        noisy_suffix = torch.where(masked_indices, self.mask_token_id, clean_suffix)

        aux = {
            "masked_indices": masked_indices,
            "p_mask": (1.0 - alpha_t),
            "alpha_t": alpha_t,
            "t": t,
        }
        return noisy_suffix, aux

    # ----- Training loss --------------------------------------------------

    def training_loss(
        self, logits: torch.FloatTensor, labels: torch.LongTensor, aux: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Mean cross-entropy over masked positions (standard MDLM loss).

        Only positions where ``aux["masked_indices"]`` is ``True`` contribute.
        Returns a **scalar** so that Megatron's ``loss_func`` (which multiplies
        by ``loss_mask.sum()`` then divides by the same ``num_tokens``) faithfully
        reproduces the scalar without changing the gradient magnitude.

        Args:
            logits (FloatTensor): Model predictions ``(B, S, V)``.
            labels (LongTensor): Ground-truth token IDs ``(B, S)``.
            aux (Dict[str, Tensor]): Must contain ``"masked_indices"``
                ``(B, S)`` boolean tensor from :meth:`corrupt_suffix`.

        Returns:
            Tensor: Scalar mean cross-entropy over masked positions.
        """
        masked = aux.get("masked_indices", None)
        per_token_loss = self._compute_cross_entropy(logits, labels, reduction="none")

        if masked is None or masked.sum() == 0:
            return per_token_loss.sum() * 0.0

        masked_loss = (per_token_loss * masked.float()).sum()
        return masked_loss / masked.float().sum()

    # ----- Noise schedule -------------------------------------------------

    def _noise_schedule(
        self, t: torch.FloatTensor, schedule: str = "linear"
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Evaluate the noise schedule at time *t*.

        Returns ``(alpha_t, alpha_t_prime)`` where ``alpha_t`` is the
        retention probability and ``alpha_t_prime`` is its derivative.
        Satisfies ``alpha_t(0) = 1`` (clean) and ``alpha_t(1) ~ 0`` (masked).

        Args:
            t (FloatTensor): Scalar or batched timestep(s) in ``[0, 1]``.
            schedule (str): One of ``"linear"``, ``"cosine"``, ``"exponential"``.

        Returns:
            Tuple[FloatTensor, FloatTensor]: ``(alpha_t, alpha_t_prime)``.

        Raises:
            ValueError: If *schedule* is not recognised.
        """
        if schedule == "linear":
            alpha_t = 1.0 - t
            alpha_t_prime = torch.ones_like(alpha_t)
        elif schedule == "cosine":
            alpha_t = torch.cos(t * (math.pi / 2.0))
            alpha_t_prime = (math.pi / 2.0) * torch.sin(t * (math.pi / 2.0))
        elif schedule == "exponential":
            _rate = 3.0
            _exp_neg_rate = math.exp(-_rate)
            raw = torch.exp(-_rate * t)
            alpha_t = (raw - _exp_neg_rate) / (1.0 - _exp_neg_rate)
            alpha_t_prime = _rate * raw / (1.0 - _exp_neg_rate)
        else:
            raise ValueError(
                f"Unknown noise schedule: {schedule}. " f"Choose from: linear, cosine, exponential"
            )
        return alpha_t, alpha_t_prime

    # ----- MDLM logit processing -----------------------------------------

    def _mdlm_forward(self, logits: torch.FloatTensor, xt: torch.LongTensor) -> torch.FloatTensor:
        """Post-process logits for MDLM reverse sampling.

        Adapted from E2D2: https://github.com/kuleshov-group/e2d2

        Two constraints are enforced:
        1. The mask token is assigned ``-inf`` probability (the model should
           never predict the mask token as output).
        2. Positions that are already unmasked in *xt* are clamped to a
           one-hot distribution over their current token, preventing the
           reverse process from overwriting committed tokens.

        Args:
            logits (FloatTensor): Raw model logits ``(B, S, V)``.
            xt (LongTensor): Current noisy token IDs ``(B, S)``.

        Returns:
            FloatTensor: Log-probabilities ``(B, S, V)`` with the above
                constraints applied.
        """
        logits = logits.clone()
        logits[..., self.mask_token_id] = self.neg_infinity
        log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

        unmasked_indices = xt != self.mask_token_id
        if unmasked_indices.any():
            log_probs[unmasked_indices] = self.neg_infinity
            B, L = xt.shape
            batch_idx = (
                torch.arange(B, device=xt.device).unsqueeze(1).expand(B, L)[unmasked_indices]
            )
            seq_idx = torch.arange(L, device=xt.device).unsqueeze(0).expand(B, L)[unmasked_indices]
            vocab_idx = xt[unmasked_indices]
            log_probs[batch_idx, seq_idx, vocab_idx] = 0.0

        return log_probs

    # ----- Posterior -------------------------------------------------------

    def _compute_posterior(
        self,
        x_theta: torch.FloatTensor,
        xt: torch.LongTensor,
        alpha_t: torch.FloatTensor,
        alpha_s: torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Compute the reverse posterior ``q(x_s | x_t, x_theta)`` for absorbing diffusion.

        Adapted from E2D2: https://github.com/kuleshov-group/e2d2

        For non-mask tokens the posterior probability scales with
        ``(alpha_s - alpha_t) / (1 - alpha_t)``.  The mask token's posterior
        is set to ``(1 - alpha_s) / (1 - alpha_t)``.

        Args:
            x_theta (FloatTensor): Predicted clean-token probabilities ``(B, S, V)``.
            xt (LongTensor): Current noisy token IDs ``(B, S)``.
            alpha_t (FloatTensor): Retention probability at current time ``(1, 1, 1)``.
            alpha_s (FloatTensor): Retention probability at target time ``(1, 1, 1)``.

        Returns:
            FloatTensor: Unnormalised posterior distribution ``(B, S, V)``.
        """
        q_xs = x_theta * (alpha_s - alpha_t) / (1.0 - alpha_t)
        q_xs[..., self.mask_token_id] = (1.0 - alpha_s[..., 0]) / (1.0 - alpha_t[..., 0])
        return q_xs

    def _gumbel_sample(self, probs: torch.FloatTensor) -> torch.LongTensor:
        """Draw Gumbel-argmax samples from *probs* without explicit log-space.

        Args:
            probs (FloatTensor): Token probabilities ``(B, S, V)``.

        Returns:
            LongTensor: Sampled token indices ``(B, S)``.
        """
        gumbel_norm = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
        return (probs / gumbel_norm).argmax(dim=-1)

    def _sample_categorical(
        self, logits: torch.FloatTensor, temperature: float = 1.0, top_k: int = 0
    ) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """Sample from a categorical distribution with temperature and top-k.

        Adapted from E2D2: https://github.com/kuleshov-group/e2d2

        When ``temperature <= 0`` the function falls back to deterministic
        argmax (greedy decoding).

        Args:
            logits (FloatTensor): Unnormalised log-probabilities ``(B, S, V)``.
            temperature (float): Softmax temperature. ``<= 0`` for greedy.
            top_k (int): If positive, only the top-k tokens are kept before
                sampling.

        Returns:
            Tuple[LongTensor, FloatTensor]: ``(sampled_ids [B, S], probs [B, S, V])``.
        """
        B, L, V = logits.shape
        logits = logits.clone()

        if temperature <= 0:
            probs = F.softmax(logits, dim=-1)
            return logits.argmax(dim=-1), probs

        logits = logits / max(temperature, 1e-8)
        if top_k > 0 and top_k < V:
            top_k_vals, _ = torch.topk(logits, min(top_k, V), dim=-1)
            threshold = top_k_vals[..., -1:]
            logits = torch.where(logits < threshold, self.neg_infinity, logits)

        probs = F.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs.view(-1, V), num_samples=1).view(B, L)
        return sampled, probs

    # ----- Reverse sampling -----------------------------------------------

    def sample_block(
        self,
        run_denoiser: Callable[[torch.LongTensor], torch.FloatTensor],
        *,
        init_ids: torch.LongTensor,
        num_steps: int = 10,
        sampling_strategy: Literal[
            "posterior", "predict_and_noise", "confidence_unmasking"
        ] = "predict_and_noise",
        min_t: float = 1e-5,
        temperature: float = 1.0,
        top_k: int = 0,
        confidence_based_noising: bool = True,
        confidence_margin_based_noising: bool = False,
        confidence_threshold: float = 1e6,
        use_model_output_cache: bool = True,
        step_callback: Optional[Callable] = None,
        adaptive_unmasking: bool = False,
        posterior_float64: bool = False,
        noise_schedule: str = "linear",
        **kwargs,
    ) -> torch.LongTensor:
        """Run the iterative reverse diffusion process on a single block.

        Starts from *init_ids* (typically fully masked at ``t = 1``) and
        progressively unmasks tokens over *num_steps* steps toward ``t ~ 0``.

        Three sampling strategies are supported:

        * ``"posterior"`` — sample from the analytic reverse posterior
          ``q(x_s | x_t, p_theta(x_0 | x_t))``.
        * ``"predict_and_noise"`` — predict clean tokens, then strategically
          re-mask the lowest-confidence positions.
        * ``"confidence_unmasking"`` — greedily commit high-confidence
          predictions and re-mask the rest.

        Args:
            run_denoiser (Callable): ``(token_ids [B, S], t [B]) -> logits [B, S, V]``.
            init_ids (LongTensor): Starting token IDs ``(B, S)``, usually all
                ``mask_token_id``.
            num_steps (int): Number of reverse steps.
            sampling_strategy (str): One of ``"posterior"``,
                ``"predict_and_noise"``, ``"confidence_unmasking"``.
            min_t (float): Terminal timestep (avoids exact ``t = 0``).
            temperature (float): Sampling temperature.
            top_k (int): Top-k filtering for ``_sample_categorical``.
            confidence_based_noising (bool): Re-mask lowest-confidence tokens
                in predict-and-noise mode.
            confidence_margin_based_noising (bool): Use top-2 margin instead of
                raw confidence for re-masking. Mutually exclusive with
                *confidence_based_noising*.
            confidence_threshold (float): Threshold for confidence-based
                unmasking strategy.
            use_model_output_cache (bool): Reuse model output when tokens
                haven't changed between steps.
            step_callback (Optional[Callable]): Called after each step for
                logging / visualisation.
            adaptive_unmasking (bool): Vary tokens-to-commit per step
                heuristically.
            posterior_float64 (bool): Use float64 for posterior computation to
                improve numerical stability.
            noise_schedule (str): Schedule for ``_noise_schedule``.

        Returns:
            LongTensor: Denoised token IDs ``(B, S)``.
        """
        assert not (confidence_based_noising and confidence_margin_based_noising)

        xt = init_ids.clone()
        B, L = xt.shape
        device = xt.device

        timesteps = torch.linspace(1.0, min_t, num_steps + 1, device=device)[:-1]
        dt = (1.0 - min_t) / len(timesteps)

        model_output_cache = None

        if step_callback is not None:
            try:
                step_callback(0, len(timesteps), xt, t=1.0, logits=None, init_ids=init_ids)
            except TypeError:
                step_callback(0, len(timesteps), xt, init_ids=init_ids)

        for step_idx, t in enumerate(timesteps):
            if (xt == self.mask_token_id).sum().item() == 0:
                break

            alpha_t, _ = self._noise_schedule(t, noise_schedule)
            alpha_s, _ = self._noise_schedule(t - dt, noise_schedule)
            alpha_t = alpha_t.view(1, 1, 1)
            alpha_s = alpha_s.view(1, 1, 1)

            if model_output_cache is None:
                if sampling_strategy == "confidence_unmasking":
                    t_model = (xt == self.mask_token_id).float().mean(dim=-1)
                elif noise_schedule == "linear":
                    t_model = t
                else:
                    t_model = 1.0 - alpha_t.squeeze()
                if t_model.dim() == 0:
                    t_model = t_model.expand(B)
                try:
                    logits = run_denoiser(xt, t_model)
                except TypeError:
                    logits = run_denoiser(xt)
                log_x_theta = self._mdlm_forward(logits, xt)
                x_theta = log_x_theta.exp()
                if use_model_output_cache:
                    model_output_cache = {"x_theta": x_theta, "log_x_theta": log_x_theta}
            else:
                x_theta = model_output_cache["x_theta"]
                log_x_theta = model_output_cache["log_x_theta"]

            if sampling_strategy == "posterior":
                if posterior_float64:
                    q_xs = self._compute_posterior(
                        x_theta.double(), xt, alpha_t.double(), alpha_s.double()
                    ).float()
                else:
                    q_xs = self._compute_posterior(x_theta, xt, alpha_t, alpha_s)
                q_xs = q_xs / q_xs.sum(dim=-1, keepdim=True)
                log_q_xs = torch.log(q_xs + 1e-10)
                xs, _ = self._sample_categorical(log_q_xs, temperature=temperature, top_k=top_k)
                output = torch.where(xt != self.mask_token_id, xt, xs)

            elif sampling_strategy == "predict_and_noise":
                xs, sampled_probs = self._sample_categorical(
                    log_x_theta, temperature=temperature, top_k=top_k
                )
                xs_probs = sampled_probs.gather(-1, xs.unsqueeze(-1)).squeeze(-1)
                output = xs.clone()

                if step_idx < num_steps - 1:
                    num_masks_remaining = (xt == self.mask_token_id).sum(dim=-1)

                    if adaptive_unmasking:
                        remaining_steps = num_steps - step_idx
                        progress = step_idx / num_steps
                        if progress < 0.4:
                            tokens_to_commit = torch.ones_like(num_masks_remaining)
                        else:
                            tokens_to_commit = torch.ceil(
                                num_masks_remaining.float() / float(remaining_steps)
                            ).to(torch.int)
                        tokens_to_commit = torch.minimum(tokens_to_commit, num_masks_remaining)
                        num_noise_indices = torch.clamp(
                            num_masks_remaining - tokens_to_commit, min=0
                        )
                    else:
                        num_noise_indices = torch.minimum(
                            ((1.0 - alpha_s[0, 0, 0]) * L).to(torch.int), num_masks_remaining - 1
                        )
                        num_noise_indices = torch.clamp(num_noise_indices, min=0)

                    if int(num_noise_indices.max().item()) > 0:
                        if confidence_based_noising:
                            conf = x_theta.gather(-1, xs.unsqueeze(-1)).squeeze(-1)
                            conf = torch.where(xt == self.mask_token_id, conf, torch.inf)
                            scores = conf
                        elif confidence_margin_based_noising:
                            top2 = torch.topk(x_theta, k=2, dim=-1).values
                            conf = (top2[..., 0] - top2[..., 1]).abs()
                            conf = torch.where(xt == self.mask_token_id, conf, torch.inf)
                            scores = conf
                        else:
                            mask_positions = xt == self.mask_token_id
                            rand_vals = torch.rand_like(xs_probs)
                            rand_vals[~mask_positions] = torch.inf
                            scores = rand_vals
                        noise_order = scores.argsort(dim=-1)
                        noise_ranks = torch.empty_like(noise_order)
                        noise_ranks.scatter_(
                            -1, noise_order, torch.arange(L, device=device).view(1, L).expand(B, -1)
                        )
                        noise_mask = (xt == self.mask_token_id) & (
                            noise_ranks < num_noise_indices.unsqueeze(-1)
                        )
                        output = torch.where(
                            noise_mask, torch.full_like(output, self.mask_token_id), output
                        )

            elif sampling_strategy == "confidence_unmasking":
                if temperature > 0 and temperature != 1.0:
                    scaled_logits = logits / max(temperature, 1e-8)
                    scaled_log = self._mdlm_forward(scaled_logits, xt)
                    _x_theta = scaled_log.exp()
                else:
                    _x_theta = x_theta

                if temperature <= 0:
                    predicted = _x_theta.argmax(dim=-1)
                else:
                    predicted = self._gumbel_sample(_x_theta.float())

                confidence = x_theta.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
                is_masked = xt == self.mask_token_id
                num_masked = is_masked.sum(dim=-1)
                confidence = torch.where(
                    is_masked, confidence, torch.tensor(float("inf"), device=device)
                )
                remaining_steps = max(1, num_steps - step_idx)
                output = torch.where(is_masked, predicted, xt)

                if step_idx == num_steps - 1:
                    tokens_to_commit = num_masked
                else:
                    num_above = (is_masked & (confidence > confidence_threshold)).sum(dim=-1)
                    tokens_to_commit = torch.where(
                        num_masked > 0,
                        torch.maximum(num_above, torch.ones_like(num_above)),
                        torch.zeros_like(num_above),
                    )
                    min_to_finish = torch.ceil(num_masked.float() / float(remaining_steps)).to(
                        torch.int
                    )
                    tokens_to_commit = torch.maximum(tokens_to_commit, min_to_finish)
                    tokens_to_commit = torch.minimum(tokens_to_commit, num_masked)

                num_to_remask = torch.clamp(num_masked - tokens_to_commit, min=0)
                if int(num_to_remask.max().item()) > 0:
                    remask_order = confidence.argsort(dim=-1)
                    remask_ranks = torch.empty_like(remask_order)
                    remask_ranks.scatter_(
                        -1, remask_order, torch.arange(L, device=device).view(1, L).expand(B, -1)
                    )
                    remask_mask = is_masked & (remask_ranks < num_to_remask.unsqueeze(-1))
                    output = torch.where(
                        remask_mask, torch.full_like(output, self.mask_token_id), output
                    )
            else:
                raise ValueError(f"Unknown sampling_strategy: {sampling_strategy}")

            if not torch.allclose(output, xt) or not use_model_output_cache:
                model_output_cache = None
            xt = output

            if step_callback is not None:
                try:
                    step_callback(step_idx + 1, len(timesteps), xt, t=float(t), logits=logits)
                except TypeError:
                    step_callback(step_idx + 1, len(timesteps), xt)

        return xt


# ---------------------------------------------------------------------------
# Registry — maps ``--tt-diffusion-mode`` CLI values to process classes.
# ---------------------------------------------------------------------------

DIFFUSION_REGISTRY: Dict[str, type] = {"mask_diffusion": MaskDiffusionProcess}
