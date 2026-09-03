# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Optional

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

from megatron.core.inference.sampling.base import Sampling
from megatron.core.inference.sampling_params import (
    MIN_SAMPLING_TEMPERATURE,
    is_no_op_top_k,
    is_no_op_top_p,
)


def _greedy_batch_flags(context) -> tuple[bool, bool]:
    """Return whether any/all active requests use greedy sampling.

    Sampling parameters have a pinned CPU source of truth, so these checks add
    no GPU synchronization to the sampling path.
    """
    active_count = context.total_request_count - context.paused_request_count
    if active_count == 0:
        return False, False
    metadata = context.active_request_metadata
    top_k = metadata["top_k"][:active_count]
    top_p = metadata["top_p"][:active_count]
    greedy = (top_k == 1) & is_no_op_top_p(top_p)
    return bool(greedy.any()), bool(greedy.all())


class FlashInferSampling(Sampling):
    """FlashInfer sampling with per-step unfiltered / top-p-only / top-k-only / joint dispatch.

    Each step selects a kernel from the batch's active filters: the logits kernel
    when nothing filters, the dedicated exact top-p or top-k kernel when only one filter is in use,
    and the joint kernel only for genuinely mixed batches.
    The dispatch flags are read from the pinned CPU sampling metadata.

    The sampler runs eagerly. Its kernel choice is data-dependent (it varies with
    which filters the batch uses), so it cannot be captured in a CUDA graph; running
    eagerly also lets the controller's seeded RNG generator advance its philox offset
    normally between steps -- fresh randomness per step, reproducible from the seed.
    (FlashInfer bakes the philox state into a graph as a by-value constant at capture,
    so a captured sampler replays identical random numbers; see
    https://www.linkedin.com/pulse/pinned-rng-drifting-crash-from-cuda-graph-chenyang-zhao-csuac/)
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, config=None, enable_cuda_graph: bool = False
    ) -> None:
        # `config` / `enable_cuda_graph` are accepted for factory API symmetry but
        # intentionally unused: the sampler is never graphed (see class docstring).
        del config, enable_cuda_graph
        self._vocab_size = vocab_size
        self._rng = rng

    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        no_top_k: bool,
        no_top_p: bool,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
        output: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Sample tokens, dispatching top-p-only / top-k-only / joint by filter flags.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            no_top_k, no_top_p: Required batch-level dispatch flags (whether NO active
                request uses top-k / top-p). The caller computes them once from the
                pinned CPU sampling metadata (the context's
                `active_sampling_filter_flags`).
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, sampling parameters are gathered
                per-token rather than per-request (speculative decoding path).
            output: Optional caller-owned destination tensor of shape `[n]`.
            eager, cache_key: Accepted for API symmetry; ignored (no CUDA graph).

        Returns:
            Sampled token IDs in `output`, or a newly allocated tensor when it is not provided.
        """
        del eager, cache_key

        # Per-row sampling params (GPU) for the kernel. gpu_view mirrors the pinned
        # CPU `active_request_metadata` via the per-step coalesced H2D.
        gv = context.gpu_view
        if token_to_request_index is None:
            temperature = gv.temperature[:n]
            top_k = gv.top_k[:n]
            top_p = gv.top_p[:n]
        else:
            temperature = gv.temperature[token_to_request_index]
            top_k = gv.top_k[token_to_request_index]
            top_p = gv.top_p[token_to_request_index]

        # Temperature scale. `temperature` is a float32 tensor, so `bf16 logits /
        # temperature` promotes `scaled` to fp32 -- the softmax / nucleus math must
        # run in fp32 (a bf16 softmax over the vocab loses precision in exactly the
        # tail region top-p depends on). The assert pins that guarantee.
        temperature = temperature.clamp(min=MIN_SAMPLING_TEMPERATURE)
        if gather_indices is None:
            scaled = logits[:n] / temperature.unsqueeze(1)
        else:
            scaled = logits[gather_indices[:n], :] / temperature.unsqueeze(1)
        assert scaled.dtype == torch.float32, f"sampling math must be fp32, got {scaled.dtype}"

        # SamplingParams defines top_k=1 as greedy, matching the torch backend's
        # argmax fast path. FlashInfer's top-k kernels can retain multiple tokens
        # when logits tie at the kth threshold and then sample among them. Besides
        # violating greedy semantics, that makes output depend on unrelated RNG
        # consumption (for example, a different prefix-cache graph shape).
        # The common unconstrained path already tells us top-k is absent. Avoid
        # adding either CPU metadata scans or GPU pointwise work to that path.
        has_greedy_rows, all_greedy_rows = (
            (False, False) if no_top_k else _greedy_batch_flags(context)
        )
        if all_greedy_rows:
            sampled_tokens = torch.argmax(scaled, dim=-1)
            if output is None:
                return sampled_tokens
            output.copy_(sampled_tokens)
            return output

        # `no_top_k` / `no_top_p` are the caller-supplied batch-level dispatch flags:
        # a filter is absent only when NO active request uses it. Per-row sentinels
        # disable a filter for a row (top_k=vocab keeps all tokens, top_p=1.0 keeps
        # the full mass). Every kernel gets `self._rng` so sampling is seeded and its
        # philox offset advances per launch.
        if no_top_k and no_top_p:
            # No filtering: sample the temperature-scaled dist with the Gumbel-race logits kernel.
            sampled_tokens = flashinfer.sampling.sampling_from_logits(
                scaled, deterministic=True, generator=self._rng
            ).long()
        elif no_top_k:
            # Top-p only -> dedicated exact nucleus kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
            sampled_tokens = flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p_safe, deterministic=True, generator=self._rng
            ).long()
        elif no_top_p:
            # Top-k only -> dedicated exact top-k kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), self._vocab_size)
            sampled_tokens = flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k_safe, deterministic=True, generator=self._rng
            ).long()
        else:
            # Mixed batch (some top-k, some top-p, or requests using both) -> joint
            # kernel, fed the temperature-scaled logits.
            top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), self._vocab_size)
            top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
            sampled_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                scaled, top_k_safe, top_p_safe, deterministic=True, generator=self._rng
            ).long()

        # Mixed batches still use FlashInfer for stochastic rows, then repair
        # greedy rows with deterministic first-argmax tie breaking.
        if has_greedy_rows:
            greedy_mask = (top_k == 1) & is_no_op_top_p(top_p)
            sampled_tokens = torch.where(greedy_mask, torch.argmax(scaled, dim=-1), sampled_tokens)

        if output is None:
            return sampled_tokens
        output.copy_(sampled_tokens)
        return output

    def log_probs_kernel(
        self, logits: Tensor, context, *, token_to_request_index: Optional[Tensor] = None
    ) -> Tensor:
        """Per-row log-probs of the FlashInfer top-k / top-p sampling distribution.

        Args:
            logits (Tensor): Raw logits with shape `[num_rows, vocab_size]`.
            context: Active dynamic inference context providing GPU sampling metadata.
            token_to_request_index (Optional[Tensor]): Optional mapping from each
                logits row to its request index.

        Returns:
            Tensor: Per-row log probabilities for the processed distribution.
        """
        gpu_view = context.gpu_view
        if token_to_request_index is None:
            num_rows = logits.size(0)
            temperature = gpu_view.temperature[:num_rows]
            top_k = gpu_view.top_k[:num_rows]
            top_p = gpu_view.top_p[:num_rows]
        else:
            token_to_request_index = token_to_request_index.to(logits.device, non_blocking=True)
            temperature = gpu_view.temperature[token_to_request_index]
            top_k = gpu_view.top_k[token_to_request_index]
            top_p = gpu_view.top_p[token_to_request_index]

        temperature = temperature.clamp(min=MIN_SAMPLING_TEMPERATURE)
        scaled = logits / temperature.unsqueeze(1)

        # Batch-level no-op check. This is also the overwhelmingly common
        # production path, so return before checking for greedy rows.
        no_top_k_batch, no_top_p_batch = context.active_sampling_filter_flags()
        if no_top_k_batch and no_top_p_batch:
            return torch.log_softmax(scaled, dim=-1)

        has_greedy_rows, all_greedy_rows = (
            (False, False) if no_top_k_batch else _greedy_batch_flags(context)
        )
        if all_greedy_rows:
            greedy_log_probs = torch.full_like(scaled, float("-inf"))
            greedy_log_probs.scatter_(1, torch.argmax(scaled, dim=-1, keepdim=True), 0.0)
            return greedy_log_probs

        # Sentinel / no-op values disable filtering:
        # top_k=vocab_size keeps all tokens, top_p=1.0 keeps the full probability mass.
        no_top_k = is_no_op_top_k(top_k) | (top_k >= self._vocab_size)
        no_top_p = is_no_op_top_p(top_p)
        top_k_safe = top_k.masked_fill(no_top_k, self._vocab_size)
        top_p_safe = top_p.masked_fill(no_top_p, 1.0)

        probs = torch.softmax(scaled, dim=-1)
        # Renormalize to the kept set (top-k first, then top-p) to match
        renormed = flashinfer.sampling.top_k_renorm_probs(probs, top_k_safe)
        renormed = flashinfer.sampling.top_p_renorm_probs(renormed, top_p_safe)
        # Unfiltered rows of a mixed batch bypass the renorm rounding entirely.
        log_probs = torch.where(
            (no_top_k & no_top_p).unsqueeze(1),
            torch.log_softmax(scaled, dim=-1),
            torch.log(renormed),
        )
        if has_greedy_rows:
            greedy_mask = (top_k == 1) & is_no_op_top_p(top_p)
            greedy_log_probs = torch.full_like(scaled, float("-inf"))
            greedy_log_probs.scatter_(1, torch.argmax(scaled, dim=-1, keepdim=True), 0.0)
            log_probs = torch.where(greedy_mask.unsqueeze(1), greedy_log_probs, log_probs)
        return log_probs
