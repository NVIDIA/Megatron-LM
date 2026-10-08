# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import hashlib
import struct
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


def _request_token_seed(seed: int, position: int) -> int:
    """Give each request/token pair its own rejection-sampling stream."""
    digest = hashlib.sha256(struct.pack("<QQ", seed, position)).digest()
    return int.from_bytes(digest[:8], "little") & ((1 << 63) - 1)


class FlashInferSampling(Sampling):
    """FlashInfer sampling with per-step unfiltered / top-p-only / top-k-only / joint dispatch.

    Each step selects a kernel from the batch's active filters: the logits kernel
    when nothing filters, the dedicated exact top-p or top-k kernel when only one filter is in use,
    and the joint kernel only for genuinely mixed batches.
    The dispatch flags are read from the pinned CPU sampling metadata.

    Explicit request seeds use one row per launch: FlashInfer includes the batch
    row in its RNG stream, so per-row seed tensors alone are not batch invariant.
    Each absolute token position gets a fresh stream with an explicit zero offset;
    these draws never advance the shared generator. Unseeded requests stay batched.

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
        sequence_lengths: Optional[Tensor] = None,
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
            sequence_lengths: Next-token positions saved with the logits for async sampling.
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

        metadata = context.active_request_metadata
        seed_metadata = metadata.get("seed")
        active_count = context.total_request_count - context.paused_request_count
        seeds = (
            [-1] * active_count if seed_metadata is None else seed_metadata[:active_count].tolist()
        )
        if any(seed >= 0 for seed in seeds):
            if token_to_request_index is not None or context.config.num_speculative_tokens:
                raise ValueError("Request-local seeds do not yet support speculative decoding")
            positions = (
                context.get_active_sequence_lengths()
                if sequence_lengths is None
                else sequence_lengths
            ).tolist()
            cpu_top_k = metadata["top_k"][:n].tolist()
            cpu_top_p = metadata["top_p"][:n].tolist()
            sampled_tokens = torch.empty(n, dtype=torch.long, device=logits.device)
            # FlashInfer includes the output row in its Philox subsequence, even
            # with per-row seeds/offsets (flashinfer-ai/flashinfer#5745). A seeded
            # request must therefore occupy row zero and choose its own kernel,
            # independent of its neighbors' filters. Keep unseeded rows batched.
            for row, seed in enumerate(seeds):
                if seed < 0:
                    continue
                # Rejection sampling consumes a variable number of draws. Derive
                # a distinct stream per absolute token position instead of using
                # adjacent offsets, which could reuse draws between tokens.
                sampled_tokens[row : row + 1] = self._sample(
                    scaled[row : row + 1],
                    top_k[row : row + 1],
                    top_p[row : row + 1],
                    no_top_k=is_no_op_top_k(cpu_top_k[row]),
                    no_top_p=is_no_op_top_p(cpu_top_p[row]),
                    seed=_request_token_seed(seed, positions[row]),
                    offset=0,
                )
            unseeded = [row for row, seed in enumerate(seeds) if seed < 0]
            if unseeded:
                indices = torch.tensor(unseeded, device=logits.device)
                sampled_tokens[indices] = self._sample(
                    scaled[indices],
                    top_k[indices],
                    top_p[indices],
                    no_top_k=all(is_no_op_top_k(cpu_top_k[row]) for row in unseeded),
                    no_top_p=all(is_no_op_top_p(cpu_top_p[row]) for row in unseeded),
                    generator=self._rng,
                )
        else:
            sampled_tokens = self._sample(
                scaled, top_k, top_p, no_top_k=no_top_k, no_top_p=no_top_p, generator=self._rng
            )

        if output is None:
            return sampled_tokens
        output.copy_(sampled_tokens)
        return output

    def _sample(self, scaled, top_k, top_p, *, no_top_k, no_top_p, **rng_kwargs):
        """Dispatch with either an explicit seed/offset or the shared generator."""
        if no_top_k and no_top_p:
            # No filtering: sample the temperature-scaled dist with the Gumbel-race logits kernel.
            sampled_tokens = flashinfer.sampling.sampling_from_logits(
                scaled, deterministic=True, **rng_kwargs
            ).long()
        elif no_top_k:
            # Top-p only -> dedicated exact nucleus kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
            sampled_tokens = flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p_safe, deterministic=True, **rng_kwargs
            ).long()
        elif no_top_p:
            # Top-k only -> dedicated exact top-k kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), self._vocab_size)
            sampled_tokens = flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k_safe, deterministic=True, **rng_kwargs
            ).long()
        else:
            # Mixed batch (some top-k, some top-p, or requests using both) -> joint
            # kernel, fed the temperature-scaled logits.
            top_k_safe = top_k.masked_fill(is_no_op_top_k(top_k), self._vocab_size)
            top_p_safe = top_p.masked_fill(is_no_op_top_p(top_p), 1.0)
            sampled_tokens = flashinfer.sampling.top_k_top_p_sampling_from_logits(
                scaled, top_k_safe, top_p_safe, deterministic=True, **rng_kwargs
            ).long()

        return sampled_tokens

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

        # Batch-level no-op check.
        no_top_k_batch, no_top_p_batch = context.active_sampling_filter_flags()
        if no_top_k_batch and no_top_p_batch:
            return torch.log_softmax(scaled, dim=-1)

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
        return torch.where(
            (no_top_k & no_top_p).unsqueeze(1),
            torch.log_softmax(scaled, dim=-1),
            torch.log(renormed),
        )
