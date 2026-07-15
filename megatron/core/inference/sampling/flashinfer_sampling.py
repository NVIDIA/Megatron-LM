# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Optional

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

from megatron.core.inference.sampling.base import Sampling


class FlashInferSampling(Sampling):
    """FlashInfer sampling with per-step top-p-only / top-k-only / joint dispatch.

    Each step selects a kernel from the batch's active filters: the dedicated exact
    top-p or top-k kernel when only one filter is in use, and the joint kernel only
    for genuinely mixed batches. The dispatch flags are read from the pinned CPU
    sampling metadata, so evaluating them costs no GPU sync.

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
                pinned CPU sampling metadata (the controller's
                `_active_requests_sampling_filter_flags`).
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, sampling parameters are gathered
                per-token rather than per-request (speculative decoding path).
            eager, cache_key: Accepted for API symmetry; ignored (no CUDA graph).

        Returns:
            Sampled token ids of shape `[n]`.
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
        temperature = temperature.clamp(min=1e-6)
        if gather_indices is None:
            scaled = logits[:n] / temperature.unsqueeze(1)
        else:
            scaled = logits[gather_indices[:n], :] / temperature.unsqueeze(1)
        assert scaled.dtype == torch.float32, f"sampling math must be fp32, got {scaled.dtype}"

        # `no_top_k` / `no_top_p` are the caller-supplied batch-level dispatch flags:
        # a filter is absent only when NO active request uses it. Per-row sentinels
        # disable a filter for a row (top_k=vocab keeps all tokens, top_p=1.0 keeps
        # the full mass). Every kernel gets `self._rng` so sampling is seeded and its
        # philox offset advances per launch.
        if no_top_k and no_top_p:
            # No nucleus / top-k filtering: sample the full temperature-scaled
            # distribution. Use FlashInfer's kernel rather than torch.multinomial:
            # multinomial forces a device-to-host sync, whereas sampling_from_probs
            # stays on-device and keeps the RNG's philox offset advancing per launch.
            probs = torch.softmax(scaled, dim=-1)
            return flashinfer.sampling.sampling_from_probs(
                probs, deterministic=True, generator=self._rng
            )
        elif no_top_k:
            # Top-p only -> dedicated exact nucleus kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
            return flashinfer.sampling.top_p_sampling_from_probs(
                probs, top_p_safe, deterministic=True, generator=self._rng
            )
        elif no_top_p:
            # Top-k only -> dedicated exact top-k kernel.
            probs = torch.softmax(scaled, dim=-1)
            top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
            return flashinfer.sampling.top_k_sampling_from_probs(
                probs, top_k_safe, deterministic=True, generator=self._rng
            )
        else:
            # Mixed batch (some top-k, some top-p, or requests using both) -> joint
            # kernel, fed the temperature-scaled logits.
            top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
            top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
            return flashinfer.sampling.top_k_top_p_sampling_from_logits(
                scaled, top_k_safe, top_p_safe, deterministic=True, generator=self._rng
            )

    def log_probs_kernel(
        self, logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor
    ) -> Tensor:
        """Per-row log-probs of the FlashInfer top-k / top-p sampling distribution."""
        temperature = temperature.clamp(min=1e-6)
        probs = torch.softmax(logits / temperature.unsqueeze(1), dim=-1)

        # Sentinel values disable filtering:
        # top_k=vocab_size keeps all tokens, top_p=1.0 keeps the full probability mass.
        top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
        top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)

        # Renormalize to the kept set (top-k first, then top-p) to match
        renormed = flashinfer.sampling.top_k_renorm_probs(probs, top_k_safe)
        renormed = flashinfer.sampling.top_p_renorm_probs(renormed, top_p_safe)
        return torch.log(renormed)
