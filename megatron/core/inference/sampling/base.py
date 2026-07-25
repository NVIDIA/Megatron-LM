# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling backends.

    Subclasses implement `sample_kernel` and `log_probs_kernel`.
    CUDA graphs are added via `CudaGraphManager`.
    """

    @abstractmethod
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
        """Sample `n` tokens from `logits` and return them.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            no_top_k, no_top_p: Required batch-level dispatch flags (whether NO active
                request uses top-k / top-p). The caller computes them once from the
                pinned CPU sampling metadata (see the controller's
                `_active_requests_sampling_filter_flags`), so the kernel never has to.
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling
                parameters are gathered per-token instead of per-request.
            eager, cache_key: Accepted for API symmetry; ignored (no CUDA graph).

        Returns:
            Sampled token ids of shape `[n]`.
        """
        ...

    def sample_speculative(
        self,
        required_logits: Tensor,
        num_decode: int,
        num_prefill: int,
        num_speculative_tokens: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Sample tokens for the speculative-verify path.

        Decode requests contribute `1 + num_speculative_tokens` rows; prefill requests contribute 1.
        Builds the per-token request mapping and dispatches to `sample_kernel`.
        The `sample_kernel` is forced eager so its own `CudaGraphManager` wrapper does not fire.

        When `gather_indices` is supplied, the kernel selects via `logits[gather_indices[:n], :]`.
        When `gather_indices` is None, `required_logits` is expected to be already pre-gathered to
        the layout described above (e.g. when `materialize_only_last_token_logits=True` upstream).
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        n_spec = num_speculative_tokens
        num_decode_tokens = num_decode * (1 + n_spec)
        num_tokens = num_decode_tokens + num_prefill
        device = required_logits.device

        token_to_request_index = torch.cat(
            [
                torch.arange(num_decode, device=device).repeat_interleave(
                    1 + n_spec, output_size=num_decode_tokens
                ),
                torch.arange(num_decode, num_decode + num_prefill, device=device),
            ]
        )
        # Batch-level dispatch flags, required by `sample_kernel`. Read from the same
        # pinned CPU sampling metadata as the controller's filter flags (sync-free): a
        # filter is absent only when NO active request uses it.
        active_request_count = context.total_request_count - context.paused_request_count
        md = context.active_request_metadata
        no_top_k = bool((md["top_k"][:active_request_count] == 0).all())
        no_top_p = bool((md["top_p"][:active_request_count] == 0.0).all())
        return self.sample_kernel(
            required_logits,
            num_tokens,
            context,
            no_top_k=no_top_k,
            no_top_p=no_top_p,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
            eager=True,
        )

    @abstractmethod
    def log_probs_kernel(
        self, logits: Tensor, temperature: Tensor, top_k: Tensor, top_p: Tensor
    ) -> Tensor:
        """Per-row log-probs of the distribution this backend samples from.

        Args:
            logits: `[num_rows, vocab_size]` raw logits.
            temperature, top_k, top_p: `[num_rows]` per-row sampling params.

        Returns:
            `[num_rows, vocab_size]` log-probs; filtered-out tokens are `-inf`.
        """
        ...
