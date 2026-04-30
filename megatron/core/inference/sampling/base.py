# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling backends.

    Subclasses implement `sample_kernel` (the GPU kernel). CUDA graph wrapping, when
    applicable, is added by the subclass via `CudaGraphManager`. The wrapper consumes
    `eager` and `cache_key` kwargs; concrete subclasses without a wrapper still accept
    and ignore them.
    """

    @abstractmethod
    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        eager: bool = False,
        cache_key: Any = None,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample `n` tokens from `logits` and return them.

        `eager` and `cache_key` are consumed by the `CudaGraphManager` wrapper when one
        is installed; unwrapped subclasses accept and ignore them.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay (consumed by the wrapper).
            cache_key: Hashable key for runner lookup (consumed by the wrapper).
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling
                parameters are gathered per-token instead of per-request.

        Returns:
            Sampled token ids of shape `[n]`. Under CUDA graph replay, this is a static buffer.
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
        eager: bool = False,
        cache_key: Any = None,
        gather_indices: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample tokens for the speculative-verify path.

        Decode requests contribute `1 + num_speculative_tokens` rows; prefill requests contribute 1.
        Builds the per-token request mapping and dispatches to `sample_kernel`.

        When `gather_indices` is supplied, `required_logits` is the full per-token logits buffer
        (constant shape across steps); the kernel selects rows via `logits[gather_indices[:n], :]`.
        When `gather_indices` is None, `required_logits` is expected to be already pre-gathered to
        the layout described above (e.g. when `materialize_only_last_token_logits=True` upstream).
        """
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
        return self.sample_kernel(
            required_logits,
            num_tokens,
            context,
            eager=eager,
            cache_key=cache_key,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
        )
