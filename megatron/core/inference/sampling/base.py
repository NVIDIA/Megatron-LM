# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling backends.

    Subclasses implement `pre_forward_bookkeeping` (per-step setup) and `sample_kernel`
    (the GPU kernel). CUDA graph wrapping, when applicable, is added by the subclass via
    `CudaGraphManager`. The wrapper consumes `eager` and `cache_key` kwargs; concrete
    subclasses without a wrapper still accept and ignore them.
    """

    @abstractmethod
    def pre_forward_bookkeeping(self, context) -> None:
        """Prepare sampling state before the forward pass."""
        ...

    @abstractmethod
    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        eager: bool = False,
        cache_key: Any = None,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> None:
        """Sample n tokens from logits into `output[:n]`.

        `eager` and `cache_key` are consumed by the `CudaGraphManager` wrapper when one
        is installed; unwrapped subclasses accept and ignore them.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            output: Destination buffer for sampled token ids.
            context: The active DynamicInferenceContext.
            eager: If True, skip CUDA graph capture/replay (consumed by the wrapper).
            cache_key: Hashable key for runner lookup (consumed by the wrapper).
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling
                parameters are gathered per-token instead of per-request.
        """
        ...

    def sample(
        self,
        logits: Tensor,
        n: int,
        output: Tensor,
        context,
        *,
        eager: bool = False,
        cache_key: Any = None,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> None:
        """Sample `n` tokens, optionally with CUDA graph capture/replay."""
        self.sample_kernel(
            logits,
            n,
            output,
            context,
            eager=eager,
            cache_key=cache_key,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
        )

    def sample_speculative(
        self,
        required_logits: Tensor,
        request_in_prefill_status: Tensor,
        num_speculative_tokens: int,
        context,
    ) -> Tensor:
        """Sample tokens for the speculative-verify path.

        Decode requests contribute `1 + num_speculative_tokens` rows; prefill requests
        contribute one row. Builds the per-token request mapping and dispatches to
        `sample_kernel(eager=True)` so the call is safe inside an outer captured kernel.
        """
        repeats = torch.where(
            request_in_prefill_status == 0, 1 + num_speculative_tokens, 1
        )
        token_to_request_index = torch.repeat_interleave(
            torch.arange(
                len(request_in_prefill_status),
                device=request_in_prefill_status.device,
            ),
            repeats,
        )
        n = token_to_request_index.shape[0]
        output_tokens = torch.empty(n, device=required_logits.device, dtype=torch.int64)
        self.sample_kernel(
            required_logits,
            n,
            output_tokens,
            context,
            eager=True,
            token_to_request_index=token_to_request_index,
        )
        return output_tokens
