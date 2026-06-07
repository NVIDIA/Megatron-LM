# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch import Tensor


class Sampling(ABC):
    """Abstract base for inference sampling backends.

    Subclasses implement `sample_kernel`. CUDA graphs are added via `CudaGraphManager`.
    """

    @abstractmethod
    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
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
            gather_indices: If provided, only sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: Per-token request mapping; when set, sampling
                parameters are gathered per-token instead of per-request.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

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
        return self.sample_kernel(
            required_logits,
            num_tokens,
            context,
            gather_indices=gather_indices,
            token_to_request_index=token_to_request_index,
            eager=True,
        )
