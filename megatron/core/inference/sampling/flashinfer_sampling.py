# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Any, Optional

import torch
from torch import Tensor

try:
    import flashinfer
except ImportError:
    flashinfer = None

from megatron.core.inference.sampling.base import Sampling
from megatron.core.transformer.cuda_graphs import CudaGraphManager


class FlashInferSampling(Sampling):
    """Fused FlashInfer sampling, with optional CUDA graph capture/replay."""

    def __init__(
        self, vocab_size: int, rng: torch.Generator, config=None, enable_cuda_graph: bool = False
    ) -> None:
        self._vocab_size = vocab_size
        self._rng = rng
        if enable_cuda_graph and config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config,
                self,
                function_name="sample_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config,
                self,
                function_name="sample_speculative",
                need_backward=False,
                inline_capture=True,
            )

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
        """FlashInfer fused top-k / top-p sampling kernel.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, sampling parameters are gathered per-token
                rather than per-request (used by the speculative path).
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            Sampled token ids of shape `[n]`. Under CUDA graph replay, this is a static buffer.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        # Read GPU sampling parameters directly from `context.active_request_metadata`.
        md = context.active_request_metadata
        if token_to_request_index is None:
            temperature = md["temperature"][:n]
            top_k = md["top_k"][:n]
            top_p = md["top_p"][:n]
        else:
            temperature = md["temperature"][token_to_request_index]
            top_k = md["top_k"][token_to_request_index]
            top_p = md["top_p"][token_to_request_index]

        # Clamp temperature to avoid division by 0.
        temperature = temperature.clamp(min=1e-6)
        if gather_indices is None:
            scaled = logits[:n] / temperature.unsqueeze(1)
        else:
            scaled = logits[gather_indices[:n], :] / temperature.unsqueeze(1)
        probs = torch.softmax(scaled, dim=-1)

        # Sentinel values disable filtering:
        # top_k=vocab_size keeps all tokens, top_p=1.0 keeps the full probability mass.
        # TODO: Consider changing the disable flags in the `InferenceRequest`.
        top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
        top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
        output = torch.empty(n, device=logits.device, dtype=torch.int64)
        output.copy_(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k_safe, top_p_safe, generator=self._rng
            )
        )
        return output
