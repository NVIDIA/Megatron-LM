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
    """Fused FlashInfer sampling, with optional CUDA graph capture/replay.

    Unlike `TorchSampling`, FlashInfer kernels accept per-row parameter tensors
    (temperature, top_k, top_p) directly, so no bucketing is required.
    """

    def __init__(
        self, vocab_size: int, rng: torch.Generator, config=None, enable_cuda_graph: bool = False
    ) -> None:
        self._vocab_size = vocab_size
        self._rng = rng
        self._enable_cuda_graph = enable_cuda_graph
        if enable_cuda_graph and config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config,
                self,
                function_name="sample_kernel",
                need_backward=False,
                inline_capture=True,
            )

    def pre_forward_bookkeeping(self, context) -> None:
        """No-op; FlashInfer needs no per-step bookkeeping."""

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
        """FlashInfer fused top-k / top-p sampling kernel.

        Reads sampling parameters per-row from `context.active_request_metadata`,
        applies temperature scaling and top-k/top-p filtering, then samples via
        `flashinfer.sampling.top_k_top_p_sampling_from_probs`.

        When wrapped by `CudaGraphManager`, `eager` and `cache_key` are consumed by
        the wrapper before this body runs. When unwrapped (no CUDA graphs), they are
        accepted and ignored so callers can pass them unconditionally.

        Returns:
            Sampled token ids of shape `[n]`. Under CUDA graph replay, this is a static buffer.
        """
        del eager, cache_key
        md = context.active_request_metadata
        if token_to_request_index is None:
            temperature = md["temperature"][:n]
            top_k = md["top_k"][:n]
            top_p = md["top_p"][:n]
        else:
            temperature = md["temperature"][token_to_request_index]
            top_k = md["top_k"][token_to_request_index]
            top_p = md["top_p"][token_to_request_index]

        if gather_indices is None:
            # Slice is a view; clone before in-place div_ to avoid mutating the caller.
            scaled = logits[:n].clone()
        else:
            # Advanced indexing already returns a new tensor.
            scaled = logits[gather_indices[:n], :]
        scaled.div_(temperature.unsqueeze(1))
        probs = torch.softmax(scaled, dim=-1)

        # Sentinel values disable filtering: top_k=vocab_size keeps all
        # tokens, top_p=1.0 keeps the full probability mass.
        top_k_safe = top_k.masked_fill(top_k == 0, self._vocab_size)
        top_p_safe = top_p.masked_fill(top_p == 0.0, 1.0)
        output = torch.empty(n, device=logits.device, dtype=torch.int64)
        output.copy_(
            flashinfer.sampling.top_k_top_p_sampling_from_probs(
                probs, top_k_safe, top_p_safe, generator=self._rng
            )
        )
        return output
