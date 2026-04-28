# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import functools
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.transformer.cuda_graphs import CudaGraphManager

try:
    from flashinfer import top_k as _flashinfer_top_k

    def _topk(values: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        return _flashinfer_top_k(values, k=k, sorted=False)

except ImportError:

    def _topk(values: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        return torch.topk(values, k=k, dim=-1, sorted=False)


class LogProbsDecode:
    """Log probability computation for decode-only steps.

    Kernel methods are static so they can be called directly for eager (graph-unaware) computation.
    Instance methods wrap the kernels in CUDA-graph capture/replay.
    """

    def __init__(self, config=None):
        if config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config, self,
                function_name="indexing_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config, self,
                function_name="softmax_kernel",
                need_backward=False,
                inline_capture=True,
            )

    @staticmethod
    def indexing_kernel(context) -> Tuple[Tensor, Tensor]:
        """Select which requests need decode log probs.

        Args:
            context: The active DynamicInferenceContext.

        Returns:
            padded (request_indices, padded_arange)
        """
        padded_count = context.padded_active_request_count
        request_indices = torch.nonzero_static(
            context.active_request_metadata["return_log_probs"][:padded_count], size=padded_count
        ).squeeze(1)
        padded_arange = torch.arange(padded_count, device=request_indices.device)
        return request_indices, padded_arange

    @staticmethod
    def softmax_kernel(
        logits: Tensor, new_tokens: Tensor, request_indices: Tensor, padded_arange: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """Gather logits, compute decode log probs and per-row lse.

        Args:
            logits: shape [1, total_active_requests, vocab_size]
            new_tokens: shape [total_active_requests]
            request_indices: shape [padded_active_requests]
            padded_arange: shape [padded_active_requests]

        Returns:
            (selected_log_probs, lse)
        """
        selected_tokens = new_tokens[request_indices]
        selected_logits = logits.squeeze(0)[request_indices].float()
        lse = torch.logsumexp(selected_logits, dim=-1)
        selected_log_probs = selected_logits[padded_arange, selected_tokens] - lse
        return selected_log_probs, lse

    @staticmethod
    def extract(
        context,
        request_indices: Tensor,
        selected_log_probs: Tensor,
        lse: Tensor,
        log_prob_request_count: int,
        active_request_count: int,
        logits: Optional[Tensor] = None,
        top_n_max: int = 0,
    ) -> Tuple[List[Optional[List[float]]], Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]]:
        """Extract decode log-prob kernel outputs into a per-request list.

        Args:
            context: The active DynamicInferenceContext.
            request_indices (Tensor): Padded indices from indexing_kernel.
            selected_log_probs (Tensor): Log probs from softmax_kernel.
            lse (Tensor): Per-row log-sum-exp from softmax_kernel.
            log_prob_request_count (int): Number of real (non-padding) requests wanting log probs.
            active_request_count (int): Total number of active requests.
            logits (Optional[Tensor]): Raw logits from the forward pass (for top-n re-gather).
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            (per_request_log_probs, top_n_dict) tuple.
        """
        req_idx_list = request_indices[:log_prob_request_count].tolist()
        lp_list = selected_log_probs[:log_prob_request_count].tolist()
        result: List[Optional[List[float]]] = [None] * active_request_count
        for i, req_idx in enumerate(req_idx_list):
            result[req_idx] = [lp_list[i]]

        top_n_dict = None
        if top_n_max > 0 and logits is not None:
            raw = logits.squeeze(0)[request_indices[:log_prob_request_count]].float()
            top_n_v, top_n_i = _topk(raw, k=top_n_max)
            top_n_v = top_n_v - lse[:log_prob_request_count].unsqueeze(-1)
            top_n_dict = LogProbsDecode._build_top_n_dict(context, req_idx_list, top_n_v, top_n_i)
        return result, top_n_dict

    @staticmethod
    def _build_top_n_dict(
        context, req_idx_list: List[int], top_n_values: Tensor, top_n_indices: Tensor
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Build per-request top-n dict for decode mode.

        Args:
            context: The active DynamicInferenceContext.
            req_idx_list (List[int]): Request indices that wanted log probs.
            top_n_values (Tensor): Top-n log-prob values on GPU.
            top_n_indices (Tensor): Top-n token indices on GPU.
        """
        active_request_count = context.total_request_count - context.paused_request_count
        top_n_per_req: List[int] = context.active_request_metadata["top_n_logprobs"][
            :active_request_count
        ].tolist()
        top_n_v_cpu = top_n_values.cpu()
        top_n_i_cpu = top_n_indices.cpu()
        result: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
        for i, req_idx in enumerate(req_idx_list):
            n = top_n_per_req[req_idx]
            if n > 0:
                result[req_idx] = [(top_n_v_cpu[i, :n], top_n_i_cpu[i, :n])]
        return result if result else None

    # -- public API --

    def indexing(self, context, *, eager: bool = False) -> None:
        """Run indexing kernel with optional CUDA graph capture/replay."""
        key = ("decode_idx", context.padded_batch_dimensions)
        self._ri, self._padded_arange = self.indexing_kernel(
            context, eager=eager, cache_key=key,
        )

    def calculate(
        self,
        context,
        logits: Tensor,
        new_tokens: Tensor,
        log_prob_request_count: int,
        *,
        eager: bool = False,
        top_n_max: int = 0,
    ):
        """Run softmax kernel and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, padded_active_requests, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            eager (bool): If True, skip CUDA graph capture/replay.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        ri, padded_arange = self._ri, self._padded_arange
        key = ("decode_sm", context.padded_batch_dimensions)
        slp, lse = self.softmax_kernel(
            logits, new_tokens, ri, padded_arange, eager=eager, cache_key=key,
        )
        active_request_count = context.total_request_count - context.paused_request_count
        return functools.partial(
            self.extract,
            context,
            ri,
            slp,
            lse,
            log_prob_request_count,
            active_request_count,
            logits=logits,
            top_n_max=top_n_max,
        )
