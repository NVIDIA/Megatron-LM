# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import functools
from contextlib import nullcontext
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

    def __init__(
        self,
        config=None,
        *,
        side_stream: Optional[torch.cuda.Stream] = None,
        side_pool_anchor: Optional[CudaGraphManager] = None,
    ):
        """
        Args:
            config: Optional MegatronConfig for CUDA graph capture configuration.
            side_stream: Optional CUDA stream the controller hosts indexing, LSE, and top-n on.
            side_pool_anchor: Optional anchor whose mempool the side-stream graphs share.
        """
        self._side_stream = side_stream
        self.side_pool_anchor: Optional[CudaGraphManager] = None
        if config is not None and config.cuda_graph_impl == "local":
            self.side_pool_anchor = CudaGraphManager(
                config,
                self,
                function_name="indexing_kernel",
                need_backward=False,
                inline_capture=True,
                **(
                    {"share_mempool_with": side_pool_anchor}
                    if side_pool_anchor is not None
                    else {"new_mempool": True}
                ),
            )
            CudaGraphManager(
                config,
                self,
                function_name="lse_kernel",
                need_backward=False,
                inline_capture=True,
                share_mempool_with=side_pool_anchor or self.side_pool_anchor,
            )
            CudaGraphManager(
                config,
                self,
                function_name="gather_kernel",
                need_backward=False,
                inline_capture=True,
                share_mempool_with=side_pool_anchor or self.side_pool_anchor,
            )

    @staticmethod
    def indexing_kernel(context, *, eager: bool = False, cache_key=None) -> Tensor:
        """Select which requests need decode log probs.

        Args:
            context: The active DynamicInferenceContext.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            request_indices [padded_active_requests]
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        # Pick rows whose request asked for log probs.
        # nonzero_static keeps the output shape fixed (= padded_count) for graph capture;
        # trailing entries past log_prob_request_count are sliced off later, during extract.
        padded_count = context.padded_active_request_count
        return torch.nonzero_static(
            context.gpu_return_log_probs_mask[:padded_count], size=padded_count
        ).squeeze(1)

    @staticmethod
    def lse_kernel(
        logits: Tensor, request_indices: Tensor, *, eager: bool = False, cache_key=None
    ) -> Tensor:
        """Per-row logsumexp over the rows of `logits` selected by `request_indices`.

        Args:
            logits: shape [1, total_active_requests, vocab_size].
            request_indices: shape [padded_active_requests].
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            lse [padded_active_requests]
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        selected_logits = logits.squeeze(0)[request_indices].float()
        return torch.logsumexp(selected_logits, dim=-1)

    @staticmethod
    def gather_kernel(
        logits: Tensor,
        new_tokens: Tensor,
        request_indices: Tensor,
        lse: Tensor,
        *,
        eager: bool = False,
        cache_key=None,
    ) -> Tensor:
        """Gather the per-row logit at the sampled token position; subtract LSE.

        Args:
            logits: shape [1, total_active_requests, vocab_size].
            new_tokens: shape [max_requests].
            request_indices: shape [padded_active_requests].
            lse: shape [padded_active_requests], from `lse_kernel`.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            selected_log_probs [padded_active_requests]
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        selected_tokens = new_tokens[request_indices]
        # Advanced-indexing per-row gather: logit at (request_indices[i], selected_tokens[i]).
        selected_logit = logits.squeeze(0)[request_indices, selected_tokens].float()
        return selected_logit - lse

    @staticmethod
    def extract(
        context,
        request_indices: Tensor,
        selected_log_probs: Tensor,
        log_prob_request_count: int,
        active_request_count: int,
        top_n_values: Optional[Tensor] = None,
        top_n_indices: Optional[Tensor] = None,
    ) -> Tuple[List[Optional[List[float]]], Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]]:
        """Extract decode log-prob kernel outputs into a per-request list.

        Args:
            context: The active DynamicInferenceContext.
            request_indices (Tensor): Padded indices from indexing_kernel.
            selected_log_probs (Tensor): Log probs from gather_kernel.
            log_prob_request_count (int): Number of real (non-padding) requests wanting log probs.
            active_request_count (int): Total number of active requests.
            top_n_values (Optional[Tensor]): Pre-computed top-n log-prob values.
            top_n_indices (Optional[Tensor]): Pre-computed top-n token indices.

        Returns:
            (per_request_log_probs, top_n_dict) tuple.
        """
        # Single D2H per tensor; trailing entries past log_prob_request_count are filler.
        req_idx_list = request_indices[:log_prob_request_count].tolist()
        lp_list = selected_log_probs[:log_prob_request_count].tolist()
        # Pre-fill with None; only requests that asked for log probs are populated.
        result: List[Optional[List[float]]] = [None] * active_request_count
        for i, req_idx in enumerate(req_idx_list):
            result[req_idx] = [lp_list[i]]

        top_n_dict: Optional[Dict[int, List[Tuple[Tensor, Tensor]]]] = None
        if top_n_values is not None and top_n_indices is not None:
            # Top-n was computed at top_n_max columns for every requesting row;
            # per-request top_n_per_req[i] tells us how many to actually return.
            top_n_v_cpu = top_n_values[:log_prob_request_count].cpu()
            top_n_i_cpu = top_n_indices[:log_prob_request_count].cpu()
            top_n_per_req: List[int] = context.active_request_metadata["top_n_logprobs"][
                :active_request_count
            ].tolist()
            built: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
            for i, req_idx in enumerate(req_idx_list):
                n = top_n_per_req[req_idx]
                if n > 0:
                    built[req_idx] = [(top_n_v_cpu[i, :n], top_n_i_cpu[i, :n])]
            top_n_dict = built or None
        return result, top_n_dict

    def indexing(self, context, *, eager: bool = False) -> Tensor:
        """Run indexing kernel with optional CUDA graph capture/replay."""
        key = ("decode_idx", context.padded_batch_dimensions)
        return self.indexing_kernel(context, eager=eager, cache_key=key)

    def lse(self, context, logits: Tensor, ri: Tensor, *, eager: bool = False) -> Tensor:
        """Run LSE precompute on the side stream so it overlaps with sampling."""
        key = ("decode_lse", context.padded_batch_dimensions)
        return self.lse_kernel(logits, ri, eager=eager, cache_key=key)

    def calculate(
        self,
        context,
        logits: Tensor,
        new_tokens: Tensor,
        ri: Tensor,
        lse: Tensor,
        log_prob_request_count: int,
        *,
        eager: bool = False,
        top_n_max: int = 0,
    ):
        """Run gather kernel + top-n and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, padded_active_requests, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            ri (Tensor): Output of `indexing`.
            lse (Tensor): Output of `lse`.
            log_prob_request_count (int): Number of requests wanting log probs.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        # Gather is cheap (per-row scalar lookup + subtract), runs on the main stream.
        key = ("decode_gather", context.padded_batch_dimensions)
        slp = self.gather_kernel(logits, new_tokens, ri, lse, eager=eager, cache_key=key)

        top_n_v = top_n_i = None
        if top_n_max > 0:
            # Top-n on the side stream. `lse` is side-stream-resident (lse_kernel),
            # `ri` is too (indexing_kernel), and `logits` is the long-lived
            # `_all_logits_cuda` buffer — no cross-stream sync needed: side FIFO
            # already ensures `indexing → lse → topn`, and that chain inherits the
            # earlier `wait_stream(main)` that gated `lse` on forward.
            stream_ctx = (
                torch.cuda.stream(self._side_stream)
                if self._side_stream is not None
                else nullcontext()
            )
            with stream_ctx:
                # top_n_log_probs = topk(raw_logits) - lse.
                raw = logits.squeeze(0)[ri].float()
                top_n_v_raw, top_n_i = _topk(raw, k=top_n_max)
                top_n_v = top_n_v_raw - lse.unsqueeze(-1)

        active_request_count = context.total_request_count - context.paused_request_count
        # Defer the CPU-side extract: caller invokes the partial after step bookkeeping
        # so D2H copies pay their synchronization cost as late as possible.
        return functools.partial(
            self.extract,
            context,
            ri,
            slp,
            log_prob_request_count,
            active_request_count,
            top_n_values=top_n_v,
            top_n_indices=top_n_i,
        )
