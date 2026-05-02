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

    def __init__(self, config=None, topn_stream=None, topn_event=None):
        """
        Args:
            config: Optional MegatronConfig for CUDA graph capture configuration.
            topn_stream: Optional CUDA stream for running top-n computation asynchronously.
            topn_event: Optional CUDA event to signal completion of top-n computation.
        """
        self._topn_stream = topn_stream
        self._topn_event = topn_event
        if config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config,
                self,
                function_name="indexing_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config,
                self,
                function_name="softmax_kernel",
                need_backward=False,
                inline_capture=True,
            )

    @staticmethod
    def indexing_kernel(context, *, eager: bool = False, cache_key=None) -> Tuple[Tensor, Tensor]:
        """Select which requests need decode log probs.

        Args:
            context: The active DynamicInferenceContext.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            padded (request_indices, padded_arange)
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        # Pick rows whose request asked for log probs.
        # nonzero_static keeps the output shape fixed (= padded_count) for graph capture;
        # trailing entries past log_prob_request_count are sliced off later, during extract.
        padded_count = context.padded_active_request_count
        request_indices = torch.nonzero_static(
            context.active_request_metadata["return_log_probs"][:padded_count], size=padded_count
        ).squeeze(1)
        # Row range over the gather space; used by softmax_kernel to address per-row sampled tokens.
        padded_arange = torch.arange(padded_count, device=request_indices.device)
        return request_indices, padded_arange

    @staticmethod
    def softmax_kernel(
        logits: Tensor,
        new_tokens: Tensor,
        request_indices: Tensor,
        padded_arange: Tensor,
        *,
        eager: bool = False,
        cache_key=None,
    ) -> Tuple[Tensor, Tensor]:
        """Gather logits, compute decode log probs and per-row lse.

        Args:
            logits: shape [1, total_active_requests, vocab_size]
            new_tokens: shape [total_active_requests]
            request_indices: shape [padded_active_requests]
            padded_arange: shape [padded_active_requests]
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (selected_log_probs, lse)
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        # Gather only the rows we need; the same indices select the matching sampled token per row.
        selected_tokens = new_tokens[request_indices]
        selected_logits = logits.squeeze(0)[request_indices].float()
        # LSE trick: log_prob(t) = logit(t) - logsumexp(logits).
        # Avoids materializing a full [padded, V] log_softmax tensor.
        lse = torch.logsumexp(selected_logits, dim=-1)
        selected_log_probs = selected_logits[padded_arange, selected_tokens] - lse
        return selected_log_probs, lse

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
            selected_log_probs (Tensor): Log probs from softmax_kernel.
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

    def indexing(self, context, *, eager: bool = False) -> None:
        """Run indexing kernel with optional CUDA graph capture/replay.

        Args:
            context: The active DynamicInferenceContext.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
        """
        key = ("decode_idx", context.padded_batch_dimensions)
        self._ri, self._padded_arange = self.indexing_kernel(context, eager=eager, cache_key=key)

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
        """Run softmax kernel + top-n (on side stream) and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, padded_active_requests, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        # Indices were stashed by the earlier `indexing` call.
        # Run softmax on the main stream (graph-captured under a per-shape key).
        ri, padded_arange = self._ri, self._padded_arange
        key = ("decode_sm", context.padded_batch_dimensions)
        slp, lse = self.softmax_kernel(
            logits, new_tokens, ri, padded_arange, eager=eager, cache_key=key
        )

        top_n_v = top_n_i = None
        if top_n_max > 0:
            # Run top-n on a side stream so it overlaps the main-stream
            # downstream sampling / extract work. wait_stream defers the side
            # stream until the softmax above is queued so it can read `logits`.
            if self._topn_stream is not None:
                self._topn_stream.wait_stream(torch.cuda.current_stream())
                stream_ctx = torch.cuda.stream(self._topn_stream)
            else:
                stream_ctx = nullcontext()
            with stream_ctx:
                # Reuse the LSE from softmax_kernel: top_n_log_probs = topk_raw - lse.
                raw = logits.squeeze(0)[ri].float()
                top_n_v_raw, top_n_i = _topk(raw, k=top_n_max)
                top_n_v = top_n_v_raw - lse.unsqueeze(-1)
            if self._topn_event is not None:
                # Signals the next step's main stream that side-stream reads
                # of `_all_logits_cuda` are done, so it can safely overwrite it.
                self._topn_event.record(self._topn_stream)

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
