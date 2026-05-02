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


class LogProbsPrefill:
    """Log probability computation for prefill steps.

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
    def indexing_kernel(
        context, *, eager: bool = False, cache_key=None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build per-token indices for prefill log probs.

        Args:
            context: The active DynamicInferenceContext.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (request_indices, cu_masked_lengths, logit_indices, logit_indices_range, masked_tokens)
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        padded_count = context.padded_active_request_count
        padded_token_count = context.padded_active_token_count

        return_log_probs_mask = context.active_request_metadata["return_log_probs"][:padded_count]
        active_query_lengths = context.active_request_query_lengths
        last_token_idxs = context.active_request_last_token_idxs

        # Pick requests asking for log probs.
        # Size is padded_count + 1 so there's always at least one trailing sentinel slot
        # (filled with max_requests) used by the slack-absorbing logic below.
        request_indices = torch.nonzero_static(
            return_log_probs_mask, size=padded_count + 1, fill_value=context.max_requests
        ).squeeze(1)

        # Per-request: how many tokens contribute log probs and where their last token sits.
        # Sentinel slots index the permanent zero at the max_requests tail of these tensors.
        masked_lengths = active_query_lengths[request_indices]
        masked_ends = last_token_idxs[request_indices]

        # Slack absorbs filler rows so the kernel emits exactly padded_token_count rows
        # (graph padding + tokens of requests that didn't ask for log probs).
        # Dumping it into the trailing sentinel's masked_length makes cu_masked_lengths hit
        # padded_token_count exactly while leaving real entries untouched. Non-negative!!!
        slack = padded_token_count - masked_lengths.sum()
        masked_lengths[-1] = masked_lengths[-1] + slack

        # Build per-token logit indices via a base offset (per request) plus
        # a [0, padded_token_count) range.
        # offset_i = last_token_i - cu_i + 1 repeated masked_length_i times
        # places the right base under each of request i's slots;
        # adding the range walks across the request's tokens.
        cu_masked_lengths = masked_lengths.cumsum(0)
        logit_indices_offset = torch.repeat_interleave(
            masked_ends - cu_masked_lengths + 1, masked_lengths, output_size=padded_token_count
        )
        logit_indices_range = torch.arange(padded_token_count, device=torch.cuda.current_device())
        logit_indices = logit_indices_offset + logit_indices_range
        # Roll left by 1: logits[k] predicts token[k+1], so we want token[k+1] at slot k.
        # Each request's last slot ends up holding the wrong token (wrapped in from next request).
        # softmax_kernel overwrites those slots with the freshly sampled tokens.
        # The trailing scratch slot is the redirect target for sentinel writes (see softmax_kernel).
        masked_tokens = torch.nn.functional.pad(
            context.token_to_input_ids[logit_indices].roll(-1, 0), (0, 1)
        )

        return request_indices, cu_masked_lengths, logit_indices, logit_indices_range, masked_tokens

    @staticmethod
    def softmax_kernel(
        logits: Tensor,
        new_tokens: Tensor,
        request_indices: Tensor,
        cu_masked_lengths: Tensor,
        logit_indices: Tensor,
        logit_indices_range: Tensor,
        masked_tokens: Tensor,
        *,
        eager: bool = False,
        cache_key=None,
    ) -> Tuple[Tensor, Tensor]:
        """Insert sampled tokens, compute prefill log probs and per-row lse.

        Args:
            logits (Tensor): Raw model output logits [1, total_active_tokens, vocab_size].
            new_tokens (Tensor): Newly sampled tokens [total_active_requests].
            request_indices (Tensor): Padded request indices from indexing_kernel.
            cu_masked_lengths (Tensor): Cumulative masked lengths from indexing_kernel.
            logit_indices (Tensor): Indices into the logits tensor from indexing_kernel.
            logit_indices_range (Tensor): Arange for padded indexing from indexing_kernel.
            masked_tokens (Tensor): Token IDs to gather log probs for from indexing_kernel.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (selected_log_probs, lse)
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        # nonzero_static has filled unused entries of `request_indices` with `context.max_requests`.
        # Unselected entries in the middle have masked_length == 0 so they collide with the
        # real-last entry, and also out-of-bounds read `new_tokens` at index max_requests.
        # Redirect every sentinel write to the scratch slot at the end of `masked_tokens` and clamp.
        real_mask = request_indices < (cu_masked_lengths.shape[0] - 1)
        write_idx = torch.where(real_mask, cu_masked_lengths - 1, masked_tokens.shape[0] - 1)
        safe_indices = torch.where(real_mask, request_indices, 0)
        # One write per request_indices entry: stamp the freshly sampled token
        # into the request's last slot (overwriting the wrong rolled token).
        # Sentinels redirect to the scratch tail.
        masked_tokens[write_idx] = new_tokens[safe_indices]

        # Drop the scratch slot for the log-prob computation; per-token
        # compute uses padded_token_count rows.
        masked_tokens_real = masked_tokens[:-1]
        selected_logits = logits.squeeze(0)[logit_indices].float()
        # LSE trick (per row); see decode.py for the same pattern.
        lse = torch.logsumexp(selected_logits, dim=-1)
        selected_log_probs = selected_logits[logit_indices_range, masked_tokens_real] - lse
        return selected_log_probs, lse

    @staticmethod
    def extract(
        context,
        request_indices: Tensor,
        cu_masked_lengths: Tensor,
        token_log_probs: Tensor,
        log_prob_request_count: int,
        active_request_count: int,
        top_n_values: Optional[Tensor] = None,
        top_n_indices: Optional[Tensor] = None,
    ) -> Tuple[List[Optional[List[float]]], Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]]:
        """Extract prefill log-prob kernel outputs into a per-request list.

        Args:
            context: The active DynamicInferenceContext.
            request_indices (Tensor): Padded indices from indexing_kernel.
            cu_masked_lengths (Tensor): Cumulative masked lengths from indexing_kernel.
            token_log_probs (Tensor): Per-token log probs from softmax_kernel.
            log_prob_request_count (int): Number of real (non-padding) requests wanting log probs.
            active_request_count (int): Total number of active requests.
            top_n_values (Optional[Tensor]): Pre-computed per-token top-n log-prob values.
            top_n_indices (Optional[Tensor]): Pre-computed per-token top-n token indices.

        Returns:
            (per_request_log_probs, top_n_dict) tuple.
        """
        ri = request_indices[:log_prob_request_count]
        # Total real tokens from real requests = the last real cumulative entry.
        # The trailing sentinel rows (slack-absorbed) live past this count
        # and are dropped by the [:selected_token_count] slice below.
        selected_token_count = cu_masked_lengths[log_prob_request_count - 1].item()
        # Recover per-request lengths from the cumulative sum so we can split the log-probs.
        cu_ml_cpu = cu_masked_lengths[:log_prob_request_count].cpu()
        masked_lengths_cpu = torch.diff(cu_ml_cpu, prepend=cu_ml_cpu.new_zeros(1)).tolist()

        # Single D2H + split by per-request lengths -> per-request log-prob lists.
        per_request = token_log_probs[:selected_token_count].cpu().split(masked_lengths_cpu, dim=0)
        req_idx_list = ri.tolist()
        result: List[Optional[List[float]]] = [None] * active_request_count
        for i, req_idx in enumerate(req_idx_list):
            result[req_idx] = per_request[i].tolist()

        top_n_dict: Optional[Dict[int, List[Tuple[Tensor, Tensor]]]] = None
        if top_n_values is not None and top_n_indices is not None:
            # Top-n was computed per logit row at top_n_max columns;
            # per-request top_n_per_req[i] tells us how many to keep;
            # skip_prompt determines whether to emit per-token or only last-token.
            top_n_v_cpu = top_n_values[:selected_token_count].cpu()
            top_n_i_cpu = top_n_indices[:selected_token_count].cpu()
            top_n_per_req: List[int] = context.active_request_metadata["top_n_logprobs"][
                :active_request_count
            ].tolist()
            skip_prompt_per_req: List[bool] = context.active_request_metadata[
                "skip_prompt_log_probs"
            ][:active_request_count].tolist()
            built: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
            token_offset = 0
            for i, req_idx in enumerate(req_idx_list):
                req_len = masked_lengths_cpu[i]
                n = top_n_per_req[req_idx]
                if n > 0:
                    # skip_prompt: only the last token (= first generated one) gets top-n.
                    if skip_prompt_per_req[req_idx] and req_len > 1:
                        last = token_offset + req_len - 1
                        built[req_idx] = [(top_n_v_cpu[last, :n], top_n_i_cpu[last, :n])]
                    else:
                        built[req_idx] = [
                            (top_n_v_cpu[token_offset + j, :n], top_n_i_cpu[token_offset + j, :n])
                            for j in range(req_len)
                        ]
                token_offset += req_len
            top_n_dict = built or None
        return result, top_n_dict

    def indexing(self, context, *, eager: bool = False) -> None:
        """Run indexing kernel with optional CUDA graph capture/replay.

        Args:
            context: The active DynamicInferenceContext.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
        """
        key = ("prefill_idx", context.padded_batch_dimensions)
        result = self.indexing_kernel(context, eager=eager, cache_key=key)
        self._ri, self._cu_ml, self._li, self._li_range, self._mt = result

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
            logits (Tensor): Raw model output logits [1, padded_active_tokens, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        # Indices were stashed by the earlier `indexing` call.
        # Run softmax on the main stream (graph-captured under a per-shape key).
        ri, cu_ml, li, li_range, mt = (self._ri, self._cu_ml, self._li, self._li_range, self._mt)
        key = ("prefill_sm", context.padded_batch_dimensions)
        slp, lse = self.softmax_kernel(
            logits, new_tokens, ri, cu_ml, li, li_range, mt, eager=eager, cache_key=key
        )

        top_n_v = top_n_i = None
        if top_n_max > 0:
            # Run top-n on a side stream so it overlaps the main-stream sampling work.
            # wait_stream defers the side stream until the softmax above is queued.
            if self._topn_stream is not None:
                self._topn_stream.wait_stream(torch.cuda.current_stream())
                stream_ctx = torch.cuda.stream(self._topn_stream)
            else:
                stream_ctx = nullcontext()
            with stream_ctx:
                # Topk on all padded rows; extract slices to selected_token_count.
                # The sentinel-fill rows produce throwaway top-n that extract drops.
                # Reuse the LSE from softmax_kernel: top_n_log_probs = topk_raw - lse.
                raw = logits.squeeze(0)[li].float()
                top_n_v_raw, top_n_i = _topk(raw, k=top_n_max)
                top_n_v = top_n_v_raw - lse.unsqueeze(-1)
            if self._topn_event is not None:
                # Signals the next step's main stream that side-stream reads
                # of `_all_logits_cuda` are done.
                self._topn_event.record(self._topn_stream)

        active_request_count = context.total_request_count - context.paused_request_count
        # Defer the CPU-side extract: caller invokes the partial after step bookkeeping
        # so D2H copies pay their synchronization cost as late as possible.
        return functools.partial(
            self.extract,
            context,
            ri,
            cu_ml,
            slp,
            log_prob_request_count,
            active_request_count,
            top_n_values=top_n_v,
            top_n_indices=top_n_i,
        )
