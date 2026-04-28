# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import functools
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.cuda_graphs import CudaGraphManager


class LogProbsPrefill:
    """Log probability computation for prefill steps.

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
    def indexing_kernel(context) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build per-token indices for prefill log probs.

        Args:
            context: The active DynamicInferenceContext.

        Returns:
            (request_indices, cu_masked_lengths, logit_indices, logit_indices_range, masked_tokens)
        """
        padded_count = context.padded_active_request_count
        padded_token_count = context.padded_active_token_count

        return_log_probs_mask = context.active_request_metadata["return_log_probs"][:padded_count]
        active_query_lengths = context.active_request_query_lengths
        last_token_idxs = context.active_request_last_token_idxs

        request_indices = torch.nonzero_static(
            return_log_probs_mask, size=padded_count + 1, fill_value=context.max_requests
        ).squeeze(1)

        masked_lengths = active_query_lengths[request_indices]
        masked_ends = last_token_idxs[request_indices]

        # Slack is non-negative because padded_token_count >= sum_real.
        slack = padded_token_count - masked_lengths.sum()
        masked_lengths[-1] = masked_lengths[-1] + slack

        cu_masked_lengths = masked_lengths.cumsum(0)
        logit_indices_offset = torch.repeat_interleave(
            masked_ends - cu_masked_lengths + 1, masked_lengths, output_size=padded_token_count
        )
        logit_indices_range = torch.arange(padded_token_count, device=torch.cuda.current_device())
        logit_indices = logit_indices_offset + logit_indices_range
        # Roll by 1 because the newly-generated tokens are not present yet.
        masked_tokens = context.token_to_input_ids[logit_indices].roll(-1, 0)

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
    ) -> Tensor:
        """Insert sampled tokens and compute prefill log probs.

        Args:
            logits (Tensor): Raw model output logits [1, total_active_tokens, vocab_size].
            new_tokens (Tensor): Newly sampled tokens [total_active_requests].
            request_indices (Tensor): Padded request indices from indexing_kernel.
            cu_masked_lengths (Tensor): Cumulative masked lengths from indexing_kernel.
            logit_indices (Tensor): Indices into the logits tensor from indexing_kernel.
            logit_indices_range (Tensor): Arange for padded indexing from indexing_kernel.
            masked_tokens (Tensor): Token IDs to gather log probs for from indexing_kernel.

        Returns:
            selected_log_probs tensor.
        """
        # Insert the newly-generated tokens at request boundaries.
        masked_tokens[cu_masked_lengths - 1] = new_tokens[request_indices]
        selected_logits = logits.squeeze(0)[logit_indices].float()
        log_softmax_result = F.log_softmax(selected_logits, dim=-1)
        selected_log_probs = log_softmax_result[logit_indices_range, masked_tokens]
        return selected_log_probs

    @staticmethod
    def extract(
        context,
        request_indices: Tensor,
        cu_masked_lengths: Tensor,
        token_log_probs: Tensor,
        log_prob_request_count: int,
        active_request_count: int,
        top_n_max: int = 0,
        logits: Optional[Tensor] = None,
        logit_indices: Optional[Tensor] = None,
    ) -> Tuple[List[Optional[List[float]]], Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]]:
        """Extract prefill log-prob kernel outputs into a per-request list.

        Args:
            context: The active DynamicInferenceContext.
            request_indices (Tensor): Padded indices from indexing_kernel.
            cu_masked_lengths (Tensor): Cumulative masked lengths from indexing_kernel.
            token_log_probs (Tensor): Per-token log probs from softmax_kernel.
            log_prob_request_count (int): Number of real (non-padding) requests wanting log probs.
            active_request_count (int): Total number of active requests.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).
            logits (Optional[Tensor]): Raw logits for top-n computation.
            logit_indices (Optional[Tensor]): Logit indices for top-n computation.

        Returns:
            (per_request_log_probs, top_n_dict) tuple.
        """
        ri = request_indices[:log_prob_request_count]
        selected_token_count = cu_masked_lengths[log_prob_request_count - 1].item()
        # Recover per-request lengths from the cumulative sum.
        cu_ml_cpu = cu_masked_lengths[:log_prob_request_count].cpu()
        masked_lengths_cpu = torch.diff(cu_ml_cpu, prepend=cu_ml_cpu.new_zeros(1)).tolist()

        per_request = token_log_probs[:selected_token_count].cpu().split(masked_lengths_cpu, dim=0)
        req_idx_list = ri.tolist()
        result: List[Optional[List[float]]] = [None] * active_request_count
        for i, req_idx in enumerate(req_idx_list):
            result[req_idx] = per_request[i].tolist()

        top_n_dict = None
        if top_n_max > 0 and logits is not None and logit_indices is not None:
            raw = logits.squeeze(0)[logit_indices[:selected_token_count]].float()
            lse = torch.logsumexp(raw, dim=-1, keepdim=True)
            top_n_v, top_n_i = torch.topk(raw, k=top_n_max, dim=-1)
            top_n_v = top_n_v - lse
            top_n_dict = LogProbsPrefill._build_top_n_dict(
                context, req_idx_list, masked_lengths_cpu, top_n_v, top_n_i
            )
        return result, top_n_dict

    @staticmethod
    def _build_top_n_dict(
        context,
        req_idx_list: List[int],
        masked_lengths_cpu: List[int],
        top_n_values: Tensor,
        top_n_indices: Tensor,
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """Build per-request top-n dict for prefill mode.

        Args:
            context: The active DynamicInferenceContext.
            req_idx_list (List[int]): Request indices that wanted log probs.
            masked_lengths_cpu (List[int]): Per-request token counts.
            top_n_values (Tensor): Top-n log-prob values on GPU.
            top_n_indices (Tensor): Top-n token indices on GPU.
        """
        active_request_count = context.total_request_count - context.paused_request_count
        top_n_per_req: List[int] = context.active_request_metadata["top_n_logprobs"][
            :active_request_count
        ].tolist()
        skip_prompt_per_req: List[bool] = context.active_request_metadata["skip_prompt_log_probs"][
            :active_request_count
        ].tolist()
        top_n_v_cpu = top_n_values.cpu()
        top_n_i_cpu = top_n_indices.cpu()
        result: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
        token_offset = 0
        for i, req_idx in enumerate(req_idx_list):
            req_len = masked_lengths_cpu[i]
            n = top_n_per_req[req_idx]
            if n > 0:
                if skip_prompt_per_req[req_idx] and req_len > 1:
                    last = token_offset + req_len - 1
                    result[req_idx] = [(top_n_v_cpu[last, :n], top_n_i_cpu[last, :n])]
                else:
                    result[req_idx] = [
                        (top_n_v_cpu[token_offset + j, :n], top_n_i_cpu[token_offset + j, :n])
                        for j in range(req_len)
                    ]
            token_offset += req_len
        return result if result else None

    # -- public API --

    def indexing(self, context, *, eager: bool = False) -> None:
        """Run indexing kernel with optional CUDA graph capture/replay."""
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
        """Run softmax kernel and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, padded_active_tokens, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            eager (bool): If True, skip CUDA graph capture/replay.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        ri, cu_ml, li, li_range, mt = (
            self._ri, self._cu_ml, self._li, self._li_range, self._mt,
        )
        key = ("prefill_sm", context.padded_batch_dimensions)
        slp = self.softmax_kernel(
            logits, new_tokens, ri, cu_ml, li, li_range, mt,
            eager=eager, cache_key=key,
        )
        active_request_count = context.total_request_count - context.paused_request_count
        return functools.partial(
            self.extract,
            context,
            ri,
            cu_ml,
            slp,
            log_prob_request_count,
            active_request_count,
            top_n_max=top_n_max,
            logits=logits,
            logit_indices=li,
        )
