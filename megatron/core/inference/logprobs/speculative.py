# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.inference.logprobs.prefill import LogProbsPrefill
from megatron.core.transformer.cuda_graphs import CudaGraphManager


class LogProbsSpeculative:
    """Log probability computation for speculative decoding.

    Handles the post-forward softmax, post-verification gather, and CPU extraction.
    When full prefill logits are available it delegates to `LogProbsPrefill`.
    """

    def __init__(self, config=None):
        if config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config, self,
                function_name="softmax_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config, self,
                function_name="gather_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config, self,
                function_name="prefill_indexing_kernel",
                need_backward=False,
                inline_capture=True,
            )
            # Thin wrapper: delegates to LogProbsPrefill's static method.
            CudaGraphManager(
                config, self,
                function_name="_prefill_softmax",
                need_backward=False,
                inline_capture=True,
            )

    @staticmethod
    def softmax_kernel(context, logits: Tensor) -> Tuple[Tensor, Tensor]:
        """Post-forward: log-softmax over all speculative logit rows.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].

        Returns:
            (decode_log_probs [padded_decode, spec+1, vocab],
             prefill_log_probs [padded_prefill, vocab])
        """
        padded_decode_count = context.padded_batch_dimensions.decode_req_count
        padded_prefill_count = context.padded_batch_dimensions.prefill_req_count
        spec_plus_one = context.num_speculative_tokens + 1
        decode_len = padded_decode_count * spec_plus_one
        total_len = decode_len + padded_prefill_count

        all_log_probs = F.log_softmax(logits.squeeze(0)[:total_len].float(), dim=-1)
        decode_log_probs = all_log_probs[:decode_len].reshape(
            padded_decode_count, spec_plus_one, -1
        )
        prefill_log_probs = all_log_probs[decode_len:]
        return decode_log_probs, prefill_log_probs

    @staticmethod
    def gather_kernel(
        context,
        decode_log_probs: Tensor,
        prefill_log_probs: Tensor,
        new_tokens: Tensor,
        accepted_tokens: Tensor,
        accepted_token_counts: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Post-verification: gather log probs for speculative decode and materialized prefill.

        Args:
            context: The active DynamicInferenceContext.
            decode_log_probs (Tensor): Decode log probs [padded_decode, spec+1, vocab].
            prefill_log_probs (Tensor): Prefill log probs [padded_prefill, vocab].
            new_tokens (Tensor): Newly sampled tokens [padded_active_requests].
            accepted_tokens (Tensor): Speculative-verified token IDs.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.

        Returns:
            (decode_gathered [padded_decode, spec+1], prefill_gathered [padded_prefill])
        """
        padded_decode_count = context.padded_batch_dimensions.decode_req_count
        padded_prefill_count = context.padded_batch_dimensions.prefill_req_count
        spec_plus_one = context.num_speculative_tokens + 1
        device = decode_log_probs.device

        gather_tokens = torch.zeros(
            padded_decode_count, spec_plus_one, device=device, dtype=torch.long
        )
        gather_tokens[:, : context.num_speculative_tokens] = accepted_tokens[
            :padded_decode_count
        ].clamp(min=0)
        decode_row_range = torch.arange(padded_decode_count, device=device)
        gather_tokens[decode_row_range, accepted_token_counts[:padded_decode_count]] = new_tokens[
            :padded_decode_count
        ]
        decode_gathered = decode_log_probs.gather(2, gather_tokens.unsqueeze(-1)).squeeze(-1)

        prefill_new_tokens = new_tokens[
            padded_decode_count : padded_decode_count + padded_prefill_count
        ]
        prefill_row_range = torch.arange(padded_prefill_count, device=device)
        prefill_gathered = prefill_log_probs[prefill_row_range, prefill_new_tokens]

        return decode_gathered, prefill_gathered

    @staticmethod
    def prefill_indexing_kernel(context) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Mask out decode entries, build prefill indices, restore mask.

        Args:
            context: The active DynamicInferenceContext.

        Returns:
            (request_indices, cu_masked_lengths, logit_indices,
             logit_indices_range, masked_tokens, prefill_log_prob_count_gpu)
        """
        padded_decode_count = context.padded_batch_dimensions.decode_req_count

        saved_mask = context.active_request_metadata["return_log_probs"][
            :padded_decode_count
        ].clone()
        context.active_request_metadata["return_log_probs"][:padded_decode_count] = False

        prefill_log_prob_count_gpu = context.active_request_metadata["return_log_probs"][
            : context.padded_active_request_count
        ].sum()

        ri, cu_ml, li, li_range, mt = LogProbsPrefill.indexing_kernel(context)

        context.active_request_metadata["return_log_probs"][:padded_decode_count] = saved_mask

        return ri, cu_ml, li, li_range, mt, prefill_log_prob_count_gpu

    def _prefill_softmax(self, logits, new_tokens, ri, cu_ml, li, li_range, mt):
        """Delegate to LogProbsPrefill's softmax kernel (cross-class call)."""
        return LogProbsPrefill.softmax_kernel(logits, new_tokens, ri, cu_ml, li, li_range, mt)

    @staticmethod
    def extract(
        context,
        decode_gathered: Tensor,
        prefill_gathered: Tensor,
        accepted_token_counts: Tensor,
        result: List[Optional[List[float]]],
        logits: Optional[Tensor] = None,
        top_n_max: int = 0,
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """CPU extraction for speculative log probs. Populates `result` in-place.

        Args:
            context: The active DynamicInferenceContext.
            decode_gathered (Tensor): Gathered decode log probs from gather_kernel.
            prefill_gathered (Tensor): Gathered prefill log probs from gather_kernel.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.
            result (List): Mutable list to populate with per-request log probs.
            logits (Optional[Tensor]): Raw logits for top-n computation.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            top_n_dict if top_n_max > 0, else None.
        """
        num_decode = context.num_decode_requests
        num_prefill = context.total_request_count - context.paused_request_count - num_decode

        # Decode: variable-length list per request.
        decode_gathered_cpu = decode_gathered[:num_decode].cpu()
        accepted_counts_cpu = accepted_token_counts[:num_decode].tolist()
        mask_cpu: List[bool] = context.active_request_metadata["return_log_probs"][
            :num_decode
        ].tolist()
        for i in range(num_decode):
            if mask_cpu[i]:
                result[i] = decode_gathered_cpu[i, : accepted_counts_cpu[i] + 1].tolist()

        top_n_dict: Dict[int, List[Tuple[Tensor, Tensor]]] = {}

        # Decode top-n.
        if top_n_max > 0 and logits is not None and num_decode > 0:
            spec_plus_one = context.num_speculative_tokens + 1
            raw_decode = logits.squeeze(0)[: num_decode * spec_plus_one].float()
            raw_reshaped = raw_decode.reshape(num_decode, spec_plus_one, -1)
            lse = torch.logsumexp(raw_reshaped, dim=-1, keepdim=True)
            top_n_v, top_n_i = torch.topk(raw_reshaped, k=top_n_max, dim=-1)
            top_n_v = top_n_v - lse
            top_n_v_cpu = top_n_v.cpu()
            top_n_i_cpu = top_n_i.cpu()
            top_n_per_req_cpu: List[int] = context.active_request_metadata[
                "top_n_logprobs"
            ].tolist()
            for i in range(num_decode):
                if not mask_cpu[i]:
                    continue
                req_top_n = top_n_per_req_cpu[i]
                if req_top_n == 0:
                    continue
                n_emitted = accepted_counts_cpu[i] + 1
                top_n_dict[i] = [
                    (top_n_v_cpu[i, j, :req_top_n], top_n_i_cpu[i, j, :req_top_n])
                    for j in range(n_emitted)
                ]

        # Materialized prefill: single log prob per request.
        prefill_gathered_cpu = prefill_gathered[:num_prefill].cpu()
        prefill_mask_cpu: List[bool] = context.active_request_metadata["return_log_probs"][
            num_decode : num_decode + num_prefill
        ].tolist()
        for i in range(num_prefill):
            if prefill_mask_cpu[i]:
                result[num_decode + i] = [prefill_gathered_cpu[i].item()]

        # Materialized prefill top-n.
        if top_n_max > 0 and logits is not None and num_prefill > 0:
            spec_plus_one = context.num_speculative_tokens + 1
            raw_decode_len = num_decode * spec_plus_one
            raw_prefill = logits.squeeze(0)[raw_decode_len : raw_decode_len + num_prefill].float()
            lse = torch.logsumexp(raw_prefill, dim=-1, keepdim=True)
            top_n_v, top_n_i = torch.topk(raw_prefill, k=top_n_max, dim=-1)
            top_n_v = top_n_v - lse
            top_n_v_cpu = top_n_v.cpu()
            top_n_i_cpu = top_n_i.cpu()
            top_n_per_req_cpu2: List[int] = context.active_request_metadata[
                "top_n_logprobs"
            ].tolist()
            for i in range(num_prefill):
                req_idx = num_decode + i
                if not prefill_mask_cpu[i]:
                    continue
                req_top_n = top_n_per_req_cpu2[req_idx]
                if req_top_n == 0:
                    continue
                top_n_dict[req_idx] = [(top_n_v_cpu[i, :req_top_n], top_n_i_cpu[i, :req_top_n])]

        return top_n_dict if top_n_dict else None

    # -- public API --

    def prefill_indexing(self, context, *, eager: bool = False) -> None:
        """Run prefill indexing kernel with optional CUDA graph capture/replay.

        Stores results on ``self`` for use by :meth:`calculate`.
        """
        key = ("spec_fp_idx", context.padded_batch_dimensions)
        result = self.prefill_indexing_kernel(context, eager=eager, cache_key=key)
        self._fp_ri, self._fp_cu_ml, self._fp_li, self._fp_li_range, self._fp_mt, self._fp_count_gpu = result

    def softmax(self, context, logits: Tensor, *, eager: bool = False) -> None:
        """Run post-forward softmax with optional CUDA graph capture/replay.

        Stores results on ``self`` for use by :meth:`calculate`.
        """
        key = ("spec_sm", context.padded_batch_dimensions)
        self._decode_log_probs, self._prefill_log_probs = self.softmax_kernel(
            context, logits, eager=eager, cache_key=key,
        )

    def calculate(
        self,
        context,
        logits: Tensor,
        new_tokens: Tensor,
        log_prob_request_count: int,
        accepted_tokens: Tensor,
        accepted_token_counts: Tensor,
        *,
        eager: bool = False,
        top_n_max: int = 0,
    ):
        """Run gather kernel and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            accepted_tokens (Tensor): Speculative-verified token IDs.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.
            eager (bool): If True, skip CUDA graph capture/replay.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        active_request_count = context.total_request_count - context.paused_request_count
        num_prefill = active_request_count - context.num_decode_requests
        only_last = context.config.materialize_only_last_token_logits

        # Speculative gather: runs post-verification.
        decode_log_probs = self._decode_log_probs
        prefill_log_probs = self._prefill_log_probs

        key = ("spec_ga", context.padded_batch_dimensions)
        decode_gathered, prefill_gathered = self.gather_kernel(
            context,
            decode_log_probs,
            prefill_log_probs,
            new_tokens,
            accepted_tokens,
            accepted_token_counts,
            eager=eager,
            cache_key=key,
        )

        # Full prefill softmax (if needed); runs post-verification, before extract.
        fp_slp = None
        fp_ri = fp_cu_ml = fp_li = fp_count_gpu = None
        if not only_last and num_prefill > 0:
            fp_ri = self._fp_ri
            fp_cu_ml = self._fp_cu_ml
            fp_li = self._fp_li
            li_range = self._fp_li_range
            mt = self._fp_mt
            fp_count_gpu = self._fp_count_gpu

            fp_key = ("spec_fp_sm", context.padded_batch_dimensions)
            fp_slp = self._prefill_softmax(
                logits, new_tokens, fp_ri, fp_cu_ml, fp_li, li_range, mt,
                eager=eager, cache_key=fp_key,
            )

        def extract_fn():
            result: List[Optional[List[float]]] = [None] * active_request_count
            top_n_dict = LogProbsSpeculative.extract(
                context,
                decode_gathered,
                prefill_gathered,
                accepted_token_counts,
                result,
                logits=logits,
                top_n_max=top_n_max,
            )

            if fp_slp is not None:
                prefill_result, prefill_top_n = LogProbsPrefill.extract(
                    context,
                    fp_ri,
                    fp_cu_ml,
                    fp_slp,
                    fp_count_gpu.item(),
                    active_request_count,
                    top_n_max=top_n_max,
                    logits=logits,
                    logit_indices=fp_li,
                )
                for i, lp in enumerate(prefill_result):
                    if lp is not None:
                        result[i] = lp
                if prefill_top_n:
                    if top_n_dict is None:
                        top_n_dict = prefill_top_n
                    else:
                        top_n_dict.update(prefill_top_n)

            return result, top_n_dict

        return extract_fn
