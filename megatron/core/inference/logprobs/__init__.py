# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from megatron.core.inference.logprobs.decode import LogProbsDecode
from megatron.core.inference.logprobs.prefill import LogProbsPrefill
from megatron.core.inference.logprobs.speculative import LogProbsSpeculative

__all__ = ["LogProbsDecode", "LogProbsPrefill", "LogProbsSpeculative", "calculate_log_probs"]


def calculate_log_probs(
    context,
    logits,
    new_tokens,
    only_last_token_logits=False,
    log_prob_request_count=0,
    accepted_tokens=None,
    accepted_token_counts=None,
    top_n_max=0,
):
    """Graph-unaware entry point for log probability computation.

    Runs all kernels eagerly and consecutively, ignoring the D2H syncs between them.

    Args:
        context: The active DynamicInferenceContext.
        logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
        new_tokens (Tensor): Newly sampled tokens for active requests.
        only_last_token_logits (bool): Whether logits contain only last-token outputs.
        log_prob_request_count (int): Number of requests wanting log probs.
        accepted_tokens (Optional[Tensor]): Speculative-verified token IDs.
        accepted_token_counts (Optional[Tensor]): Per-decode-request accepted counts.
        top_n_max (int): Maximum top-n logprobs requested across the batch.

    Returns:
        (per_request_log_probs, top_n_dict) tuple.
    """
    if log_prob_request_count == 0:
        return [], None

    if context.num_speculative_tokens > 0:
        assert (
            accepted_tokens is not None and accepted_token_counts is not None
        ), "accepted_tokens and accepted_token_counts are required when num_speculative_tokens > 0"
        return _calculate_log_probs_speculative(
            context,
            logits,
            new_tokens,
            only_last_token_logits,
            accepted_tokens,
            accepted_token_counts,
            top_n_max,
        )

    active_request_count = context.total_request_count - context.paused_request_count

    if only_last_token_logits or context.is_decode_only():
        ri, padded_arange = LogProbsDecode.indexing_kernel(context)
        slp, lse = LogProbsDecode.softmax_kernel(logits, new_tokens, ri, padded_arange)
        return LogProbsDecode.extract(
            context,
            ri,
            slp,
            lse,
            log_prob_request_count,
            active_request_count,
            logits=logits,
            top_n_max=top_n_max,
        )
    else:
        ri, cu_ml, li, li_range, mt = LogProbsPrefill.indexing_kernel(context)
        slp, lse = LogProbsPrefill.softmax_kernel(logits, new_tokens, ri, cu_ml, li, li_range, mt)
        return LogProbsPrefill.extract(
            context,
            ri,
            cu_ml,
            slp,
            lse,
            log_prob_request_count,
            active_request_count,
            top_n_max=top_n_max,
            logits=logits,
            logit_indices=li,
        )


def _calculate_log_probs_speculative(
    context,
    logits,
    new_tokens,
    only_last_token_logits,
    accepted_tokens,
    accepted_token_counts,
    top_n_max,
):
    """Graph-unaware speculative log-prob calculation."""
    active_request_count = context.total_request_count - context.paused_request_count
    num_decode = context.num_decode_requests
    num_prefill = active_request_count - num_decode
    result = [None] * active_request_count

    decode_lse, prefill_lse = LogProbsSpeculative.softmax_kernel(context, logits)
    decode_gathered, prefill_gathered = LogProbsSpeculative.gather_kernel(
        context,
        logits,
        decode_lse,
        prefill_lse,
        new_tokens,
        accepted_tokens,
        accepted_token_counts,
    )
    top_n_dict = LogProbsSpeculative.extract(
        context,
        decode_gathered,
        prefill_gathered,
        accepted_token_counts,
        result,
        logits=logits,
        decode_lse=decode_lse,
        prefill_lse=prefill_lse,
        top_n_max=top_n_max,
    )

    # Full prefill path when logits are materialized for all tokens.
    if not only_last_token_logits and num_prefill > 0:
        ri, cu_ml, li, li_range, mt, count_gpu = LogProbsSpeculative.prefill_indexing_kernel(
            context
        )
        slp, fp_lse = LogProbsPrefill.softmax_kernel(
            logits, new_tokens, ri, cu_ml, li, li_range, mt
        )
        prefill_result, prefill_top_n = LogProbsPrefill.extract(
            context,
            ri,
            cu_ml,
            slp,
            fp_lse,
            count_gpu.item(),
            active_request_count,
            top_n_max=top_n_max,
            logits=logits,
            logit_indices=li,
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
