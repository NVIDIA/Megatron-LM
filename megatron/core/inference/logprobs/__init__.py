# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-request log probability computation for the dynamic-batching engine.

Three execution paths, each a three-stage pipeline:

    path         class                  when
    -----------  ---------------------  --------------------------------------
    decode       LogProbsDecode         decode-only step or only-last logits
    prefill      LogProbsPrefill        mixed prefill+decode step
    speculative  LogProbsSpeculative    num_speculative_tokens > 0

    stage 1: indexing kernel  - which logit rows / which token IDs to look up;
                                runs pre-forward on a side stream.
    stage 2: softmax kernel   - LSE trick: gather selected logits, subtract
                                logsumexp per row. No full softmax materialized.
    stage 3: extract          - CPU-side D2H + per-request packaging.

Splitting the kernels lets the controller graph each stage and overlap GPU
work with the forward pass / sampling on side streams; the CPU-side extract
stage is deferred to the engine via a returned callable. `calculate_log_probs`
below is the graph-unaware entry point that runs all three stages eagerly
back-to-back; used by tests and any non-graph caller.

Prefill-path example. Logits at row k predict token k+1, but the newly-sampled
token isn't in `token_to_input_ids` yet, so the kernel shifts the active token
window left by one and overwrites each request's final slot with the sampled
token. Mixed prefill+decode, all four requests want log probs:

    request roles          : [ decode | decode | prefill | prefill         ]
    request_query_lengths  : [   1    |   1    |    2    |    5            ]
    new_tokens (sampled)   : [   52   |   12   |    3    |   86            ]
    token_to_input_ids     : [   31   |   75   |  45 16  |  90 12 72 24 88 ]   active_token_count=9

Indexing kernel (LogProbsPrefill.indexing_kernel):

    request_indices        : [ 0 | 1 | 2 | 3 | sentinel ]   nonzero_static of return_log_probs;
                                                            trailing fill = max_requests
    masked_lengths         : [ 1 | 1 | 2 | 5 |    0     ]   active_query_lengths[request_indices]
    masked_ends            : [ 0 | 1 | 3 | 8 |    0     ]   active_request_last_token_idxs[...]
    cu_masked_lengths      : [ 1 | 2 | 4 | 9 |    9     ]   slack absorbed into the last entry
    logit_indices          : [ 0 | 1 | 2 3 | 4 5 6 7 8 ]    rows of `logits` to gather
    masked_tokens (rolled) : [ XX| XX| 16 XX | 12 72 24 88 XX | scratch ]   XX = wrong rolled token

Softmax kernel (LogProbsPrefill.softmax_kernel) overwrites each request's last
slot with the sampled token; the trailing sentinel is redirected to the scratch
slot to avoid colliding with request 3's write at slot 8:

    masked_tokens (final)  : [ 52 | 12 | 16 3 | 12 72 24 88 86 | scratch ]

    selected_logits = logits.squeeze(0)[logit_indices].float()       # [9, V]
    lse             = logsumexp(selected_logits, dim=-1)
    log_probs       = selected_logits[range(9), masked_tokens[:9]] - lse

Extract (LogProbsPrefill.extract) splits along masked_lengths and packages
per request:

    [[lp(52)], [lp(12)], [lp(16), lp(3)], [lp(12), lp(72), lp(24), lp(88), lp(86)]]
"""

import torch

from megatron.core.inference.logprobs.decode import LogProbsDecode
from megatron.core.inference.logprobs.prefill import LogProbsPrefill
from megatron.core.inference.logprobs.speculative import LogProbsSpeculative

__all__ = ["LogProbsDecode", "LogProbsPrefill", "LogProbsSpeculative", "calculate_log_probs"]


try:
    from flashinfer import top_k as _flashinfer_top_k

    def _topk(values, k):
        return _flashinfer_top_k(values, k=k, sorted=False)

except ImportError:

    def _topk(values, k):
        return torch.topk(values, k=k, dim=-1, sorted=False)


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

    Runs all kernels eagerly and consecutively.

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

    # Local naming (kept short to keep kernel call sites compact):
    #   ri              request_indices (output of indexing_kernel)
    #   padded_arange   arange over padded_active_request_count
    #   cu_ml           cu_masked_lengths (cumulative per-request masked lengths)
    #   li              logit_indices (rows of `logits` to gather, one per active token)
    #   li_range        arange over padded_active_token_count
    #   mt              masked_tokens (token IDs after roll-and-overwrite)
    #   slp             selected_log_probs (output of softmax_kernel)
    #   lse             logsumexp per row (subtracted to convert raw logits -> log-probs)
    #   raw             raw (un-LSE-adjusted) logits selected by `ri` or `li`
    #   top_n_v / _i    top-n log-prob values / token indices

    # Speculative decoding has its own helper because the logit layout is different
    # (decode rows are [num_decode, spec+1] grouped, prefill rows follow at the real decode offset).
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

    # Decode path: one logit row per request. Used both for decode-only steps
    # and when the model only materialized last-token logits during prefill.
    if only_last_token_logits or context.is_decode_only():
        ri, padded_arange = LogProbsDecode.indexing_kernel(context)
        slp, lse = LogProbsDecode.softmax_kernel(logits, new_tokens, ri, padded_arange)

        top_n_v = top_n_i = None
        if top_n_max > 0:
            # Reuse the LSE: top_n_log_probs = topk(raw_logits) - lse.
            raw = logits.squeeze(0)[ri].float()
            top_n_v_raw, top_n_i = _topk(raw, k=top_n_max)
            top_n_v = top_n_v_raw - lse.unsqueeze(-1)

        return LogProbsDecode.extract(
            context,
            ri,
            slp,
            log_prob_request_count,
            active_request_count,
            top_n_values=top_n_v,
            top_n_indices=top_n_i,
        )
    else:
        # Mixed prefill+decode path: per-token logits across each request's query length.
        # See module docstring for the indexing/roll example.
        ri, cu_ml, li, li_range, mt = LogProbsPrefill.indexing_kernel(context)
        slp, lse = LogProbsPrefill.softmax_kernel(logits, new_tokens, ri, cu_ml, li, li_range, mt)

        top_n_v = top_n_i = None
        if top_n_max > 0:
            # Per-token top-n; LSE-adjusted using the same lse as softmax_kernel.
            raw = logits.squeeze(0)[li].float()
            top_n_v_raw, top_n_i = _topk(raw, k=top_n_max)
            top_n_v = top_n_v_raw - lse.unsqueeze(-1)

        return LogProbsPrefill.extract(
            context,
            ri,
            cu_ml,
            slp,
            log_prob_request_count,
            active_request_count,
            top_n_values=top_n_v,
            top_n_indices=top_n_i,
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
    """Graph-unaware speculative log-prob calculation.

    Args:
        context: The active DynamicInferenceContext.
        logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
        new_tokens (Tensor): Newly sampled tokens for active requests.
        only_last_token_logits (bool): Whether logits contain only last-token outputs.
        accepted_tokens (Tensor): Speculative-verified token IDs.
        accepted_token_counts (Tensor): Per-decode-request accepted counts.
        top_n_max (int): Maximum top-n logprobs requested across the batch.

    Returns:
        (per_request_log_probs, top_n_dict) tuple.
    """
    # Local naming:
    #   spec_plus_one        num_speculative_tokens + 1 (rows per decode request)
    #   prefill_offset_*     num_decode * spec_plus_one; where the prefill region
    #                        begins in the flattened logits tensor (_gpu / _pinned)
    #   {decode,prefill}_lse        per-row logsumexp for each region
    #   {decode,prefill}_gathered   log-probs gathered at accepted/sampled tokens
    #   {decode,prefill}_top_n_v/_i top-n values / token indices per region
    #   raw_{decode,prefill}        raw logits slice fed into top-n
    #   top_n_v_raw / _i_raw        topk output before LSE adjustment
    #   top_n_v_flat                decode top-n flattened to [num_decode * spec+1, k]
    #   fp_*                        "full prefill": per-token outputs when
    #                               materialize_only_last_token_logits=False
    #   ri, cu_ml, li, li_range, mt, slp, raw   as in calculate_log_probs (prefill kernel)
    #   count_gpu                   prefill log-prob request count from prefill_indexing_kernel
    active_request_count = context.total_request_count - context.paused_request_count
    num_decode = context.num_decode_requests
    num_prefill = active_request_count - num_decode
    spec_plus_one = context.num_speculative_tokens + 1
    result = [None] * active_request_count

    # Eager equivalent of the pinned scalar that flows through the production path:
    # prefill rows in `logits` start at this offset (real, not padded).
    prefill_offset_gpu = torch.tensor(
        num_decode * spec_plus_one, dtype=torch.int64, device=logits.device
    )

    # Stage 1: per-row LSE for both decode and prefill regions.
    decode_lse, prefill_lse = LogProbsSpeculative.softmax_kernel(
        context, logits, prefill_offset_gpu
    )
    # Stage 2: gather log-probs at (accepted_tokens + new_token) for decode,
    # and (new_token) for prefill. LSE-subtracted inside the kernel.
    decode_gathered, prefill_gathered = LogProbsSpeculative.gather_kernel(
        context,
        logits,
        decode_lse,
        prefill_lse,
        new_tokens,
        accepted_tokens,
        accepted_token_counts,
        prefill_offset_gpu,
    )

    # Stage 3 (optional): top-n. Decode emits a [num_decode, spec+1, top_n] block;
    # prefill emits last-token-only top-n (overridden below in full-prefill mode).
    decode_top_n_v = decode_top_n_i = None
    prefill_top_n_v = prefill_top_n_i = None
    if top_n_max > 0:
        if num_decode > 0:
            decode_len = num_decode * spec_plus_one
            raw_decode = logits.squeeze(0)[:decode_len].float()
            top_n_v_raw, top_n_i_raw = _topk(raw_decode, k=top_n_max)
            # Reuse decode_lse to convert raw logits -> log-probs row by row.
            top_n_v_flat = top_n_v_raw - decode_lse[:num_decode].reshape(decode_len).unsqueeze(-1)
            decode_top_n_v = top_n_v_flat.reshape(num_decode, spec_plus_one, -1)
            decode_top_n_i = top_n_i_raw.reshape(num_decode, spec_plus_one, -1)
        if num_prefill > 0:
            decode_len = num_decode * spec_plus_one
            raw_prefill = logits.squeeze(0)[decode_len : decode_len + num_prefill].float()
            top_n_v_raw, top_n_i_raw = _topk(raw_prefill, k=top_n_max)
            prefill_top_n_v = top_n_v_raw - prefill_lse[:num_prefill].unsqueeze(-1)
            prefill_top_n_i = top_n_i_raw

    # Stage 4: CPU extract for speculative + only-last prefill.
    top_n_dict = LogProbsSpeculative.extract(
        context,
        decode_gathered,
        prefill_gathered,
        accepted_token_counts,
        result,
        decode_top_n_values=decode_top_n_v,
        decode_top_n_indices=decode_top_n_i,
        prefill_top_n_values=prefill_top_n_v,
        prefill_top_n_indices=prefill_top_n_i,
    )

    # Full-prefill overlay: when the model materialized per-token prefill logits,
    # run the prefill kernel to compute per-token log-probs and overwrite the single-log-prob.
    if not only_last_token_logits and num_prefill > 0:
        prefill_offset_pinned = torch.tensor([num_decode * spec_plus_one], dtype=torch.int64)
        # The kernel also returns a GPU copy of the offset; we already have one above, drop it.
        _, ri, cu_ml, li, li_range, mt, count_gpu = LogProbsSpeculative.prefill_indexing_kernel(
            context, prefill_offset_pinned
        )
        slp, fp_lse = LogProbsPrefill.softmax_kernel(
            logits, new_tokens, ri, cu_ml, li, li_range, mt
        )

        fp_top_n_v = fp_top_n_i = None
        if top_n_max > 0:
            raw = logits.squeeze(0)[li].float()
            top_n_v_raw, top_n_i_raw = _topk(raw, k=top_n_max)
            fp_top_n_v = top_n_v_raw - fp_lse.unsqueeze(-1)
            fp_top_n_i = top_n_i_raw

        prefill_result, prefill_top_n = LogProbsPrefill.extract(
            context,
            ri,
            cu_ml,
            slp,
            count_gpu.item(),
            active_request_count,
            top_n_values=fp_top_n_v,
            top_n_indices=fp_top_n_i,
        )
        # Overlay per-token prefill values onto the result list (and top-n dict) from extract.
        for i, lp in enumerate(prefill_result):
            if lp is not None:
                result[i] = lp
        if prefill_top_n:
            if top_n_dict is None:
                top_n_dict = prefill_top_n
            else:
                top_n_dict.update(prefill_top_n)

    return result, top_n_dict
