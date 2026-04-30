# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.logprobs.prefill import LogProbsPrefill
from megatron.core.transformer.cuda_graphs import CudaGraphManager

try:
    from flashinfer import top_k as _flashinfer_top_k

    def _topk(values: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        return _flashinfer_top_k(values, k=k, sorted=False)

except ImportError:

    def _topk(values: Tensor, k: int) -> Tuple[Tensor, Tensor]:
        return torch.topk(values, k=k, dim=-1, sorted=False)


class LogProbsSpeculative:
    """Log probability computation for speculative decoding.

    Handles the post-forward softmax (lse only), post-verification gather, and CPU extraction.
    When prefill logits are available it delegates to `LogProbsPrefill` for the prefill portion.
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
        # Pinned CPU scalar holding `num_decode * spec_plus_one`.
        self._prefill_offset_pinned = torch.zeros(1, dtype=torch.int64).pin_memory()
        if config is not None and config.cuda_graph_impl == "local":
            CudaGraphManager(
                config,
                self,
                function_name="softmax_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config,
                self,
                function_name="gather_kernel",
                need_backward=False,
                inline_capture=True,
            )
            CudaGraphManager(
                config,
                self,
                function_name="prefill_indexing_kernel",
                need_backward=False,
                inline_capture=True,
            )
            # Thin wrapper: delegates to LogProbsPrefill's static method.
            CudaGraphManager(
                config,
                self,
                function_name="_prefill_softmax",
                need_backward=False,
                inline_capture=True,
            )

    @staticmethod
    def softmax_kernel(
        context, logits: Tensor, prefill_offset_gpu: Tensor, *, eager: bool = False, cache_key=None
    ) -> Tuple[Tensor, Tensor]:
        """Post-forward: per-row log-sum-exp over all speculative logit rows.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
            prefill_offset_gpu (Tensor): scalar holding `num_decode * spec_plus_one`.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (decode_lse [padded_decode, spec+1], prefill_lse [padded_prefill])
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        padded_decode_count = context.padded_batch_dimensions.decode_req_count
        padded_prefill_count = context.padded_batch_dimensions.prefill_req_count
        spec_plus_one = context.num_speculative_tokens + 1
        decode_len = padded_decode_count * spec_plus_one
        device = logits.device

        # Decode region: padded_decode * spec_plus_one rows of [.., V].
        # Per-row LSE, reshaped so each request's spec rows stay grouped.
        # Runs over the padded shape for graphability; rows past num_decode are filtered by extract.
        decode_logits = logits.squeeze(0)[:decode_len].float()
        decode_lse = torch.logsumexp(decode_logits, dim=-1).reshape(
            padded_decode_count, spec_plus_one
        )

        # Prefill region: rows live at the real decode offset, not the padded one.
        # The slice [num_decode * (K+1) : padded_decode * (K+1)] is undefined memory.
        prefill_indices = prefill_offset_gpu + torch.arange(padded_prefill_count, device=device)
        prefill_logits = logits.squeeze(0)[prefill_indices].float()
        prefill_lse = torch.logsumexp(prefill_logits, dim=-1)
        return decode_lse, prefill_lse

    @staticmethod
    def gather_kernel(
        context,
        logits: Tensor,
        decode_lse: Tensor,
        prefill_lse: Tensor,
        new_tokens: Tensor,
        accepted_tokens: Tensor,
        accepted_token_counts: Tensor,
        prefill_offset_gpu: Tensor,
        *,
        eager: bool = False,
        cache_key=None,
    ) -> Tuple[Tensor, Tensor]:
        """Post-verification: gather log probs for speculative decode and materialized prefill.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
            decode_lse (Tensor): Decode lse [padded_decode, spec+1].
            prefill_lse (Tensor): Prefill lse [padded_prefill].
            new_tokens (Tensor): Newly sampled tokens [padded_active_requests].
            accepted_tokens (Tensor): Speculative-verified token IDs.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.
            prefill_offset_gpu (Tensor): scalar holding `num_decode * spec_plus_one`.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (decode_gathered [padded_decode, spec+1], prefill_gathered [padded_prefill])
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        padded_decode_count = context.padded_batch_dimensions.decode_req_count
        padded_prefill_count = context.padded_batch_dimensions.prefill_req_count
        spec_plus_one = context.num_speculative_tokens + 1
        decode_len = padded_decode_count * spec_plus_one
        device = decode_lse.device

        # Build a [padded_decode, spec+1] matrix of token IDs to gather log
        # probs for, one per (request, position) cell:
        #   columns < num_spec : the speculative tokens.
        #     Rejected slots hold -1 in `accepted_tokens`;
        #     clamp to 0 to keep gather indices in [0, V).
        #     Their gathered values are throwaway because extract slices to accepted_count + 1.
        #   column = accepted_count : overwrite with the freshly sampled token (from verifier).
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

        # Gather along the vocab axis to pick one logit per (request, position),
        # then LSE-adjust using the values precomputed in softmax_kernel.
        decode_logits = (
            logits.squeeze(0)[:decode_len]
            .float()
            .reshape(padded_decode_count, spec_plus_one, logits.size(-1))
        )
        decode_gathered_raw = decode_logits.gather(2, gather_tokens.unsqueeze(-1)).squeeze(-1)
        decode_gathered = decode_gathered_raw - decode_lse

        # Prefill: one log prob per request: the freshly sampled token at
        # the real decode offset (not padded). Same LSE-trick subtraction.
        prefill_new_tokens = new_tokens[
            padded_decode_count : padded_decode_count + padded_prefill_count
        ]
        prefill_row_range = torch.arange(padded_prefill_count, device=device)
        prefill_indices = prefill_offset_gpu + prefill_row_range
        prefill_logits = logits.squeeze(0)[prefill_indices].float()
        prefill_gathered_raw = prefill_logits[prefill_row_range, prefill_new_tokens]
        prefill_gathered = prefill_gathered_raw - prefill_lse

        return decode_gathered, prefill_gathered

    @staticmethod
    def prefill_indexing_kernel(
        context, prefill_offset_pinned: Tensor, *, eager: bool = False, cache_key=None
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Build prefill offset, mask out decode entries, build prefill indices, restore mask.

        Args:
            context: The active DynamicInferenceContext.
            prefill_offset_pinned (Tensor): pinned CPU scalar holding `num_decode * spec_plus_one`.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.

        Returns:
            (prefill_offset_gpu, request_indices, cu_masked_lengths, logit_indices,
             logit_indices_range, masked_tokens, prefill_log_prob_count_gpu).
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        device = torch.cuda.current_device()
        # H2D the offset; non-blocking because Python wrote the pinned scalar already.
        # The GPU copy is reused by softmax_kernel and gather_kernel.
        prefill_offset_gpu = prefill_offset_pinned.to(device, non_blocking=True)

        if context.config.materialize_only_last_token_logits:
            # Per-token prefill log probs aren't computed when model only emits last-token logits.
            # Return zero-shape placeholders so the call site can unpack without branching.
            empty = torch.zeros(0, dtype=torch.int64, device=device)
            return prefill_offset_gpu, empty, empty, empty, empty, empty, empty

        padded_decode_count = context.padded_batch_dimensions.decode_req_count

        # LogProbsPrefill.indexing_kernel reads `return_log_probs` to pick which requests to index.
        # In speculative mode the prefill kernel must only process *prefill* requests
        # (decode is handled by gather_kernel above), so temporarily mask out the decode entries.
        # Saved mask is restored before returning so the rest of the system never sees it.
        saved_mask = context.active_request_metadata["return_log_probs"][
            :padded_decode_count
        ].clone()
        context.active_request_metadata["return_log_probs"][:padded_decode_count] = False

        # Count of prefill requests asking for log probs.
        # Stays on GPU; the engine .item()s it later when it actually needs the value.
        prefill_log_prob_count_gpu = context.active_request_metadata["return_log_probs"][
            : context.padded_active_request_count
        ].sum()

        ri, cu_ml, li, li_range, mt = LogProbsPrefill.indexing_kernel(context)

        context.active_request_metadata["return_log_probs"][:padded_decode_count] = saved_mask

        return prefill_offset_gpu, ri, cu_ml, li, li_range, mt, prefill_log_prob_count_gpu

    def _prefill_softmax(
        self,
        logits,
        new_tokens,
        ri,
        cu_ml,
        li,
        li_range,
        mt,
        *,
        eager: bool = False,
        cache_key=None,
    ):
        """Delegate to LogProbsPrefill's softmax kernel (cross-class call).

        Args:
            logits, new_tokens, ri, cu_ml, li, li_range, mt:
                Forwarded as-is to LogProbsPrefill.softmax_kernel; see there for shapes.
            eager, cache_key: Consumed by `CudaGraphManager` when it wraps this kernel.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key
        # Wrapper exists so CudaGraphManager can graph-cache this call under a separate key.
        return LogProbsPrefill.softmax_kernel(logits, new_tokens, ri, cu_ml, li, li_range, mt)

    @staticmethod
    def extract(
        context,
        decode_gathered: Tensor,
        prefill_gathered: Tensor,
        accepted_token_counts: Tensor,
        result: List[Optional[List[float]]],
        decode_top_n_values: Optional[Tensor] = None,
        decode_top_n_indices: Optional[Tensor] = None,
        prefill_top_n_values: Optional[Tensor] = None,
        prefill_top_n_indices: Optional[Tensor] = None,
    ) -> Optional[Dict[int, List[Tuple[Tensor, Tensor]]]]:
        """CPU extraction for speculative log probs. Populates `result` in-place.

        Args:
            context: The active DynamicInferenceContext.
            decode_gathered (Tensor): Gathered decode log probs from gather_kernel.
            prefill_gathered (Tensor): Gathered prefill log probs from gather_kernel.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.
            result (List): Mutable list to populate with per-request log probs.
            decode_top_n_values / decode_top_n_indices (Optional[Tensor]):
                Pre-computed decode top-n shape [num_decode, spec+1, top_n_max].
            prefill_top_n_values / prefill_top_n_indices (Optional[Tensor]):
                Pre-computed last-token prefill top-n shape [num_prefill, top_n_max].

        Returns:
            top_n_dict if any top-n entries were produced, else None.
        """
        num_decode = context.num_decode_requests
        num_prefill = context.total_request_count - context.paused_request_count - num_decode
        active_request_count = num_decode + num_prefill

        # D2H pulls. Batched up front so we hit GPU<->CPU traffic only once for the metadata;
        # per-tensor copies happen in their respective sections.
        full_mask_cpu: List[bool] = context.active_request_metadata["return_log_probs"][
            :active_request_count
        ].tolist()
        need_top_n_per_req = (
            decode_top_n_values is not None and decode_top_n_indices is not None and num_decode > 0
        ) or (
            prefill_top_n_values is not None
            and prefill_top_n_indices is not None
            and num_prefill > 0
        )
        if need_top_n_per_req:
            top_n_per_req_cpu: List[int] = context.active_request_metadata["top_n_logprobs"][
                :active_request_count
            ].tolist()
        mask_cpu = full_mask_cpu[:num_decode]
        prefill_mask_cpu = full_mask_cpu[num_decode:]

        # Decode: each request gets accepted_count[i] + 1 log probs
        # (the accepted speculative tokens followed by the verifier's freshly sampled token).
        # Slots past accepted_count are dropped.
        decode_gathered_cpu = decode_gathered[:num_decode].cpu()
        accepted_counts_cpu = accepted_token_counts[:num_decode].tolist()
        for i in range(num_decode):
            if mask_cpu[i]:
                result[i] = decode_gathered_cpu[i, : accepted_counts_cpu[i] + 1].tolist()

        # Decode top-n: shape [num_decode, spec+1, top_n_max];
        # emit accepted_count[i] + 1 tuples per request (one per emitted token).
        top_n_dict: Dict[int, List[Tuple[Tensor, Tensor]]] = {}
        if decode_top_n_values is not None and decode_top_n_indices is not None and num_decode > 0:
            top_n_v_cpu = decode_top_n_values.cpu()
            top_n_i_cpu = decode_top_n_indices.cpu()
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

        # Materialized (only-last) prefill: single log prob per request (the freshly sampled token).
        # In full-prefill mode these get overwritten with per-token lists by extract.
        prefill_gathered_cpu = prefill_gathered[:num_prefill].cpu()
        for i in range(num_prefill):
            if prefill_mask_cpu[i]:
                result[num_decode + i] = [prefill_gathered_cpu[i].item()]

        # Materialized prefill top-n: only_last mode emits a single tuple per request.
        # Full-prefill top-n overrides in calculate().
        if (
            prefill_top_n_values is not None
            and prefill_top_n_indices is not None
            and num_prefill > 0
        ):
            top_n_v_cpu = prefill_top_n_values.cpu()
            top_n_i_cpu = prefill_top_n_indices.cpu()
            for i in range(num_prefill):
                req_idx = num_decode + i
                if not prefill_mask_cpu[i]:
                    continue
                req_top_n = top_n_per_req_cpu[req_idx]
                if req_top_n == 0:
                    continue
                top_n_dict[req_idx] = [(top_n_v_cpu[i, :req_top_n], top_n_i_cpu[i, :req_top_n])]

        return top_n_dict if top_n_dict else None

    def prefill_indexing(self, context, *, eager: bool = False) -> None:
        """Run prefill indexing kernel with optional CUDA graph capture/replay.

        Args:
            context: The active DynamicInferenceContext.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
        """
        spec_plus_one = context.num_speculative_tokens + 1
        # CPU-side stamp into the pinned scalar;
        # the H2D itself runs inside the graphed kernel below so it gets captured.
        self._prefill_offset_pinned[0] = context.num_decode_requests * spec_plus_one
        key = ("spec_fp_idx", context.padded_batch_dimensions)
        result = self.prefill_indexing_kernel(
            context, self._prefill_offset_pinned, eager=eager, cache_key=key
        )
        (
            self._prefill_offset_gpu,
            self._fp_ri,
            self._fp_cu_ml,
            self._fp_li,
            self._fp_li_range,
            self._fp_mt,
            self._fp_count_gpu,
        ) = result

    def softmax(self, context, logits: Tensor, *, eager: bool = False) -> None:
        """Run post-forward softmax with optional CUDA graph capture/replay.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
        """
        # Computes per-row LSE only; gather happens later in `calculate` post-verification.
        # Runs on a side stream so it overlaps with verification/sampling on the main stream.
        key = ("spec_sm", context.padded_batch_dimensions)
        self._decode_lse, self._prefill_lse = self.softmax_kernel(
            context, logits, self._prefill_offset_gpu, eager=eager, cache_key=key
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
        """Run gather kernel + top-n (on side stream) and return a deferred extract callable.

        Args:
            context: The active DynamicInferenceContext.
            logits (Tensor): Raw model output logits [1, seq_len, vocab_size].
            new_tokens (Tensor): Newly sampled tokens.
            log_prob_request_count (int): Number of requests wanting log probs.
            accepted_tokens (Tensor): Speculative-verified token IDs.
            accepted_token_counts (Tensor): Per-decode-request accepted counts.
            eager (bool): If True, skip CUDA graph capture/replay for the kernels.
            top_n_max (int): Maximum top-n logprobs to compute (0 to skip).

        Returns:
            A callable that returns (per_request_log_probs, top_n_dict).
        """
        active_request_count = context.total_request_count - context.paused_request_count
        num_decode = context.num_decode_requests
        num_prefill = active_request_count - num_decode
        only_last = context.config.materialize_only_last_token_logits
        spec_plus_one = context.num_speculative_tokens + 1

        # LSE values were precomputed in the post-forward `softmax` stage,
        # which ran concurrently with verification/sampling on a side stream.
        decode_lse = self._decode_lse
        prefill_lse = self._prefill_lse

        # Gather depends on accepted_tokens / new_tokens which only exist
        # after verification, so it runs on the main stream post-verification.
        key = ("spec_ga", context.padded_batch_dimensions)
        decode_gathered, prefill_gathered = self.gather_kernel(
            context,
            logits,
            decode_lse,
            prefill_lse,
            new_tokens,
            accepted_tokens,
            accepted_token_counts,
            self._prefill_offset_gpu,
            eager=eager,
            cache_key=key,
        )

        # Full-prefill softmax: only when the model materialized per-token logits for prefills.
        # Reuses LogProbsPrefill via the _prefill_softmax wrapper.
        fp_slp = fp_lse = None
        fp_ri = fp_cu_ml = fp_li = fp_count_gpu = None
        if not only_last and num_prefill > 0:
            fp_ri = self._fp_ri
            fp_cu_ml = self._fp_cu_ml
            fp_li = self._fp_li
            li_range = self._fp_li_range
            mt = self._fp_mt
            fp_count_gpu = self._fp_count_gpu

            fp_key = ("spec_fp_sm", context.padded_batch_dimensions)
            fp_slp, fp_lse = self._prefill_softmax(
                logits,
                new_tokens,
                fp_ri,
                fp_cu_ml,
                fp_li,
                li_range,
                mt,
                eager=eager,
                cache_key=fp_key,
            )

        # Top-n on the side stream so it overlaps the main-stream extract / next-step bookkeeping.
        # Three independent topks (decode, last-token prefill, optional full-prefill rows)
        # all gated on the same stream/event.
        decode_top_n_v = decode_top_n_i = None
        prefill_top_n_v = prefill_top_n_i = None
        fp_top_n_v = fp_top_n_i = None

        if top_n_max > 0:
            if self._topn_stream is not None:
                self._topn_stream.wait_stream(torch.cuda.current_stream())
                stream_ctx = torch.cuda.stream(self._topn_stream)
            else:
                stream_ctx = nullcontext()
            with stream_ctx:
                # Decode region: topk on [num_decode * spec_plus_one] real rows,
                # LSE-adjust using the pre-computed decode_lse,
                # reshape to [num_decode, spec+1, top_n_max] for per-position addressing in extract.
                if num_decode > 0:
                    decode_len = num_decode * spec_plus_one
                    raw_decode = logits.squeeze(0)[:decode_len].float()
                    top_n_v_raw, top_n_i_raw = _topk(raw_decode, k=top_n_max)
                    top_n_v_flat = top_n_v_raw - decode_lse[:num_decode].reshape(
                        decode_len
                    ).unsqueeze(-1)
                    decode_top_n_v = top_n_v_flat.reshape(num_decode, spec_plus_one, -1)
                    decode_top_n_i = top_n_i_raw.reshape(num_decode, spec_plus_one, -1)

                # Last-token prefill region.
                # In full-prefill mode this is overridden per-request by the full-prefill top-n;
                # it's still emitted here so only-last mode has it without an extra branch.
                if num_prefill > 0:
                    decode_len = num_decode * spec_plus_one
                    raw_prefill = logits.squeeze(0)[decode_len : decode_len + num_prefill].float()
                    top_n_v_raw, top_n_i_raw = _topk(raw_prefill, k=top_n_max)
                    prefill_top_n_v = top_n_v_raw - prefill_lse[:num_prefill].unsqueeze(-1)
                    prefill_top_n_i = top_n_i_raw

                # Full-prefill region: per-token top-n addressed by fp_li.
                if fp_slp is not None:
                    raw_full_prefill = logits.squeeze(0)[fp_li].float()
                    top_n_v_raw, top_n_i_raw = _topk(raw_full_prefill, k=top_n_max)
                    fp_top_n_v = top_n_v_raw - fp_lse.unsqueeze(-1)
                    fp_top_n_i = top_n_i_raw
            if self._topn_event is not None:
                # Signals the next step's main stream that side-stream logits read is done.
                self._topn_event.record(self._topn_stream)

        # Defer the CPU-side extract: caller invokes the partial after step bookkeeping
        # so D2H copies pay their synchronization cost as late as possible.
        def extract_fn():
            result: List[Optional[List[float]]] = [None] * active_request_count
            # Speculative-decode + only-last-prefill extract.
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

            # Full-prefill overrides:
            # replace the single-log-prob prefill entries with per-token lists,
            # and per-request top-n with the per-token version.
            if fp_slp is not None:
                prefill_result, prefill_top_n = LogProbsPrefill.extract(
                    context,
                    fp_ri,
                    fp_cu_ml,
                    fp_slp,
                    fp_count_gpu.item(),
                    active_request_count,
                    top_n_values=fp_top_n_v,
                    top_n_indices=fp_top_n_i,
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
