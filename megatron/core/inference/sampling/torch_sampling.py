# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from megatron.core.inference.sampling.base import Sampling


def _modify_logits_for_top_k_filtering(logits: Tensor, top_k: int) -> None:
    """Set the logits for non-top-k values to -inf (in place)."""
    filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits.masked_fill_(filter_, float("-Inf"))


def _modify_logits_for_top_p_filtering(logits: Tensor, top_p: float) -> None:
    """Set the logits for non-top-p values to -inf (in place)."""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    filter_ = cumulative_probs > top_p
    # Clone needed: filter_[:, 1:] and filter_[:, :-1] are overlapping views;
    # without clone, each write would corrupt the next read during the shift.
    filter_[:, 1:] = filter_[:, :-1].clone()
    filter_[..., 0] = 0

    filter_ = filter_.scatter(1, sorted_indices, filter_)
    logits.masked_fill_(filter_, float("-Inf"))


class TorchSampling(Sampling):
    """Sampling via bucketed `torch.multinomial`.

    Groups requests into unique buckets by `(temperature, top_k, top_p)` for separate
    launches. Within a bucket the (temperature / top-k / top-p) filtering is vectorized,
    but the random draw is **per-request keyed**: each request draws from its own
    `torch.Generator` seeded by `(seed + request_id)`. A request's draw therefore depends
    only on `(request_id, its own draw count)` and is invariant to the batch composition
    and row order. This is exactly the property the launch-before-commit async scheduler
    needs (pre-commit sampling re-orders rows vs. the post-`update_requests` compacted
    order), and it is what makes `async == serial` token-exact *by construction* for the
    torch backend. Greedy rows (`top_k == 1`) short-circuit to `argmax` and draw no random
    numbers, so greedy is unaffected by the keying.
    """

    def __init__(
        self, rng: torch.Generator, vocab_size: int, seed: Optional[int] = None
    ) -> None:
        # `rng` is retained for the static-batching path (`sample_from_logits`); dynamic
        # decode / speculative sampling uses per-request keyed generators instead.
        self._rng = rng
        self._vocab_size = vocab_size
        # Base seed for the per-request keyed generators. Defaults to the shared rng's
        # seed so behavior is well-defined even if a caller omits it.
        self._seed = int(seed if seed is not None else rng.initial_seed())
        # request_id -> its dedicated generator (memoized; retired when a request finishes).
        self._request_rngs: Dict[int, torch.Generator] = {}

    def _generator_for_request(self, request_id: int) -> torch.Generator:
        """Return ``request_id``'s dedicated generator, creating it on first use.

        Seeded by ``(seed + request_id) % 2**63`` so a request's random stream depends only
        on its id (and its own draw count), never on which other requests share the batch.
        """
        request_id = int(request_id)
        gen = self._request_rngs.get(request_id)
        if gen is None:
            gen = torch.Generator(device=torch.cuda.current_device())
            gen.manual_seed((self._seed + request_id) % (2**63))
            self._request_rngs[request_id] = gen
        return gen

    def retire_requests(self, request_ids: Sequence[int]) -> None:
        """Drop finished requests' generators (memory hygiene).

        Per-request streams are independent, so retiring one never perturbs a survivor's
        draws. A request id that is later reused gets a fresh, identically-seeded
        generator, preserving determinism.
        """
        for request_id in request_ids:
            self._request_rngs.pop(int(request_id), None)

    @staticmethod
    def sample_from_logits(
        last_token_logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        *,
        generator: torch.Generator,
        vocab_size: Optional[int] = None,
    ) -> Tensor:
        """Sample tokens from logits with temperature, top-k, and top-p filtering.

        Used by the static-batching path (one shared generator for the whole batch).
        Dynamic decode / speculative sampling goes through :meth:`sample_kernel`, which
        draws per-request-keyed instead.

        Args:
            last_token_logits: Logits of shape `[batch_size, vocab_size]`.
            temperature: Temperature scaling factor.
            top_k: Top-k filtering value (0 = disabled).
            top_p: Top-p (nucleus) filtering value (0.0 = disabled).
            generator: RNG used by `torch.multinomial`.
            vocab_size: When provided, asserts `top_k < vocab_size` and clamps the
                sampled ids to `[0, vocab_size - 1]`.

        Returns:
            Sampled token ids of shape `[batch_size]`.
        """
        assert isinstance(top_p, float)
        assert isinstance(top_k, int)
        assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
        assert top_p <= 1.0, "top-p should be in (0,1]"

        if top_k == 1:
            return torch.argmax(last_token_logits, dim=-1)

        # Clone needed: .div_() and masked_fill_() below modify in-place.
        last_token_logits = last_token_logits.clone()
        if temperature != 1.0:
            last_token_logits.div_(temperature)
        if top_k > 1:
            assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
            if vocab_size:
                assert top_k < vocab_size, "top-k is larger than vocab size."
            _modify_logits_for_top_k_filtering(last_token_logits, top_k)
        elif top_p > 0.0:
            _modify_logits_for_top_p_filtering(last_token_logits, top_p)

        probabilities = last_token_logits.softmax(dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).view(-1)

        if vocab_size:
            sampled = torch.clamp(sampled, min=0, max=(vocab_size - 1))

        return sampled

    def _sample_bucket_per_request(
        self,
        bucket_logits: Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
        row_request_ids: Sequence[int],
    ) -> Tensor:
        """Sample one token per row of a single `(temperature, top_k, top_p)` bucket.

        Greedy (`top_k == 1`) short-circuits to `argmax` with no generator -- a strict
        no-op vs. the shared-rng path. Otherwise the bucket is filtered once (vectorized),
        then each row is drawn from its own request's keyed generator, so the draw is
        invariant to batch composition / row order.
        """
        assert isinstance(top_p, float)
        assert isinstance(top_k, int)
        assert not (top_k > 0 and top_p > 0.0), "Cannot have top-p and top-k both greater than zero"
        assert top_p <= 1.0, "top-p should be in (0,1]"

        if top_k == 1:
            return torch.argmax(bucket_logits, dim=-1)

        # Clone needed: .div_() and masked_fill_() below modify in-place.
        logits = bucket_logits.clone()
        if temperature != 1.0:
            logits.div_(temperature)
        if top_k > 1:
            assert top_k <= logits.size(1), "top-k is larger than logit size."
            if self._vocab_size:
                assert top_k < self._vocab_size, "top-k is larger than vocab size."
            _modify_logits_for_top_k_filtering(logits, top_k)
        elif top_p > 0.0:
            _modify_logits_for_top_p_filtering(logits, top_p)

        probabilities = logits.softmax(dim=-1)
        sampled = torch.empty(probabilities.shape[0], device=logits.device, dtype=torch.int64)
        for row, request_id in enumerate(row_request_ids):
            sampled[row : row + 1] = torch.multinomial(
                probabilities[row : row + 1],
                num_samples=1,
                generator=self._generator_for_request(request_id),
            ).view(-1)

        if self._vocab_size:
            sampled = torch.clamp(sampled, min=0, max=(self._vocab_size - 1))

        return sampled

    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
        eager: bool = False,
        cache_key: Any = None,
    ) -> Tensor:
        """Bucket active requests by `(temperature, top_k, top_p)` and sample each bucket.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, the loop dispatches per-token rather than
                per-request (used by the speculative path).
            eager: Accepted for API symmetry; ignored (TorchSampling has no graph wrapper).
            cache_key: Accepted for API symmetry; ignored.

        Returns:
            Sampled token ids of shape `[n]`.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        # Group active requests into sampling buckets by (temperature, top_k, top_p).
        active_request_count = context.total_request_count - context.paused_request_count
        md = context.active_request_metadata
        device = torch.cuda.current_device()

        # Active request ids in row order: active row i == request_ids[paused + i] (see
        # build_active_slices). Used to key each row's generator. TorchSampling always runs
        # eager (no captured sampling graph), so this host read is safe.
        paused = context.paused_request_count
        active_request_ids = context.request_ids[paused : paused + active_request_count].tolist()

        bucket_map: dict = defaultdict(list)
        temp = md["temperature"][:active_request_count].tolist()
        top_k = md["top_k"][:active_request_count].tolist()
        top_p = md["top_p"][:active_request_count].tolist()
        for request_index, (t, k, p) in enumerate(zip(temp, top_k, top_p)):
            bucket_map[(t, k, p)].append(request_index)

        buckets: List[Tuple] = [(indices, *params) for params, indices in bucket_map.items()]
        bucket_index_tensors: List[Tensor] = [
            torch.tensor(indices, device=device, dtype=torch.long) for indices, *_ in buckets
        ]

        if gather_indices is not None:
            logits = logits[gather_indices[:n], :]

        output = torch.empty(n, device=logits.device, dtype=torch.int64)
        token_list = []
        indices_list = []
        for idx_tensor, (request_indices, temp_val, top_k_val, top_p_val) in zip(
            bucket_index_tensors, buckets
        ):
            if token_to_request_index is None:
                row_indices = idx_tensor
                # Each row is one request; map its request index to its id.
                row_request_ids = [active_request_ids[ri] for ri in request_indices]
            else:
                row_indices = torch.where(torch.isin(token_to_request_index, idx_tensor))[0]
                # Speculative: a request contributes 1+n_spec consecutive rows; map each
                # selected token row (in ascending row order) to its request's id.
                row_request_ids = [
                    active_request_ids[ri] for ri in token_to_request_index[row_indices].tolist()
                ]
            token_list.append(
                self._sample_bucket_per_request(
                    logits[row_indices, :], temp_val, top_k_val, top_p_val, row_request_ids
                )
            )
            indices_list.append(row_indices)

        sampled_tokens = torch.cat(token_list, dim=0)
        sampled_indices = torch.cat(indices_list, dim=0)
        output[sampled_indices] = sampled_tokens
        return output
