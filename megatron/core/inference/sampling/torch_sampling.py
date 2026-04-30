# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from collections import defaultdict
from typing import Any, List, Optional, Tuple

import torch
from torch import Tensor

from megatron.core.inference.sampling.base import Sampling


class TorchSampling(Sampling):
    """Sampling via bucketed `torch.multinomial`.

    Groups requests into unique buckets by `(temperature, top_k, top_p)` for separate launches.
    """

    def __init__(self, rng: torch.Generator, vocab_size: int) -> None:
        self._rng = rng
        self._vocab_size = vocab_size

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

        Shared between dynamic batching (`TorchSampling.sample_kernel`) and static
        batching (`TextGenerationController.sample_from_logits`).

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

        def modify_logits_for_top_k_filtering(logits, top_k):
            """Set the logits for none top-k values to -inf."""
            filter_ = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits.masked_fill_(filter_, float("-Inf"))

        def modify_logits_for_top_p_filtering(logits, top_p):
            """Set the logits for none top-p values to -inf."""
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            filter_ = cumulative_probs > top_p
            # Clone needed: filter_[:, 1:] and filter_[:, :-1] are overlapping views;
            # without clone, each write would corrupt the next read during the shift.
            filter_[:, 1:] = filter_[:, :-1].clone()
            filter_[..., 0] = 0

            filter_ = filter_.scatter(1, sorted_indices, filter_)
            logits.masked_fill_(filter_, float("-Inf"))

        if top_k == 1:
            return torch.argmax(last_token_logits, dim=-1)

        # Clone needed: .div_() and masked_fill_() below modify in-place,
        # which would mutate the caller's tensor without this clone.
        last_token_logits = last_token_logits.clone()
        if temperature != 1.0:
            last_token_logits.div_(temperature)
        if top_k > 1:
            assert top_k <= last_token_logits.size(1), "top-k is larger than logit size."
            if vocab_size:
                assert top_k < vocab_size, "top-k is larger than vocab size."
            modify_logits_for_top_k_filtering(last_token_logits, top_k)
        elif top_p > 0.0:
            modify_logits_for_top_p_filtering(last_token_logits, top_p)

        probabilities = last_token_logits.softmax(dim=-1)
        sampled = torch.multinomial(probabilities, num_samples=1, generator=generator).view(-1)

        if vocab_size:
            sampled = torch.clamp(sampled, min=0, max=(vocab_size - 1))

        return sampled

    def sample_kernel(
        self,
        logits: Tensor,
        n: int,
        context,
        *,
        eager: bool = False,
        cache_key: Any = None,
        gather_indices: Optional[Tensor] = None,
        token_to_request_index: Optional[Tensor] = None,
    ) -> Tensor:
        """Bucket active requests by `(temperature, top_k, top_p)` and sample each bucket.

        Args:
            logits: Logits tensor of shape `[>=n, vocab_size]`.
            n: Number of rows to sample.
            context: The active DynamicInferenceContext.
            eager: Accepted for API symmetry; ignored (TorchSampling has no graph wrapper).
            cache_key: Accepted for API symmetry; ignored.
            gather_indices: When set, sample from `logits[gather_indices[:n], :]`.
            token_to_request_index: When set, the loop dispatches per-token rather than
                per-request (used by the speculative path).

        Returns:
            Sampled token ids of shape `[n]`.
        """
        # CudaGraphManager consumes these args, if it exists.
        del eager, cache_key

        # Group active requests into sampling buckets by (temperature, top_k, top_p).
        active_request_count = context.total_request_count - context.paused_request_count
        md = context.active_request_metadata
        device = torch.cuda.current_device()

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
        for idx_tensor, (_, temp, top_k, top_p) in zip(bucket_index_tensors, buckets):
            if token_to_request_index is None:
                row_indices = idx_tensor
            else:
                row_indices = torch.where(torch.isin(token_to_request_index, idx_tensor))[0]
            token_list.append(self._sampling_func(logits[row_indices, :], temp, top_k, top_p))
            indices_list.append(row_indices)

        sampled_tokens = torch.cat(token_list, dim=0)
        sampled_indices = torch.cat(indices_list, dim=0)
        output[sampled_indices] = sampled_tokens
        return output

    def _sampling_func(
        self, last_token_logits: Tensor, temperature: float, top_k: int, top_p: float
    ) -> Tensor:
        return TorchSampling.sample_from_logits(
            last_token_logits,
            temperature,
            top_k,
            top_p,
            generator=self._rng,
            vocab_size=self._vocab_size,
        )
