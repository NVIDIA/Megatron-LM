# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Cached, deterministic routing assignments for rank-skew performance benchmarks."""

import math

import torch


def validate_mock_route_parameters(num_experts, topk, ep_size, maxvio):
    """Validate the attainable max/mean rank load with distinct experts per token."""
    if not num_experts or ep_size < 1 or num_experts % ep_size:
        raise ValueError("Mock routing requires experts evenly divided across EP ranks.")
    if not 1 <= topk <= num_experts:
        raise ValueError("Mock routing requires 1 <= topk <= num_experts.")
    upper = ep_size * min(topk, num_experts // ep_size) / topk
    if not math.isfinite(maxvio) or not 1 <= maxvio <= upper:
        raise ValueError(
            f"moe_router_mock_maxvio must be finite and in [1, {upper}], got {maxvio}."
        )


class MockRouteCache:
    """Cache semantic expert IDs, independent of logits, RNG, and source rank.

    EP rank 0 is the hot destination on *every* source rank. For each valid-token
    prefix of length V, it gets round(V * topk * maxvio / EP) assignments; other
    destinations differ by at most one assignment. Expert IDs are interleaved
    within each destination and never repeated within a token's top-k.

    Cache entries are retained for this router's lifetime, including tensors
    referenced by CUDA graphs. Warm every physical token shape/device eagerly
    before capture. Entries are ordinary tensors, not checkpoint buffers.
    """

    def __init__(self, num_experts, topk, ep_size, maxvio):
        validate_mock_route_parameters(num_experts, topk, ep_size, maxvio)
        self.num_experts = num_experts
        self.topk = topk
        self.ep_size = ep_size
        self.maxvio = maxvio
        self._entries = {}

    def get(self, num_tokens, device):
        """Return read-only cached [physical_tokens, topk] int64 IDs."""
        device = torch.device(device)
        if device.type == 'cuda' and device.index is None:
            device = torch.device('cuda', torch.cuda.current_device())
        key = (num_tokens, device)
        if key not in self._entries:
            if device.type == 'cuda' and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Mock route cache miss during CUDA Graph capture: warm this token "
                    "shape/device with an eager router forward before capture."
                )
            # Build once on CPU, without advancing the training RNG. Disable
            # inference mode so the IDs can later be saved by gather backward.
            with torch.inference_mode(False), torch.no_grad():
                self._entries[key] = self._build(num_tokens).to(device)
        return self._entries[key]

    def _build(self, num_tokens):
        if num_tokens < 0:
            raise ValueError("num_tokens must be nonnegative.")
        if self.ep_size == 1:
            return (
                torch.arange(num_tokens * self.topk, device='cpu').view(num_tokens, self.topk)
                % self.num_experts
            )

        local_experts = self.num_experts // self.ep_size
        hot_per_token = self.topk * self.maxvio / self.ep_size
        prefix = torch.arange(num_tokens + 1, dtype=torch.float64, device='cpu')
        hot_prefix = torch.floor(prefix * hot_per_token + 0.5).long()
        hot_count = (hot_prefix[1:] - hot_prefix[:-1]).unsqueeze(1)
        slots = torch.arange(self.topk, device='cpu').unsqueeze(0)
        hot_ids = (hot_prefix[:-1].unsqueeze(1) + slots) % local_experts
        cold_offset = torch.arange(num_tokens, device='cpu') * self.topk - hot_prefix[:-1]
        cold_index = cold_offset.unsqueeze(1) + slots - hot_count
        cold_rank = 1 + cold_index % (self.ep_size - 1)
        cold_expert = (cold_index // (self.ep_size - 1)) % local_experts
        return torch.where(slots < hot_count, hot_ids, cold_rank * local_experts + cold_expert)


def mock_route_scores(logits, indices, score_function, pre_softmax, scaling_factor, hash_layer):
    """Use cached choices with fresh differentiable weights, matching router score semantics."""
    selected = logits.gather(1, indices).float()
    if score_function == 'softmax':
        if pre_softmax or hash_layer:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32).gather(1, indices)
        else:
            probs = torch.softmax(selected, dim=-1)
    elif score_function in ('sigmoid', 'sqrtsoftplus'):
        probs = (
            torch.sigmoid(selected)
            if score_function == 'sigmoid'
            else torch.nn.functional.softplus(selected).sqrt()
        )
        if indices.shape[1] > 1 or hash_layer:
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-20)
    else:
        raise ValueError(f"Invalid score_function: {score_function}")
    if scaling_factor:
        probs = probs * scaling_factor
    return probs.type_as(logits)
