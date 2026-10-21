# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core-owned experts, groups and autograd collectives; R1's slot arithmetic.

Core's dispatcher unpermutes by summing expert slots then adds the shared
expert. MegaMoE starts from the shared expert and adds slots in slot order.
Keep that arithmetic here, dispatching actual routed rows with Core A2A.
Stable expert/source/token/slot order and copies preserve signed-zero bits.
"""
import torch

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.transformer.moe.moe_layer import MoELayer

from . import experts
from .linear import linear
from .routing import route


def combine(slots, indices, shared):
    result = torch.zeros_like(shared, dtype=torch.float32) + shared.float()
    for slot in range(indices.shape[1]):
        result = torch.where(
            (indices[:, slot] >= 0).unsqueeze(-1), result + slots[:, slot].float(), result
        )
    return result.bfloat16()


def dispatch_rows(x, scores, indices, encoded, local_experts, group):
    # Stable global expert order also makes contiguous expert ownership peer-major.
    valid = (indices.flatten() >= 0).nonzero().flatten()
    position = valid[torch.argsort(indices.flatten()[valid], stable=True)]
    expert = indices.flatten()[position]
    sends = torch.bincount(expert // local_experts, minlength=group.size())
    receives = all_to_all(group, sends).tolist()
    sends = sends.tolist()
    token = position // indices.shape[1]
    received_ids = all_to_all(group, expert, receives, sends)
    # Incoming chunks are source-rank-major. Stable expert sort reproduces the
    # old all-gather's expert/source/token/slot order, including padded gaps.
    order = torch.argsort(received_ids, stable=True)
    counts = torch.bincount(received_ids % local_experts, minlength=local_experts).tolist()
    live = all_to_all(group, x[token], receives, sends)[order]
    weights = all_to_all(group, scores.flatten()[position], receives, sends)[order]
    codes = all_to_all(group, encoded[0].view(torch.uint8)[token], receives, sends)[order]
    scale = all_to_all(group, encoded[1][token], receives, sends)[order]
    state = position, torch.argsort(order), sends, receives
    return live, weights, (codes.view(torch.float8_e4m3fn), scale), counts, state


def return_rows(rows, indices, state, group):
    position, inverse, sends, receives = state
    returned = all_to_all(group, rows[inverse], sends, receives)
    slots = returned.new_zeros((indices.numel(), rows.shape[-1]))
    return slots.index_copy(0, position, returned).reshape(*indices.shape, rows.shape[-1])


class MegaMoE(MoELayer):
    def forward(self, hidden_states, input_ids=None, **kwargs):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        group, rank = self.ep_group, self.ep_group.rank()
        expected = list(range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts))
        if self.local_expert_indices != expected:
            raise ValueError('MegaMoE requires the Core contiguous global expert ownership map')
        logits = linear(x, self.router.weight, fp8=False)
        weights, indices = route(logits, self.router, input_ids)
        codes, scale = experts.quantize_activation(x)
        limit = self.config.activation_func_clamp_value
        shared = (
            experts.shared_rows(x, self.shared_experts, (codes, scale), limit)
            if self.shared_experts is not None
            else x * 0
        )

        live, scores, encoded, splits, dispatch = dispatch_rows(
            x, weights, indices, (codes, scale), self.num_local_experts, group
        )
        fc1 = [
            getattr(self.experts.linear_fc1, f'weight{i}') for i in range(self.num_local_experts)
        ]
        fc2 = [
            getattr(self.experts.linear_fc2, f'weight{i}') for i in range(self.num_local_experts)
        ]
        rows = experts.expert_rows(live, fc1, fc2, splits, scores, encoded, limit)
        selected = return_rows(rows, indices, dispatch, group)
        return combine(selected, indices, shared).reshape_as(hidden_states), None
