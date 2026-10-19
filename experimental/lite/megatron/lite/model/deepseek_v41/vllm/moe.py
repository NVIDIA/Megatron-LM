# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Core-owned experts, groups and autograd collectives; R1's slot arithmetic.

Core's dispatcher unpermutes by summing expert slots then adds the shared
expert. MegaMoE starts from the shared expert and adds slots in slot order.
Keep that arithmetic here, reusing Core AG/RS and A2A rather than another
collective implementation. Owner selection is a copy, including signed zero.
"""
import torch
from torch.nn.functional import pad

from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region
from megatron.core.transformer.moe.moe_layer import MoELayer

from . import experts
from .linear import linear
from .routing import route


def _gather(x, group):
    return gather_from_sequence_parallel_region(x, group=group)


def combine(slots, indices, shared):
    result = torch.zeros_like(shared, dtype=torch.float32) + shared.float()
    for slot in range(indices.shape[1]):
        result = torch.where(
            (indices[:, slot] >= 0).unsqueeze(-1), result + slots[:, slot].float(), result
        )
    return result.bfloat16()


def return_slots(slots, local_ids, local_experts, group):
    received = all_to_all(group, slots)
    parts = received.permute(1, 2, 0, 3)
    owner = local_ids.clamp_min(0).long() // local_experts
    return parts.gather(2, owner[:, :, None, None].expand(-1, -1, 1, slots.shape[-1])).squeeze(2)


class MegaMoE(MoELayer):
    def forward(self, hidden_states, input_ids=None, **kwargs):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        group, world, rank = self.ep_group, self.ep_group.size(), self.ep_group.rank()
        expected = list(range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts))
        if self.local_expert_indices != expected:
            raise ValueError('MegaMoE requires the Core contiguous global expert ownership map')
        logits = linear(x, self.router.weight, fp8=False)
        weights, indices = route(logits, self.router, input_ids)
        counts = _gather(torch.tensor([x.shape[0]], dtype=torch.int64, device=x.device), group)
        capacity = max(1, int(counts.max()))
        codes, scale = experts.quantize_activation(x)
        limit = self.config.activation_func_clamp_value
        shared = (
            experts.shared_rows(x, self.shared_experts, (codes, scale), limit)
            if self.shared_experts is not None
            else x * 0
        )

        def gather(value, fill=0):
            return _gather(pad(value, (0, 0, 0, capacity - value.shape[0]), value=fill), group)

        encoded = gather(codes.view(torch.uint8)), gather(scale, 1)
        live, scores, routes = gather(x), gather(weights), gather(indices, -1)
        positions = [(routes.flatten() == expert).nonzero().flatten() for expert in expected]
        splits = tuple(p.numel() for p in positions)
        position = torch.cat(positions)
        token = position // routes.shape[1]
        fc1 = [
            getattr(self.experts.linear_fc1, f'weight{i}') for i in range(self.num_local_experts)
        ]
        fc2 = [
            getattr(self.experts.linear_fc2, f'weight{i}') for i in range(self.num_local_experts)
        ]
        rows = experts.expert_rows(
            live[token],
            fc1,
            fc2,
            splits,
            scores.flatten()[position],
            (encoded[0][token].view(torch.float8_e4m3fn), encoded[1][token]),
            limit,
        )
        slots = rows.new_zeros((routes.numel(), x.shape[1])).index_copy(0, position, rows)
        selected = return_slots(
            slots.reshape(world, capacity, routes.shape[1], x.shape[1]),
            pad(indices, (0, 0, 0, capacity - x.shape[0]), value=-1),
            self.num_local_experts,
            group,
        )[: x.shape[0]]
        return combine(selected, indices, shared).reshape_as(hidden_states), None
