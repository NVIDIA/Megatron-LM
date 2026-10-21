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


def actual_rows(x, scores, indices, masters, limit, group, has_shared=True):
    local = len(masters) // 2 - int(has_shared)
    codes, scale = experts.quantize_activation(x)
    shared = x * 0
    if has_shared:
        first = experts._Shared.apply(x, masters[-2], (codes, scale))
        hidden, encoded = experts.activation(first, x.new_ones(x.shape[0]), limit)
        shared = experts._Shared.apply(hidden, masters[-1], encoded)
    live, weights, encoded, splits, state = dispatch_rows(
        x, scores, indices, (codes, scale), local, group
    )
    rows = experts.expert_rows(
        live, masters[:local], masters[local : 2 * local], splits, weights, encoded, limit
    )
    return combine(return_rows(rows, indices, state, group), indices, shared)


class _FusedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scores, indices, options, *masters):
        group, limit, buffer, packed = options
        ctx.options = group, limit
        ctx.save_for_backward(x, scores, indices, *masters)
        return experts.launch(x, scores, indices, packed, buffer, limit, group.size())

    @staticmethod
    def backward(ctx, grad):
        x, scores, indices, *masters = ctx.saved_tensors
        group, limit = ctx.options
        need = (*ctx.needs_input_grad[:2], *ctx.needs_input_grad[4:])
        with torch.enable_grad():
            leaves = [t.detach().requires_grad_(n) for t, n in zip((x, scores, *masters), need)]
            out = actual_rows(leaves[0], leaves[1], indices, leaves[2:], limit, group)
            wanted = [t for t in leaves if t.requires_grad]
            gradients = iter(torch.autograd.grad(out, wanted, grad))
            result = [next(gradients) if n else None for n in need]
        return result[0], result[1], None, None, *result[2:]


class MegaMoE(MoELayer):
    def forward(self, hidden_states, input_ids=None, **kwargs):
        x = hidden_states.reshape(-1, hidden_states.shape[-1])
        group, rank = self.ep_group, self.ep_group.rank()
        expected = list(range(rank * self.num_local_experts, (rank + 1) * self.num_local_experts))
        if self.local_expert_indices != expected:
            raise ValueError('MegaMoE requires the Core contiguous global expert ownership map')
        logits = linear(x, self.router.weight, fp8=False)
        weights, indices = route(logits, self.router, input_ids)
        limit = self.config.activation_func_clamp_value
        fc1 = [
            getattr(self.experts.linear_fc1, f'weight{i}') for i in range(self.num_local_experts)
        ]
        fc2 = [
            getattr(self.experts.linear_fc2, f'weight{i}') for i in range(self.num_local_experts)
        ]
        masters = (*fc1, *fc2)
        if self.shared_experts is None:
            out = actual_rows(x, weights, indices, masters, limit, group, False)
        else:
            masters += (
                self.shared_experts.linear_fc1.weight,
                self.shared_experts.linear_fc2.weight,
            )
            # Quantized operands are fresh on every call; only the symmetric buffer is cached.
            packed = experts.kernel_weights(fc1, fc2, *masters[-2:])
            capacity = 8192
            buffer = experts.symm_buffer(
                group,
                len(fc1) * group.size(),
                capacity,
                indices.shape[1],
                x.shape[1],
                fc1[0].shape[0] // 2,
            )
            chunks = torch.tensor(max(1, (x.shape[0] + capacity - 1) // capacity), device=x.device)
            torch.distributed.all_reduce(chunks, torch.distributed.ReduceOp.MAX, group=group)
            outputs = []
            for chunk in range(int(chunks)):
                rows = slice(chunk * capacity, (chunk + 1) * capacity)
                outputs.append(
                    _FusedMoE.apply(
                        x[rows],
                        weights[rows],
                        indices[rows],
                        (group, limit, buffer, packed),
                        *masters,
                    )
                )
            out = outputs[0] if len(outputs) == 1 else torch.cat(outputs)
        return out.reshape_as(hidden_states), None
