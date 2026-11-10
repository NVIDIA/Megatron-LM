# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# License for these changes: BSD-3-Clause.

"""Bounded duplicated-mode expert NS batching with native optimizer updates."""

import torch

from megatron.core.optimizer.muon_expert_normalization import reference_norm_batched_ns


class ExpertMuonBatch:
    """Serve independent expert orthogonalizations in native parameter traversal order."""

    def __init__(self, optimizer, replay, block=32):
        self.optimizer, self.replay = optimizer, replay
        self.chunks, self.groups, self.outputs, self.visited = {}, {}, {}, set()
        for group in optimizer.param_groups:
            shapes = {}
            for p in group['params']:
                if (
                    not group.get('is_expert_parallel', False)
                    or p.grad is None
                    or tuple(p.shape) not in ((1856, 672), (672, 1856))
                ):
                    continue
                if optimizer.split_qkv and optimizer.is_qkv_fn(p):
                    raise ValueError('Expert batching cannot split QKV parameters')
                if (
                    p.dtype != torch.float32
                    or p.grad.dtype != torch.float32
                    or not p.is_contiguous()
                    or not p.grad.is_contiguous()
                    or not getattr(p, 'expert_tp', False)
                ):
                    raise ValueError(
                        'Expert batching requires contiguous local FP32 expert '
                        'parameters and gradients'
                    )
                state = optimizer.state.get(p, {})
                if not state:
                    continue  # Native first step initializes each momentum itself.
                if (
                    set(state) != {'momentum_buffer'}
                    or state['momentum_buffer'].dtype != torch.float32
                ):
                    raise ValueError('Unsupported expert optimizer state')
                shapes.setdefault(tuple(p.shape), []).append(p)
            for values in shapes.values():
                for start in range(0, len(values), block):
                    chunk = values[start : start + block]
                    for p in chunk:
                        if id(p) in self.chunks:
                            raise ValueError('Duplicate expert parameter')
                        self.chunks[id(p)], self.groups[id(p)] = chunk, group

    def consume(self, p, grad):
        """Return a batched result or None for the unmodified native fallback."""
        key = id(p)
        if key not in self.chunks:
            return None
        if key in self.visited:
            raise RuntimeError('Repeated native expert callback')
        self.visited.add(key)
        if key not in self.outputs:
            chunk, group = self.chunks[key], self.groups[key]
            if chunk[0] is not p:
                raise RuntimeError('Native expert parameter traversal changed')
            previews = [grad] + [
                self.optimizer.state[q]['momentum_buffer'].lerp(q.grad, 1 - group['momentum'])
                for q in chunk[1:]
            ]
            packed = torch.stack(previews)
            result = self.replay(
                packed,
                steps=16,
                coefficient_type='polar_express',
                tp_group=self.optimizer.pg_collection.expt_tp,
                partition_dim=None,
                tp_mode='duplicated',
            )
            self.outputs.update(
                {id(q): value for q, value in zip(chunk, result.unbind(0), strict=True)}
            )
        from megatron.core.optimizer.emerging_optimizers import get_muon_scale_factor

        return self.outputs.pop(key) * get_muon_scale_factor(*p.shape, mode='spectral') * 0.2

    def finish(self):
        """Reject missing callbacks without masking an active native error."""
        if self.visited != set(self.chunks) or self.outputs:
            raise RuntimeError('Missing native expert callbacks')


def new_expert_replay():
    """Bind native per-matrix normalization without requiring the separate graph-replay PR."""
    return reference_norm_batched_ns
