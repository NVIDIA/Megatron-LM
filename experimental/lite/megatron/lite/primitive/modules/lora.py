# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""LoRA helpers for Megatron Lite native model implementations.

This module is intentionally narrow: it supports the Qwen3-MoE lite path's
Megatron-style sharded linear surfaces, not arbitrary PEFT injection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

_DEFAULT_TARGET_MODULES = ("linear_qkv", "linear_proj", "linear_fc1", "linear_fc2")
_TARGET_ALIASES = {
    "qkv": "linear_qkv",
    "proj": "linear_proj",
    "fc1": "linear_fc1",
    "fc2": "linear_fc2",
}


@dataclass(frozen=True)
class LoraConfig:
    rank: int = 0
    alpha: int | None = None
    dropout: float = 0.0
    target_modules: tuple[str, ...] = field(default_factory=lambda: _DEFAULT_TARGET_MODULES)

    @property
    def enabled(self) -> bool:
        return self.rank > 0

    @property
    def scale(self) -> float:
        return float(self.rank if self.alpha is None else self.alpha) / float(self.rank)

    def targets(self) -> set[str]:
        out = set()
        for target in self.target_modules:
            out.add(_TARGET_ALIASES.get(target, target))
        return out

    def targets_module(self, name: str) -> bool:
        canonical = _TARGET_ALIASES.get(name, name)
        return canonical in self.targets()


def normalize_lora_config(config: LoraConfig | dict[str, Any] | None) -> LoraConfig:
    if config is None:
        return LoraConfig()
    if isinstance(config, LoraConfig):
        return config
    if not isinstance(config, dict):
        raise TypeError(f"LoRA config must be LoraConfig, dict, or None, got {type(config)!r}.")
    values = dict(config)
    enabled = values.pop("enabled", None)
    if enabled is False:
        values["rank"] = 0
    if "targets" in values and "target_modules" not in values:
        values["target_modules"] = values.pop("targets")
    else:
        values.pop("targets", None)
    if "target_modules" in values and not isinstance(values["target_modules"], tuple):
        values["target_modules"] = tuple(values["target_modules"])
    return LoraConfig(**values)


def freeze_non_lora_params(model: nn.Module) -> dict[str, int]:
    """Freeze base parameters and leave adapter parameters trainable."""

    lora_tensors = 0
    lora_numel = 0
    frozen_tensors = 0
    frozen_numel = 0
    for name, param in model.named_parameters():
        if "lora" in name.lower() or "adapter" in name.lower():
            param.requires_grad_(True)
            lora_tensors += 1
            lora_numel += param.numel()
        else:
            param.requires_grad_(False)
            frozen_tensors += 1
            frozen_numel += param.numel()
    return {
        "lora_tensors": lora_tensors,
        "lora_numel": lora_numel,
        "frozen_tensors": frozen_tensors,
        "frozen_numel": frozen_numel,
    }


def trainable_param_stats(model: nn.Module) -> dict[str, int]:
    tensors = 0
    numel = 0
    for param in model.parameters():
        if param.requires_grad:
            tensors += 1
            numel += param.numel()
    return {"trainable_tensors": tensors, "trainable_numel": numel}


def _gather_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherSequence.apply(x, group)


def _reduce_scatter_sequence_parallel(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ReduceScatterSequence.apply(x, group)


def _scatter_sequence_parallel(x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _ScatterSequence.apply(x, group, group_rank)


def _all_reduce_sum(x: torch.Tensor, group) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllReduceSum.apply(x, group)


class _AllGatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_seq = x.shape[0]
        out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.all_gather_into_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty((ctx.local_seq, *grad.shape[1:]), dtype=grad.dtype, device=grad.device)
        dist.reduce_scatter_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ReduceScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(
                f"Cannot reduce-scatter sequence dim {x.shape[0]} over TP={world_size}."
            )
        ctx.group = group
        ctx.world_size = world_size
        out = torch.empty((x.shape[0] // world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
        dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None


class _ScatterSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group, group_rank: int) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        if x.shape[0] % world_size != 0:
            raise ValueError(f"Cannot scatter sequence dim {x.shape[0]} over TP={world_size}.")
        ctx.group = group
        ctx.world_size = world_size
        local_seq = x.shape[0] // world_size
        start = int(group_rank) * local_seq
        return x[start : start + local_seq].contiguous()

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = torch.empty(
            (grad.shape[0] * ctx.world_size, *grad.shape[1:]), dtype=grad.dtype, device=grad.device
        )
        dist.all_gather_into_tensor(out, grad.contiguous(), group=ctx.group)
        return out, None, None


class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, group) -> torch.Tensor:
        ctx.group = group
        out = x.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        out = grad.contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out, None


def _all_gather_last_dim(x: torch.Tensor, group, *, reduce_backward: bool = False) -> torch.Tensor:
    if group is None or dist.get_world_size(group) == 1:
        return x
    return _AllGatherLastDim.apply(x, group, reduce_backward)


class _AllGatherLastDim(torch.autograd.Function):
    """All-gather last dim with Megatron tensor-parallel split backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, reduce_backward: bool) -> torch.Tensor:
        world_size = dist.get_world_size(group)
        ctx.group = group
        ctx.local_width = x.shape[-1]
        ctx.group_rank = dist.get_rank(group)
        ctx.reduce_backward = bool(reduce_backward)
        flat = x.movedim(-1, 0).contiguous().view(ctx.local_width, -1)
        gathered = torch.empty(
            (ctx.local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
        )
        dist.all_gather_into_tensor(gathered, flat, group=group)
        return (
            gathered.view(ctx.local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()
        )

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        flat = grad.movedim(-1, 0).contiguous().view(grad.shape[-1], -1)
        start = ctx.group_rank * ctx.local_width
        out = flat.narrow(0, start, ctx.local_width).contiguous()
        if ctx.reduce_backward:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=ctx.group)
        return out.view(ctx.local_width, *grad.shape[:-1]).movedim(0, -1).contiguous(), None, None


class _SequenceParallelRankPartitionedLoRA(torch.autograd.Function):
    """QKV LoRA path that recomputes gathered activations in backward.

    The ordinary composition of all-gather + matmul saves the full
    sequence-parallel gathered input for every layer. For QKV LoRA that input
    is much larger than the low-rank hidden activation. This function saves
    only the local input plus LoRA weights, then repeats the small gather/matmul
    sequence during backward.
    """

    @staticmethod
    def forward(
        ctx, x: torch.Tensor, lora_a: torch.Tensor, lora_b: torch.Tensor, scale: float, group
    ):
        world_size = dist.get_world_size(group) if group is not None else 1
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)
        out = hidden.matmul(lora_b.t()) * scale
        ctx.save_for_backward(x, lora_a, lora_b)
        ctx.group = group
        ctx.world_size = world_size
        ctx.local_seq = x.shape[0]
        ctx.local_rank_width = hidden_local.shape[-1]
        ctx.scale = float(scale)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        x, lora_a, lora_b = ctx.saved_tensors
        world_size = ctx.world_size
        group = ctx.group
        if world_size > 1:
            gathered = _all_gather_sequence_forward(x, group, world_size)
        else:
            gathered = x
        hidden_local = gathered.matmul(lora_a.t())
        hidden = _all_gather_last_dim_forward(hidden_local, group, world_size)

        grad_out_scaled = grad_out * ctx.scale
        grad_b = (
            grad_out_scaled.reshape(-1, grad_out_scaled.shape[-1])
            .t()
            .matmul(hidden.reshape(-1, hidden.shape[-1]))
        )
        grad_hidden = grad_out_scaled.matmul(lora_b)
        if world_size > 1:
            grad_hidden_local = _split_last_dim(
                grad_hidden, dist.get_rank(group), ctx.local_rank_width
            )
            dist.all_reduce(grad_hidden_local, op=dist.ReduceOp.SUM, group=group)
        else:
            grad_hidden_local = grad_hidden
        grad_a = (
            grad_hidden_local.reshape(-1, grad_hidden_local.shape[-1])
            .t()
            .matmul(gathered.reshape(-1, gathered.shape[-1]))
        )
        grad_gathered = grad_hidden_local.matmul(lora_a)
        if world_size > 1:
            grad_x = _reduce_scatter_sequence_forward(grad_gathered, group, ctx.local_seq)
        else:
            grad_x = grad_gathered
        return grad_x, grad_a, grad_b, None, None


def _all_gather_sequence_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    out = torch.empty((x.shape[0] * world_size, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.all_gather_into_tensor(out, x.contiguous(), group=group)
    return out


def _reduce_scatter_sequence_forward(x: torch.Tensor, group, local_seq: int) -> torch.Tensor:
    out = torch.empty((local_seq, *x.shape[1:]), dtype=x.dtype, device=x.device)
    dist.reduce_scatter_tensor(out, x.contiguous(), group=group)
    return out


def _all_gather_last_dim_forward(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    if world_size == 1:
        return x
    local_width = x.shape[-1]
    flat = x.movedim(-1, 0).contiguous().view(local_width, -1)
    gathered = torch.empty(
        (local_width * world_size, flat.shape[1]), dtype=x.dtype, device=x.device
    )
    dist.all_gather_into_tensor(gathered, flat, group=group)
    return gathered.view(local_width * world_size, *x.shape[:-1]).movedim(0, -1).contiguous()


def _split_last_dim(x: torch.Tensor, group_rank: int, local_width: int) -> torch.Tensor:
    start = int(group_rank) * local_width
    return x.narrow(-1, start, local_width).contiguous()


class LinearLoRA(nn.Module):
    """Low-rank delta for a sharded linear layer.

    `a` is replicated unless the caller feeds a row-parallel local input. `b`
    has the local output shard for column-parallel surfaces, and the replicated
    full output for row-parallel surfaces.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
        sequence_parallel_input: bool = False,
        row_parallel_output: bool = False,
        sequence_parallel_scatter_output: bool = False,
        tp_group=None,
        tp_rank: int = 0,
        rank_partition_size: int | None = None,
        rank_partitioned_a: bool = False,
        input_parallel_reduce: bool = False,
        output_partition_size: int | None = None,
        output_partitioned_b: bool = False,
        a_tensor_model_parallel: bool = False,
        b_tensor_model_parallel: bool = False,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for LinearLoRA.")
        self.rank = int(rank)
        self.rank_partitioned_a = bool(rank_partitioned_a)
        if self.rank_partitioned_a:
            partition_size = (
                int(rank_partition_size)
                if rank_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA rank partition size must be positive.")
            if self.rank % partition_size != 0:
                raise ValueError(
                    f"LoRA rank {self.rank} must be divisible by rank partition size {partition_size}."
                )
            self.rank_partition_size = partition_size
            self.local_rank = self.rank // partition_size
        else:
            self.rank_partition_size = 1
            self.local_rank = self.rank
        self.scale = float(rank if alpha is None else alpha) / float(rank)
        self.dropout_p = float(dropout)
        self.sequence_parallel_input = bool(sequence_parallel_input)
        self.row_parallel_output = bool(row_parallel_output)
        self.sequence_parallel_scatter_output = bool(sequence_parallel_scatter_output)
        if self.row_parallel_output and self.sequence_parallel_scatter_output:
            raise ValueError(
                "Use either row_parallel_output or sequence_parallel_scatter_output, not both."
            )
        self.tp_group = tp_group
        self.tp_rank = int(tp_rank)
        self.input_parallel_reduce = bool(input_parallel_reduce)
        self.output_partitioned_b = bool(output_partitioned_b)
        if self.output_partitioned_b:
            partition_size = (
                int(output_partition_size)
                if output_partition_size is not None
                else (dist.get_world_size(tp_group) if tp_group is not None else 1)
            )
            if partition_size <= 0:
                raise ValueError("LoRA output partition size must be positive.")
            if out_features % partition_size != 0:
                raise ValueError(
                    f"LoRA output features {out_features} must be divisible by {partition_size}."
                )
            self.output_partition_size = partition_size
            self.local_out_features = out_features // partition_size
        else:
            self.output_partition_size = 1
            self.local_out_features = out_features
        self.lora_a = nn.Parameter(torch.empty(self.local_rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(self.local_out_features, rank))
        self.lora_a.tensor_model_parallel = bool(a_tensor_model_parallel)
        self.lora_b.tensor_model_parallel = bool(b_tensor_model_parallel)
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sequence_parallel_input and self.rank_partitioned_a and not self.training:
            # Keep eval/inference on the simple path; the memory optimization
            # matters only when autograd needs to retain forward activations.
            pass
        elif (
            self.sequence_parallel_input
            and self.rank_partitioned_a
            and not self.input_parallel_reduce
            and not self.output_partitioned_b
            and not self.row_parallel_output
            and not self.sequence_parallel_scatter_output
            and self.dropout_p == 0.0
        ):
            return _SequenceParallelRankPartitionedLoRA.apply(
                x, self.lora_a, self.lora_b, self.scale, self.tp_group
            )
        if self.sequence_parallel_input:
            x = _gather_sequence_parallel(x, self.tp_group)
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        hidden = dropped.matmul(self.lora_a.t())
        if self.rank_partitioned_a:
            hidden = _all_gather_last_dim(hidden, self.tp_group, reduce_backward=True)
        if self.input_parallel_reduce:
            hidden = _all_reduce_sum(hidden, self.tp_group)
        out = hidden.matmul(self.lora_b.t()) * self.scale
        if self.output_partitioned_b:
            out = _all_gather_last_dim(out, self.tp_group)
        if self.row_parallel_output:
            out = _reduce_scatter_sequence_parallel(out, self.tp_group)
        if self.sequence_parallel_scatter_output:
            out = _scatter_sequence_parallel(out, self.tp_group, self.tp_rank)
        return out


class GroupedLinearLoRA(nn.Module):
    """Per-local-expert LoRA delta for `te.GroupedLinear` expert surfaces."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for GroupedLinearLoRA.")
        self.num_local_experts = int(num_local_experts)
        self.rank = int(rank)
        self.scale = float(rank if alpha is None else alpha) / float(rank)
        self.dropout_p = float(dropout)
        self.lora_a = nn.Parameter(torch.empty(num_local_experts, rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(num_local_experts, out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        if len(splits) != self.num_local_experts:
            raise ValueError(
                f"GroupedLinearLoRA expected {self.num_local_experts} splits, got {len(splits)}."
            )
        outputs = []
        offset = 0
        for expert_idx, size in enumerate(splits):
            x_i = x[offset : offset + size]
            if size == 0:
                outputs.append(x_i.new_empty((0, self.lora_b.shape[1])))
            else:
                dropped = (
                    F.dropout(x_i, p=self.dropout_p, training=self.training)
                    if self.dropout_p
                    else x_i
                )
                h_i = dropped.matmul(self.lora_a[expert_idx].t())
                outputs.append(h_i.matmul(self.lora_b[expert_idx].t()) * self.scale)
            offset += size
        return torch.cat(outputs, dim=0) if outputs else x.new_empty((0, self.lora_b.shape[1]))


class SharedGroupedLinearLoRA(nn.Module):
    """LoRA delta shared by all local experts in a GroupedLinear."""

    def __init__(
        self,
        num_local_experts: int,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        alpha: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive for SharedGroupedLinearLoRA.")
        self.num_local_experts = int(num_local_experts)
        self.rank = int(rank)
        self.scale = float(rank if alpha is None else alpha) / float(rank)
        self.dropout_p = float(dropout)
        self.shared_across_experts = True
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.empty(out_features, rank))
        self.lora_a.tensor_model_parallel = False
        self.lora_b.tensor_model_parallel = False
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, splits: list[int]) -> torch.Tensor:
        if len(splits) != self.num_local_experts:
            raise ValueError(
                f"SharedGroupedLinearLoRA expected {self.num_local_experts} splits, got {len(splits)}."
            )
        dropped = F.dropout(x, p=self.dropout_p, training=self.training) if self.dropout_p else x
        return dropped.matmul(self.lora_a.t()).matmul(self.lora_b.t()) * self.scale


__all__ = [
    "GroupedLinearLoRA",
    "LinearLoRA",
    "LoraConfig",
    "SharedGroupedLinearLoRA",
    "freeze_non_lora_params",
    "normalize_lora_config",
    "trainable_param_stats",
]
