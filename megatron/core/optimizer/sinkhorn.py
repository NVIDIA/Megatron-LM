# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Momentum optimizer using Emerging-Optimizers' Sinkhorn balancing.

Logical table boundaries and row-shard metadata are attached before optimizer
wrapping and copied to master parameters by the TP attribute-copy helper.
"""

import math

import torch
import torch.distributed as dist
from emerging_optimizers.orthogonalized_optimizers import OrthogonalizedOptimizer
from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import sinkhorn_balance

from megatron.core.ops.optimizers.sinkhorn import balance_row_shards


class SinkhornOptimizer(OrthogonalizedOptimizer):
    """Nesterov EMA followed by independent logical-table Sinkhorn updates.

    Only FP32 master parameters with complete columns are supported. Global mode
    reduces axis statistics over distinct row owners; local mode deliberately
    balances each row partition separately. Both retain just one momentum tensor.
    """

    def __init__(
        self,
        params,
        lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        num_normalization_steps: int = 11,
        zero_row_threshold: float = 1e-3,
        eps: float = 1e-20,
        extra_scale_factor: float = 0.18,
        balance_mode: str = "global",
    ) -> None:
        if not 0 <= momentum < 1 or not math.isfinite(momentum):
            raise ValueError("Sinkhorn momentum must be finite and in [0, 1)")
        if (
            type(num_normalization_steps) is not int
            or num_normalization_steps < 1
            or num_normalization_steps % 2 == 0
        ):
            raise ValueError("Sinkhorn normalization steps must be a positive odd integer")
        if eps <= 0 or not math.isfinite(eps):
            raise ValueError("Sinkhorn eps must be positive and finite")
        if zero_row_threshold < 0 or not math.isfinite(zero_row_threshold):
            raise ValueError("Sinkhorn row threshold must be nonnegative and finite")
        if extra_scale_factor <= 0 or not math.isfinite(extra_scale_factor):
            raise ValueError("Sinkhorn scale factor must be positive and finite")
        if balance_mode not in ("global", "local"):
            raise ValueError("Sinkhorn balance_mode must be global or local")
        super().__init__(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=0.0,
            nesterov=nesterov,
            weight_decay_method="decoupled",
            fp32_matmul_prec="highest",
            scaled_orthogonalize_fn=sinkhorn_balance,
        )
        self.balance_mode = balance_mode
        settings = dict(
            num_normalization_steps=num_normalization_steps,
            zero_row_threshold=zero_row_threshold,
            eps=eps,
            extra_scale_factor=extra_scale_factor,
            balance_mode=balance_mode,
            nesterov=nesterov,
        )
        for group in self.param_groups:
            group.update(settings)
            group["weight_decay"] = 0.0
            group["wd_mult"] = 0.0
            for p in group["params"]:
                self._validate_parameter(p)

    def _validate_parameter(self, p) -> None:
        if p.ndim != 2 or p.dtype not in (torch.float32, torch.bfloat16) or p.is_sparse:
            raise ValueError("Sinkhorn requires dense 2D FP32/BF16 model parameters")
        ranges = getattr(p, "sinkhorn_table_ranges", None)
        row_start = getattr(p, "sinkhorn_row_start", None)
        if not ranges or row_start is None:
            raise ValueError("Sinkhorn requires logical table ranges and row-shard offsets")
        previous = 0
        for start, end in ranges:
            if start != previous or end <= start or end - start < p.shape[1]:
                raise ValueError("Sinkhorn logical tables must be contiguous, nonempty and tall")
            previous = end
        if row_start < 0 or row_start + p.shape[0] > previous:
            raise ValueError("Sinkhorn row shard lies outside its logical tables")
        if row_start != 0 or p.shape[0] != previous:
            if self.balance_mode == "global":
                group = getattr(p, "sinkhorn_group", None)
                if group is None or group.size() <= 1:
                    raise ValueError(
                        "Global Sinkhorn row shards require an explicit multi-rank process group"
                    )
        if getattr(p, "is_gtp_weight_remat", False):
            raise ValueError("Sinkhorn GTP-rematerialized tables are not supported")
        if getattr(p, "partition_dim", -1) not in (-1, 0):
            raise ValueError("Sinkhorn prototype requires complete feature rows")

    def orthogonalize(self, p, grad, **kwargs):
        """Balance each logical table without altering persistent momentum or packing."""
        row_start = p.sinkhorn_row_start
        result = torch.empty_like(grad)
        for start, end in p.sinkhorn_table_ranges:
            # Include empty intersections in the global collective schedule.
            begin = min(max(start - row_start, 0), p.shape[0])
            stop = min(max(end - row_start, 0), p.shape[0])
            result[begin:stop] = (
                balance_row_shards(
                    grad[begin:stop],
                    global_rows=end - start,
                    group=p.sinkhorn_group,
                    upstream_balance=self.scaled_orthogonalize_fn,
                    num_normalization_steps=kwargs["num_normalization_steps"],
                    zero_row_threshold=kwargs["zero_row_threshold"],
                    eps=kwargs["eps"],
                    local=self.balance_mode == "local",
                )
                * kwargs["extra_scale_factor"]
            )
        return result

    def step(self, closure=None):
        """Update finalized gradients; coordinate unused parameters over row owners."""
        if closure is not None:
            raise ValueError("Sinkhorn does not support closures")
        for group in self.param_groups:
            if group["weight_decay"] != 0:
                raise ValueError("Sinkhorn updates do not apply weight decay")
            for p in group["params"]:
                if p.dtype != torch.float32 or (p.grad is not None and p.grad.is_sparse):
                    raise ValueError("Sinkhorn requires FP32 masters and finalized dense gradients")
                if self.balance_mode == "global" and p.sinkhorn_group is not None:
                    if dist.get_world_size(p.sinkhorn_group) > 1:
                        # Ranks with no lookup contribution must still participate. Only
                        # skip when the parameter is unused on every distinct row owner.
                        active = torch.tensor(int(p.grad is not None), device=p.device)
                        dist.all_reduce(active, op=dist.ReduceOp.MAX, group=p.sinkhorn_group)
                        if active.item() and p.grad is None:
                            p.grad = torch.zeros_like(p)
        return super().step(closure)

    def load_state_dict(self, state_dict):
        """Prevent silent changes to balancing geometry or Nesterov on resume."""
        for group in state_dict["param_groups"]:
            if group["balance_mode"] != self.balance_mode or group["nesterov"] != self.nesterov:
                raise ValueError(
                    "Sinkhorn balancing mode/Nesterov changed on resume; reset optimizer state explicitly"
                )
            if group["weight_decay"] != 0:
                raise ValueError("Sinkhorn checkpoint has nonzero weight decay")
        return super().load_state_dict(state_dict)
