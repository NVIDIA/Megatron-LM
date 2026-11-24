# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bind logical parameter norm constraints to distributed optimizer shards."""

from collections.abc import Callable
from dataclasses import dataclass, replace

import torch

from .. import tensor_parallel
from ..fp8_utils import is_grouped_tensor
from .param_norm_constraints import (
    EmbeddingMinNorm,
    EmbeddingMinNormStats,
    ParamNormConstraint,
    ParamNormRecipe,
    collect_param_norm_constraints,
)
from .param_norm_validation import ParamNormAgreement


@dataclass(frozen=True)
class _Rows:
    """Track which matrix row each value in this rank's flat shard belongs to.

    With width=3, start=2, and end=7, this rank skips the first two matrix elements
    and owns the next five. Dots mark elements on other ranks.

               matrix row     this rank's sum of squares
        row 0   .  .  1       1² = 1
        row 1   2  3  4       2² + 3² + 4² = 29
        row 2   5  .  .       5² = 25

    If only row 1 needs to double, scale_ changes the flat shard from
    [1, 2, 3, 4, 5] to [1, 4, 6, 8, 5].
    """

    start: int
    end: int
    width: int

    def add_square_sums(self, sums: torch.Tensor, shard: torch.Tensor) -> None:
        """Add each shard element's square to the sum for its matrix row."""
        row, offset = divmod(self.start, self.width)
        position = 0
        if offset:
            head = min(self.width - offset, shard.numel())
            sums[row] += shard[:head].square().sum()
            position = head
            row += 1
        num_full_rows = (shard.numel() - position) // self.width
        if num_full_rows:
            full_end = position + num_full_rows * self.width
            sums[row : row + num_full_rows] += (
                shard[position:full_end].view(num_full_rows, self.width).square().sum(dim=1)
            )
            position = full_end
            row += num_full_rows
        if position < shard.numel():
            sums[row] += shard[position:].square().sum()

    def scale_(self, shard: torch.Tensor, scales: torch.Tensor) -> None:
        """Multiply each shard element in place by the scale for its row."""
        rows = torch.arange(self.start, self.end, dtype=torch.int64, device=shard.device).div_(
            self.width, rounding_mode="floor"
        )
        shard.mul_(scales[rows])


@dataclass(frozen=True, eq=False)
class _ParamNormPlanEntry:
    """Describe how this rank applies one row norm rule to one parameter.

    At setup, ``constraint`` identifies the weight and rule. ``rows`` records which flat
    indices this rank owns and the row width. Each step reuses ``square_sums``, which has
    one FP32 slot per valid row in this model-parallel partition. Ranks in
    ``data_parallel_group`` sum their contributions there before the rule computes each
    row's multiplier.
    """

    constraint: ParamNormConstraint
    rows: _Rows
    data_parallel_group: torch.distributed.ProcessGroup
    square_sums: torch.Tensor

    def local_master_shard(self) -> torch.Tensor | None:
        """Return a writable view of this rank's FP32 values, or None for an empty shard.

        For the flat range [2, 7), an FP32 weight uses ``parameter.view(-1)[2:7]``.
        A BF16 or FP16 weight uses ``main_param[:5]`` because its master is already local.
        """
        start, end = self.rows.start, self.rows.end
        if start == end:
            return None
        parameter = self.constraint.parameter
        if parameter.dtype == torch.float32:
            return parameter.view(-1)[start:end]
        master = getattr(parameter, "main_param", None)
        if master is None or master.dtype != torch.float32:
            raise RuntimeError("Parameter norm constraint could not find the FP32 master shard")
        return master[: end - start]


@dataclass(frozen=True, eq=False)
class ParamNormPlan:
    """Apply one optimizer's norm constraints after each successful update.

    Entries follow parameter-name order so DP peers reduce the same tables in the same order.
    """

    entries: tuple[_ParamNormPlanEntry, ...]

    @torch.no_grad()
    def apply_(self) -> None:
        """Compute full row norms and scale the optimizer-owned elements in place."""
        for entry in self.entries:
            sums = entry.square_sums.zero_()
            shard = entry.local_master_shard()
            if shard is not None:
                entry.rows.add_square_sums(sums, shard)
            # Empty DP shards still participate in the table's reduction.
            torch.distributed.all_reduce(sums, group=entry.data_parallel_group)
            scales = entry.constraint.rule.multipliers(sums.sqrt())
            if shard is not None:
                entry.rows.scale_(shard, scales)


def bind_param_norm_constraint(
    constraint: ParamNormConstraint,
    *,
    local_range: Callable[[torch.nn.Parameter], tuple[int, int]],
    data_parallel_group: torch.distributed.ProcessGroup,
) -> _ParamNormPlanEntry | None:
    """Build this rank's plan entry for one parameter's row norm constraint.

    Each row must fit within one model-parallel partition. The optimizer's flat slice,
    given by ``local_range``, can start or end inside a row. Exclude padding from that
    slice and allocate one FP32 sum per valid row in the partition.

    Even a rank with an empty slice needs an entry to join the data-parallel reduction.
    Return ``None`` only when the entire model-parallel partition has zero valid rows.
    """
    parameter = constraint.parameter
    if parameter.ndim != 2 or is_grouped_tensor(parameter) or parameter.shape[1] == 0:
        raise ValueError("Row norm constraints require a dense matrix with positive row width")
    if parameter.dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise TypeError("Row norm constraints require an FP32, FP16, or BF16 parameter")
    if (
        getattr(parameter, "tensor_model_parallel", False)
        and getattr(parameter, "partition_dim", None) != 0
    ):
        raise ValueError("Row norm constraints require complete rows on each tensor-parallel rank")
    if (
        getattr(parameter, "is_gtp_weight_remat", False)
        and getattr(parameter, "group", None) is None
    ):
        raise ValueError("A GTP-sharded embedding requires its weight-rematerialization group")
    num_rows, width = parameter.shape
    padding = tensor_parallel.gtp_local_pad_zero_count(parameter, 0, parameter.numel())
    if padding % width:
        raise ValueError("GTP embedding padding must contain complete rows")
    num_rows -= padding // width
    if num_rows == 0:
        return None
    start, end = local_range(parameter)
    if not 0 <= start <= end <= parameter.numel():
        raise ValueError("The local optimizer range is outside the selected parameter")
    valid_end = num_rows * width
    return _ParamNormPlanEntry(
        constraint,
        _Rows(min(start, valid_end), min(end, valid_end), width),
        data_parallel_group,
        torch.zeros(num_rows, dtype=torch.float32, device=parameter.device),
    )


class ParamNormSetup:
    """Build and validate norm plans as the factory constructs the optimizers.

    The factory shares one setup object across its optimizer instances. It collects the
    tables selected by the recipe, records their owners, and attaches a frozen
    ``ParamNormPlan`` to each distributed optimizer that owns selected tables.

    Once all optimizers are built, ``validate`` checks ownership, agreement on DP
    reductions, and consistent rules for tied pipeline copies. Setup saves local errors
    until this check so peers can reach the same validation step.
    """

    def __init__(self, model_chunks, recipe: ParamNormRecipe) -> None:
        self.constraints = []
        self.owners = {}
        self.agreement = ParamNormAgreement()
        self.tied_rules = set()
        self.error = None
        try:
            self.constraints = collect_param_norm_constraints(model_chunks, recipe)
            rules = {c.parameter: c.rule.validation_key() for c in self.constraints}
            # A frozen physical copy must participate in the pipeline consistency check too.
            self.tied_rules = {
                rules.get(p)
                for chunk in model_chunks
                for p in chunk.parameters()
                if getattr(p, "shared_embedding", False)
            }
            if len(self.tied_rules) > 1:
                raise ValueError("Local tied word embedding copies have different norm constraints")
        except Exception as exc:
            # All construction peers must still reach the setup exchange.
            self.error = str(exc)

    def ownership(self, param_groups, owner: str, *, supported: bool):
        """Record the owner of matching constraints and return them for binding."""
        owned = {p for group in param_groups for p in group["params"]}
        selected = [c for c in self.constraints if c.parameter in owned]
        for c in selected:
            if c.parameter in self.owners:
                self.error = (
                    f"{c.name} has multiple optimizer owners: "
                    f"{self.owners[c.parameter]} and {owner}"
                )
            elif not supported:
                self.error = (
                    f"{c.name} requests embedding_min_norm but its owner {owner} cannot apply it"
                )
            self.owners[c.parameter] = owner
        return selected

    def bind_to_distrib_optimizer(self, optimizer, *, data_parallel_group, pg_collection):
        """Bind this optimizer's tables and record their DP reductions for validation."""
        owner = f"distributed {optimizer.config.optimizer}"
        selected = self.ownership(optimizer.optimizer.param_groups, owner, supported=True)
        if not selected or self.error is not None:
            return
        constraint = selected[0]
        try:
            if optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8:
                raise ValueError(
                    "embedding_min_norm requires optimizer-managed FP32 master parameters"
                )
            buffered = {p for buffer in optimizer.buffers for p in buffer.params}
            reporter = (
                data_parallel_group.rank() == 0 and optimizer.distributed_optimizer_instance_id == 0
            )
            stats = None
            entries = []
            for constraint in selected:
                param = constraint.parameter
                if param not in buffered:
                    raise ValueError("selected parameter is missing from its parameter buffers")
                if (
                    reporter
                    and isinstance(constraint.rule, EmbeddingMinNorm)
                    and not getattr(param, "shared", False)
                ):
                    if stats is None:
                        stats = EmbeddingMinNormStats()
                    constraint = replace(constraint, rule=replace(constraint.rule, stats=stats))
                entry = bind_param_norm_constraint(
                    constraint,
                    local_range=optimizer._local_param_range,
                    data_parallel_group=data_parallel_group,
                )
                if entry is not None:
                    entries.append(entry)
                self.agreement.add(
                    constraint,
                    owner,
                    data_parallel_group,
                    num_rows=entry.square_sums.numel() if entry is not None else 0,
                    pg_collection=pg_collection,
                )
            optimizer._param_norm_plan = ParamNormPlan(tuple(entries))
            optimizer._embedding_min_norm_stats = stats
        except Exception as exc:
            self.error = f"Cannot bind parameter norm constraint for {constraint.name}: {exc}"

    def validate(self, pg_collection, process_group=None):
        """Check optimizer coverage and agreement before the first step."""
        for constraint in self.constraints:
            if constraint.parameter not in self.owners and self.error is None:
                self.error = f"No optimizer owns selected parameter {constraint.name}"
                break
        self.agreement.validate(
            process_group,
            self.error,
            tied_group=getattr(pg_collection, "embd", None) if self.tied_rules else None,
            tied_rule=next(iter(self.tied_rules), None),
        )
