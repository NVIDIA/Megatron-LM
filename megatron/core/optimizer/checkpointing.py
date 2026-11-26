# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed optimizer checkpoint compatibility."""

from collections.abc import Iterator, Mapping

from .distrib_optimizer import DistributedOptimizer
from .optimizer import MegatronOptimizer


def contains_distributed_optimizer(optimizer: MegatronOptimizer | None) -> bool:
    """Whether the active optimizer tree includes a tensor-sharded DistributedOptimizer."""
    pending = [optimizer]
    while pending:
        current = pending.pop()
        if isinstance(current, DistributedOptimizer):
            return True
        if isinstance(current, MegatronOptimizer):
            pending.extend(getattr(current, "chained_optimizers", ()))
    return False


def _saved_sharding_types(state: object) -> Iterator[str]:
    """Read DistOpt layout tags from common state, including nested optimizer chains."""
    if isinstance(state, Mapping):
        layout = state.get("param_state_sharding_type")
        if isinstance(layout, str):
            yield layout
        else:
            for value in state.values():
                yield from _saved_sharding_types(value)
    elif isinstance(state, (tuple, list)):
        for value in state:
            yield from _saved_sharding_types(value)


def validate_optimizer_checkpoint_topology(
    optimizer: MegatronOptimizer | None,
    saved_optimizer_state: object,
    *,
    saved_tp_pp: tuple[int, int],
    current_tp_pp: tuple[int, int],
    sharding_type: str,
    distributed_optimizer_configured: bool,
) -> None:
    """Check a torch_dist optimizer restore before constructing its loading template.

    LayerWise can share the checkpoint's layout metadata without using tensor-sharded
    buffers. Apply the restriction to saved DistOpt leaves and to the layout requested
    by the target. The ordinary DistOpt config also covers checkpoints without leaf tags.
    """
    if optimizer is None or saved_tp_pp == current_tp_pp:
        return
    layouts = set(_saved_sharding_types(saved_optimizer_state))
    if distributed_optimizer_configured or contains_distributed_optimizer(optimizer):
        layouts.add(sharding_type)
    unsupported = layouts - DistributedOptimizer.checkpoint_fully_reshardable_formats
    if unsupported:
        raise RuntimeError(
            f"Cannot restore DistributedOptimizer state from (TP, PP) {saved_tp_pp} "
            f"to {current_tp_pp} with sharding type {', '.join(sorted(unsupported))}. "
            "Resume with the original TP/PP and re-save with "
            "checkpoint.dist_ckpt_optim_fully_reshardable=True "
            "(--dist-ckpt-optim-fully-reshardable), or load only the model weights."
        )
