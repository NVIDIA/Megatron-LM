# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Layer and stage CP partition-mode policy helpers."""

from typing import Any, Optional

import torch

from megatron.core.context_parallel_layout import CpPartitionMode


def get_context_parallel_layout_chunk_indices(
    cp_size: int, cp_rank: int, cp_partition_mode: str
) -> torch.Tensor:
    """Return the two global chunk indices owned by this CP rank in ``cp_partition_mode``."""
    if cp_size < 1:
        raise ValueError(f"cp_size must be >= 1, got {cp_size}.")
    if not 0 <= cp_rank < cp_size:
        raise ValueError(f"cp_rank must be in [0, {cp_size}), got {cp_rank}.")

    if cp_partition_mode == "zigzag":
        return torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], dtype=torch.long)
    if cp_partition_mode == "contiguous":
        return torch.tensor([2 * cp_rank, 2 * cp_rank + 1], dtype=torch.long)
    raise ValueError(
        f"Unsupported context-parallel partition mode {cp_partition_mode!r} for "
        f"cp_size={cp_size}, cp_rank={cp_rank}."
    )


################################################################################
# Layer-to-CP-partition-mode mapping
################################################################################
# ``None`` is a meaningful result here: it means the module is token-layout
# agnostic and preserves whichever CP partition mode it receives.  It must not
# be used as the fallback for an unrecognized module type; unknown types should
# fail loudly so new layer implementations add an explicit partition-mode policy.


def _validate_required_cp_partition_mode(
    mode: Optional[str], *, module_name: str
) -> Optional[CpPartitionMode]:
    if mode is None or mode in ("zigzag", "contiguous"):
        return mode
    raise ValueError(f"Invalid CP partition mode requirement {mode!r} declared by {module_name}.")


def get_required_cp_partition_mode_for_layer(
    layer: Any, config: Any, *, cp_comm_type: Optional[str] = None
) -> Optional[CpPartitionMode]:
    """Return the CP partition mode required by a layer or attention-like module.

    Modules must declare their layout contract via ``required_cp_partition_mode``
    or ``get_required_cp_partition_mode()``.  Wrapper modules may delegate to
    ``inner_layer`` or ``self_attention``.
    """
    if cp_comm_type is None:
        cp_comm_type = getattr(config, "cp_comm_type", None)

    if layer is None:
        raise ValueError("Cannot determine CP partition mode for None.")

    module_name = layer.__class__.__name__
    get_required_mode = getattr(layer, "get_required_cp_partition_mode", None)
    if callable(get_required_mode):
        return _validate_required_cp_partition_mode(get_required_mode(), module_name=module_name)

    if hasattr(layer, "required_cp_partition_mode"):
        return _validate_required_cp_partition_mode(
            getattr(layer, "required_cp_partition_mode"), module_name=module_name
        )

    if hasattr(layer, "inner_layer"):
        return get_required_cp_partition_mode_for_layer(
            layer.inner_layer, getattr(layer, "config", config), cp_comm_type=cp_comm_type
        )
    if hasattr(layer, "self_attention"):
        return get_required_cp_partition_mode_for_layer(
            layer.self_attention, getattr(layer, "config", config), cp_comm_type=cp_comm_type
        )

    raise ValueError(f"Cannot determine CP partition mode for layer/module type {module_name!r}.")


def get_stage_entry_partition_mode(
    packed_seq_params: Optional[Any],
    expected_stage_entry_partition_mode: Optional[CpPartitionMode],
    *,
    owner_name: str,
    cp_group: Optional[Any] = None,
) -> Optional[CpPartitionMode]:
    """Return and validate the CP partition mode at a stage input boundary."""
    expected_stage_entry_partition_mode = _validate_required_cp_partition_mode(
        expected_stage_entry_partition_mode, module_name=owner_name
    )
    packed_partition_mode = _validate_required_cp_partition_mode(
        getattr(packed_seq_params, "cp_partition_mode", None), module_name=owner_name
    )

    stage_entry_partition_mode = (
        packed_partition_mode
        if packed_partition_mode is not None
        else expected_stage_entry_partition_mode
    )
    if expected_stage_entry_partition_mode is not None and stage_entry_partition_mode is not None:
        assert stage_entry_partition_mode == expected_stage_entry_partition_mode, (
            f"{owner_name} expected CP stage entry partition mode "
            f"{expected_stage_entry_partition_mode!r}, but packed_seq_params carries "
            f"{stage_entry_partition_mode!r}."
        )

    effective_cp_group = (
        cp_group if cp_group is not None else getattr(packed_seq_params, "cp_group", None)
    )
    if (
        effective_cp_group is not None
        and effective_cp_group.size() > 1
        and stage_entry_partition_mode is None
    ):
        raise ValueError(
            f"{owner_name} requires a CP stage entry partition mode when context "
            "parallelism is active."
        )

    return stage_entry_partition_mode
