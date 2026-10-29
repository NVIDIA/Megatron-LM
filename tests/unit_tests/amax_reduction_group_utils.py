# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Helpers for tests that check which group FP8/FP4 contexts reduce amaxes over."""

from contextlib import contextmanager, nullcontext
from unittest.mock import patch

import torch

from megatron.core import parallel_state
from megatron.core.enums import Fp8Recipe
from megatron.core.process_groups_config import ProcessGroupCollection


def global_amax_group(tp_only_amax_red):
    """Return the amax reduction group of the global parallel state."""
    return parallel_state.get_amax_reduction_group(
        with_context_parallel=True, tp_only_amax_red=tp_only_amax_red
    )


def copy_of(group):
    """Create a distinct communicator over the ranks of ``group``; every rank must call it."""
    all_ranks = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(all_ranks, torch.distributed.get_process_group_ranks(group))
    enumeration = [list(ranks) for ranks in sorted({tuple(ranks) for ranks in all_ranks})]
    copy, _ = torch.distributed.new_subgroups_by_enumeration(enumeration)
    return copy


def with_copied_amax_groups(pg_collection=None):
    """Set amax groups that match the global ones in ranks but not in identity.

    A context that reads the global group instead of the collection's then fails an identity
    check, although both groups reduce over the same ranks.
    """
    if pg_collection is None:
        pg_collection = ProcessGroupCollection()
    pg_collection.tp_cp = copy_of(global_amax_group(True))
    pg_collection.tp_dp_cp = copy_of(global_amax_group(False))
    return pg_collection


@contextmanager
def forbid_global_amax_group():
    """Fail on any read of the global amax reduction group."""
    from megatron.core.extensions import transformer_engine as te_extension

    error = AssertionError("the global amax reduction group was read")
    with (
        patch.object(parallel_state, "get_amax_reduction_group", side_effect=error),
        patch.object(te_extension, "get_amax_reduction_group", side_effect=error),
    ):
        yield


@contextmanager
def record_amax_groups():
    """Replace the Transformer Engine autocasts that Megatron opens with no-op contexts.

    Yields a list that receives the ``fp8_group`` of every enabled autocast. Modules then run
    unquantized, so the test does not depend on FP8 or FP4 hardware support.
    """
    import transformer_engine.pytorch as te_pytorch

    from megatron.core.extensions import transformer_engine as te_extension

    groups = []

    def autocast(*args, **kwargs):
        if kwargs.get("enabled"):
            groups.append(kwargs["fp8_group"])
        return nullcontext()

    with (
        patch.object(te_pytorch, "fp8_autocast", side_effect=autocast),
        patch.object(te_extension, "fp8_autocast", side_effect=autocast),
    ):
        yield groups


def te_quantization_params(training_tp_only, evaluation_tp_only):
    """Per-module FP8 recipes that quantize even outside a quantized autocast."""
    from megatron.core.extensions.transformer_engine import (
        TEQuantizationParams,
        TEQuantizationRecipe,
    )

    def recipe(tp_only_amax_red):
        return TEQuantizationRecipe(
            fp8_quantization_recipe=Fp8Recipe.tensorwise,
            override_nonquantized_autocast=True,
            tp_only_amax_red=tp_only_amax_red,
        )

    return TEQuantizationParams(
        training_recipe=recipe(training_tp_only), evaluation_recipe=recipe(evaluation_tp_only)
    )
