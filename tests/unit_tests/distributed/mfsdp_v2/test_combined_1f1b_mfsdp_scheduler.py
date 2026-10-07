"""Gradient callback totals declared by the combined 1F1B scheduler."""

from types import SimpleNamespace

import torch
from torch import nn

from megatron.core.models.common.combined_1f1b_mfsdp_scheduler import _unit_grad_accumulation_count


class TestUnitGradAccumulationCount:
    """Scheduler totals account for ownership, aliases, and additive consumers."""

    def test_additive_shared_weight_extras(self):
        """Borrowed embedding and projection consumers add to one physical weight."""
        embedding = nn.Parameter(torch.ones(2, 2))
        projection = nn.Parameter(torch.ones(2, 2))
        ordinary = nn.Parameter(torch.ones(2, 2))
        parameters = [
            SimpleNamespace(unsharded=embedding, sharded=object()),
            SimpleNamespace(unsharded=object(), sharded=projection),
            SimpleNamespace(unsharded=ordinary, sharded=object()),
        ]
        unit = SimpleNamespace(_trainable_fsdp_parameters=lambda: iter(parameters))
        extras = [(embedding, 1), (embedding, 1), (projection, 1), (None, 7)]
        assert _unit_grad_accumulation_count(unit, extras) == 6
        assert _unit_grad_accumulation_count(unit, []) == 3

    def test_no_trainable_parameters(self):
        """Parameter-free units retain the module-hook fallback."""
        unit = SimpleNamespace(_trainable_fsdp_parameters=lambda: iter(()))
        assert _unit_grad_accumulation_count(unit, []) == 0
