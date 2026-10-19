# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Writer ownership for replicated MLP activation state with native TP and ETP."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import ShardedObject, is_main_replica
from megatron.core.transformer.mlp import MLP
from megatron.core.transformer.moe.experts import TEGroupedMLP


class _ActivationWithExtraState(torch.nn.Module):
    def get_extra_state(self):
        # TE activations expose an empty extra-state tensor in BF16 training.
        return torch.empty(0, dtype=torch.uint8)


def _group(rank, size):
    return SimpleNamespace(rank=lambda: rank, size=lambda: size)


@pytest.mark.parametrize(
    "module_class,expert_tp_size", [(MLP, None), (TEGroupedMLP, 1), (TEGroupedMLP, 2)]
)
def test_activation_extra_state_has_one_writer(monkeypatch, module_class, expert_tp_size):
    """A shared activation key must elect one writer across TP2 x DP2.

    Exercise the real module checkpoint methods without constructing linear layers:
    those have their own checkpoint methods and cannot establish the activation's
    replica ownership. In particular, ETP1 must not elect both dense TP ranks.
    """
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    entries = []
    for dp_rank in range(2):
        for tp_rank in range(2):
            module = module_class.__new__(module_class)
            torch.nn.Module.__init__(module)
            module.config = SimpleNamespace(gated_linear_unit=False)
            module.dense_tp_group = _group(tp_rank, 2)
            module.tp_group = (
                _group(tp_rank % expert_tp_size, expert_tp_size)
                if module_class is TEGroupedMLP
                else module.dense_tp_group
            )
            module.activation_func = _ActivationWithExtraState()
            state = module.sharded_state_dict(
                prefix="mlp.", metadata={"dp_cp_group": _group(dp_rank, 2)}
            )
            assert len(state) == 1
            entry = next(iter(state.values()))
            assert isinstance(entry, ShardedObject)
            entries.append(entry)
    assert len({entry.key for entry in entries}) == 1
    assert len({entry.replica_id for entry in entries}) == 4
    assert sum(is_main_replica(entry.replica_id) for entry in entries) == 1
