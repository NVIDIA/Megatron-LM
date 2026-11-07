# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression tests for ``DistributedOptimizer.load_state_dict`` (issue #8046).

When ``dist_checkpointing.load`` populates the inner optimizer state in place
(via sharded-state-dict wrappers around the live state tensors) and
``load_checkpoint`` then calls ``DistributedOptimizer.load_state_dict``, the
latter used to take the populated-state branch and pass the optimizer's own
current state back through ``self.optimizer.load_state_dict``. That call

  (a) cleared ``self.state`` and rebuilt it from copies (with dtype casts), and
  (b) for TE ``FusedAdam`` allocated fresh FP32 state and copied into it.

The originals are still referenced through ``state_dict["param_state"]`` until
the call returns, so peak GPU memory is roughly doubled during checkpoint
restore and large MoE runs OOM.

The fix replaces that path with a param-group-only update that leaves the
live state tensors untouched. These tests pin:

  - ``self.optimizer.load_state_dict`` is NOT called in the populated branch
    (the structural regression invariant),
  - the live state tensors' storage is preserved,
  - the param-group hyperparameters are still synced from the saved state,
  - the ``step`` counter is still updated in place for native PyTorch.
"""

from collections import defaultdict
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer, Range
from megatron.core.optimizer.optimizer_config import OptimizerConfig


class _LoadTrackingAdamW(torch.optim.AdamW):
    """``AdamW`` that counts ``load_state_dict`` invocations.

    The populated branch of ``DistributedOptimizer.load_state_dict`` must
    not delegate to ``self.optimizer.load_state_dict`` at all (issue #8046);
    this counter makes that contract observable.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.load_state_dict_calls = 0

    def load_state_dict(self, state_dict):
        self.load_state_dict_calls += 1
        return super().load_state_dict(state_dict)


def _build_distopt(optimizer_cls, n: int = 8):
    """Build a minimal ``DistributedOptimizer`` whose inner optimizer already has
    state allocated, mirroring the post-``dist_checkpointing.load`` state.

    Uses ``DistributedOptimizer.__new__`` to skip ``__init__`` (which requires
    a real distributed setup). The pattern matches
    ``tests/unit_tests/optimizer/test_distrib_optimizer_mixed_precision.py``.
    """
    parameters = [torch.nn.Parameter(torch.full((n,), 1.0, dtype=torch.float32, device='cuda'))]
    groups = [{'params': parameters}]
    ranges = [
        {
            (torch.float32, torch.float32): [
                {'param_map': {parameters[0]: {'param': Range(0, n), 'gbuf_local': Range(0, n)}}}
            ]
        }
    ]

    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    optimizer.config = OptimizerConfig()
    optimizer.gbuf_ranges = ranges
    optimizer.per_bucket_numel = []
    optimizer.per_bucket_numel_unpadded = []
    # ``load_state_dict`` short-circuits to the FSDP path via
    # ``self.ddp_config.use_megatron_fsdp``; a non-FSDP stub is enough here.
    optimizer.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
    optimizer.model_param_group_index_map, group_ranges = optimizer._build_optimizer_group_ranges(
        groups, ranges
    )
    optimizer._build_model_and_main_param_groups(
        ranges, optimizer._build_model_param_gbuf_map(ranges), group_ranges, optimizer.config
    )
    optimizer.optimizer = optimizer_cls([group['orig_group'] for group in group_ranges], lr=1e-3)
    main_param = optimizer.optimizer.param_groups[0]['params'][0]
    optimizer.optimizer.state[main_param] = {
        'step': torch.tensor(0, dtype=torch.float32),
        'exp_avg': torch.full_like(main_param, 0.5),
        'exp_avg_sq': torch.full_like(main_param, 1.0),
    }
    return optimizer, main_param


@pytest.mark.internal
def test_load_state_dict_into_allocated_state_does_not_reload_optimizer_state():
    """REGRESSION (issue #8046): when the inner optimizer's state is already
    populated, ``DistributedOptimizer.load_state_dict`` must NOT call
    ``self.optimizer.load_state_dict``. Re-feeding the optimizer's own state
    through it clears ``self.state``, casts every state tensor to the param
    dtype, and (for TE ``FusedAdam``) allocates fresh FP32 copies — together
    roughly doubling peak GPU memory during checkpoint restore."""
    optimizer, main_param = _build_distopt(_LoadTrackingAdamW, n=8)

    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)

    assert optimizer.optimizer.load_state_dict_calls == 0, (
        f"self.optimizer.load_state_dict was called "
        f"{optimizer.optimizer.load_state_dict_calls} times; the populated "
        "branch must not re-feed the live state back through "
        "self.optimizer.load_state_dict (issue #8046)."
    )


@pytest.mark.internal
def test_load_state_dict_into_allocated_state_preserves_state_dict_identity():
    """The post-fix path leaves ``self.optimizer.state`` (the per-param state
    mapping) untouched. Pin its identity so a future maintainer who
    accidentally reintroduces the self-copy OOM sees this fail."""
    optimizer, main_param = _build_distopt(torch.optim.AdamW, n=8)

    pre_state_id = id(optimizer.optimizer.state)
    pre_per_param_state_id = id(optimizer.optimizer.state[main_param])

    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)

    assert id(optimizer.optimizer.state) == pre_state_id, (
        "self.optimizer.state was reassigned during load_state_dict. The "
        "populated branch must leave it untouched (issue #8046)."
    )
    assert id(optimizer.optimizer.state[main_param]) == pre_per_param_state_id, (
        "self.optimizer.state[param] was reassigned during load_state_dict. "
        "The populated branch must leave it untouched (issue #8046)."
    )


@pytest.mark.internal
def test_load_state_dict_into_allocated_state_preserves_state_tensor_storage():
    """When ``self.optimizer.load_state_dict`` is the wrong code path, the
    subsequent dtype cast (and TE FusedAdam FP32 allocations) reallocate the
    state tensors. Pin the storage identity of the non-step tensors so the
    fix cannot regress to that path."""
    optimizer, main_param = _build_distopt(torch.optim.AdamW, n=8)

    pre_exp_avg_ptr = optimizer.optimizer.state[main_param]['exp_avg'].data_ptr()
    pre_exp_avg_sq_ptr = optimizer.optimizer.state[main_param]['exp_avg_sq'].data_ptr()

    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)

    assert (
        optimizer.optimizer.state[main_param]['exp_avg'].data_ptr() == pre_exp_avg_ptr
    ), "exp_avg storage was reallocated during load_state_dict (issue #8046)."
    assert (
        optimizer.optimizer.state[main_param]['exp_avg_sq'].data_ptr() == pre_exp_avg_sq_ptr
    ), "exp_avg_sq storage was reallocated during load_state_dict (issue #8046)."


@pytest.mark.internal
def test_load_state_dict_into_allocated_state_updates_param_group_hyperparameters():
    """When the state is already populated, ``load_state_dict`` must still
    update the param-group hyperparameters from the saved state dict — that
    is the only thing left to sync, since the state tensors are already
    correct."""
    optimizer, main_param = _build_distopt(torch.optim.AdamW, n=8)

    saved = optimizer.state_dict()
    saved['optimizer']['param_groups'][0]['lr'] = 9.999e-5
    saved['optimizer']['param_groups'][0]['weight_decay'] = 0.05

    optimizer.load_state_dict(saved)

    assert optimizer.optimizer.param_groups[0]['lr'] == 9.999e-5
    assert optimizer.optimizer.param_groups[0]['weight_decay'] == 0.05


@pytest.mark.internal
def test_load_state_dict_into_allocated_state_syncs_step_in_place():
    """Native-PyTorch optimizers store ``step`` inside each per-param state
    dict. With the state already populated, ``load_state_dict`` must update
    that key on the LIVE state (rather than rebuilding state) so the next
    optimizer step sees the restored ``step``."""
    optimizer, main_param = _build_distopt(torch.optim.AdamW, n=8)

    # Bump ``step`` so the saved state carries a non-zero value.
    optimizer.optimizer.state[main_param]['step'].fill_(7)

    saved = optimizer.state_dict()
    # Restore a different step value to confirm it actually flows through.
    saved['optimizer']['param_groups'][0]['step'] = 42

    optimizer.load_state_dict(saved)

    assert int(optimizer.optimizer.state[main_param]['step'].item()) == 42
