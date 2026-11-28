# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for relinking ``overlap_param_gather`` bucket order to match forward access."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from megatron.core.distributed.distributed_data_parallel import (
    DistributedDataParallel,
    relink_param_gather_bucket_groups,
)
from megatron.core.utils import PARAM_READY_CALLBACK_ATTR, ensure_params_ready


class _DummyBucketGroup:
    def __init__(self, name: str):
        self.name = name
        self.next_param_gather_bucket_group = None


def _chain(group: _DummyBucketGroup | None) -> list[str]:
    names = []
    cur = group
    while cur is not None:
        names.append(cur.name)
        cur = cur.next_param_gather_bucket_group
    return names


def test_relink_param_gather_bucket_groups_valid_and_invalid_orders():
    groups = [_DummyBucketGroup("g0"), _DummyBucketGroup("g1"), _DummyBucketGroup("g2")]
    groups[2].next_param_gather_bucket_group = groups[1]
    groups[1].next_param_gather_bucket_group = groups[0]

    assert relink_param_gather_bucket_groups(groups, [1, 2, 0]) is True
    assert _chain(groups[1]) == ["g1", "g2", "g0"]
    assert groups[0].next_param_gather_bucket_group is None

    # Incomplete or duplicate orders leave existing links untouched.
    assert relink_param_gather_bucket_groups(groups, [0, 1]) is False
    assert relink_param_gather_bucket_groups(groups, [0, 0, 2]) is False
    assert _chain(groups[1]) == ["g1", "g2", "g0"]

    # Empty list is a valid trivial no-op.
    assert relink_param_gather_bucket_groups([], []) is True


def test_relink_param_gather_bucket_groups_rejects_rank_disagreement():
    groups = [_DummyBucketGroup("g0"), _DummyBucketGroup("g1"), _DummyBucketGroup("g2")]
    groups[2].next_param_gather_bucket_group = groups[1]
    groups[1].next_param_gather_bucket_group = groups[0]
    fake_pg = object()

    def _fake_all_gather_object(gathered, local_obj, group=None):
        assert group is fake_pg
        gathered[0] = local_obj
        gathered[1] = [0, 2, 1]

    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.all_gather_object", side_effect=_fake_all_gather_object),
    ):
        assert relink_param_gather_bucket_groups(groups, [1, 2, 0], fake_pg) is False

    assert _chain(groups[2]) == ["g2", "g1", "g0"]


class _EarlyConsumerBlock(torch.nn.Module):
    """Simulates a module whose forward reads another module's weight ahead of registration."""

    def __init__(self, first_linear: torch.nn.Linear, second_linear: torch.nn.Linear):
        super().__init__()
        self.first_linear = first_linear
        self.second_linear = second_linear

    def forward(self, x: torch.Tensor, *, fail: bool = False) -> torch.Tensor:
        ensure_params_ready([self.second_linear.weight])
        if fail:
            raise RuntimeError("forward failure")
        y = self.first_linear(x)
        return self.second_linear(y)


def _make_fake_ddp_for_probe(module: torch.nn.Module):
    first = _DummyBucketGroup("first")
    second = _DummyBucketGroup("second")
    first.next_param_gather_bucket_group = second  # Deliberately wrong before alignment.
    calls = []

    cb_first = lambda: calls.append("first")  # noqa: E731
    cb_second = lambda: calls.append("second")  # noqa: E731
    setattr(module.first_linear.weight, PARAM_READY_CALLBACK_ATTR, cb_first)
    setattr(module.second_linear.weight, PARAM_READY_CALLBACK_ATTR, cb_second)

    ddp = object.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(ddp)
    ddp.ddp_config = SimpleNamespace(overlap_param_gather=True)
    ddp.module = module
    ddp.bucket_groups = [first, second]
    ddp.expert_parallel_bucket_groups = []
    ddp.param_to_bucket_group = {
        module.first_linear.weight: first,
        module.second_linear.weight: second,
    }
    ddp.intra_dp_cp_group = None
    ddp.intra_expt_dp_group = None
    return ddp, first, second, cb_first, cb_second, calls


def test_align_param_gather_order_captures_ready_callbacks_and_restores_hooks():
    module = _EarlyConsumerBlock(
        torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)
    )
    ddp, first, second, cb_first, cb_second, calls = _make_fake_ddp_for_probe(module)

    ddp.align_param_gather_order_on_next_forward()
    ddp.align_param_gather_order_on_next_forward()  # Idempotent when already armed.
    out = module(torch.ones(2, 4))
    assert out.shape == (2, 4)

    # second_linear.weight was read via ensure_params_ready before first_linear ran.
    assert ddp.last_param_gather_order_alignment["dense"] == {
        "observed_order": [1, 0],
        "applied": True,
    }
    assert second.next_param_gather_bucket_group is first
    assert first.next_param_gather_bucket_group is None

    # Original callbacks and clean hook state must be restored.
    assert getattr(module.first_linear.weight, PARAM_READY_CALLBACK_ATTR) is cb_first
    assert getattr(module.second_linear.weight, PARAM_READY_CALLBACK_ATTR) is cb_second
    assert calls == ["second"]
    assert not module._forward_hooks
    assert not module.first_linear._forward_pre_hooks
    assert not module.second_linear._forward_pre_hooks


def test_align_param_gather_order_cleans_up_even_if_forward_raises():
    module = _EarlyConsumerBlock(
        torch.nn.Linear(4, 4, bias=False), torch.nn.Linear(4, 4, bias=False)
    )
    ddp, first, second, cb_first, cb_second, calls = _make_fake_ddp_for_probe(module)

    ddp.align_param_gather_order_on_next_forward()
    with pytest.raises(RuntimeError, match="forward failure"):
        module(torch.ones(2, 4), fail=True)

    # Relinking must NOT run on failed forward, while hooks/callbacks must be restored.
    assert not hasattr(ddp, "last_param_gather_order_alignment")
    assert first.next_param_gather_bucket_group is second
    assert getattr(module.first_linear.weight, PARAM_READY_CALLBACK_ATTR) is cb_first
    assert getattr(module.second_linear.weight, PARAM_READY_CALLBACK_ATTR) is cb_second
    assert calls == ["second"]
    assert not module._forward_hooks
    assert not module.first_linear._forward_pre_hooks
    assert not module.second_linear._forward_pre_hooks
