# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Keep external batch-helper callers working during the Dynamic-CP API migration."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.utils import get_batches_on_this_cp_rank
from megatron.core.utils import get_batch_on_this_cp_rank, get_batch_on_this_tp_rank


@pytest.fixture
def batch_helpers(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group: 1)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: 0)
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *args, **kwargs: None)
    return {
        "tokens": torch.arange(8).reshape(1, -1),
        "labels": torch.arange(1, 9).reshape(1, -1),
        "loss_mask": torch.ones(1, 8),
        "position_ids": torch.arange(8).reshape(1, -1),
        "attention_mask": None,
        "cu_seqlens": None,
        "cu_seqlens_padded": None,
        "max_seqlen": None,
        "local_cp_size": None,
        "hybrid_cp_group": None,
    }


@pytest.mark.parametrize(
    "helper, options",
    [
        (get_batch_on_this_cp_rank, {}),
        (get_batches_on_this_cp_rank, {"boundary_layout": "zigzag", "cp_group": None}),
        (
            get_batch_on_this_tp_rank,
            {
                "has_cu_seqlens": False,
                "create_attention_mask_in_dataloader": False,
                "broadcast_src_rank": 0,
                "broadcast_group": None,
                "cp_size": 1,
                "tp_rank": 0,
                "micro_batch_size": 1,
                "seq_length": 8,
                "mtp_on_this_rank": False,
            },
        ),
    ],
)
def test_batch_helpers_accept_deprecated_keywords(batch_helpers, helper, options):
    signature = inspect.signature(helper)
    assert "dynamic_cp" in signature.parameters
    # MBridge's packed-sequence helper uses this exact feature-detection check.
    assert "is_hybrid_cp" in signature.parameters
    assert signature.parameters["is_hybrid_cp"].kind is inspect.Parameter.KEYWORD_ONLY
    assert "is_hybrid_cp" not in inspect.signature(helper.__wrapped__).parameters
    expected = helper(dict(batch_helpers), dynamic_cp=False, **options)
    with pytest.warns(DeprecationWarning, match="is_hybrid_cp is deprecated"):
        actual = helper(dict(batch_helpers), is_hybrid_cp=False, **options)
    if hasattr(expected, "get_batch"):
        expected, actual = expected.get_batch(), actual.get_batch()
    assert expected.keys() == actual.keys()
    for key, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value)
        else:
            assert actual[key] is value
    with pytest.raises(TypeError, match="Cannot pass both"):
        helper(dict(batch_helpers), dynamic_cp=False, is_hybrid_cp=False, **options)


@pytest.mark.parametrize("helper", [get_batch_on_this_cp_rank, get_batches_on_this_cp_rank])
def test_batch_helpers_keep_legacy_runtime_group_factory(batch_helpers, helper):
    group = SimpleNamespace(size=lambda: 1)

    def group_factory(group_size):
        assert group_size == 1
        return group

    options = (
        {"boundary_layout": "zigzag", "cp_group": None}
        if helper is get_batches_on_this_cp_rank
        else {}
    )
    batch_helpers["local_cp_size"] = 1
    expected = helper(
        dict(batch_helpers), dynamic_cp=True, dynamic_cp_group_func=group_factory, **options
    )
    with pytest.warns(DeprecationWarning, match="deprecated") as captured:
        actual = helper(
            dict(batch_helpers), is_hybrid_cp=True, hybrid_cp_group_func=group_factory, **options
        )
    assert len(captured) == 2
    if hasattr(expected, "get_batch"):
        expected, actual = expected.get_batch(), actual.get_batch()
    assert expected["hybrid_cp_group"] is actual["hybrid_cp_group"] is group
    torch.testing.assert_close(expected["tokens"], actual["tokens"])


def test_batch_helpers_reject_ambiguous_group_factory(batch_helpers):
    with pytest.raises(TypeError, match="Cannot pass both"):
        get_batch_on_this_cp_rank(
            batch_helpers,
            dynamic_cp=True,
            dynamic_cp_group_func=lambda group_size: None,
            hybrid_cp_group_func=lambda group_size: None,
        )
