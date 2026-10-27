# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from megatron.lite.primitive.modules.router_replay import build_r3_replay_mask


def _jagged_input_ids() -> torch.Tensor:
    return torch.nested.as_nested_tensor(
        [torch.arange(4), torch.arange(3)], layout=torch.jagged
    )


def test_r3_replay_mask_uses_full_causal_prefix_for_each_response() -> None:
    actual = build_r3_replay_mask(
        _jagged_input_ids(), torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    )

    assert [row.tolist() for row in actual.unbind()] == [
        [True, True, True, False],
        [False, False, False],
    ]


def test_r3_replay_mask_rejects_non_jagged_input_ids() -> None:
    with pytest.raises(TypeError, match="jagged input_ids"):
        build_r3_replay_mask(torch.ones(2, 4), torch.ones(2, 2))


def test_r3_replay_mask_rejects_batch_size_mismatch() -> None:
    with pytest.raises(ValueError, match="batch size"):
        build_r3_replay_mask(_jagged_input_ids(), torch.ones(3, 2))
