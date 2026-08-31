# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression coverage for response-mask selection in MLite VERL packing."""

import pytest
import torch
from tensordict import TensorDict

from verl_mlite.engine.mlite_engine import MegatronLiteEngine

pytestmark = [pytest.mark.mlite, pytest.mark.optional]


def test_response_mask_is_used_when_loss_mask_is_absent():
    input_ids = torch.nested.as_nested_tensor(
        [torch.arange(12), torch.arange(9)], layout=torch.jagged
    )
    response_mask = torch.nested.as_nested_tensor(
        [torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]), torch.ones(3)],
        layout=torch.jagged,
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "response_mask": response_mask}, batch_size=[2]
    )

    packed = MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)

    assert packed is not None
    assert packed.offsets().diff().tolist() == [12, 9]
    rows = packed.unbind(0)
    torch.testing.assert_close(
        rows[0], torch.tensor([0.0] * 7 + [1.0, 1.0, 0.0, 1.0, 1.0])
    )
    torch.testing.assert_close(rows[1], torch.tensor([0.0] * 6 + [1.0, 1.0, 1.0]))
