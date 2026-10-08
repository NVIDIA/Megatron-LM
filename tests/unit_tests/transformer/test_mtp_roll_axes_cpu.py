# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import roll_tensor


@pytest.mark.parametrize("axis", [0, -3])
def test_packed_first_axis_roll_respects_valid_ends_and_gradients(axis):
    values = torch.arange(12, dtype=torch.float32).view(6, 1, 2).requires_grad_()
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=torch.tensor([0, 2, 4], dtype=torch.int32),
        cu_seqlens_q_padded=torch.tensor([0, 3, 6], dtype=torch.int32),
    )
    rolled, total = roll_tensor(values, dims=axis, packed_seq_params=packed)
    expected = torch.tensor(
        [[2, 3], [0, 0], [0, 0], [8, 9], [0, 0], [0, 0]], dtype=torch.float32
    ).view(6, 1, 2)
    torch.testing.assert_close(rolled, expected)
    torch.testing.assert_close(total, expected.sum())
    total.backward()
    expected_grad = torch.zeros_like(values)
    expected_grad[1] = 1
    expected_grad[4] = 1
    torch.testing.assert_close(values.grad, expected_grad)


def test_mtp_padding_layout_conversion_uses_activation_tp_ownership(monkeypatch):
    from types import SimpleNamespace

    from megatron.core.transformer import multi_token_prediction as mtp

    cp_group = SimpleNamespace(size=lambda: 2)
    tp_group = object()
    tp_cp_group = object()
    plan = object()
    block = SimpleNamespace(
        config=SimpleNamespace(
            _linear_cp_layout_explicit=True,
            linear_cp_layout="contiguous",
            attention_cp_layout="zigzag",
        ),
        cp_group=cp_group,
        tp_group=tp_group,
        tp_cp_group=tp_cp_group,
        sequence_parallel=True,
    )
    target_batch = {name: None for name in ("tokens", "position_ids", "labels", "loss_mask")}
    cp_batch = SimpleNamespace(
        boundary_layout="contiguous",
        thd_plan=plan,
        get_batch=lambda layout: target_batch,
        get_packed_seq_params=lambda layout: None,
    )
    seen = []

    def scatter(mask, group):
        assert group is tp_group
        assert mask.shape == (4, 1, 1)
        seen.append("scatter")
        return mask[:2]

    def convert(value, source, target, cp, sp, tp, tp_cp, thd_plan):
        assert (source, target) == ("contiguous", "zigzag")
        assert cp is cp_group and tp is tp_group and tp_cp is tp_cp_group
        assert sp is True and thd_plan is plan
        assert value.shape[0] == 2
        if value.dtype == torch.bool:
            seen.append("convert_mask")
            return value.flip(0)
        return value

    def gather(mask, group):
        assert group is tp_group
        seen.append("gather")
        return torch.cat((mask, torch.tensor([False, True]).view(2, 1, 1)))

    monkeypatch.setattr(mtp, "scatter_to_sequence_parallel_region", scatter)
    monkeypatch.setattr(mtp, "gather_from_sequence_parallel_region", gather)
    monkeypatch.setattr(mtp, "convert_cp_layout", convert)
    result = mtp.MultiTokenPredictionBlock.prepare_cp_layout(
        block,
        input_ids=None,
        position_ids=None,
        hidden_states=torch.zeros(2, 1, 8),
        decoder_input=None,
        mhc_multistream=None,
        labels=None,
        loss_mask=None,
        mtp_input_mask=None,
        packed_seq_params=None,
        cp_batch=cp_batch,
        padding_mask=torch.tensor([[False, True, True, False]]),
    )
    assert seen == ["scatter", "convert_mask", "gather"]
    torch.testing.assert_close(result.padding_mask, torch.tensor([[True, False, False, True]]))
