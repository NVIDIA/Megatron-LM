# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Regression coverage for THD loss-mask packing in the MLite VERL engine.

VERL hands the actor-update batch a *response-only* ``loss_mask`` /
``response_mask`` nested tensor (shape ``[bsz, response_len]``), while
``input_ids`` is the full ``[prompt; response]`` packed sequence. The engine
must expand that mask to the full sequence length before the model protocol
packs it against the ``input_ids`` seq_lens. Returning the response-only mask
unchanged left ``loss_mask`` shorter than the declared seq_lens and crashed the
update step inside ``_nested_from_packed_tensor``::

    torch.narrow(0, offset, 206): start + length exceeds dimension size (128)

These tests exercise the pure ``_loss_mask_for_packing`` static logic end-to-end
with ``_nested_from_packed_tensor``; no GPU, model init, or torch.distributed.
"""

from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict

from verl_mlite.engine.mlite_engine import MegatronLiteEngine
from megatron.lite.model.deepseek_v4.lite.protocol import _nested_from_packed_tensor

pytestmark = pytest.mark.mlite


def _full_input_ids(full_lengths: list[int]) -> torch.Tensor:
    return torch.nested.as_nested_tensor(
        [torch.arange(length) for length in full_lengths], layout=torch.jagged
    )


def _response_mask_row(response_len: int) -> torch.Tensor:
    # Include an internal zero to model a masked span (e.g. tool output); the
    # whole valid response span must be preserved, not collapsed to its sum.
    row = torch.ones(response_len, dtype=torch.float32)
    if response_len > 4:
        row[3] = 0.0
    return row


def _assert_packs_to_full(packed_mask, input_ids, full_lengths, response_lengths):
    seq_lens = input_ids.offsets().diff().to(torch.int64)
    # Must be full-length: sum of response lengths alone would be too short.
    assert int(packed_mask.values().numel()) == sum(full_lengths)
    # Packing against the full seq_lens must not overflow (the original crash).
    nested = _nested_from_packed_tensor(packed_mask.values().contiguous(), seq_lens)
    rows = nested.unbind(0)
    for i, (total, resp) in enumerate(zip(full_lengths, response_lengths, strict=True)):
        prompt_len = total - resp
        # Prompt positions are masked out (zeros); response span is kept intact.
        assert torch.count_nonzero(rows[i][:prompt_len]) == 0
        assert rows[i][prompt_len:].numel() == resp


def test_nested_response_only_loss_mask_expands_to_full_length():
    """The production crash case: response-only nested mask + full input_ids."""
    full_lengths = [206, 130, 40]
    response_lengths = [78, 30, 20]  # sum 128 < a single full length (206)
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [_response_mask_row(r) for r in response_lengths], layout=torch.jagged
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[len(full_lengths)]
    )

    packed = MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)
    _assert_packs_to_full(packed, input_ids, full_lengths, response_lengths)


def test_full_length_nested_loss_mask_is_unchanged():
    """A response covering the whole sequence (prompt_len == 0) still round-trips."""
    full_lengths = [100, 100]
    response_lengths = [100, 100]
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(r, dtype=torch.float32) for r in response_lengths], layout=torch.jagged
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[len(full_lengths)]
    )

    packed = MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)
    _assert_packs_to_full(packed, input_ids, full_lengths, response_lengths)


def test_dense_response_loss_mask_expands_to_full_length():
    """The dense (V0 left_right_2_no_padding) path stays full-length too."""
    full_lengths = [206, 130, 40]
    response_lengths = [78, 30, 20]
    input_ids = _full_input_ids(full_lengths)
    max_response = max(response_lengths)
    dense = torch.zeros(len(response_lengths), max_response, dtype=torch.float32)
    for i, r in enumerate(response_lengths):
        dense[i, :r] = torch.ones(r, dtype=torch.float32)
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": dense}, batch_size=[len(full_lengths)]
    )

    packed = MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)
    seq_lens = input_ids.offsets().diff().to(torch.int64)
    assert int(packed.values().numel()) == sum(full_lengths)
    # Must not overflow when packed against the full seq_lens.
    _nested_from_packed_tensor(packed.values().contiguous(), seq_lens)


def test_response_longer_than_input_is_rejected():
    """A response mask longer than its input sequence is a hard error, not silent."""
    full_lengths = [40]
    input_ids = _full_input_ids(full_lengths)
    loss_mask = torch.nested.as_nested_tensor(
        [torch.ones(64, dtype=torch.float32)], layout=torch.jagged
    )
    micro_batch = TensorDict(
        {"input_ids": input_ids, "loss_mask": loss_mask}, batch_size=[1]
    )
    with pytest.raises(ValueError, match="tokens but packed input"):
        MegatronLiteEngine._loss_mask_for_packing(micro_batch, input_ids)
