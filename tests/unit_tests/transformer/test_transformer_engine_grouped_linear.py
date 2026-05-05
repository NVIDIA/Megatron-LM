# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.extensions import transformer_engine as te_ext

pytestmark = pytest.mark.skipif(
    not te_ext.HAVE_TE or not te_ext.is_te_min_version("1.9.0.dev0"),
    reason="TE GroupedLinear is only supported in TE 1.9.0.dev0 and later.",
)


class _FakeGroupedCheckpointTensor:
    def __init__(self, members, quantized_tensors=None):
        self._members = members
        self.quantized_tensors = quantized_tensors

    def split_into_quantized_tensors(self):
        return self._members


def _grouped_linear_stub(num_gemms):
    module = te_ext.TEGroupedLinear.__new__(te_ext.TEGroupedLinear)
    module.num_gemms = num_gemms
    return module


def test_split_grouped_checkpoint_tensor_uses_quantized_members():
    module = _grouped_linear_stub(num_gemms=2)
    members = [torch.tensor([1, 2]), torch.tensor([3, 4])]
    tensor = _FakeGroupedCheckpointTensor(members)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == len(members)
    assert all(split is member for split, member in zip(splits, members))


def test_split_grouped_checkpoint_tensor_unbinds_grouped_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(12).view(3, 4)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == 3
    for gemm_idx, split in enumerate(splits):
        torch.testing.assert_close(split, tensor[gemm_idx])


def test_split_grouped_checkpoint_tensor_chunks_packed_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(18).view(6, 3)

    splits = module._split_grouped_checkpoint_tensor(tensor, "weight")

    assert len(splits) == 3
    for split, expected in zip(splits, torch.chunk(tensor, 3, dim=0)):
        torch.testing.assert_close(split, expected)


def test_split_grouped_checkpoint_tensor_rejects_bad_group_count():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = _FakeGroupedCheckpointTensor([torch.tensor([1]), torch.tensor([2])])

    with pytest.raises(RuntimeError, match="has 2 groups, expected 3"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")
