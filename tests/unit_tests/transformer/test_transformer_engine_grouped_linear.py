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


def _grouped_linear_stub(num_gemms, *, use_bias=False, single_grouped_weight=False):
    module = te_ext.TEGroupedLinear.__new__(te_ext.TEGroupedLinear)
    module.num_gemms = num_gemms
    module.use_bias = use_bias
    module.single_grouped_weight = single_grouped_weight
    return module


def _empty_load_args():
    """Standard load_state_dict pre-hook trailing arguments."""
    return {}, True, [], [], []


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


def test_split_grouped_checkpoint_tensor_rejects_unsplittable_first_dim():
    module = _grouped_linear_stub(num_gemms=3)
    tensor = torch.arange(8).view(4, 2)

    with pytest.raises(RuntimeError, match="Cannot split checkpoint tensor"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")


def test_split_grouped_checkpoint_tensor_rejects_zero_dim():
    module = _grouped_linear_stub(num_gemms=2)
    tensor = torch.tensor(7)

    with pytest.raises(RuntimeError, match="Cannot split checkpoint tensor"):
        module._split_grouped_checkpoint_tensor(tensor, "weight")


def test_normalize_grouped_parameter_keys_indexed_to_grouped_weight_only():
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=True)
    indexed = [torch.tensor([float(i), float(i) + 0.5]) for i in range(3)]
    state_dict = {f"layer.weight{i}": indexed[i] for i in range(3)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight"}
    torch.testing.assert_close(state_dict["layer.weight"], torch.stack(indexed, dim=0))


def test_normalize_grouped_parameter_keys_indexed_to_grouped_with_bias():
    module = _grouped_linear_stub(num_gemms=2, use_bias=True, single_grouped_weight=True)
    weights = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    biases = [torch.tensor([10.0]), torch.tensor([20.0])]
    state_dict = {
        "layer.weight0": weights[0],
        "layer.weight1": weights[1],
        "layer.bias0": biases[0],
        "layer.bias1": biases[1],
    }

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight", "layer.bias"}
    torch.testing.assert_close(state_dict["layer.weight"], torch.stack(weights, dim=0))
    torch.testing.assert_close(state_dict["layer.bias"], torch.stack(biases, dim=0))


def test_normalize_grouped_parameter_keys_grouped_to_indexed_weight_only():
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=False)
    grouped = torch.arange(9, dtype=torch.float32).view(3, 3)
    state_dict = {"layer.weight": grouped}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight1", "layer.weight2"}
    for i in range(3):
        torch.testing.assert_close(state_dict[f"layer.weight{i}"], grouped[i])


def test_normalize_grouped_parameter_keys_grouped_to_indexed_with_bias():
    module = _grouped_linear_stub(num_gemms=2, use_bias=True, single_grouped_weight=False)
    grouped_weight = torch.arange(8, dtype=torch.float32).view(2, 4)
    grouped_bias = torch.tensor([[10.0, 20.0], [30.0, 40.0]])
    state_dict = {"layer.weight": grouped_weight, "layer.bias": grouped_bias}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {
        "layer.weight0",
        "layer.weight1",
        "layer.bias0",
        "layer.bias1",
    }
    torch.testing.assert_close(state_dict["layer.weight0"], grouped_weight[0])
    torch.testing.assert_close(state_dict["layer.weight1"], grouped_weight[1])
    torch.testing.assert_close(state_dict["layer.bias0"], grouped_bias[0])
    torch.testing.assert_close(state_dict["layer.bias1"], grouped_bias[1])


def test_normalize_grouped_parameter_keys_skips_bias_when_use_bias_false():
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    state_dict = {
        "layer.weight0": torch.zeros(2),
        "layer.weight1": torch.ones(2),
        "layer.bias0": torch.tensor([99.0]),
        "layer.bias1": torch.tensor([42.0]),
    }

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert "layer.weight" in state_dict
    assert state_dict["layer.bias0"].item() == 99.0
    assert state_dict["layer.bias1"].item() == 42.0


def test_normalize_grouped_parameter_keys_returns_when_target_layout_already_present():
    # single_grouped_weight=True and grouped key already present → no-op
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    grouped = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state_dict = {"layer.weight": grouped, "layer.weight0": torch.zeros(2)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    torch.testing.assert_close(state_dict["layer.weight"], grouped)
    torch.testing.assert_close(state_dict["layer.weight0"], torch.zeros(2))

    # single_grouped_weight=False and any indexed key present → no-op
    module = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=False)
    grouped = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    state_dict = {"layer.weight": grouped, "layer.weight0": torch.tensor([99.0])}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    torch.testing.assert_close(state_dict["layer.weight"], grouped)
    torch.testing.assert_close(state_dict["layer.weight0"], torch.tensor([99.0]))


def test_normalize_grouped_parameter_keys_returns_when_indexed_set_incomplete():
    # single_grouped_weight=True needs ALL indexed keys to fold; partial → no-op
    module = _grouped_linear_stub(num_gemms=3, use_bias=False, single_grouped_weight=True)
    state_dict = {"layer.weight0": torch.zeros(2), "layer.weight2": torch.ones(2)}

    module._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert "layer.weight" not in state_dict
    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight2"}


def test_normalize_grouped_parameter_keys_round_trips_via_chained_hooks():
    """Save in one layout, load in the other, then back: tensors survive intact."""
    members = [torch.tensor([1.0, 2.0, 3.0]), torch.tensor([4.0, 5.0, 6.0])]

    # Start from indexed checkpoint, target a single-grouped model.
    state_dict = {f"layer.weight{i}": members[i] for i in range(2)}
    folder = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=True)
    folder._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    grouped = state_dict["layer.weight"]
    torch.testing.assert_close(grouped, torch.stack(members, dim=0))

    # Now use that grouped checkpoint with an indexed model.
    splitter = _grouped_linear_stub(num_gemms=2, use_bias=False, single_grouped_weight=False)
    splitter._normalize_grouped_parameter_keys(state_dict, "layer.", *_empty_load_args())

    assert set(state_dict.keys()) == {"layer.weight0", "layer.weight1"}
    for i in range(2):
        torch.testing.assert_close(state_dict[f"layer.weight{i}"], members[i])
