# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


class _FakeGroupedQuantizedTensor:
    def __init__(self, members, quantized_tensors=None):
        self._members = members
        self.quantized_tensors = quantized_tensors
        self.quantizer = object()

    def split_into_quantized_tensors(self):
        return self._members


def test_expand_quantized_param_shard_for_cast_splits_grouped_wrapper():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    members = [torch.empty(3), torch.empty(5), torch.empty(2)]
    grouped_param = _FakeGroupedQuantizedTensor(members, quantized_tensors=members)
    shard_main_param = torch.arange(6)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            grouped_param, shard_main_param, start_offset=2
        )
    )

    assert len(expanded_params) == len(members)
    assert all(expanded is member for expanded, member in zip(expanded_params, members))
    torch.testing.assert_close(expanded_main_params[0], torch.tensor([0]))
    torch.testing.assert_close(expanded_main_params[1], torch.tensor([1, 2, 3, 4, 5]))
    assert expanded_main_params[2] is None
    assert expanded_offsets == [2, 0, None]


def test_expand_quantized_param_shard_for_cast_keeps_plain_param_unchanged():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    model_param = torch.empty(4)
    shard_main_param = torch.arange(4)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            model_param, shard_main_param, start_offset=1
        )
    )

    assert len(expanded_params) == 1 and expanded_params[0] is model_param
    assert len(expanded_main_params) == 1 and expanded_main_params[0] is shard_main_param
    assert expanded_offsets == [1]


def test_grouped_quantized_tensor_detection_allows_lazy_split_members():
    grouped_param = _FakeGroupedQuantizedTensor([torch.empty(1)], quantized_tensors=None)

    assert DistributedOptimizer._is_grouped_quantized_tensor(grouped_param)
    assert DistributedOptimizer._is_distopt_quantized_param(grouped_param)


def test_grouped_quantized_tensor_detection_requires_quantizer():
    grouped_param = _FakeGroupedQuantizedTensor([torch.empty(1)], quantized_tensors=None)
    grouped_param.quantizer = None

    assert not DistributedOptimizer._is_grouped_quantized_tensor(grouped_param)
    assert not DistributedOptimizer._is_distopt_quantized_param(grouped_param)


def test_expand_quantized_param_shard_for_cast_lazy_splits_when_members_unset():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    members = [torch.empty(2), torch.empty(2)]
    grouped_param = _FakeGroupedQuantizedTensor(members, quantized_tensors=None)
    shard_main_param = torch.arange(4)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            grouped_param, shard_main_param, start_offset=0
        )
    )

    assert expanded_params == members
    torch.testing.assert_close(expanded_main_params[0], torch.tensor([0, 1]))
    torch.testing.assert_close(expanded_main_params[1], torch.tensor([2, 3]))
    assert expanded_offsets == [0, 0]


def test_expand_quantized_param_shard_for_cast_handles_none_shard_for_grouped_param():
    optimizer = DistributedOptimizer.__new__(DistributedOptimizer)
    members = [torch.empty(3), torch.empty(2)]
    grouped_param = _FakeGroupedQuantizedTensor(members, quantized_tensors=members)

    expanded_params, expanded_main_params, expanded_offsets = (
        optimizer._expand_quantized_param_shard_for_cast(
            grouped_param, shard_main_param=None, start_offset=None
        )
    )

    assert expanded_params == members
    assert expanded_main_params == [None, None]
    assert expanded_offsets == [None, None]


# -----------------------------------------------------------------------------
# Tests for `_normalize_state_dict_for_grouped_params`.
#
# The method's duck-type discriminator is `hasattr(module, 'num_gemms')` and
# `hasattr(module, '_split_grouped_checkpoint_tensor')`, so we don't need a real
# `TEGroupedLinear` instance — a `torch.nn.Module` subclass with those attrs
# stands in. Building real model trees lets us exercise `named_modules()` paths
# and the `module.`-prefix stripping logic.
# -----------------------------------------------------------------------------


class _FakeTEGroupedLinear(torch.nn.Module):
    """Stub matching the duck-type the optimizer keys on."""

    def __init__(
        self, num_gemms, *, single_grouped_weight, use_bias=False, single_grouped_bias=False
    ):
        super().__init__()
        self.num_gemms = num_gemms
        self.single_grouped_weight = single_grouped_weight
        self.use_bias = use_bias
        self.single_grouped_bias = single_grouped_bias

    def _split_grouped_checkpoint_tensor(self, tensor, checkpoint_key):
        # Mirror the real method's first-dim split (sufficient for round-trip tests).
        if tensor.shape[0] == self.num_gemms:
            return list(tensor.unbind(dim=0))
        if tensor.shape[0] % self.num_gemms == 0:
            return list(torch.chunk(tensor, self.num_gemms, dim=0))
        raise RuntimeError(f"cannot split {checkpoint_key}")


class _Wrapper(torch.nn.Module):
    """Minimal model_chunk wrapping a child module under a configurable name path."""

    def __init__(self, child, child_path: str = "experts.linear_fc1"):
        super().__init__()
        # Build the path one level at a time using nested Modules so that
        # named_modules() emits an entry for each path segment (matching real
        # mcore module trees that look like `decoder.layers.0.mlp.experts.linear_fc1`).
        cursor = self
        parts = child_path.split(".")
        for part in parts[:-1]:
            inner = torch.nn.Module()
            setattr(cursor, part, inner)
            cursor = inner
        setattr(cursor, parts[-1], child)


def test_normalize_state_dict_indexed_to_grouped_weight():
    grouped = _FakeTEGroupedLinear(num_gemms=3, single_grouped_weight=True)
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    indexed = [torch.tensor([float(i), float(i) + 0.5]) for i in range(3)]
    state_dict = {f"experts.linear_fc1.weight{i}": indexed[i] for i in range(3)}

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert set(state_dict.keys()) == {"experts.linear_fc1.weight"}
    torch.testing.assert_close(state_dict["experts.linear_fc1.weight"], torch.stack(indexed, dim=0))


def test_normalize_state_dict_grouped_to_indexed_weight():
    grouped = _FakeTEGroupedLinear(num_gemms=3, single_grouped_weight=False)
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    grouped_tensor = torch.arange(9, dtype=torch.float32).view(3, 3)
    state_dict = {"experts.linear_fc1.weight": grouped_tensor}

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert set(state_dict.keys()) == {
        "experts.linear_fc1.weight0",
        "experts.linear_fc1.weight1",
        "experts.linear_fc1.weight2",
    }
    for i in range(3):
        torch.testing.assert_close(state_dict[f"experts.linear_fc1.weight{i}"], grouped_tensor[i])


def test_normalize_state_dict_indexed_to_grouped_with_bias():
    grouped = _FakeTEGroupedLinear(
        num_gemms=2, single_grouped_weight=True, use_bias=True, single_grouped_bias=True
    )
    chunk = _Wrapper(grouped, child_path="layer.linear_fc1")
    weights = [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
    biases = [torch.tensor([10.0]), torch.tensor([20.0])]
    state_dict = {
        "layer.linear_fc1.weight0": weights[0],
        "layer.linear_fc1.weight1": weights[1],
        "layer.linear_fc1.bias0": biases[0],
        "layer.linear_fc1.bias1": biases[1],
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert set(state_dict.keys()) == {"layer.linear_fc1.weight", "layer.linear_fc1.bias"}
    torch.testing.assert_close(state_dict["layer.linear_fc1.weight"], torch.stack(weights, dim=0))
    torch.testing.assert_close(state_dict["layer.linear_fc1.bias"], torch.stack(biases, dim=0))


def test_normalize_state_dict_skips_bias_when_use_bias_false():
    grouped = _FakeTEGroupedLinear(
        num_gemms=2, single_grouped_weight=True, use_bias=False, single_grouped_bias=False
    )
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    state_dict = {
        "experts.linear_fc1.weight0": torch.zeros(2),
        "experts.linear_fc1.weight1": torch.ones(2),
        # Stray bias keys that don't belong on this module — must remain untouched.
        "experts.linear_fc1.bias0": torch.tensor([99.0]),
        "experts.linear_fc1.bias1": torch.tensor([42.0]),
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert "experts.linear_fc1.weight" in state_dict
    assert state_dict["experts.linear_fc1.bias0"].item() == 99.0
    assert state_dict["experts.linear_fc1.bias1"].item() == 42.0


def test_normalize_state_dict_strips_module_prefix_from_named_modules_path():
    """`named_modules()` may emit `module.foo.bar` (Float16Module wrapping); the
    suffix lookup must use `foo.bar` after stripping the leading `module.`.
    """

    class _ModuleWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner  # named_modules() now emits paths like `module.experts.linear_fc1`.

    grouped = _FakeTEGroupedLinear(num_gemms=2, single_grouped_weight=True)
    inner = _Wrapper(grouped, child_path="experts.linear_fc1")
    chunk = _ModuleWrapper(inner)

    indexed = [torch.tensor([1.0]), torch.tensor([2.0])]
    state_dict = {
        "experts.linear_fc1.weight0": indexed[0],
        "experts.linear_fc1.weight1": indexed[1],
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert set(state_dict.keys()) == {"experts.linear_fc1.weight"}
    torch.testing.assert_close(state_dict["experts.linear_fc1.weight"], torch.stack(indexed, dim=0))


def test_normalize_state_dict_skips_when_grouped_target_already_present():
    grouped = _FakeTEGroupedLinear(num_gemms=2, single_grouped_weight=True)
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    grouped_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    state_dict = {
        "experts.linear_fc1.weight": grouped_tensor,
        "experts.linear_fc1.weight0": torch.zeros(2),
        "experts.linear_fc1.weight1": torch.ones(2),
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    # No mutation: target layout already present.
    torch.testing.assert_close(state_dict["experts.linear_fc1.weight"], grouped_tensor)
    torch.testing.assert_close(state_dict["experts.linear_fc1.weight0"], torch.zeros(2))
    torch.testing.assert_close(state_dict["experts.linear_fc1.weight1"], torch.ones(2))


def test_normalize_state_dict_skips_when_indexed_set_incomplete():
    """`single_grouped=True` requires ALL indexed keys to fold; partial set is left alone."""
    grouped = _FakeTEGroupedLinear(num_gemms=3, single_grouped_weight=True)
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    state_dict = {
        "experts.linear_fc1.weight0": torch.zeros(2),
        "experts.linear_fc1.weight2": torch.ones(2),  # weight1 missing
    }

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert "experts.linear_fc1.weight" not in state_dict
    assert set(state_dict.keys()) == {"experts.linear_fc1.weight0", "experts.linear_fc1.weight2"}


def test_normalize_state_dict_skips_ambiguous_indexed_match():
    """Two state_dict keys ending in the same indexed suffix (e.g. duplicate paths
    across model chunks under one flat dict) must NOT be folded — the
    `len(matches) == 1` guard in the optimizer leaves an incomplete `indexed_match_map`,
    which fails the `len == num_gemms` check and the function bails out safely.
    """
    grouped = _FakeTEGroupedLinear(num_gemms=2, single_grouped_weight=True)
    chunk = _Wrapper(grouped, child_path="experts.linear_fc1")
    state_dict = {
        # Both keys end in `experts.linear_fc1.weight0` → endswith ambiguity.
        "chunk_a.experts.linear_fc1.weight0": torch.tensor([1.0]),
        "chunk_b.experts.linear_fc1.weight0": torch.tensor([2.0]),
        # Single weight1 key — would-be match, but weight0 is ambiguous.
        "chunk_a.experts.linear_fc1.weight1": torch.tensor([3.0]),
    }
    snapshot = dict(state_dict)

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    # No remap: ambiguous weight0 means indexed_match_map is incomplete → skipped.
    assert state_dict.keys() == snapshot.keys()
    for k, v in snapshot.items():
        torch.testing.assert_close(state_dict[k], v)


def test_normalize_state_dict_ignores_non_te_grouped_modules():
    """Modules without `num_gemms`/`_split_grouped_checkpoint_tensor` (e.g. `torch.nn.Linear`)
    must be skipped by the duck-type check — no traversal of their state_dict keys.
    """
    plain = torch.nn.Linear(4, 4)
    chunk = _Wrapper(plain, child_path="layer.linear")
    # state_dict that, if naively matched, looks indexed.
    state_dict = {
        "layer.linear.weight0": torch.zeros(4, 4),
        "layer.linear.weight1": torch.ones(4, 4),
    }
    snapshot = dict(state_dict)

    DistributedOptimizer._normalize_state_dict_for_grouped_params(state_dict, chunk)

    assert state_dict.keys() == snapshot.keys()
