# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the ``--drop-redundant-extra-state`` optimization.

These cover the two pure-logic helpers that decide which TE ``_extra_state``
artifacts are redundant and rewrite them as local (non-persistent) objects:
``_fp8_extra_state_is_persistent`` and ``_localize_redundant_extra_states``.

The logic operates on the raw serialized ``_extra_state`` bytes, so it is
exercised here with TE-format payloads built the same way TE's
``get_extra_state`` does (a ``torch.save`` of a small dict into a ``uint8``
tensor). This needs no FP8 hardware, so it runs identically on FP8-capable CI
(H100 / GB200) and elsewhere.
"""

import io

import pytest
import torch

from megatron.core.dist_checkpointing.mapping import (
    LocalNonpersistentObject,
    ShardedObject,
    ShardedTensor,
)
from megatron.training.checkpointing import (
    _fp8_extra_state_is_persistent,
    _localize_redundant_extra_states,
)


def _te_like_extra_state(payload: dict) -> torch.Tensor:
    """Serialize ``payload`` the way TE's ``get_extra_state`` does: a
    ``torch.save`` of a dict into a ``uint8`` byte tensor."""
    buf = io.BytesIO()
    torch.save(payload, buf)
    return torch.frombuffer(bytearray(buf.getvalue()), dtype=torch.uint8)


# A delayed-scaling FP8 payload carries amax history + scale buffers -> persistent.
_DELAYED = {
    "amax_history_fwd": torch.zeros(3),
    "scale_fwd": torch.ones(1),
    "scale_bwd": torch.ones(1),
}
# Block / current scaling (NVFP4, MXFP8, Float8CurrentScaling) only stores
# recipe + config-derived scalars, none of the delayed-scaling markers.
_RECIPE_ONLY = {"recipe": "Float8CurrentScaling", "global_steps": 7}


class TestFp8ExtraStateIsPersistent:
    def test_none_is_not_persistent(self):
        assert _fp8_extra_state_is_persistent(None) is False

    def test_empty_tensor_is_not_persistent(self):
        # FP8 disabled: get_extra_state returns an empty uint8 tensor.
        assert _fp8_extra_state_is_persistent(torch.empty(0, dtype=torch.uint8)) is False

    def test_recipe_only_is_not_persistent(self):
        # Block / current scaling: no delayed-scaling markers in the payload.
        assert _fp8_extra_state_is_persistent(_te_like_extra_state(_RECIPE_ONLY)) is False

    @pytest.mark.parametrize("marker", ["amax_history_fwd", "scale_fwd", "scale_bwd"])
    def test_delayed_scaling_markers_are_persistent(self, marker):
        # Any one delayed-scaling marker is enough to keep the payload.
        assert (
            _fp8_extra_state_is_persistent(_te_like_extra_state({marker: torch.zeros(2)})) is True
        )

    def test_full_delayed_payload_is_persistent(self):
        assert _fp8_extra_state_is_persistent(_te_like_extra_state(_DELAYED)) is True

    def test_unknown_payload_defaults_to_persistent(self):
        # Anything that is not a tensor is kept, so real state is never dropped.
        assert _fp8_extra_state_is_persistent({"not": "a tensor"}) is True


class TestLocalizeRedundantExtraStates:
    def _sharded_object(self, key, data):
        return ShardedObject(key, data, (1,), (0,))

    def test_localizes_only_redundant_extra_states(self):
        empty_es = self._sharded_object("decoder.0.self_attention._extra_state", None)
        recipe_es = self._sharded_object(
            "decoder.0.mlp._extra_state", _te_like_extra_state(_RECIPE_ONLY)
        )
        delayed_es = self._sharded_object(
            "decoder.1.mlp._extra_state", _te_like_extra_state(_DELAYED)
        )
        # A ShardedObject that is NOT an _extra_state must be left alone.
        other_obj = self._sharded_object("decoder.0.metadata", {"foo": "bar"})
        weight = ShardedTensor.from_rank_offsets("decoder.0.weight", torch.zeros(4), replica_id=0)

        state_dict = {
            "empty_es": empty_es,
            "recipe_es": recipe_es,
            "delayed_es": delayed_es,
            "other_obj": other_obj,
            "weight": weight,
        }
        _localize_redundant_extra_states(state_dict)

        # Redundant (empty / recipe-only) _extra_state -> dropped on save.
        assert isinstance(state_dict["empty_es"], LocalNonpersistentObject)
        assert isinstance(state_dict["recipe_es"], LocalNonpersistentObject)
        # The wrapped value is preserved so load can restore the key locally.
        assert state_dict["recipe_es"].unwrap() is recipe_es.data
        # Delayed-scaling _extra_state stays a ShardedObject (still checkpointed).
        assert state_dict["delayed_es"] is delayed_es
        # Non-_extra_state objects and tensors are untouched.
        assert state_dict["other_obj"] is other_obj
        assert state_dict["weight"] is weight

    def test_noop_when_no_redundant_extra_states(self):
        delayed_es = self._sharded_object(
            "decoder.0.mlp._extra_state", _te_like_extra_state(_DELAYED)
        )
        state_dict = {"delayed_es": delayed_es}
        _localize_redundant_extra_states(state_dict)
        assert state_dict["delayed_es"] is delayed_es
