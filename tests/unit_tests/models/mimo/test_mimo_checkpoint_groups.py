# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""MIMO checkpointing elects shard writers over each module's own process groups."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.mimo.model.base import MimoModel
from megatron.core.models.mimo.optimizer import _get_replica_id
from megatron.core.models.mimo.submodules.vision import VisionModalitySubmodules
from megatron.core.process_groups_config import ProcessGroupCollection


class _Group:
    """Stand-in for a process group with a fixed rank and size."""

    def __init__(self, rank=0, size=1):
        self._rank = rank
        self._size = size

    def rank(self):
        return self._rank

    def size(self):
        return self._size


class _MetadataRecorder(torch.nn.Module):
    """Records the checkpoint metadata that its parent passes to it."""

    def __init__(self, pg_collection=None):
        super().__init__()
        self.pg_collection = pg_collection
        self.metadata = None

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        self.metadata = metadata
        return {}


def _collection_without(field, state, **groups):
    """Return a collection of ``groups`` in which ``field`` is unset or set to None."""
    if state == "unset":
        del groups[field]
    else:
        groups[field] = None
    return ProcessGroupCollection(**groups)


class TestOptimizerReplicaId:
    """MimoOptimizer elects one writer of its per-module state from the module's groups."""

    def test_uses_the_groups_of_the_collection(self):
        pg_collection = ProcessGroupCollection(tp=_Group(1, 2), pp=_Group(2, 4), dp_cp=_Group(3, 8))
        assert _get_replica_id(pg_collection) == (1, 2, 3)

    def test_gtp_remat_group_extends_the_tensor_parallel_rank(self):
        pg_collection = ProcessGroupCollection(
            tp=_Group(1, 2), pp=_Group(), dp_cp=_Group(), gtp_remat=_Group(1, 2)
        )
        assert _get_replica_id(pg_collection) == (1 * 2 + 1, 0, 0)

    @pytest.mark.parametrize("gtp_remat", ["unset", "none"])
    def test_gtp_remat_unset_or_none_is_off(self, gtp_remat):
        groups = dict(tp=_Group(1, 2), pp=_Group(), dp_cp=_Group())
        if gtp_remat == "none":
            groups["gtp_remat"] = None
        assert _get_replica_id(ProcessGroupCollection(**groups)) == (1, 0, 0)

    @pytest.mark.parametrize("field", ["tp", "pp", "dp_cp"])
    @pytest.mark.parametrize("state", ["unset", "none"])
    def test_rejects_a_missing_group(self, field, state):
        pg_collection = _collection_without(field, state, tp=_Group(), pp=_Group(), dp_cp=_Group())
        with pytest.raises(ValueError, match=f"pg_collection to set {field}"):
            _get_replica_id(pg_collection)

    def test_rejects_no_collection(self):
        with pytest.raises(ValueError, match="requires a ProcessGroupCollection"):
            _get_replica_id(None)


def _submodule_metadata(pg_collection):
    encoder = _MetadataRecorder()
    submodules = VisionModalitySubmodules(
        encoders={"encoder": encoder}, pg_collection=pg_collection
    )
    submodules.sharded_state_dict()
    return encoder.metadata


def _language_model_metadata(pg_collection):
    language_model = _MetadataRecorder(pg_collection)
    model = SimpleNamespace(
        named_children=lambda: [("language_model", language_model)],
        mimo_config=SimpleNamespace(language_model_input_projections_spec=None),
    )
    MimoModel.sharded_state_dict(model)
    return language_model.metadata


_OWNERS = [
    pytest.param(_submodule_metadata, id="modality_submodules"),
    pytest.param(_language_model_metadata, id="language_model"),
]


class TestModelCheckpointGroups:
    """Modality submodules and the language model pass their own DP x CP group down."""

    @pytest.mark.parametrize("read_metadata", _OWNERS)
    def test_prefers_the_gtp_inclusive_group(self, read_metadata):
        dp_cp, dp_cp_gtp_remat = object(), object()
        metadata = read_metadata(
            ProcessGroupCollection(dp_cp=dp_cp, dp_cp_gtp_remat=dp_cp_gtp_remat)
        )
        assert metadata["dp_cp_group"] is dp_cp_gtp_remat
        # dp_cp_gtp_remat accompanies dp_cp; a collection without it elects over dp_cp.
        assert read_metadata(ProcessGroupCollection(dp_cp=dp_cp))["dp_cp_group"] is dp_cp

    @pytest.mark.parametrize("read_metadata", _OWNERS)
    @pytest.mark.parametrize("state", ["unset", "none"])
    def test_rejects_a_collection_without_dp_cp(self, read_metadata, state):
        pg_collection = _collection_without(
            "dp_cp", state, dp_cp=object(), dp_cp_gtp_remat=object()
        )
        with pytest.raises(ValueError, match="dp_cp"):
            read_metadata(pg_collection)
