# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The CUDA graph GTP finalize plan takes its groups from the graphed module's collection.

The tests drive `_CudaGraphRunner._set_gtp_finalize_hook_plan` on a runner built without
`__init__`, with stand-ins for the graphed module and its parameters. GTP needs a newer
Transformer Engine than some environments have, so the two GTP symbols the plan uses are
stubbed. No GPU or process group is needed.
"""

from types import SimpleNamespace
from unittest.mock import sentinel

import pytest

import megatron.core.transformer.cuda_graphs as cuda_graphs_module
from megatron.core import parallel_state
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.cuda_graphs import _CudaGraphRunner


class _Param:
    """A parameter stand-in: expert parameters set `allreduce=False`, dense ones may omit it."""

    def __init__(self, **attributes):
        vars(self).update(attributes)


@pytest.fixture(autouse=True)
def stub_gtp(monkeypatch):
    """Stub the GTP chain ids and make each reduce-scatter stream name its chain and group."""
    monkeypatch.setattr(
        cuda_graphs_module, "GTPChain", SimpleNamespace(GRAPHED=SimpleNamespace(value="graphed"))
    )
    monkeypatch.setattr(
        cuda_graphs_module, "get_rs_stream", lambda chain_id, group: (chain_id, group)
    )


def _forbid_global_process_groups(monkeypatch):
    """Make the global group, rank and world-size accessors and the collection shim fail."""

    def forbidden(*args, **kwargs):
        pytest.fail("the global parallel state was read")

    for name in dir(parallel_state):
        if name.startswith("get_") and name.endswith(
            ("_rank", "_ranks", "_world_size", "_group", "_groups")
        ):
            monkeypatch.setattr(parallel_state, name, forbidden)
    monkeypatch.setattr(ProcessGroupCollection, "use_mpu_process_groups", forbidden)


def _finalize_hook_plan(base_module, params):
    """Return the finalize hook plan of a GTP runner that graphs `base_module`."""
    runner = object.__new__(_CudaGraphRunner)
    runner.gtp_remat = True
    runner.base_module = base_module
    runner._set_gtp_finalize_hook_plan(params)
    assert runner.finalized_during_bwd_capture == params
    return runner._gtp_finalize_hook_plan


class TestGtpFinalizeHookPlanGroups:

    def test_uses_the_module_collection(self, monkeypatch):
        """Dense and expert parameters use the GTP groups of the graphed module's collection."""
        dense, untagged, expert = _Param(allreduce=True), _Param(), _Param(allreduce=False)
        base_module = SimpleNamespace(
            pg_collection=ProcessGroupCollection(
                gtp_remat=sentinel.dense_group, expt_gtp_remat=sentinel.expert_group
            )
        )
        _forbid_global_process_groups(monkeypatch)

        plan = _finalize_hook_plan(base_module, [dense, expert, untagged, dense])

        # Repeated parameters stay repeated: replay must match eager grad-ready counts.
        assert plan == [
            (("graphed", sentinel.dense_group), [dense, untagged, dense]),
            (("graphed", sentinel.expert_group), [expert]),
        ]

    def test_none_field_is_used_as_given(self, monkeypatch):
        """A field set to None turns that axis off; it is not replaced by a global group."""
        dense, expert = _Param(), _Param(allreduce=False)
        base_module = SimpleNamespace(
            pg_collection=ProcessGroupCollection(
                gtp_remat=sentinel.dense_group, expt_gtp_remat=None
            )
        )
        _forbid_global_process_groups(monkeypatch)

        plan = _finalize_hook_plan(base_module, [dense, expert])

        assert plan == [(("graphed", sentinel.dense_group), [dense]), (("graphed", None), [expert])]

    @pytest.mark.parametrize(
        "base_module",
        [SimpleNamespace(), SimpleNamespace(pg_collection=ProcessGroupCollection(tp=sentinel.tp))],
        ids=["module_without_collection", "collection_without_gtp_fields"],
    )
    def test_falls_back_to_the_global_groups(self, monkeypatch, base_module):
        """A module without the GTP fields keeps using the global GTP groups."""
        monkeypatch.setattr(
            parallel_state,
            "get_gtp_weight_remat_group",
            lambda check_initialized=True: sentinel.global_dense_group,
        )
        monkeypatch.setattr(
            parallel_state,
            "get_expert_gtp_weight_remat_group",
            lambda check_initialized=True: sentinel.global_expert_group,
        )
        dense, expert = _Param(), _Param(allreduce=False)

        plan = _finalize_hook_plan(base_module, [dense, expert])

        assert plan == [
            (("graphed", sentinel.global_dense_group), [dense]),
            (("graphed", sentinel.global_expert_group), [expert]),
        ]
