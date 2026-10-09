# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for the process-group helpers in tests/unit_tests/test_utilities.py."""

import re
import sys
import types
from dataclasses import fields

import pytest
import torch

import megatron.core.parallel_state as ps
from megatron.core import mpu
from megatron.core.process_groups_config import ProcessGroupCollection, resolve_gtp_remat_group
from megatron.core.tensor_parallel import cross_entropy, vocab_parallel_cross_entropy
from tests.unit_tests.test_utilities import (
    Utils,
    build_test_pg_collection,
    destroy_test_pg_collection,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

pytestmark = pytest.mark.skipif(Utils.world_size % 4 != 0, reason="needs a multiple of 4 ranks")

# Collection fields that build_test_pg_collection sets to None.
_NOT_BUILT = {"hcp", "gtp_remat", "expt_gtp_remat", "inter_dist_opt", "dp_cp_ag", "expt_dp_ag"}


def _ranks(group):
    return None if group is None else torch.distributed.get_process_group_ranks(group)


class TestForbidGlobalProcessGroups:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_counted_accessors_raise_and_are_restored(self):
        # One accessor for each name suffix the process-group checker counts.
        names = [
            "get_tensor_model_parallel_group",
            "get_hierarchical_context_parallel_groups",
            "get_data_parallel_group_gloo",
            "get_pipeline_model_parallel_rank",
            "get_context_parallel_global_ranks",
            "get_expert_model_parallel_world_size",
            "get_data_parallel_src_rank",
        ]
        originals = {name: getattr(ps, name) for name in names}
        all_ranks = ps.get_all_ranks()
        with forbid_global_process_groups():
            for name in names:
                with pytest.raises(AssertionError, match=re.escape(f"parallel_state.{name}()")):
                    getattr(ps, name)()
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_world_size"):
                mpu.get_tensor_model_parallel_world_size()
            # Not counted by the checker. get_all_ranks calls counted accessors itself.
            assert ps.is_initialized()
            assert ps.get_global_memory_buffer() is not None
            ps.get_virtual_pipeline_model_parallel_world_size()
            assert ps.get_all_ranks() == all_ranks
        assert {name: getattr(ps, name) for name in names} == originals
        assert ps.get_tensor_model_parallel_world_size() == 2

    def test_by_name_import_in_a_megatron_module(self):
        tp_group = ps.get_tensor_model_parallel_group()
        logits = torch.randn(3, 2, 8, device="cuda")
        target = torch.arange(6, device="cuda").view(3, 2)
        with forbid_global_process_groups():
            # cross_entropy imported get_tensor_model_parallel_group by name; without a group
            # it falls back to that binding.
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_group"):
                vocab_parallel_cross_entropy(logits.clone(), target)
            loss = vocab_parallel_cross_entropy(logits.clone(), target, tp_group=tp_group)
        assert cross_entropy.get_tensor_model_parallel_group is ps.get_tensor_model_parallel_group
        torch.testing.assert_close(loss, vocab_parallel_cross_entropy(logits.clone(), target))

    def test_bindings_made_before_or_inside_the_block_are_restored(self, monkeypatch):
        original = ps.get_tensor_model_parallel_world_size
        module = types.ModuleType("megatron.forbid_global_process_groups_probe")
        # An import under another name: `from ...parallel_state import <accessor> as tp_size`.
        module.tp_size = original
        monkeypatch.setitem(sys.modules, module.__name__, module)
        with forbid_global_process_groups():
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_world_size"):
                module.tp_size()
            # What a module imported for the first time inside the block binds.
            module.bound_inside = ps.get_tensor_model_parallel_world_size
            kept_reference = ps.get_tensor_model_parallel_world_size
        assert module.tp_size is original
        assert module.bound_inside is original
        assert kept_reference() == 2

    def test_parallel_state_lifecycle_inside_the_block(self):
        with forbid_global_process_groups():
            # initialize_model_parallel calls accessors itself, which stays allowed.
            Utils.destroy_model_parallel()
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=2, pipeline_model_parallel_size=2
            )
            assert ps.is_initialized()
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_group"):
                ps.get_tensor_model_parallel_group()

    def test_allow_and_opt_outs(self):
        tp_group = ps.get_tensor_model_parallel_group()
        pp_group = ps.get_pipeline_model_parallel_group()
        first_stage = ps.is_pipeline_first_stage()
        with forbid_global_process_groups(
            allow=["get_tensor_model_parallel_group"],
            forbid_shim=False,
            forbid_stage_predicates=False,
        ):
            assert ps.get_tensor_model_parallel_group() is tp_group
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_world_size"):
                ps.get_tensor_model_parallel_world_size()
            assert ps.is_pipeline_first_stage() == first_stage
            assert ProcessGroupCollection.use_mpu_process_groups(["pp"]).pp is pp_group
        with forbid_global_process_groups():
            with pytest.raises(AssertionError, match="use_mpu_process_groups"):
                ProcessGroupCollection.use_mpu_process_groups(["pp"])
            with pytest.raises(AssertionError, match="is_pipeline_first_stage"):
                ps.is_pipeline_first_stage()
        with pytest.raises(ValueError, match="get_tensor_parallel_group"):
            with forbid_global_process_groups(allow=["get_tensor_parallel_group"]):
                pass

    def test_nested_blocks(self):
        accessor = ps.get_tensor_model_parallel_group
        all_ranks = ps.get_all_ranks()
        first_stage = ps.is_pipeline_first_stage()
        with forbid_global_process_groups(forbid_stage_predicates=False):
            with forbid_global_process_groups(allow=["get_tensor_model_parallel_group"]):
                # The enclosing block still forbids what this one allows.
                with pytest.raises(AssertionError, match="get_tensor_model_parallel_group"):
                    ps.get_tensor_model_parallel_group()
                with pytest.raises(AssertionError, match="is_pipeline_first_stage"):
                    ps.is_pipeline_first_stage()
                # Calls made by parallel_state itself pass through both blocks.
                assert ps.get_all_ranks() == all_ranks
            assert ps.is_pipeline_first_stage() == first_stage
            with pytest.raises(AssertionError, match="get_tensor_model_parallel_group"):
                ps.get_tensor_model_parallel_group()
        assert ps.get_tensor_model_parallel_group is accessor

    def test_restores_after_an_exception(self):
        accessor = ps.get_tensor_model_parallel_group
        shim = vars(ProcessGroupCollection)["use_mpu_process_groups"]
        with pytest.raises(RuntimeError, match="test body failed"):
            with forbid_global_process_groups():
                raise RuntimeError("test body failed")
        assert ps.get_tensor_model_parallel_group is accessor
        assert vars(ProcessGroupCollection)["use_mpu_process_groups"] is shim


class TestNewGroupWithSameRanks:
    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, pipeline_model_parallel_size=2
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_distinct_group_with_the_same_ranks(self):
        tp_group = ps.get_tensor_model_parallel_group()
        # Only the first pipeline stage holds the position embedding; the other ranks pass None.
        pos_embd = ps.get_position_embedding_group(check_initialized=False)
        new_tp = new_group_with_same_ranks(tp_group)
        new_pos_embd = new_group_with_same_ranks(pos_embd)
        try:
            assert new_tp is not tp_group
            assert _ranks(new_tp) == _ranks(tp_group)
            ones = torch.ones(1, device="cuda")
            torch.distributed.all_reduce(ones, group=new_tp)
            assert ones.item() == 2
            assert (new_pos_embd is None) == (pos_embd is None)
            assert _ranks(new_pos_embd) == _ranks(pos_embd)
        finally:
            for group in (new_tp, new_pos_embd):
                if group is not None:
                    torch.distributed.destroy_process_group(group)


@pytest.mark.parametrize(
    "tp, pp, cp, ep, expt_tp, order",
    [
        (1, 1, 1, 1, None, "tp-cp-ep-dp-pp"),
        (2, 2, 1, 1, None, "tp-cp-ep-dp-pp"),
        (2, 1, 2, 1, None, "tp-cp-ep-dp-pp"),
        # The middle pipeline stages belong to no embedding group.
        (1, 4, 1, 1, None, "tp-cp-ep-dp-pp"),
        (2, 1, 1, 2, 1, "tp-cp-ep-dp-pp"),
        (1, 2, 1, 2, 1, "tp-cp-ep-dp-pp"),
        (1, 1, 2, 2, None, "tp-cp-ep-dp-pp"),
        (1, 2, 1, 1, None, "tp-cp-ep-pp-dp"),
    ],
)
def test_build_test_pg_collection_matches_initialize_model_parallel(tp, pp, cp, ep, expt_tp, order):
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        context_parallel_size=cp,
        expert_model_parallel_size=ep,
        expert_tensor_parallel_size=expt_tp,
        order=order,
    )
    reference = ProcessGroupCollection.use_mpu_process_groups()
    with forbid_global_process_groups():
        built = build_test_pg_collection(tp=tp, pp=pp, cp=cp, ep=ep, expt_tp=expt_tp, order=order)
    try:
        names = [field.name for field in fields(ProcessGroupCollection)]
        assert set(vars(built)) == set(names)
        expected = {
            name: None if name in _NOT_BUILT else _ranks(getattr(reference, name)) for name in names
        }
        assert {name: _ranks(vars(built)[name]) for name in names} == expected
    finally:
        destroy_test_pg_collection(built)
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("grid_pg_collection", [{"tp": 2}], indirect=True)
def test_built_collection_needs_no_global_grid(grid_pg_collection):
    with forbid_global_process_groups():
        # Every field is set, so a resolver that falls back for an absent field does not.
        assert resolve_gtp_remat_group(grid_pg_collection, is_expert=False) is None
        with pytest.raises(AssertionError, match="use_mpu_process_groups"):
            resolve_gtp_remat_group(
                ProcessGroupCollection(tp=grid_pg_collection.tp), is_expert=False
            )
        ones = torch.ones(1, device="cuda")
        torch.distributed.all_reduce(ones, group=grid_pg_collection.tp)
    assert ones.item() == 2


def test_build_test_pg_collection_rejects_a_grid_that_does_not_fit():
    Utils.initialize_distributed()
    with pytest.raises(ValueError, match="must be divisible"):
        build_test_pg_collection(tp=3)
    with pytest.raises(ValueError, match="order"):
        build_test_pg_collection(order="tp-dp-pp")
