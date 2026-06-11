# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the MIMO optimizer's process-group resolution and cross-rank
gradient statistics, focused on the non-colocated / heterogeneous grid case.

The ``item1`` (process-group resolution) and ``item2`` (cross-rank ``count_zeros``)
tests are REAL distributed tests: they initialize the process group, build a real
``HyperCommGrid`` spanning all 8 ranks, register the expert factorization via the
``register_view(...)`` rank-view API, create the base + expert-view process groups,
and assert actual group membership / actual collective results. They run on the
standard 1-node x 8-GPU unit-test lane and skip on any other world size, matching
the skip-gating convention used by the rest of the MIMO test suite.

Run with 8 GPUs:
    uv run python -m torch.distributed.run --nproc-per-node=8 \
        -m pytest tests/unit_tests/models/mimo/test_mimo_optimizer.py -v -s
"""

import pytest
import torch
import torch.distributed as dist

from megatron.core.dist_checkpointing.mapping import ShardedObject
from megatron.core.dist_checkpointing.utils import add_prefix_for_sharding
from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.optimizer import (
    EXPERT_VIEW,
    MimoOptimizer,
    ModuleOptimizerInfo,
    _get_pg_collection_for_optimizer,
)
from megatron.core.optimizer.optimizer_config import OptimizerConfig
from tests.unit_tests.test_utilities import Utils


def _ranks(pg):
    return sorted(dist.get_process_group_ranks(pg))


def _build_dense_grid(tp=1, cp=1, pp=1, dp=1, ep=1, expt_dp=1, register_expert=False):
    """Build a HyperCommGrid spanning the whole world with optimizer groups.

    The base view carries the dense ``tp/cp/pp/dp`` factorization (plus degenerate
    ``ep``/``expt_dp`` placeholders for backward compatibility). When ``register_expert``
    is set, a separate ``"expert"`` rank-view (``expt_tp/ep/expt_dp/pp``) is registered
    via ``register_view`` with ``pp`` declared a shared dim, and its groups are created
    through the ``view="expert"`` kwarg, exercising the heterogeneous path.
    """
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp, 1, 1],
        dim_names=["tp", "cp", "pp", "dp", "ep", "expt_dp"],
        rank_offset=0,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    grid.create_pg(["dp", "cp"])
    grid.create_pg(["tp", "pp"])
    grid.create_pg(["tp", "cp", "dp", "pp"])
    if register_expert:
        # expt_tp * ep * expt_dp * pp must equal the grid size. ``pp`` is shared with the
        # base view: ``register_view`` enforces that a shared dim enumerate to the SAME
        # ranks under both factorizations, so the expert dim ordering is chosen
        # (``[expt_tp, pp, expt_dp, ep]``) such that its ``pp`` groups match the base grid's
        # ``pp`` groups exactly. The expert ``[expt_tp, ep, pp]`` group then reuses the
        # base pp ranks for its pp component.
        grid.register_view(
            EXPERT_VIEW,
            shape=[tp, pp, expt_dp, ep],
            dim_names=["expt_tp", "pp", "expt_dp", "ep"],
            shared_dims=["pp"],
        )
        grid.create_pg(["expt_tp", "ep", "pp"], view=EXPERT_VIEW)
        grid.create_pg(["expt_dp"], view=EXPERT_VIEW)
    else:
        grid.create_pg(["ep"])
        grid.create_pg(["tp", "ep", "pp"])
        grid.create_pg(["dp", "ep"])
    return grid


@pytest.fixture
def dist_env():
    Utils.initialize_distributed()
    yield dist.get_world_size()
    # Process groups created per-test are torn down inside each test.


class TestGetPgCollectionForOptimizer:
    """item1: expert-view group resolution + narrowed intra_dist_opt.

    REAL 8-GPU distributed test: builds real process groups and asserts on actual
    rank membership. Runs unconditionally on the 8-GPU unit-test lane.
    """

    def test_expert_view_groups_and_dense_intra_dist_opt(self, dist_env):
        world_size = dist_env
        if world_size != 8:
            pytest.skip(f"This real distributed test requires 8 GPUs, got {world_size}")

        # Dense: tp=2, dp=2, pp=2. Expert: expt_tp=2, ep=2, expt_dp=1, pp=2 (same 8 ranks,
        # different factorization). ep is an INDEPENDENT axis carved from the dense dp here.
        grid = _build_dense_grid(tp=2, pp=2, dp=2, ep=2, expt_dp=1, register_expert=True)
        try:
            pg = _get_pg_collection_for_optimizer(grid)

            # Dense groups come from the base view.
            assert pg.tp is grid.get_pg("tp")
            assert pg.mp is grid.get_pg(["tp", "pp"])
            assert pg.pp is grid.get_pg("pp")

            # intra_dist_opt is the DENSE grad-stats group: tp x cp x dp x pp, NOT ep.
            # It must equal the [tp,cp,dp,pp] base group.
            assert pg.intra_dist_opt is grid.get_pg(["tp", "cp", "dp", "pp"])
            # For this config [tp,cp,dp,pp] spans all 8 ranks (cp=1), so the dense
            # grad-stats group is the full world.
            assert _ranks(pg.intra_dist_opt) == list(range(8))

            # Expert groups come from the registered "expert" view, NOT the base dims.
            assert pg.tp_ep_pp is grid.get_pg(["expt_tp", "ep", "pp"], view=EXPERT_VIEW)
            assert pg.expt_dp is grid.get_pg(["expt_dp"], view=EXPERT_VIEW)
            assert pg.intra_expt_dp is pg.expt_dp

            # pp is a SHARED dim of the expert view: the expert [expt_tp, ep, pp]
            # group's pp membership must agree with the base pp group, and the view
            # reuses the base pp group object for a pure-pp request.
            assert grid.get_pg(["pp"], view=EXPERT_VIEW) is grid.get_pg("pp")

            # The expert tp_ep_pp partition differs from the base [tp,pp] (mp) partition:
            # it additionally folds in the ep axis, so a heterogeneous expert view is
            # genuinely honored rather than silently aliased onto the dense groups.
            assert _ranks(pg.tp_ep_pp) != _ranks(pg.mp)
        finally:
            grid.destroy()

    def test_falls_back_to_base_view_without_expert_view(self, dist_env):
        world_size = dist_env
        if world_size != 8:
            pytest.skip(f"This real distributed test requires 8 GPUs, got {world_size}")

        grid = _build_dense_grid(tp=2, pp=2, dp=2, register_expert=False)
        try:
            pg = _get_pg_collection_for_optimizer(grid)
            # No expert view -> expert groups resolve against the base view.
            assert pg.tp_ep_pp is grid.get_pg(["tp", "ep", "pp"])
            assert pg.expt_dp is grid.get_pg(["dp", "ep"])
            # intra_dist_opt is still the dense [tp,cp,dp,pp] group.
            assert pg.intra_dist_opt is grid.get_pg(["tp", "cp", "dp", "pp"])
        finally:
            grid.destroy()


class _FakeModuleOptimizer:
    """Minimal stand-in for a per-module MegatronOptimizer.

    ``count_zeros`` returns a fixed per-module value to emulate the real optimizer,
    whose own SUM reduce over its grad-stats group makes the value identical across
    all ranks that own the module.
    """

    def __init__(self, zeros):
        self._zeros = zeros
        self.is_stub_optimizer = False

    def count_zeros(self):
        return self._zeros


class TestCountZeros:
    """item2: count_zeros must be consistent across ranks in non-colocated mode.

    REAL 8-GPU distributed test: builds a real ``MimoOptimizer`` over two modules
    with disjoint rank ownership and verifies the cross-rank all_reduce(MAX) makes
    ``count_zeros`` agree on every rank via a real ``all_gather_object`` collective.
    Runs unconditionally on the 8-GPU unit-test lane.
    """

    def test_count_zeros_consistent_across_ranks(self, dist_env):
        world_size = dist_env
        if world_size != 8:
            pytest.skip(f"This real distributed test requires 8 GPUs, got {world_size}")

        rank = dist.get_rank()
        # Two modules with disjoint rank ownership: "images" on ranks 0-3, "language"
        # on ranks 4-7 (the canonical non-colocated split).
        images_active = rank < 4
        language_active = rank >= 4

        IMAGES_ZEROS = 7
        LANGUAGE_ZEROS = 11

        module_infos = {
            "images": ModuleOptimizerInfo(
                optimizer=_FakeModuleOptimizer(IMAGES_ZEROS) if images_active else None,
                grid=None,
                pg_collection=None,
                is_active=images_active,
            ),
            "language": ModuleOptimizerInfo(
                optimizer=_FakeModuleOptimizer(LANGUAGE_ZEROS) if language_active else None,
                grid=None,
                pg_collection=None,
                is_active=language_active,
            ),
        }
        config = OptimizerConfig(optimizer="adam", lr=1e-3)
        mimo_opt = MimoOptimizer(module_infos, config)

        total = mimo_opt.count_zeros()

        # The cross-rank MAX recovers every module's count on every rank, so the
        # global total is the SAME on all ranks and equals the sum of both modules.
        assert total == IMAGES_ZEROS + LANGUAGE_ZEROS

        # Sanity: a naive local sum would have returned 7 on ranks 0-3 and 11 on
        # ranks 4-7 -- inconsistent. Confirm every rank agrees on the global value
        # via a real collective.
        gathered = [None] * world_size
        dist.all_gather_object(gathered, total)
        assert len(set(gathered)) == 1, f"count_zeros disagrees across ranks: {gathered}"


class TestShardedKeyCollision:
    """item3: add_prefix_for_sharding disambiguates colliding inner optimizer keys.

    DistributedOptimizer.sharded_state_dict keys its regular optimizer state as
    ``optimizer.distributed.dp_group_idx_{model_parallel_rank}.*``. In non-colocated
    mode each module has its own model-parallel group, so two modules both produce
    model_parallel_rank 0 and therefore identical keys for different data on disjoint
    ranks -- a global key collision. Prefixing with ``mimo.{module_name}.`` fixes it.

    This is a real collision repro: it constructs real ShardedObjects and checks key
    disjointness; it does not fake torch.distributed.
    """

    @staticmethod
    def _module_sharded_state_dict():
        # Emulate the colliding ShardedObjects DistributedOptimizer emits when both
        # modules land on model_parallel_rank 0 (data_parallel_group_idx == 0).
        return {
            "step": ShardedObject(
                "optimizer.distributed.dp_group_idx_0.step", torch.tensor(0), (1,), (0,)
            ),
            "param_groups": ShardedObject(
                "optimizer.distributed.dp_group_idx_0.param_groups", [], (1,), (0,)
            ),
        }

    def test_keys_collide_without_prefix(self):
        images = self._module_sharded_state_dict()
        language = self._module_sharded_state_dict()
        images_keys = {v.key for v in images.values()}
        language_keys = {v.key for v in language.values()}
        # Demonstrate the collision: the two modules share identical inner keys.
        assert images_keys & language_keys == images_keys
        assert "optimizer.distributed.dp_group_idx_0.step" in images_keys & language_keys

    def test_prefix_disambiguates_keys(self):
        images = self._module_sharded_state_dict()
        language = self._module_sharded_state_dict()

        add_prefix_for_sharding(images, "mimo.images.")
        add_prefix_for_sharding(language, "mimo.language.")

        images_keys = {v.key for v in images.values()}
        language_keys = {v.key for v in language.values()}

        # After prefixing the two modules no longer collide on any key.
        assert images_keys.isdisjoint(language_keys)
        assert "mimo.images.optimizer.distributed.dp_group_idx_0.step" in images_keys
        assert "mimo.language.optimizer.distributed.dp_group_idx_0.step" in language_keys
