# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
import logging
import os
import sys

import pytest
import torch
import torch.distributed as dist
from packaging import version

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.comm.colocated_communicator import (
    ColocatedBridgeCommunicator,
    SliceInfo,
)
from tests.unit_tests.test_utilities import Utils

logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)

_active_grids: list = []
_active_comms: list = []


def create_hypercomm_grid(offset=0, tp=1, cp=1, pp=1, dp=1):
    grid = HyperCommGrid(
        shape=[tp, cp, pp, dp],
        dim_names=["tp", "cp", "pp", "dp"],
        rank_offset=offset,
        backend="nccl",
    )
    grid.create_pg(["tp"])
    grid.create_pg(["cp"])
    grid.create_pg(["pp"])
    grid.create_pg(["dp"])
    _active_grids.append(grid)
    return grid


def make_comm(*args, **kwargs):
    comm = ColocatedBridgeCommunicator(*args, **kwargs)
    _active_comms.append(comm)
    return comm


def destroy_all_grids():
    # Destroy communicators first so their NCCL subgroups are freed before we
    # tear down the parent grids. NCCL caps concurrent communicators at ~500;
    # leaked PGs from per-test fixtures blow that budget quickly.
    for comm in _active_comms:
        comm.destroy()
    _active_comms.clear()
    for grid in _active_grids:
        grid.destroy()
    _active_grids.clear()


# ── Test 1: Rank mappings ──────────────────────────────────────────────────────


class TestRankMappings:

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, expected_src_pos, expected_dest_pos",
        [
            # Fan-in: TP2/DP4 → TP4/DP2
            (
                2,
                4,
                4,
                2,
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (1, 0),
                    3: (1, 1),
                    4: (2, 0),
                    5: (2, 1),
                    6: (3, 0),
                    7: (3, 1),
                },
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (0, 3),
                    4: (1, 0),
                    5: (1, 1),
                    6: (1, 2),
                    7: (1, 3),
                },
            ),
            # Fan-out: TP4/DP2 → TP2/DP4
            (
                4,
                2,
                2,
                4,
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (0, 3),
                    4: (1, 0),
                    5: (1, 1),
                    6: (1, 2),
                    7: (1, 3),
                },
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (1, 0),
                    3: (1, 1),
                    4: (2, 0),
                    5: (2, 1),
                    6: (3, 0),
                    7: (3, 1),
                },
            ),
            # Equal: TP4/DP2 → TP4/DP2
            (
                4,
                2,
                4,
                2,
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (0, 3),
                    4: (1, 0),
                    5: (1, 1),
                    6: (1, 2),
                    7: (1, 3),
                },
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (0, 3),
                    4: (1, 0),
                    5: (1, 1),
                    6: (1, 2),
                    7: (1, 3),
                },
            ),
            # Extreme: TP1/DP8 → TP8/DP1
            (
                1,
                8,
                8,
                1,
                {
                    0: (0, 0),
                    1: (1, 0),
                    2: (2, 0),
                    3: (3, 0),
                    4: (4, 0),
                    5: (5, 0),
                    6: (6, 0),
                    7: (7, 0),
                },
                {
                    0: (0, 0),
                    1: (0, 1),
                    2: (0, 2),
                    3: (0, 3),
                    4: (0, 4),
                    5: (0, 5),
                    6: (0, 6),
                    7: (0, 7),
                },
            ),
        ],
        ids=["fan_in", "fan_out", "equal", "extreme"],
    )
    def test_rank_mappings(
        self, src_tp, src_dp, dest_tp, dest_dp, expected_src_pos, expected_dest_pos
    ):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid)

        assert comm.rank_to_src_pos == expected_src_pos
        assert comm.rank_to_dest_pos == expected_dest_pos

    def test_rank_mappings_with_rank_offset(self):
        # 4-rank grids at offset=4 (covering ranks 4-7). Exercises the
        # rank_offset propagation that previously only ran with offset=0.
        if dist.get_world_size() < 8:
            pytest.skip("requires at least 8 ranks")
        src_grid = create_hypercomm_grid(offset=4, tp=2, dp=2)
        dest_grid = create_hypercomm_grid(offset=4, tp=1, dp=4)
        comm = make_comm(src_grid, dest_grid)

        assert comm.rank_to_src_pos == {4: (0, 0), 5: (0, 1), 6: (1, 0), 7: (1, 1)}
        assert comm.rank_to_dest_pos == {4: (0, 0), 5: (1, 0), 6: (2, 0), 7: (3, 0)}


# ── Test 2: All-gather groups ──────────────────────────────────────────────────


class TestAllGatherGroups:

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, expected_groups",
        [
            # Fan-in: TP2/DP4 → TP4/DP2
            (2, 4, 4, 2, [[0, 2], [1, 3], [4, 6], [5, 7]]),
            # Extreme: TP1/DP8 → TP8/DP1
            (1, 8, 8, 1, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        ],
        ids=["fan_in_2x", "extreme_8x"],
    )
    def test_fan_in_all_gather_groups(self, src_tp, src_dp, dest_tp, dest_dp, expected_groups):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid)

        assert comm.all_gather_group_ranks == expected_groups
        assert comm.all_gather_pg is not None

    def test_fan_out_no_all_gather(self):
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, dp=4)
        comm = make_comm(src_grid, dest_grid)

        assert comm.all_gather_group_ranks == []
        assert comm.all_gather_pg is None

    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, expected_groups",
        [
            # Fan-out: TP4/DP2 → TP2/DP4. Two src DP groups, two dest TP shards;
            # for each src_dp_idx we sweep dest_tp_idx over 2 dest DP replicas.
            # Expected groups are (src_dp_idx, dest_tp_idx) sweeping dest_dp_idx
            # in slot order.
            (4, 2, 2, 4, [[0, 2], [1, 3], [4, 6], [5, 7]]),
            # Extreme fan-out: TP8/DP1 → TP1/DP8 (one src DP group, one dest TP).
            (8, 1, 1, 8, [[0, 1, 2, 3, 4, 5, 6, 7]]),
        ],
        ids=["fan_out_2x", "extreme_8x"],
    )
    def test_fan_out_gather_groups(self, src_tp, src_dp, dest_tp, dest_dp, expected_groups):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid)

        # Direct equality check enforces both membership and slot order —
        # all_gather_into_tensor concatenates by group-local-rank, and backward
        # relies on slot 0 of each group holding dest_dp_start's slice.
        assert comm.fan_out_gather_group_ranks == expected_groups
        assert comm.fan_out_gather_pg is not None

    def test_fan_in_no_fan_out_gather(self):
        src_grid = create_hypercomm_grid(tp=2, dp=4)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        comm = make_comm(src_grid, dest_grid)

        assert comm.fan_out_gather_group_ranks == []
        assert comm.fan_out_gather_pg is None


# ── Test 3: Slice info ─────────────────────────────────────────────────────────


class TestSliceInfo:

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.parametrize(
        "src_tp, src_dp, dest_tp, dest_dp, batch_size, expected_slices",
        [
            # Fan-out: TP4/DP2 → TP2/DP4, batch=8
            (
                4,
                2,
                2,
                4,
                8,
                {
                    0: SliceInfo(start=0, size=4),
                    1: SliceInfo(start=0, size=4),
                    2: SliceInfo(start=4, size=4),
                    3: SliceInfo(start=4, size=4),
                    4: SliceInfo(start=0, size=4),
                    5: SliceInfo(start=0, size=4),
                    6: SliceInfo(start=4, size=4),
                    7: SliceInfo(start=4, size=4),
                },
            ),
            # Fan-in: TP2/DP4 → TP4/DP2, batch=8
            (
                2,
                4,
                4,
                2,
                8,
                {
                    0: SliceInfo(start=0, size=4),
                    1: SliceInfo(start=0, size=4),
                    2: SliceInfo(start=4, size=4),
                    3: SliceInfo(start=4, size=4),
                    4: SliceInfo(start=0, size=4),
                    5: SliceInfo(start=0, size=4),
                    6: SliceInfo(start=4, size=4),
                    7: SliceInfo(start=4, size=4),
                },
            ),
        ],
        ids=["fan_out", "fan_in"],
    )
    def test_slice_info(self, src_tp, src_dp, dest_tp, dest_dp, batch_size, expected_slices):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid)

        rank = dist.get_rank()
        if rank not in expected_slices:
            pytest.skip(f"rank {rank} not in expected_slices")

        info = comm.get_slice_info(batch_size)
        expected = expected_slices[rank]
        assert info.start == expected.start, f"rank {rank}: start {info.start} != {expected.start}"
        assert info.size == expected.size, f"rank {rank}: size {info.size} != {expected.size}"

    def test_equal_dp_slice(self):
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        comm = make_comm(src_grid, dest_grid)

        info = comm.get_slice_info(batch_size=8)
        assert info == SliceInfo(start=0, size=8)


# ── Test 3b: _validate_grids negative tests ───────────────────────────────────


class TestValidateGrids:
    """One negative test per raise path in ColocatedBridgeCommunicator._validate_grids.

    Each case builds a pair of grids that violates exactly one invariant and
    asserts that the constructor raises ValueError.
    """

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    def _grid_missing_tp(self, offset=0, dp=1):
        # Build a grid without a 'tp' dim to exercise the "missing 'tp'" raise.
        grid = HyperCommGrid(
            shape=[dp], dim_names=["dp"], rank_offset=offset, backend="nccl"
        )
        grid.create_pg(["dp"])
        _active_grids.append(grid)
        return grid

    def test_missing_tp_dim(self):
        src_grid = self._grid_missing_tp(dp=8)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        with pytest.raises(ValueError, match="must have 'tp' dimension"):
            make_comm(src_grid, dest_grid)

    def test_size_mismatch(self):
        src_grid = create_hypercomm_grid(tp=2, dp=4)  # 8 ranks
        dest_grid = create_hypercomm_grid(offset=4, tp=2, dp=2)  # 4 ranks
        with pytest.raises(ValueError, match="span same number of ranks"):
            make_comm(src_grid, dest_grid)

    def test_rank_offset_mismatch(self):
        src_grid = create_hypercomm_grid(offset=0, tp=2, dp=2)
        dest_grid = create_hypercomm_grid(offset=4, tp=2, dp=2)
        with pytest.raises(ValueError, match="same rank offset"):
            make_comm(src_grid, dest_grid)

    def test_src_pp_gt_one_rejected(self):
        src_grid = create_hypercomm_grid(tp=2, pp=2, dp=2)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        with pytest.raises(ValueError, match="src PP must be 1"):
            make_comm(src_grid, dest_grid)

    def test_dest_pp_gt_one_rejected(self):
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, pp=2, dp=2)
        with pytest.raises(ValueError, match="dest PP must be 1"):
            make_comm(src_grid, dest_grid)

    def test_cp_gt_one_rejected(self):
        src_grid = create_hypercomm_grid(tp=2, cp=2, dp=2)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        with pytest.raises(ValueError, match="CP must be 1"):
            make_comm(src_grid, dest_grid)

    def test_dp_not_divisible(self):
        # 6-rank grids with DP sizes (3 vs 2) that neither divides the other.
        # Fits inside an 8-rank world (HyperCommGrid enforces size <= world - offset).
        if dist.get_world_size() < 6:
            pytest.skip("requires at least 6 ranks")
        src_grid = HyperCommGrid(
            shape=[2, 1, 1, 3], dim_names=["tp", "cp", "pp", "dp"], backend="nccl"
        )
        dest_grid = HyperCommGrid(
            shape=[3, 1, 1, 2], dim_names=["tp", "cp", "pp", "dp"], backend="nccl"
        )
        for g in (src_grid, dest_grid):
            _active_grids.append(g)
        with pytest.raises(ValueError, match="evenly divisible"):
            make_comm(src_grid, dest_grid)

    def test_pp_gt_1_rejected(self):
        # Dedicated coverage of the PP>1 guard on either grid. PP>1 is out of
        # scope for this communicator; the validator must reject on either side.
        for pp_on in ("src", "dest"):
            _active_grids.clear()
            if pp_on == "src":
                src_grid = create_hypercomm_grid(tp=2, pp=2, dp=2)
                dest_grid = create_hypercomm_grid(tp=4, dp=2)
                expected = "src PP must be 1"
            else:
                src_grid = create_hypercomm_grid(tp=4, dp=2)
                dest_grid = create_hypercomm_grid(tp=2, pp=2, dp=2)
                expected = "dest PP must be 1"
            with pytest.raises(ValueError, match=expected):
                make_comm(src_grid, dest_grid)
            # Release PGs between iterations to stay under the NCCL cap.
            for g in _active_grids:
                g.destroy()
            _active_grids.clear()


# ── Test 3c: communicate() runtime preconditions ──────────────────────────────


class TestCommunicatePreconditions:
    """Runtime-input checks enforced by ``communicate()``."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    def test_non_divisible_batch_raises_fan_out(self):
        # Fan-out: dest_dp=4, src_dp=2 → fan_out_scale=2. Pass a batch dim of
        # size 3 so 3 % 2 != 0 and we should raise before any slicing runs.
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, dp=4)
        comm = make_comm(
            src_grid, dest_grid, dim_mapping={'b': 0, 'h': 1}
        )
        tensor = torch.zeros(3, 8, device='cuda')
        with pytest.raises(ValueError, match="not divisible by fan_out_scale"):
            comm.communicate(tensor)

    def test_non_divisible_batch_raises_fan_in_backward_narrow(self):
        # Fan-in: fan_in_scale=2. Fan-in forward all-gathers (no slice), so
        # the forward path never divides. The backward path narrows the
        # post-gather output via get_slice_info, which raises on a non-
        # divisible size. Call get_slice_info directly with an odd size to
        # exercise that raise path without a full backward.
        src_grid = create_hypercomm_grid(tp=2, dp=4)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        comm = make_comm(src_grid, dest_grid)
        with pytest.raises(ValueError, match="not divisible by fan_in_scale"):
            comm.get_slice_info(batch_size=3)

    def test_non_divisible_get_slice_info_fan_out(self):
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, dp=4)
        comm = make_comm(src_grid, dest_grid)
        with pytest.raises(ValueError, match="not divisible by fan_out_scale"):
            comm.get_slice_info(batch_size=5)


# ── Test 3d: destroy() releases PGs ──────────────────────────────────────────


class TestDestroy:
    """``destroy()`` must null out both PG attributes."""

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    def test_destroy_releases_fan_in_pg(self):
        src_grid = create_hypercomm_grid(tp=2, dp=4)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        # Don't track via make_comm — destroy() is exactly what we're testing.
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)
        assert comm.all_gather_pg is not None
        assert comm.fan_out_gather_pg is None
        comm.destroy()
        assert comm.all_gather_pg is None
        assert comm.fan_out_gather_pg is None

    def test_destroy_releases_fan_out_pg(self):
        src_grid = create_hypercomm_grid(tp=4, dp=2)
        dest_grid = create_hypercomm_grid(tp=2, dp=4)
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)
        assert comm.fan_out_gather_pg is not None
        assert comm.all_gather_pg is None
        comm.destroy()
        assert comm.fan_out_gather_pg is None

    def test_destroy_is_idempotent(self):
        # Calling destroy twice must not raise — leftover test fixtures often
        # double-destroy during exception cleanup.
        src_grid = create_hypercomm_grid(tp=2, dp=4)
        dest_grid = create_hypercomm_grid(tp=4, dp=2)
        comm = ColocatedBridgeCommunicator(src_grid, dest_grid)
        comm.destroy()
        comm.destroy()


# ── Test 3e: Bridge gradient correctness (bitwise exact) ─────────────────────


def _shape_for_dim_mapping(dim_mapping, B, S, H):
    s = [0, 0, 0]
    s[dim_mapping['b']] = B
    s[dim_mapping['s']] = S
    s[dim_mapping['h']] = H
    return s


# Parametrize dim_mapping for the fan-in tests (tests 1 & 2 per AXIOM spec).
_DIM_MAPPINGS = [{'s': 0, 'b': 1, 'h': 2}, {'b': 0, 's': 1, 'h': 2}]
_DIM_MAPPING_IDS = ["sbh", "bsh"]


class TestBridgeGradients:
    """Gradient correctness for ColocatedBridgeCommunicator.

    All assertions use ``rtol=0, atol=0`` — the bridge is pure data movement
    (narrow / all-gather), so both forward output and the adjoint backward are
    exact functions of the inputs. Any deviation is a logic bug.
    """

    S = 8
    B_PER_RANK = 2
    H = 128

    @classmethod
    def setup_class(cls):
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def teardown_method(self):
        destroy_all_grids()

    # ── Test 1: fan-in forward = torch.cat of sibling inputs ─────────────────
    @pytest.mark.parametrize(
        "src_tp,src_dp,dest_tp,dest_dp",
        [(2, 4, 4, 2), (1, 8, 8, 1)],
        ids=["2x_fan_in", "8x_fan_in"],
    )
    @pytest.mark.parametrize("dim_mapping", _DIM_MAPPINGS, ids=_DIM_MAPPING_IDS)
    def test_fan_in_forward_equals_torch_cat(
        self, src_tp, src_dp, dest_tp, dest_dp, dim_mapping
    ):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid, dim_mapping=dim_mapping)

        rank = dist.get_rank()
        shape = _shape_for_dim_mapping(dim_mapping, self.B_PER_RANK, self.S, self.H)

        # Distinct inputs per rank so the cat reveals ordering bugs.
        torch.manual_seed(1000 + rank)
        local_input = torch.randn(*shape, device='cuda')

        actual = comm.communicate(local_input)

        # Expected: manual all_gather over the communicator's fan-in group,
        # then cat along batch_dim. all_gather preserves group-local-rank
        # order, which is the same order the communicator uses.
        group = comm.all_gather_pg
        gathered = [torch.empty_like(local_input) for _ in range(dist.get_world_size(group))]
        dist.all_gather(gathered, local_input, group=group)
        expected = torch.cat(gathered, dim=dim_mapping['b'])

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # ── Test 2: fan-in backward = grad_output.narrow for this rank's slot ────
    @pytest.mark.parametrize(
        "src_tp,src_dp,dest_tp,dest_dp",
        [(2, 4, 4, 2), (1, 8, 8, 1)],
        ids=["2x_fan_in", "8x_fan_in"],
    )
    @pytest.mark.parametrize("dim_mapping", _DIM_MAPPINGS, ids=_DIM_MAPPING_IDS)
    def test_fan_in_backward_equals_narrow(
        self, src_tp, src_dp, dest_tp, dest_dp, dim_mapping
    ):
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid, dim_mapping=dim_mapping)

        rank = dist.get_rank()
        batch_dim = dim_mapping['b']
        b_local = self.B_PER_RANK
        shape = _shape_for_dim_mapping(dim_mapping, b_local, self.S, self.H)

        torch.manual_seed(1000 + rank)
        local_input = torch.randn(*shape, device='cuda', requires_grad=True)
        out = comm.communicate(local_input)

        # grad_output is TP-replicated within the dest DP group: seed the same
        # on every rank so every rank in the fan-in group backward-narrows the
        # same upstream gradient. out shape is identical across group members,
        # so seeded randn produces the same tensor on each.
        torch.manual_seed(42)
        grad_output = torch.randn_like(out)
        out.backward(grad_output)

        slot = comm.rank_to_src_pos[rank][0] % comm.fan_in_scale
        expected = grad_output.narrow(batch_dim, slot * b_local, b_local).contiguous()
        torch.testing.assert_close(local_input.grad, expected, rtol=0, atol=0)

    # ── Test 3: fan-out forward = input.narrow for this rank's slot ─────────
    @pytest.mark.parametrize(
        "src_tp,src_dp,dest_tp,dest_dp",
        [(4, 2, 2, 4), (8, 1, 1, 8)],
        ids=["2x_fan_out", "8x_fan_out"],
    )
    def test_fan_out_forward_equals_narrow(self, src_tp, src_dp, dest_tp, dest_dp):
        dim_mapping = {'b': 0, 's': 1, 'h': 2}
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid, dim_mapping=dim_mapping)

        rank = dist.get_rank()
        batch_dim = dim_mapping['b']
        b_per_dest = self.B_PER_RANK
        b_full = b_per_dest * comm.fan_out_scale
        shape = _shape_for_dim_mapping(dim_mapping, b_full, self.S, self.H)

        # Input is TP-replicated on the batch dim (bridge contract). Seed
        # identically across all ranks to satisfy it.
        torch.manual_seed(42)
        input_tensor = torch.randn(*shape, device='cuda')

        actual = comm.communicate(input_tensor)

        slot = comm.rank_to_dest_pos[rank][0] % comm.fan_out_scale
        expected = input_tensor.narrow(
            batch_dim, slot * b_per_dest, b_per_dest
        ).contiguous()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # ── Test 4 (CRITICAL): fan-out backward = concat of all sibling grads ──
    @pytest.mark.parametrize(
        "src_tp,src_dp,dest_tp,dest_dp",
        [(4, 2, 2, 4), (8, 1, 1, 8)],
        ids=["2x_fan_out", "8x_fan_out"],
    )
    def test_fan_out_backward_equals_concat_of_sibling_grads(
        self, src_tp, src_dp, dest_tp, dest_dp
    ):
        """Fan-out backward must all-gather sibling grads in slot order.

        Catches four distinct regressions with a single assertion:
          * zero-pad-without-gather (other slots would be zero),
          * wrong slot order (values would be scrambled),
          * double-counting (values would be multiplied),
          * missing siblings (shape or zeros would diverge).
        """
        dim_mapping = {'b': 0, 's': 1, 'h': 2}
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid, dim_mapping=dim_mapping)

        rank = dist.get_rank()
        batch_dim = dim_mapping['b']
        scale = comm.fan_out_scale
        b_per_dest = self.B_PER_RANK
        b_full = b_per_dest * scale
        shape = _shape_for_dim_mapping(dim_mapping, b_full, self.S, self.H)

        torch.manual_seed(42)  # identical input across ranks (TP-replicated)
        input_tensor = torch.randn(*shape, device='cuda', requires_grad=True)
        out = comm.communicate(input_tensor)  # narrowed to (b_per_dest, S, H)

        # Distinct grad per slot so the cat reveals both membership and order.
        slot = comm.rank_to_dest_pos[rank][0] % scale
        grad_output = (slot + 1) * torch.ones_like(out)
        out.backward(grad_output)

        slot_shape = _shape_for_dim_mapping(dim_mapping, b_per_dest, self.S, self.H)
        expected = torch.cat(
            [(i + 1) * torch.ones(*slot_shape, device='cuda') for i in range(scale)],
            dim=batch_dim,
        )
        torch.testing.assert_close(input_tensor.grad, expected, rtol=0, atol=0)

    # ── Test 5: equal DP is a pure identity forward and backward ────────────
    @pytest.mark.parametrize(
        "src_tp,src_dp,dest_tp,dest_dp",
        [(4, 2, 4, 2), (2, 4, 2, 4)],
        ids=["tp4_dp2", "tp2_dp4"],
    )
    def test_equal_dp_is_bitwise_identity_fwd_and_bwd(
        self, src_tp, src_dp, dest_tp, dest_dp
    ):
        dim_mapping = {'b': 0, 's': 1, 'h': 2}
        src_grid = create_hypercomm_grid(tp=src_tp, dp=src_dp)
        dest_grid = create_hypercomm_grid(tp=dest_tp, dp=dest_dp)
        comm = make_comm(src_grid, dest_grid, dim_mapping=dim_mapping)

        shape = _shape_for_dim_mapping(dim_mapping, self.B_PER_RANK, self.S, self.H)
        torch.manual_seed(1000 + dist.get_rank())
        x = torch.randn(*shape, device='cuda', requires_grad=True)

        out = comm.communicate(x)
        torch.testing.assert_close(out, x, rtol=0, atol=0)

        grad_output = torch.randn_like(x)
        out.backward(grad_output)
        torch.testing.assert_close(x.grad, grad_output, rtol=0, atol=0)


# ── Test 4: Forward / backward golden test ─────────────────────────────────────


class TestGolden:

    @classmethod
    def setup_class(cls):
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        os.environ["NVTE_FLASH_ATTN"] = "0"
        os.environ["NVTE_FUSED_ATTN"] = "0"
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    @classmethod
    def teardown_class(cls):
        Utils.destroy_model_parallel()

    def teardown_method(self):
        destroy_all_grids()

    @pytest.mark.skipif(
        version.parse(torch.__version__) < version.parse("2.3.0"), reason="Requires PyTorch 2.3+"
    )
    def test_forward_backward_golden(self):
        from tests.unit_tests.pipeline_parallel.test_bridge_communicator import (
            _avg_params,
            _create_transformer_block,
            _get_pg_collection_from_grid,
            _shard_and_copy_,
        )

        hidden_size = 1024
        seq_len = 16
        micro_batch = 8
        dtype = torch.float32
        rank = dist.get_rank()

        # Encoder TP2/DP4, LLM TP4/DP2
        enc_tp, enc_dp = 2, 4
        llm_tp, llm_dp = 4, 2

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, create_gloo_process_groups=False
        )

        # Reference TP1 blocks
        ref_grid = create_hypercomm_grid(tp=1, dp=8)
        ref_pg = _get_pg_collection_from_grid(ref_grid)
        ref_enc = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg
        )
        _avg_params(ref_enc, ref_grid.get_pg("dp"))
        ref_llm = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=ref_pg
        )
        _avg_params(ref_llm, ref_grid.get_pg("dp"))

        # Sharded encoder block (TP2/DP4)
        enc_grid = create_hypercomm_grid(tp=enc_tp, dp=enc_dp)
        enc_pg = _get_pg_collection_from_grid(enc_grid)
        enc_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=enc_pg
        )
        _shard_and_copy_(ref_enc, enc_block, enc_tp, enc_pg.tp.rank())

        # Sharded LLM block (TP4/DP2)
        llm_grid = create_hypercomm_grid(tp=llm_tp, dp=llm_dp)
        llm_pg = _get_pg_collection_from_grid(llm_grid)
        llm_block = _create_transformer_block(
            dtype=dtype, hidden_size=hidden_size, pg_collection=llm_pg
        )
        _shard_and_copy_(ref_llm, llm_block, llm_tp, llm_pg.tp.rank())

        dist.barrier()

        # Communicator
        comm = make_comm(
            enc_grid,
            llm_grid,
            src_module_name="encoder",
            dest_module_name="llm",
            dim_mapping={"s": 0, "b": 1, "h": 2},
        )

        # ── Reference forward (full batch, TP1) ───────────────────────────
        torch.manual_seed(42)
        full_input = torch.randn(seq_len, micro_batch, hidden_size, device="cuda", dtype=dtype)
        full_input_ref = full_input.clone().detach().requires_grad_(True)
        ref_enc_out = ref_enc(hidden_states=full_input_ref, attention_mask=None)
        ref_llm_out = ref_llm(hidden_states=ref_enc_out, attention_mask=None)

        # ── Colocated forward ──────────────────────────────────────────────
        # Each rank gets its encoder DP slice
        enc_dp_idx = comm.rank_to_src_pos[rank][0]
        enc_slice_size = micro_batch // enc_dp
        enc_input_slice = (
            full_input[:, enc_dp_idx * enc_slice_size : (enc_dp_idx + 1) * enc_slice_size, :]
            .clone()
            .detach()
            .requires_grad_(True)
        )

        enc_out = enc_block(hidden_states=enc_input_slice, attention_mask=None)
        bridged = comm.communicate(enc_out)
        llm_out = llm_block(hidden_states=bridged, attention_mask=None)

        # ── Compare forward outputs ────────────────────────────────────────
        llm_dp_idx = comm.rank_to_dest_pos[rank][0]
        llm_slice_size = micro_batch // llm_dp
        ref_slice = ref_llm_out[
            :, llm_dp_idx * llm_slice_size : (llm_dp_idx + 1) * llm_slice_size, :
        ].detach()

        torch.testing.assert_close(llm_out.detach(), ref_slice, rtol=1e-3, atol=1e-3)

        # ── Backward ──────────────────────────────────────────────────────
        llm_out.sum().backward()
        ref_llm_out.sum().backward()

        ref_input_grad_slice = full_input_ref.grad[
            :, enc_dp_idx * enc_slice_size : (enc_dp_idx + 1) * enc_slice_size, :
        ]
        torch.testing.assert_close(enc_input_slice.grad, ref_input_grad_slice, rtol=1e-5, atol=1e-5)

        Utils.destroy_model_parallel()
