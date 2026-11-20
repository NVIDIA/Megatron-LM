# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import time
from datetime import timedelta
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core import parallel_state
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    create_row_parallel_groups,
    get_parameter_parallel_domain,
    resolve_gtp_remat_group,
)
from tests.unit_tests.test_utilities import Utils


class TestProcessGroupsConfig:
    """Simple tests for process group dataclasses."""

    def test_dp_cp_replicas_for_ddp_and_optimizer(self, mocker):
        """Weight replica groups are used directly without multiplying by CP size."""
        replicas = mocker.Mock(spec=dist.ProcessGroup)
        replicas.size.return_value = 2
        other = mocker.Mock(spec=dist.ProcessGroup)
        other.size.return_value = 1
        pg = ProcessGroupCollection(
            dp=other,
            dp_cp=replicas,
            expt_dp=other,
            tp=other,
            pp=other,
            ep=other,
            mp=other,
            tp_ep_pp=other,
            intra_dist_opt=other,
        )
        config = mocker.Mock(context_parallel_size=128)
        ddp_config = mocker.Mock(
            num_distributed_optimizer_instances=1, use_distributed_optimizer=True
        )
        chunk = mocker.Mock(ddp_config=ddp_config)
        ddp_groups = ProcessGroupCollection.setup_process_groups_for_ddp(pg, config, ddp_config)
        optimizer_groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
            pg, [chunk], use_gloo_process_groups=False
        )
        for groups in (ddp_groups, optimizer_groups):
            assert groups["dp_cp_group"] is replicas
            assert groups["intra_dp_cp_group"] is replicas

    @pytest.mark.parametrize("has_ddp_config", [True, False], ids=["fsdp", "no_ddp_config"])
    def test_intra_dist_opt_passes_through_without_ddp_distributed_optimizer(
        self, mocker, has_ddp_config
    ):
        """Megatron-FSDP leaves ddp_config.use_distributed_optimizer False but still needs the
        intra distributed-optimizer group for gradient statistics. A model chunk without a
        ddp_config gets the group from the collection too, like the parallel_state path."""
        group = mocker.Mock(spec=dist.ProcessGroup)
        group.size.return_value = 1
        intra_dist_opt = mocker.Mock(spec=dist.ProcessGroup)
        pg = ProcessGroupCollection(
            dp=group,
            dp_cp=group,
            expt_dp=group,
            tp=group,
            pp=group,
            ep=group,
            mp=group,
            tp_ep_pp=group,
            intra_dist_opt=intra_dist_opt,
        )
        if has_ddp_config:
            ddp_config = mocker.Mock(
                num_distributed_optimizer_instances=1, use_distributed_optimizer=False
            )
            chunk = mocker.Mock(ddp_config=ddp_config)
        else:
            chunk = mocker.Mock(spec=[])
        groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
            pg, [chunk], use_gloo_process_groups=False
        )
        assert groups["intra_dist_opt_group"] is intra_dist_opt

    def test_transformer_process_groups(self, mocker):
        """Test basic functionality of TransformerProcessGroups."""
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        model_pgs.tp = mock_pg1
        model_pgs.pp = mock_pg2

        # Test accessing attributes
        assert model_pgs.tp == mock_pg1
        assert model_pgs.pp == mock_pg2

        # Test attribute existence
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert model_pgs.cp is None  # Not set yet

    def test_grad_comm_process_groups(self, mocker):
        """Test basic functionality of ProcessGroupCollection."""
        # Create mock process groups
        mock_pg = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        grad_pgs = ProcessGroupCollection()

        # Test setting attributes after creation
        grad_pgs.dp = mock_pg

        # Test accessing attributes
        assert grad_pgs.dp == mock_pg

        # Test attribute existence
        assert hasattr(grad_pgs, 'dp')
        assert grad_pgs.dp_cp is None  # Not set yet

    def test_dp_fields_split_replicate_and_data_distribution_axes(self, mocker):
        """``dp`` is the replicate group; ``dp_gtp_remat`` spans the gtp_remat data axis."""
        replicate_pg = mocker.Mock(spec=dist.ProcessGroup)
        data_pg = mocker.Mock(spec=dist.ProcessGroup)

        def fake_get_data_parallel_group(*args, with_gtp_remat=True, **kwargs):
            return data_pg if with_gtp_remat else replicate_pg

        mocker.patch(
            'megatron.core.process_groups_config.parallel_state.get_data_parallel_group',
            side_effect=fake_get_data_parallel_group,
        )

        pgs = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['dp', 'dp_gtp_remat'])

        assert pgs.dp is replicate_pg
        assert pgs.dp_gtp_remat is data_pg

    def test_hierarchical_context_parallel_groups(self, mocker):
        """Test setting and accessing the hierarchical context parallel list."""
        # Create mock process groups
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)

        # Create instance
        model_pgs = ProcessGroupCollection()

        # Set the hierarchical context parallel groups
        model_pgs.hcp = [mock_pg1, mock_pg2]

        # Test list access
        assert isinstance(model_pgs.hcp, list)
        assert len(model_pgs.hcp) == 2
        assert model_pgs.hcp[0] == mock_pg1
        assert model_pgs.hcp[1] == mock_pg2

    def test_repr(self, mocker):
        """Test __repr__ shows active process groups and their sizes."""
        tp_size = 4
        pp_size = 2
        mock_tp = mocker.Mock(spec=dist.ProcessGroup)
        mock_tp.size.return_value = tp_size
        mock_pp = mocker.Mock(spec=dist.ProcessGroup)
        mock_pp.size.return_value = pp_size

        # Test empty collection
        empty_pgs = ProcessGroupCollection()
        assert repr(empty_pgs) == "ProcessGroupCollection(empty)"

        # Test collection with process groups
        model_pgs = ProcessGroupCollection()
        model_pgs.tp = mock_tp
        model_pgs.pp = mock_pp

        repr_str = repr(model_pgs)
        assert "ProcessGroupCollection(" in repr_str
        assert f"tp({tp_size})" in repr_str
        assert f"pp({pp_size})" in repr_str

    def test_repr_with_list_process_groups(self, mocker):
        """Test __repr__ handles list-typed process groups like hcp."""
        mock_pg1 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg1.size.return_value = 2
        mock_pg2 = mocker.Mock(spec=dist.ProcessGroup)
        mock_pg2.size.return_value = 4

        model_pgs = ProcessGroupCollection()
        model_pgs.hcp = [mock_pg1, mock_pg2]

        repr_str = repr(model_pgs)
        assert "ProcessGroupCollection(" in repr_str
        assert "hcp([2, 4])" in repr_str


class TestResolveGtpRematGroup:
    """resolve_gtp_remat_group must see through the multi-module wrapper.

    Only the per-module collections carry gtp_remat / expt_gtp_remat. A MIMO run hands the
    top-level MultiModuleProcessGroupCollection to callers such as
    setup_model_and_optimizer's register_gtp_symm_pool, and without the unwrap the vars()
    check misses and the MPU fallback answers None, because pretrain_mimo never initializes
    the MPU globals. The visible symptom was --gtp-remat-nccl-ub registering nothing, so the
    wgrad reduce-scatter never became symmetric and NCCL never chose an NVLS kernel.

    These run without torch.distributed: the collections only need to carry sentinels.
    """

    def _llm_collection(self, mocker):
        pgs = ProcessGroupCollection()
        pgs.gtp_remat = mocker.Mock(spec=dist.ProcessGroup)
        pgs.expt_gtp_remat = mocker.Mock(spec=dist.ProcessGroup)
        return pgs

    def test_plain_collection_is_returned_directly(self, mocker):
        """The non-MIMO path is unchanged."""
        pgs = self._llm_collection(mocker)

        assert resolve_gtp_remat_group(pgs, is_expert=False) is pgs.gtp_remat
        assert resolve_gtp_remat_group(pgs, is_expert=True) is pgs.expt_gtp_remat

    def test_multi_module_unwraps_to_language_model(self, mocker):
        """A colocated encoder + LLM rank resolves to the LLM's groups, not the encoder's."""
        llm = self._llm_collection(mocker)
        encoder = ProcessGroupCollection()
        encoder.gtp_remat = mocker.Mock(spec=dist.ProcessGroup)
        wrapper = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": encoder, "llm": llm}, language_model_module_name="llm"
        )

        assert resolve_gtp_remat_group(wrapper, is_expert=False) is llm.gtp_remat
        assert resolve_gtp_remat_group(wrapper, is_expert=True) is llm.expt_gtp_remat
        # Guards against unwrapping to whichever module happens to be first.
        assert resolve_gtp_remat_group(wrapper, is_expert=False) is not encoder.gtp_remat

    def test_multi_module_without_language_model_is_none(self, mocker):
        """Encoder-only ranks own no GTP axis, and must not raise from the unwrap."""
        encoder = ProcessGroupCollection()
        encoder.gtp_remat = mocker.Mock(spec=dist.ProcessGroup)
        wrapper = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": encoder}, language_model_module_name=None
        )

        assert resolve_gtp_remat_group(wrapper, is_expert=False) is None
        assert resolve_gtp_remat_group(wrapper, is_expert=True) is None


class TestPGConfigDefaultInitialization:

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_default_initialization(self):
        """Test default initialization of ProcessGroupCollection."""
        # Create instance
        model_pgs = ProcessGroupCollection.use_mpu_process_groups()

        # Test that instance was created successfully
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'dp')
        assert hasattr(model_pgs, 'dp_cp')

        # Test that only required process groups were initialized
        model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'cp'])
        assert hasattr(model_pgs, 'tp')
        assert hasattr(model_pgs, 'pp')
        assert hasattr(model_pgs, 'cp')
        assert model_pgs.dp is None  # Not requested, so not set

        # Test that an error is raised if an invalid process group is requested
        with pytest.raises(ValueError, match=r"Invalid process groups requested"):
            model_pgs = ProcessGroupCollection.use_mpu_process_groups(['tp', 'pp', 'foo'])

    def test_default_initialization_has_no_extra_parameter_domains(self):
        """MPU collections register no extra domain: params are only dense or expert."""
        model_pgs = ProcessGroupCollection.use_mpu_process_groups()

        assert model_pgs.parameter_replica_groups == {}
        assert model_pgs.engram is None
        assert model_pgs.engram_dp is None
        assert model_pgs.get_parameter_replica_group('dense') is model_pgs.dp_cp
        assert model_pgs.get_parameter_replica_group('expert') is model_pgs.expt_dp


class TestParameterDomains:
    """Domain names of params and param groups, and the replica group of each domain.

    These run without torch.distributed: the collections only need to carry sentinels.
    """

    def test_param_domain(self):
        """An explicit domain wins over the expert marker ``allreduce=False``."""
        param = torch.nn.Parameter(torch.zeros(1))
        assert get_parameter_parallel_domain(param) == "dense"

        param.allreduce = False
        assert get_parameter_parallel_domain(param) == "expert"

        param.parallel_domain = "engram"
        assert get_parameter_parallel_domain(param) == "engram"

    def test_param_group_domain(self):
        """Extra-domain groups report is_expert_parallel=True, but their domain still decides.

        The flag keeps their checkpoint identifiers the same in every table layout.
        """
        assert get_parameter_parallel_domain({"params": []}) == "dense"
        assert get_parameter_parallel_domain({"is_expert_parallel": True}) == "expert"
        assert (
            get_parameter_parallel_domain({"is_expert_parallel": True, "parallel_domain": None})
            == "expert"
        )
        assert (
            get_parameter_parallel_domain({"is_expert_parallel": True, "parallel_domain": "engram"})
            == "engram"
        )

    def test_parameter_replica_group(self):
        """Dense and expert map to (intra-instance) dp_cp and expt_dp, extras to their group."""
        groups = {
            name: mock.Mock(spec=dist.ProcessGroup)
            for name in ("dp_cp", "intra_dp_cp", "expt_dp", "intra_expt_dp", "engram_dp")
        }
        pgs = ProcessGroupCollection()
        for name in ("dp_cp", "intra_dp_cp", "expt_dp", "intra_expt_dp"):
            setattr(pgs, name, groups[name])
        pgs.parameter_replica_groups = {"engram": groups["engram_dp"]}

        assert pgs.get_parameter_replica_group("dense") is groups["dp_cp"]
        assert pgs.get_parameter_replica_group("dense", partial=True) is groups["intra_dp_cp"]
        assert pgs.get_parameter_replica_group("expert") is groups["expt_dp"]
        assert pgs.get_parameter_replica_group("expert", partial=True) is groups["intra_expt_dp"]
        assert pgs.get_parameter_replica_group("engram") is groups["engram_dp"]
        with pytest.raises(ValueError):
            pgs.get_parameter_replica_group("memory")

    def test_partial_replica_group_falls_back_to_the_full_group(self):
        """Without intra-instance groups, partial lookups return the full groups."""
        pgs = ProcessGroupCollection()
        pgs.dp_cp = mock.Mock(spec=dist.ProcessGroup)
        pgs.expt_dp = mock.Mock(spec=dist.ProcessGroup)

        assert pgs.get_parameter_replica_group("dense", partial=True) is pgs.dp_cp
        assert pgs.get_parameter_replica_group("expert", partial=True) is pgs.expt_dp
        with pytest.raises(ValueError):
            pgs.get_parameter_replica_group("engram")


# create_row_parallel_groups is a WORLD collective, so the cases below spawn their own 4-rank
# Gloo world on CPU, whatever the torchrun world size; every torchrun rank runs them, which keeps
# the ranks in step for the tests that follow. A rank that blocks is the failure the function
# must avoid, and new groups outlive the world timeout, so the parent kills the world and fails
# once the deadline passes instead of hanging.
_ROW_WORLD_SIZE = 4
_ROW_DEADLINE_S = 300


def _row_parallel_worker(rank, store_path, case):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=_ROW_WORLD_SIZE,
        timeout=timedelta(seconds=_ROW_DEADLINE_S),
    )
    try:
        case(rank)
    finally:
        dist.destroy_process_group()


def _spawn_row_parallel_case(tmp_path, case):
    context = mp.start_processes(
        _row_parallel_worker,
        args=(str(tmp_path / "store"), case),
        nprocs=_ROW_WORLD_SIZE,
        join=False,
        start_method="spawn",
    )
    deadline = time.monotonic() + _ROW_DEADLINE_S
    while not context.join(timeout=1):
        if time.monotonic() > deadline:
            for process in context.processes:
                process.kill()
                process.join()
            pytest.fail(f"a rank of the Gloo world was still running after {_ROW_DEADLINE_S} s")


def _new_groups(rank, ranks_per_group):
    """Create every group on every rank, in order; return the one this rank belongs to."""
    own_group = None
    for ranks in ranks_per_group:
        group = dist.new_group(ranks)
        if rank in ranks:
            own_group = group
    return own_group


def _assert_all_reduce(rank, group, members):
    probe = torch.tensor([float(rank)])
    dist.all_reduce(probe, group=group)
    assert probe.item() == sum(members)


def _shapes_and_order_case(rank):
    for parents, shard_size in (
        ([[0, 1, 2, 3]], 2),  # two row shards, two replicas each
        ([[0, 1, 2, 3]], 1),  # one-rank shards; the replica group spans the parent
        ([[0, 1], [2, 3]], 1),  # one parent per pipeline stage
        ([[0, 2], [1, 3]], 1),  # strided parents
    ):
        parent_group = _new_groups(rank, parents)
        parent = next(ranks for ranks in parents if rank in ranks)
        shard_group, replica_group, shard_ranks = create_row_parallel_groups(
            parent_group, shard_size, world_group=dist.group.WORLD, name="ROWS"
        )

        # Shards are contiguous blocks of the parent and replicas take every shard_size-th
        # rank of it, so a rank's position in its shard group is its row shard. Every rank
        # also gets the ranks of every shard, in creation order (parents sorted by ranks).
        position = parent.index(rank)
        start = position - position % shard_size
        shard = parent[start : start + shard_size]
        replica = parent[position % shard_size :: shard_size]
        assert shard_ranks == [
            ranks[block : block + shard_size]
            for ranks in sorted(parents)
            for block in range(0, len(ranks), shard_size)
        ]
        assert dist.get_process_group_ranks(shard_group) == shard
        assert dist.get_process_group_ranks(replica_group) == replica
        assert shard_group.rank() == shard.index(rank)
        assert replica_group.rank() == replica.index(rank)
        if shard_size == 1:
            # The only replica spans the parent, so the parent group itself is reused instead
            # of a second communicator over the same ranks.
            assert replica_group is parent_group

        # New groups are registered for timeout updates and described after the domain.
        for group, kind in ((shard_group, "SHARD"), (replica_group, "REPLICA")):
            if group is not parent_group:
                assert group in parallel_state._global_process_group_list
                assert group.group_desc == f"ROWS_{kind}"

        # Each group works only if every rank created every group in the same order.
        _assert_all_reduce(rank, shard_group, shard)
        _assert_all_reduce(rank, replica_group, replica)


def _parent_reuse_case(rank):
    for parents in ([[0, 1, 2, 3]], [[0, 2], [1, 3]]):
        parent_group = _new_groups(rank, parents)
        parent = next(ranks for ranks in parents if rank in ranks)
        shard_group, replica_group, shard_ranks = create_row_parallel_groups(
            parent_group, len(parent), world_group=dist.group.WORLD, name="ROWS"
        )

        # One shard spans the whole parent, so the parent group itself is the shard group and
        # every replica group is a single rank.
        assert shard_group is parent_group
        assert shard_ranks == sorted(parents)
        assert dist.get_process_group_ranks(replica_group) == [rank]
        _assert_all_reduce(rank, shard_group, parent)


def _uneven_membership_case(rank):
    # Ranks 2 and 3 are not members, so their group registries end up smaller than those of
    # ranks 0 and 1. PyTorch names members-only groups after that registry size, so every rank
    # must create every group, as parallel_state does.
    dist.new_group([0, 1])
    parent_group = dist.new_group([0, 1, 2, 3])
    shard_group, replica_group, _ = create_row_parallel_groups(
        parent_group, 2, world_group=dist.group.WORLD, name="ROWS"
    )
    _assert_all_reduce(rank, shard_group, [0, 1] if rank < 2 else [2, 3])
    _assert_all_reduce(rank, replica_group, [rank % 2, rank % 2 + 2])

    # Parents of different sizes: rank 3's parent is one shard and is reused, while the other
    # parent is split into one-rank shards. Every rank must take the same decisions, or the
    # ranks number later groups differently and a group spanning them all no longer forms.
    parents = [[0, 1, 2], [3]]
    parent_group = _new_groups(rank, parents)
    parent = next(ranks for ranks in parents if rank in ranks)
    shard_group, replica_group, shard_ranks = create_row_parallel_groups(
        parent_group, 1, world_group=dist.group.WORLD, name="UNEVEN"
    )
    assert shard_ranks == [[0], [1], [2], [3]]
    assert dist.get_process_group_ranks(shard_group) == [rank]
    assert dist.get_process_group_ranks(replica_group) == parent
    assert replica_group is parent_group
    if rank == 3:
        assert shard_group is parent_group
    _assert_all_reduce(rank, replica_group, parent)
    _assert_all_reduce(rank, dist.new_group([0, 1, 2, 3]), [0, 1, 2, 3])


def _error_after_collective_case(rank):
    whole = dist.new_group([0, 1, 2, 3])
    first, second = dist.new_group([0, 1, 2]), dist.new_group([2, 3])
    for parent_group, shard_size in (
        # Only rank 1 is wrong; had it raised before the shared collective, the other ranks
        # would wait in it forever.
        (whole, 0 if rank == 1 else 2),
        # The size divides no parent.
        (whole, 3),
        # The parents overlap on rank 2 instead of partitioning the world.
        (second if rank == 3 else first, 1),
    ):
        with pytest.raises(ValueError):
            create_row_parallel_groups(
                parent_group, shard_size, world_group=dist.group.WORLD, name="ROWS"
            )
        # Every rank raised, so all of them reach the barrier.
        dist.barrier()

    # The failed calls created no group on any rank, so the next call still lines up.
    shard_group, replica_group, _ = create_row_parallel_groups(
        whole, 2, world_group=dist.group.WORLD, name="ROWS"
    )
    _assert_all_reduce(rank, shard_group, [0, 1] if rank < 2 else [2, 3])
    _assert_all_reduce(rank, replica_group, [rank % 2, rank % 2 + 2])


class TestCreateRowParallelGroups:
    """Row-shard and replica groups split from parent groups, with Gloo on CPU."""

    def test_shapes_and_order(self, tmp_path):
        _spawn_row_parallel_case(tmp_path, _shapes_and_order_case)

    def test_parent_is_reused_as_the_only_shard(self, tmp_path):
        _spawn_row_parallel_case(tmp_path, _parent_reuse_case)

    def test_uneven_group_membership(self, tmp_path):
        _spawn_row_parallel_case(tmp_path, _uneven_membership_case)

    def test_invalid_configuration_raises_on_every_rank(self, tmp_path):
        _spawn_row_parallel_case(tmp_path, _error_after_collective_case)
