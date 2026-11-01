# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest
import torch.distributed as dist

from megatron.core import parallel_state, process_groups_config
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    ProcessGroupFallbackWarning,
    require_process_groups,
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


class TestGlobalFallbackWarnings:
    """The global fallbacks of the group helpers warn instead of reading the grid silently.

    These run without torch.distributed: the global groups are sentinels.
    """

    @pytest.fixture(autouse=True)
    def fresh_warning_registry(self, monkeypatch):
        monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())

    @staticmethod
    def _global_gtp_groups(mocker, size):
        """Make the global grid answer a GTP group of ``size`` ranks, or None for size 0."""
        group = None
        if size:
            group = mocker.Mock(spec=dist.ProcessGroup)
            group.size.return_value = size
        mocker.patch.object(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            return_value=ProcessGroupCollection(gtp_remat=group, expt_gtp_remat=group),
        )
        return group

    @pytest.mark.parametrize("is_expert", [False, True])
    def test_a_sharding_global_group_warns_naming_the_owner(self, mocker, is_expert):
        group = self._global_gtp_groups(mocker, size=2)
        with pytest.warns(FutureWarning, match="Owner was called without `pg_collection`"):
            assert resolve_gtp_remat_group(None, is_expert, owner="Owner") is group

    def test_a_collection_without_the_field_names_the_field(self, mocker):
        self._global_gtp_groups(mocker, size=2)
        pgs = ProcessGroupCollection(tp=mocker.Mock(spec=dist.ProcessGroup))
        with pytest.warns(
            FutureWarning, match=r"Owner was called without `pg_collection\.expt_gtp_remat`"
        ):
            resolve_gtp_remat_group(pgs, is_expert=True, owner="Owner")

    @pytest.mark.parametrize("size", [0, 1])
    def test_a_global_grid_without_gtp_warns_as_well(self, mocker, size):
        # A missing or single-rank group leaves GTP off, as an absent field will once the
        # fallback is removed, but the fallback still reads it from the global grid.
        group = self._global_gtp_groups(mocker, size)
        with pytest.warns(FutureWarning, match="Owner was called without `pg_collection`"):
            assert resolve_gtp_remat_group(None, is_expert=False, owner="Owner") is group

    @pytest.mark.parametrize("is_expert", [False, True])
    def test_an_uninitialized_grid_warns_and_leaves_gtp_off(self, monkeypatch, is_expert):
        monkeypatch.setattr(parallel_state, "_GTP_WEIGHT_REMAT_GROUP", None)
        monkeypatch.setattr(parallel_state, "_EXPERT_GTP_WEIGHT_REMAT_GROUP", None)
        with pytest.warns(FutureWarning, match="Owner was called without `pg_collection`"):
            assert resolve_gtp_remat_group(None, is_expert, owner="Owner") is None

    @staticmethod
    def _forbid_global_grid(mocker):
        """Make every read of the global GTP groups raise."""
        return mocker.patch.object(
            ProcessGroupCollection,
            "use_mpu_process_groups",
            side_effect=AssertionError("read the global grid"),
        )

    @pytest.mark.parametrize("is_expert", [False, True])
    @pytest.mark.parametrize("passed", ["nothing", "collection", "multi_module"])
    def test_strict_mode_raises_before_reading_the_grid(self, mocker, passed, is_expert):
        read = self._forbid_global_grid(mocker)
        pg_collection = None
        if passed != "nothing":
            pg_collection = ProcessGroupCollection(tp=mocker.Mock(spec=dist.ProcessGroup))
        if passed == "multi_module":
            pg_collection = MultiModuleProcessGroupCollection(
                module_pgs={"llm": pg_collection}, language_model_module_name="llm"
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            with pytest.raises(ProcessGroupFallbackWarning, match="Owner was called without"):
                resolve_gtp_remat_group(pg_collection, is_expert, owner="Owner")
        read.assert_not_called()

    @pytest.mark.parametrize("is_expert", [False, True])
    @pytest.mark.parametrize("gtp_on", [False, True])
    def test_a_collection_with_the_field_neither_warns_nor_reads(self, mocker, is_expert, gtp_on):
        # A collection sets the field to None when the axis is off.
        read = self._forbid_global_grid(mocker)
        group = mocker.Mock(spec=dist.ProcessGroup) if gtp_on else None
        pgs = ProcessGroupCollection(gtp_remat=group, expt_gtp_remat=group)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_gtp_remat_group(pgs, is_expert, owner="Owner") is group
        read.assert_not_called()

    def test_a_rank_outside_the_language_model_neither_warns_nor_reads(self, mocker):
        read = self._forbid_global_grid(mocker)
        wrapper = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": ProcessGroupCollection()}, language_model_module_name=None
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_gtp_remat_group(wrapper, is_expert=False, owner="Owner") is None
        read.assert_not_called()

    @pytest.mark.parametrize("size", [0, 1, 2])
    def test_owner_none_leaves_the_warning_to_the_caller(self, mocker, size):
        group = self._global_gtp_groups(mocker, size)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert resolve_gtp_remat_group(None, is_expert=False, owner=None) is group

    def test_ddp_without_a_collection_warns(self, mocker):
        sentinel = mocker.Mock(spec=dist.ProcessGroup)
        mocker.patch.multiple(
            parallel_state,
            get_data_parallel_group=mocker.Mock(return_value=sentinel),
            get_expert_data_parallel_group=mocker.Mock(return_value=sentinel),
            get_tensor_model_parallel_group=mocker.Mock(return_value=sentinel),
            get_gtp_weight_remat_group=mocker.Mock(return_value=None),
            get_pipeline_model_parallel_group=mocker.Mock(return_value=sentinel),
            get_expert_model_parallel_group=mocker.Mock(return_value=sentinel),
            get_expert_gtp_weight_remat_group=mocker.Mock(return_value=None),
        )
        ddp_config = mocker.Mock(
            num_distributed_optimizer_instances=1, use_distributed_optimizer=False
        )
        with pytest.warns(
            FutureWarning, match="DistributedDataParallel was called without `pg_collection`"
        ):
            groups = ProcessGroupCollection.setup_process_groups_for_ddp(
                None, mocker.Mock(), ddp_config
            )
        assert groups["dp_group"] is sentinel

    # Every parallel_state accessor that setup_process_groups_for_optimizer reads when it falls
    # back to the global grid.
    _OPTIMIZER_GRID_ACCESSORS = (
        "get_data_parallel_group",
        "get_data_parallel_group_gloo",
        "get_expert_data_parallel_group",
        "get_expert_data_parallel_group_gloo",
        "get_expert_gtp_weight_remat_group",
        "get_expert_tensor_model_pipeline_parallel_group",
        "get_gtp_weight_remat_group",
        "get_inter_distributed_optimizer_instance_group",
        "get_intra_distributed_optimizer_instance_group",
        "get_model_parallel_group",
    )

    @classmethod
    def _global_optimizer_groups(cls, mocker):
        """Make the global grid answer one sentinel group, with GTP off, to the optimizer helper."""
        sentinel = mocker.Mock(spec=dist.ProcessGroup)
        accessors = {
            name: mocker.Mock(return_value=sentinel) for name in cls._OPTIMIZER_GRID_ACCESSORS
        }
        # With GTP off, the fallback also reads the Gloo groups.
        accessors["get_gtp_weight_remat_group"].return_value = None
        accessors["get_expert_gtp_weight_remat_group"].return_value = None
        mocker.patch.multiple(parallel_state, **accessors)
        return sentinel, accessors

    @staticmethod
    def _model_chunk(mocker, num_distributed_optimizer_instances):
        ddp_config = mocker.Mock(
            num_distributed_optimizer_instances=num_distributed_optimizer_instances,
            use_distributed_optimizer=True,
        )
        return mocker.Mock(ddp_config=ddp_config)

    def test_optimizer_helper_without_a_collection_warns_once(self, mocker):
        sentinel, accessors = self._global_optimizer_groups(mocker)
        # Two optimizer instances make the fallback read the inter-instance group too.
        chunks = [self._model_chunk(mocker, num_distributed_optimizer_instances=2)]
        with pytest.warns(
            FutureWarning,
            match=r"ProcessGroupCollection\.setup_process_groups_for_optimizer was called "
            r"without `pg_collection`",
        ):
            groups = ProcessGroupCollection.setup_process_groups_for_optimizer(None, chunks)
        assert groups["dp_group"] is sentinel
        assert groups["inter_dist_opt_group"] is sentinel
        assert groups["intra_dp_cp_group_gloo"] is sentinel
        for accessor in accessors.values():
            accessor.assert_called()

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ProcessGroupCollection.setup_process_groups_for_optimizer(None, chunks)

    def test_strict_optimizer_helper_raises_before_reading_the_grid(self, mocker):
        _, accessors = self._global_optimizer_groups(mocker)
        chunks = [self._model_chunk(mocker, num_distributed_optimizer_instances=2)]
        with warnings.catch_warnings():
            warnings.simplefilter("error", process_groups_config.ProcessGroupFallbackWarning)
            with pytest.raises(process_groups_config.ProcessGroupFallbackWarning):
                ProcessGroupCollection.setup_process_groups_for_optimizer(None, chunks)
        for accessor in accessors.values():
            accessor.assert_not_called()

    def test_optimizer_helper_with_a_collection_is_silent(self, mocker):
        _, accessors = self._global_optimizer_groups(mocker)
        group = mocker.Mock(spec=dist.ProcessGroup)
        pgs = ProcessGroupCollection(
            dp=group, dp_cp=group, expt_dp=group, mp=group, tp_ep_pp=group, intra_dist_opt=group
        )
        chunks = [self._model_chunk(mocker, num_distributed_optimizer_instances=1)]
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            groups = ProcessGroupCollection.setup_process_groups_for_optimizer(
                pgs, chunks, use_gloo_process_groups=False
            )
        assert groups["dp_group"] is group
        for accessor in accessors.values():
            accessor.assert_not_called()


class TestRequireProcessGroups:
    """require_process_groups tells a field the caller never set from an explicit None.

    These run without torch.distributed: the collections only need to carry sentinels.
    """

    def test_returns_the_groups_in_the_requested_order(self, mocker):
        tp = mocker.Mock(spec=dist.ProcessGroup)
        cp = mocker.Mock(spec=dist.ProcessGroup)
        groups = require_process_groups(
            ProcessGroupCollection(tp=tp, cp=cp), ("cp", "tp"), owner="Owner"
        )
        assert len(groups) == 2
        assert groups[0] is cp
        assert groups[1] is tp

    def test_explicit_none_is_an_axis_that_is_off(self):
        pgs = ProcessGroupCollection(gtp_remat=None, embd=None)
        assert require_process_groups(pgs, ("gtp_remat", "embd"), owner="Owner") == (None, None)

    def test_absent_fields_raise_naming_the_owner_and_every_field(self, mocker):
        pgs = ProcessGroupCollection(tp=mocker.Mock(spec=dist.ProcessGroup))
        # Through __getattr__, a field that was never set reads like an axis that is off.
        assert pgs.cp is None
        with pytest.raises(ValueError, match=r"^Owner requires pg_collection to set cp, pp\."):
            require_process_groups(pgs, ("tp", "cp", "pp"), owner="Owner")

    def test_rejects_the_non_member_sentinel(self):
        pgs = ProcessGroupCollection(embd=dist.GroupMember.NON_GROUP_MEMBER)
        with pytest.raises(ValueError, match="Owner: pg_collection sets embd to NON_GROUP_MEMBER"):
            require_process_groups(pgs, ("embd",), owner="Owner")

    def test_rejects_a_missing_collection(self):
        with pytest.raises(ValueError, match="Owner requires a ProcessGroupCollection, got None"):
            require_process_groups(None, ("tp",), owner="Owner")

    def test_rejects_a_multi_module_collection(self, mocker):
        # The modules have different groups, so the caller must pick the module's collection.
        llm = ProcessGroupCollection(tp=mocker.Mock(spec=dist.ProcessGroup))
        wrapper = MultiModuleProcessGroupCollection(
            module_pgs={"encoder": ProcessGroupCollection(), "llm": llm},
            language_model_module_name="llm",
        )
        with pytest.raises(TypeError, match=r"^Owner requires the ProcessGroupCollection of one"):
            require_process_groups(wrapper, ("tp",), owner="Owner")


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
