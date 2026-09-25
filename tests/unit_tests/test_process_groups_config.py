# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch.distributed as dist

from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
    resolve_gtp_remat_group,
)
from tests.unit_tests.test_utilities import Utils


class TestProcessGroupsConfig:
    """Simple tests for process group dataclasses."""

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
