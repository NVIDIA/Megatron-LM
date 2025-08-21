# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# Note: --ckpt-format torch_dist has tests in tests/unit_tests/dist_checkpointing.
import os
from types import SimpleNamespace
from typing import Optional
from unittest import mock

import pytest
import torch
import torch.distributed.checkpoint

from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.checkpointing import (
    CheckpointType,
    _build_sharded_state_dict_metadata,
    _load_base_checkpoint,
    get_checkpoint_tracker_filename,
    load_checkpoint,
    save_checkpoint,
)
from megatron.training.global_vars import set_args
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils


class MockModel(MegatronModule):
    """Dummy megatron model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.l = torch.nn.Linear(1, 2)
        torch.nn.init.ones_(self.l.weight)
        torch.nn.init.zeros_(self.l.bias)
        self._called_metadata = []

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


class MockState:
    def __init__(self, state_dict):
        self._state_dict = state_dict
        self.is_stub_optimizer = False
        self._called_metadata = []

    def state_dict(self, is_loading=False):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def save_parameter_state(self, *args, **kwargs):
        pass

    def load_parameter_state(self, *args, **kwargs):
        pass

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        self._called_metadata.append(metadata)
        return self.state_dict()


def create_checkpoint(load_path, ckpt_format):
    """Setup a dummy checkpoint directory."""
    iteration = 123
    ckpt_dir = load_path / "iter_{:07d}".format(iteration)
    tracker_path = get_checkpoint_tracker_filename(load_path)
    with open(tracker_path, "w") as f:
        f.write(str(iteration))

    state_dict = {"args": "dummy", "iteration": iteration}

    if ckpt_format == "torch":
        # Torch checkpoints use a specific directory structure.
        pt_dir = ckpt_dir / "mp_rank_00"
        pt_dir.mkdir(parents=True)
        torch.save(state_dict, pt_dir / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp" and is_torch_min_version("2.4.0"):
        torch.distributed.checkpoint.save(state_dict, checkpoint_id=ckpt_dir)


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.auto_detect_ckpt_format = False
    args.retro_add_retriever = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None

    yield args


@pytest.fixture
def create_ckpt_load_args(create_args):
    """Setup dummy args allowing checkpoint load."""
    args = create_args
    args.auto_detect_ckpt_format = False
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.num_layers = 1
    args.hidden_size = 2
    args.num_attention_heads = 1
    args.add_position_embedding = False
    args.vocab_file = None
    args.tensor_model_parallel_size = 1
    args.pipeline_model_parallel_size = 1
    args.ckpt_assume_constant_structure = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.dist_ckpt_strictness = 'assume_ok_unexpected'

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(0, None, 1, 1, 1)
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.mark.parametrize("ckpt_format", ["torch_dcp"])
def test_load_base_checkpoint(init_model_parallel, create_args, ckpt_format, tmp_path_dist_ckpt):
    """Test _load_base_checkpoint."""

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    # TempNamedDir uses the same directory for all ranks in a multi-GPU setup. Cleanup is handled.
    with TempNamedDir(tmp_path_dist_ckpt / "test_load_base_checkpoint", sync=True) as load_dir:
        create_checkpoint(load_dir, ckpt_format)
        args = create_args
        args.ckpt_format = ckpt_format

        state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(
            load_dir, args, rank0=True
        )

    assert state_dict["args"] == "dummy"
    assert state_dict["iteration"] == 123

    expected_ckpt_path = None
    if ckpt_format == "torch":
        expected_ckpt_path = str(load_dir / "iter_0000123" / "mp_rank_00" / "model_optim_rng.pt")
    elif ckpt_format == "torch_dcp":
        expected_ckpt_path = str(load_dir / "iter_0000123")

    assert checkpoint_name == expected_ckpt_path
    assert not release

    expected_ckpt_type = None
    if ckpt_format == "torch":
        expected_ckpt_type = CheckpointType.LEGACY
    elif ckpt_format == "torch_dcp":
        expected_ckpt_type = CheckpointType.TORCH_DCP

    assert ckpt_type == expected_ckpt_type


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp"])
def test_save_checkpoint(init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format):
    """Test save_checkpoint."""
    args = create_args
    args.ckpt_format = ckpt_format

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_checkpoint", sync=True) as save_dir:
        args.save = save_dir
        set_args(args)

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        with open(args.save / "latest_checkpointed_iteration.txt", "r") as f:
            assert iteration == int(f.read())

        ckpt_dir = args.save / "iter_0000123"

        expected_ckpt_path = None
        if ckpt_format == "torch":
            expected_ckpt_path = ckpt_dir / "mp_rank_00" / "model_optim_rng.pt"
        elif ckpt_format == "torch_dcp":
            expected_ckpt_path = ckpt_dir / ".metadata"

        assert os.path.exists(expected_ckpt_path)


@pytest.mark.parametrize("ckpt_format", ["torch"])
def test_load_checkpoint(
    init_model_parallel, create_ckpt_load_args, tmp_path_dist_ckpt, ckpt_format
):
    """Test load_checkpoint."""
    args = create_ckpt_load_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"

    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    with TempNamedDir(tmp_path_dist_ckpt / "test_load_checkpoint", sync=True) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_floating_point_operations_so_far = 456

        save_checkpoint(
            iteration, [model], optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Create new model, optimizer, and scheduler instances to load into.
        new_model = MockModel(config)
        new_optimizer = MockState({"optimizer": "dummy1"})
        new_opt_param_scheduler = MockState({"opt_param_scheduler": "dummy2"})

        # Load checkpoint
        loaded_iter, loaded_flops = load_checkpoint(
            [new_model], new_optimizer, new_opt_param_scheduler, strict=True
        )

        assert loaded_iter == iteration
        assert loaded_flops == num_floating_point_operations_so_far

        for k in model.state_dict():
            assert torch.equal(model.state_dict()[k], new_model.state_dict()[k])

        assert new_optimizer.state_dict() == optimizer.state_dict()
        assert new_opt_param_scheduler.state_dict() == opt_param_scheduler.state_dict()


def test_dist_checkpoint_versioning(init_model_parallel, tmp_path_dist_ckpt, create_ckpt_load_args):
    """Test distributed checkpoint versioning."""
    args = create_ckpt_load_args
    args.ckpt_format = 'torch_dist'
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_dist_checkpoint_versioning", sync=True
    ) as ckpt_dir:
        args.load = ckpt_dir
        args.save = ckpt_dir
        set_args(args)

        # Create and save a checkpoint first.
        iteration = 123
        config = TransformerConfig(num_layers=1, kv_channels=1)
        model = MockModel(config)

        optimizer = MockState({"optimizer": "optimizer_state"})
        opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
        num_fp_ops = 456

        base_metadata = _build_sharded_state_dict_metadata(args)
        first_job_mock_metadata = {**base_metadata, 'metadata_A': 42, 'metadata_B_soon_removed': 43}
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=first_job_mock_metadata,
        ):
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)

        second_job_mock_metadata = {
            **base_metadata,
            'metadata_A': 'changed_default_value',
            'metadata_C_new': {'nested': 'val'},
        }
        with mock.patch(
            'megatron.training.checkpointing._build_sharded_state_dict_metadata',
            return_value=second_job_mock_metadata,
        ):
            # Load checkpoint (into the same model, we don't check load correctness here)
            load_checkpoint([model], optimizer, opt_param_scheduler, strict=True)
            assert optimizer._called_metadata[-1] == first_job_mock_metadata

            # Save the checkpoint again to check if the content metadata for the new checkpoint will be new
            save_checkpoint(iteration, [model], optimizer, opt_param_scheduler, num_fp_ops)
            assert optimizer._called_metadata[-1] == second_job_mock_metadata

        assert optimizer._called_metadata == model._called_metadata
        assert optimizer._called_metadata == [
            first_job_mock_metadata,
            first_job_mock_metadata,
            second_job_mock_metadata,
        ]
