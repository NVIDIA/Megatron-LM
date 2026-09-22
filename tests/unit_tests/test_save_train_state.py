# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for saving/loading ``train_state.pt`` (see megatron/training/state.py,
megatron/training/utils/checkpoint_utils.py and the train-state block in
megatron.training.checkpointing.save_checkpoint)."""

from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training import global_vars
from megatron.training.checkpointing import (
    get_checkpoint_tracker_filename,
    read_metadata,
    save_checkpoint,
)
from megatron.training.global_vars import set_args
from megatron.training.state import TrainState
from megatron.training.utils.checkpoint_utils import (
    TRAIN_STATE_FILE,
    get_checkpoint_train_state_filename,
    read_train_state,
)

from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

# All keys `TrainState.state_dict()` / `TrainState.load_state_dict()` operate on.
TRAIN_STATE_KEYS = {
    "step",
    "consumed_train_samples",
    "skipped_train_samples",
    "consumed_valid_samples",
    "floating_point_operations_so_far",
    "do_train",
    "do_valid",
    "do_test",
}


class MockModel(MegatronModule):
    """Dummy megatron model."""

    def __init__(self, config):
        super().__init__(config=config)
        self.l = torch.nn.Linear(1, 2)
        torch.nn.init.ones_(self.l.weight)
        torch.nn.init.zeros_(self.l.bias)

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        return self.state_dict()


class MockState:
    """Dummy stand-in for the optimizer / opt_param_scheduler passed to save_checkpoint."""

    def __init__(self, state_dict):
        self._state_dict = state_dict
        self.is_stub_optimizer = False
        self.param_groups = []

    def state_dict(self, is_loading=False):
        return self._state_dict

    def load_state_dict(self, state_dict):
        self._state_dict = state_dict

    def save_parameter_state(self, *args, **kwargs):
        pass

    def load_parameter_state(self, *args, **kwargs):
        pass

    def sharded_state_dict(self, *args, metadata: Optional[dict] = None, **kwargs):
        return self.state_dict()


@pytest.fixture
def create_args():
    """Setup dummy args required by save_checkpoint()."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.async_strategy = "nvrx"
    args.data_parallel_random_init = False
    args.no_save_optim = False
    args.no_save_rng = False
    args.no_load_optim = False
    args.no_load_rng = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False
    args.auto_detect_ckpt_format = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.override_opt_param_scheduler = False
    args.swiglu = True
    args.num_experts = 1
    args.verify_integrity = False
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = False

    yield args


@pytest.fixture
def init_model_parallel():
    """Init torch distributed."""
    Utils.initialize_model_parallel(1, 1)
    init_num_microbatches_calculator(
        rank=0, global_batch_size=1, micro_batch_size=1, data_parallel_size=1
    )
    model_parallel_cuda_manual_seed(123)
    yield  # Run the actual test.
    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


@pytest.fixture
def populated_train_state():
    """Initialize the global TrainState with distinctive, non-default values."""
    global_vars._GLOBAL_TRAIN_STATE = None
    global_vars._set_train_state()
    train_state = global_vars.get_train_state()
    train_state.iteration = 123
    train_state.consumed_train_samples = 111
    train_state.skipped_train_samples = 22
    train_state.consumed_valid_samples = 33
    train_state.num_floating_point_operations_so_far = 999
    train_state.do_train = True
    train_state.do_valid = False
    train_state.do_test = True
    yield train_state
    global_vars._GLOBAL_TRAIN_STATE = None


def _save_a_checkpoint(
    args,
    tmp_dir,
    ckpt_format,
    iteration=123,
    num_floating_point_operations_so_far=456,
    release=False,
):
    """Run save_checkpoint() with a minimal model/optimizer/scheduler setup."""
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"
    args.save = tmp_dir
    args.save_tokenizer_assets = False
    set_args(args)

    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    # Mirrors the shape of a real optimizer state dict (["state"] is accessed
    # unconditionally by handle_swiglu_in_state_dict() on the fsdp_dtensor save path).
    optimizer = MockState({"state": {}, "param_groups": []})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})

    save_checkpoint(
        iteration,
        [model],
        optimizer,
        opt_param_scheduler,
        num_floating_point_operations_so_far,
        release=release,
    )


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dist", "torch_dcp", "fsdp_dtensor"])
def test_train_state_files_saved_for_different_ckpt_types(
    init_model_parallel, create_args, populated_train_state, tmp_path_dist_ckpt, ckpt_format
):
    """train_state.pt (per-checkpoint) and latest_train_state.pt (global tracker) are written
    alongside the model/optimizer checkpoint for every supported ckpt format."""
    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args = create_args
    iteration = 123
    num_fp_ops = 456

    with TempNamedDir(
        tmp_path_dist_ckpt / f"test_train_state_files_{ckpt_format}", sync=True
    ) as save_dir:
        _save_a_checkpoint(
            args,
            save_dir,
            ckpt_format,
            iteration=iteration,
            num_floating_point_operations_so_far=num_fp_ops,
        )

        ckpt_dir = save_dir / "iter_0000123"
        local_train_state_path = ckpt_dir / TRAIN_STATE_FILE
        global_train_state_path = save_dir / f"latest_{TRAIN_STATE_FILE}"

        assert local_train_state_path.exists()
        assert global_train_state_path.exists()
        assert get_checkpoint_train_state_filename(str(ckpt_dir)) == str(local_train_state_path)

        local_state_dict = torch.load(local_train_state_path, weights_only=True)
        global_state_dict = torch.load(global_train_state_path, weights_only=True)

        # The global (latest) file is a copy of the per-checkpoint file.
        assert local_state_dict.keys() == global_state_dict.keys()
        for key in local_state_dict:
            assert torch.equal(local_state_dict[key], global_state_dict[key])


def test_train_state_values_saved_correctly(
    init_model_parallel, create_args, populated_train_state, tmp_path_dist_ckpt
):
    """Values written to train_state.pt reflect the live TrainState, with
    floating_point_operations_so_far overridden by save_checkpoint's own running total."""
    args = create_args
    iteration = 123
    num_fp_ops = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_train_state_values", sync=True) as save_dir:
        _save_a_checkpoint(
            args,
            save_dir,
            "torch",
            iteration=iteration,
            num_floating_point_operations_so_far=num_fp_ops,
        )

        global_train_state_path = save_dir / f"latest_{TRAIN_STATE_FILE}"
        state_dict = torch.load(global_train_state_path, weights_only=True)

        assert set(state_dict.keys()) == TRAIN_STATE_KEYS
        assert state_dict["step"].item() == populated_train_state.iteration
        assert (
            state_dict["consumed_train_samples"].item()
            == populated_train_state.consumed_train_samples
        )
        assert (
            state_dict["skipped_train_samples"].item()
            == populated_train_state.skipped_train_samples
        )
        assert (
            state_dict["consumed_valid_samples"].item()
            == populated_train_state.consumed_valid_samples
        )
        assert state_dict["do_train"].item() == populated_train_state.do_train
        assert state_dict["do_valid"].item() == populated_train_state.do_valid
        assert state_dict["do_test"].item() == populated_train_state.do_test

        # save_checkpoint() overwrites this field with its own running FLOPs counter,
        # which need not match the (independently tracked) TrainState field.
        assert state_dict["floating_point_operations_so_far"].item() == num_fp_ops
        assert num_fp_ops != populated_train_state.num_floating_point_operations_so_far


def test_read_train_state_returns_correct_train_state(
    init_model_parallel, create_args, populated_train_state, tmp_path_dist_ckpt
):
    """read_train_state() round-trips a saved train_state.pt back into an equivalent TrainState."""
    args = create_args
    iteration = 123
    num_fp_ops = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_read_train_state", sync=True) as save_dir:
        _save_a_checkpoint(
            args,
            save_dir,
            "torch",
            iteration=iteration,
            num_floating_point_operations_so_far=num_fp_ops,
        )

        global_train_state_path = save_dir / f"latest_{TRAIN_STATE_FILE}"
        loaded_state = read_train_state(str(global_train_state_path))

        assert isinstance(loaded_state, TrainState)
        assert loaded_state.iteration == populated_train_state.iteration
        assert loaded_state.consumed_train_samples == populated_train_state.consumed_train_samples
        assert loaded_state.skipped_train_samples == populated_train_state.skipped_train_samples
        assert loaded_state.consumed_valid_samples == populated_train_state.consumed_valid_samples
        assert loaded_state.num_floating_point_operations_so_far == num_fp_ops
        assert loaded_state.do_train == populated_train_state.do_train
        assert loaded_state.do_valid == populated_train_state.do_valid
        assert loaded_state.do_test == populated_train_state.do_test


def test_read_train_state_missing_file_raises(tmp_path):
    """read_train_state() surfaces a clear error for a missing/corrupt file."""
    missing_path = tmp_path / "does_not_exist_train_state.pt"
    with pytest.raises(RuntimeError):
        read_train_state(str(missing_path))


def test_save_checkpoint_release_writes_release_tracker_and_train_state(
    init_model_parallel, create_args, populated_train_state, tmp_path_dist_ckpt
):
    """release=True checkpoints: the Megatron-LM tracker keeps the literal 'release' marker
    (not the numeric iteration), and train_state.pt is written under the 'release/' checkpoint
    directory, matching where the model/optimizer weights are saved."""
    args = create_args
    iteration = 123
    num_fp_ops = 456

    with TempNamedDir(tmp_path_dist_ckpt / "test_save_checkpoint_release", sync=True) as save_dir:
        _save_a_checkpoint(
            args,
            save_dir,
            "torch",
            iteration=iteration,
            num_floating_point_operations_so_far=num_fp_ops,
            release=True,
        )

        tracker_filename = get_checkpoint_tracker_filename(save_dir)
        with open(tracker_filename, "r") as f:
            assert f.read().strip() == "release"

        loaded_iteration, is_release = read_metadata(tracker_filename)
        assert is_release
        assert loaded_iteration == 0

        release_ckpt_dir = save_dir / "release"
        local_train_state_path = release_ckpt_dir / TRAIN_STATE_FILE
        global_train_state_path = save_dir / f"latest_{TRAIN_STATE_FILE}"

        assert local_train_state_path.exists()
        assert global_train_state_path.exists()
        # The train_state.pt for a release checkpoint must not be dropped next to a
        # regular iter_XXXXXXX checkpoint directory.
        assert not (save_dir / "iter_0000123" / TRAIN_STATE_FILE).exists()

        state_dict = torch.load(global_train_state_path, weights_only=True)
        assert set(state_dict.keys()) == TRAIN_STATE_KEYS
        assert state_dict["step"].item() == populated_train_state.iteration
        assert state_dict["floating_point_operations_so_far"].item() == num_fp_ops

        loaded_state = read_train_state(str(global_train_state_path))
        assert loaded_state.consumed_train_samples == populated_train_state.consumed_train_samples
