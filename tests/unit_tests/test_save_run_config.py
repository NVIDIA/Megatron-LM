# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for run_config.yaml being written as part of checkpoint save."""

import os
from unittest import mock

import pytest
import yaml

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.checkpointing import save_checkpoint
from megatron.training.config import ProfilingConfig
from megatron.training.global_vars import set_args
from megatron.training.utils.checkpoint_utils import (
    get_checkpoint_run_config_filename,
    read_run_config,
)
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_checkpointing import (
    MockModel,
    MockState,
    create_args,
    init_model_parallel,
)


class MockFullConfig:
    """Minimal stand-in for ``PretrainConfigContainer`` exposing ``to_yaml``."""

    def __init__(self, data):
        self._data = data
        self.profiling = ProfilingConfig()

    def to_yaml(self, yaml_path):
        with open(yaml_path, "w") as f:
            yaml.safe_dump(self._data, f)


def test_save_checkpoint_writes_run_config(init_model_parallel, create_args, tmp_path_dist_ckpt):
    """``save_checkpoint`` writes run_config.yaml for torch_dist checkpoints, readable back."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dist checkpoint format requires torch >= 2.4.0")

    args = create_args
    args.ckpt_format = "torch_dist"
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    run_config_data = {"_target_": "dummy.Config", "train": {"global_batch_size": 8}}

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_save_checkpoint_run_config", sync=True
    ) as save_dir:
        args.save = save_dir
        args.save_tokenizer_assets = False
        set_args(args)

        with mock.patch(
            "megatron.training.checkpointing.get_run_config",
            return_value=MockFullConfig(run_config_data),
        ):
            save_checkpoint(
                iteration,
                [model],
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

        ckpt_dir = save_dir / "iter_0000123"
        run_config_filename = get_checkpoint_run_config_filename(str(ckpt_dir))

        assert os.path.exists(run_config_filename)

        loaded_config = read_run_config(run_config_filename)
        assert loaded_config == run_config_data


def test_save_checkpoint_writes_run_config_release(
    init_model_parallel, create_args, tmp_path_dist_ckpt
):
    """``save_checkpoint`` writes run_config.yaml into the ``release`` directory when
    ``release=True``, readable back."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dist checkpoint format requires torch >= 2.4.0")

    args = create_args
    args.ckpt_format = "torch_dist"
    args.use_distributed_optimizer = True
    args.use_dist_ckpt = True
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    run_config_data = {"_target_": "dummy.Config", "train": {"global_batch_size": 8}}

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_save_checkpoint_run_config_release", sync=True
    ) as save_dir:
        args.save = save_dir
        args.save_tokenizer_assets = False
        set_args(args)

        with mock.patch(
            "megatron.training.checkpointing.get_run_config",
            return_value=MockFullConfig(run_config_data),
        ):
            save_checkpoint(
                iteration,
                [model],
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
                release=True,
            )

        ckpt_dir = save_dir / "release"
        run_config_filename = get_checkpoint_run_config_filename(str(ckpt_dir))

        assert os.path.exists(run_config_filename)
        # A release checkpoint must not be written under an iter_XXXXXXX directory.
        assert not os.path.exists(save_dir / "iter_0000123")

        loaded_config = read_run_config(run_config_filename)
        assert loaded_config == run_config_data


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_save_checkpoint_writes_run_config_other_formats(
    init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format
):
    """``save_checkpoint`` writes run_config.yaml for non-torch_dist checkpoint formats."""
    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args = create_args
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True

    iteration = 123
    config = TransformerConfig(num_layers=1, kv_channels=1)
    model = MockModel(config)
    optimizer = MockState({"optimizer": "optimizer_state"})
    if ckpt_format == "fsdp_dtensor":
        model = FullyShardedDataParallel(
            config=config,
            ddp_config=DistributedDataParallelConfig(
                use_distributed_optimizer=True, use_megatron_fsdp=True
            ),
            module=model,
        )
        optimizer = MockState({"state": {}})
    opt_param_scheduler = MockState({"opt_param_scheduler": "scheduler_state"})
    num_floating_point_operations_so_far = 456

    run_config_data = {"_target_": "dummy.Config", "train": {"global_batch_size": 8}}

    with TempNamedDir(
        tmp_path_dist_ckpt / f"test_save_checkpoint_run_config_{ckpt_format}", sync=True
    ) as save_dir:
        args.save = save_dir
        args.save_tokenizer_assets = False
        set_args(args)

        with mock.patch(
            "megatron.training.checkpointing.get_run_config",
            return_value=MockFullConfig(run_config_data),
        ):
            save_checkpoint(
                iteration,
                [model],
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

        ckpt_dir = save_dir / "iter_0000123"
        run_config_filename = get_checkpoint_run_config_filename(str(ckpt_dir))

        assert os.path.exists(run_config_filename)

        loaded_config = read_run_config(run_config_filename)
        assert loaded_config == run_config_data
