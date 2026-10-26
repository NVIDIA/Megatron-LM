# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for run_config.yaml being written as part of checkpoint save."""

import os
from dataclasses import replace

import pytest

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_torch_min_version
from megatron.training.checkpointing import save_checkpoint
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
from tests.unit_tests.test_utilities import sync_checkpoint_config


def _prepare(args, run_config, save_dir, ckpt_format):
    """Register the arguments and the checkpoint section a save reads."""
    args.ckpt_format = ckpt_format
    args.use_distributed_optimizer = ckpt_format != "torch_dcp"
    args.use_dist_ckpt = ckpt_format != "torch"
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True
    args.save = str(save_dir)
    args.save_tokenizer_assets = False
    set_args(args)
    run_config.train.global_batch_size = 8
    return sync_checkpoint_config(args)


def _model_and_states(ckpt_format):
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
    return model, optimizer, opt_param_scheduler


def _assert_saved_config(ckpt_dir, run_config, checkpoint):
    run_config_filename = get_checkpoint_run_config_filename(str(ckpt_dir))
    assert os.path.exists(run_config_filename)
    loaded_config = read_run_config(run_config_filename)
    assert loaded_config["train"]["global_batch_size"] == 8
    # The saved checkpoint section describes the policy of this save.
    assert loaded_config["checkpoint"] == run_config._convert_value_to_dict(checkpoint)
    return loaded_config


def test_save_checkpoint_writes_run_config(
    init_model_parallel, create_args, tmp_path_dist_ckpt, run_config
):
    """``save_checkpoint`` writes run_config.yaml for torch_dist checkpoints, readable back."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dist checkpoint format requires torch >= 2.4.0")

    args = create_args
    model, optimizer, opt_param_scheduler = _model_and_states("torch_dist")

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_save_checkpoint_run_config", sync=True
    ) as save_dir:
        checkpoint = _prepare(args, run_config, save_dir, "torch_dist")
        save_checkpoint(123, [model], optimizer, opt_param_scheduler, 456)
        loaded = _assert_saved_config(save_dir / "iter_0000123", run_config, checkpoint)
        assert loaded["checkpoint"]["ckpt_format"] == "torch_dist"
        assert loaded["checkpoint"]["save"] == str(save_dir)


def test_save_checkpoint_writes_run_config_release(
    init_model_parallel, create_args, tmp_path_dist_ckpt, run_config
):
    """``save_checkpoint`` writes run_config.yaml into the ``release`` directory when
    ``release=True``, readable back."""
    if not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dist checkpoint format requires torch >= 2.4.0")

    args = create_args
    model, optimizer, opt_param_scheduler = _model_and_states("torch_dist")

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_save_checkpoint_run_config_release", sync=True
    ) as save_dir:
        checkpoint = _prepare(args, run_config, save_dir, "torch_dist")
        save_checkpoint(123, [model], optimizer, opt_param_scheduler, 456, release=True)
        _assert_saved_config(save_dir / "release", run_config, checkpoint)
        # A release checkpoint must not be written under an iter_XXXXXXX directory.
        assert not os.path.exists(save_dir / "iter_0000123")


@pytest.mark.parametrize("ckpt_format", ["torch", "torch_dcp", "fsdp_dtensor"])
def test_save_checkpoint_writes_run_config_other_formats(
    init_model_parallel, create_args, tmp_path_dist_ckpt, ckpt_format, run_config
):
    """``save_checkpoint`` writes run_config.yaml for non-torch_dist checkpoint formats."""
    if ckpt_format == "torch_dcp" and not is_torch_min_version("2.4.0"):
        pytest.skip("torch_dcp requires torch >= 2.4.0")

    args = create_args
    model, optimizer, opt_param_scheduler = _model_and_states(ckpt_format)

    with TempNamedDir(
        tmp_path_dist_ckpt / f"test_save_checkpoint_run_config_{ckpt_format}", sync=True
    ) as save_dir:
        checkpoint = _prepare(args, run_config, save_dir, ckpt_format)
        save_checkpoint(123, [model], optimizer, opt_param_scheduler, 456)
        _assert_saved_config(save_dir / "iter_0000123", run_config, checkpoint)


def test_save_with_an_override_records_the_effective_policy(
    init_model_parallel, create_args, tmp_path_dist_ckpt, run_config
):
    """A save that overrides the registered section (checkpoint conversion) records the format and
    directory it wrote, in the YAML and in the persisted arguments, and leaves the registered
    section and the live arguments untouched."""
    args = create_args
    model, optimizer, opt_param_scheduler = _model_and_states("torch")

    with TempNamedDir(
        tmp_path_dist_ckpt / "test_save_checkpoint_run_config_override", sync=True
    ) as save_dir:
        registered = _prepare(args, run_config, save_dir / "registered", "torch_dist")
        converted_dir = str(save_dir / "converted")
        effective = replace(registered, ckpt_format="torch", save=converted_dir)
        save_checkpoint(123, [model], optimizer, opt_param_scheduler, 456, checkpoint=effective)

        loaded = _assert_saved_config(
            save_dir / "converted" / "iter_0000123", run_config, effective
        )
        assert loaded["checkpoint"]["ckpt_format"] == "torch"
        assert loaded["checkpoint"]["save"] == converted_dir
        assert run_config.checkpoint == registered
        assert args.ckpt_format == "torch_dist" and args.save == str(save_dir / "registered")
        assert not os.path.exists(save_dir / "registered" / "iter_0000123")

        # Resource policy cannot change per save.
        with pytest.raises(ValueError, match="may override only"):
            save_checkpoint(
                124,
                [model],
                optimizer,
                opt_param_scheduler,
                456,
                checkpoint=replace(registered, async_save=True),
            )
