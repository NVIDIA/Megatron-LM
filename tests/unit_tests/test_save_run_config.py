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
from megatron.training.async_utils import maybe_finalize_async_save
from megatron.training.checkpointing import save_checkpoint
from megatron.training.config import PretrainConfigContainer
from megatron.training.global_vars import get_run_config, set_args
from megatron.training.models.gpt import GPTModelConfig
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
        from megatron.training.config import LoggerConfig, RNGConfig

        self._data = data
        self.logger = LoggerConfig()
        self.rng = RNGConfig()

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


@pytest.mark.parametrize(
    "ckpt_format,async_save,conversion",
    [
        ("torch", False, True),
        ("torch_dist", False, True),
        ("torch_dist", True, True),
        ("torch_dist", False, False),
    ],
)
def test_conversion_run_config_records_output_settings(
    init_model_parallel,
    create_args,
    tmp_path_dist_ckpt,
    run_config,
    ckpt_format,
    async_save,
    conversion,
):
    """Conversion saves record output settings without changing the active config."""
    args = create_args
    args.ckpt_format = ckpt_format
    args.ckpt_convert_format = ckpt_format if conversion else None
    args.use_dist_ckpt = ckpt_format != "torch"
    args.use_distributed_optimizer = args.use_dist_ckpt
    args.async_save = async_save
    args.save_tokenizer_assets = False
    args.ckpt_assume_constant_structure = False
    args.ckpt_load_validate_sharding_integrity = True

    source_checkpoint = run_config.checkpoint
    source_checkpoint.ckpt_format = "torch_dist" if ckpt_format == "torch" else "torch"
    source_checkpoint.save = "source-checkpoints"
    source_checkpoint.verify_integrity = ckpt_format == "torch"
    source_checkpoint.async_save = async_save
    args.verify_integrity = source_checkpoint.verify_integrity
    run_config.model = GPTModelConfig(
        transformer=TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1),
        vocab_size=256,
    )
    source_config_dict = run_config.to_dict()
    model = MockModel(TransformerConfig(num_layers=1, kv_channels=1))

    with mock.patch("megatron.training.async_utils._async_calls_queue", None):
        with TempNamedDir(tmp_path_dist_ckpt / "test_conversion_run_config", sync=True) as save_dir:
            args.save = str(save_dir)
            set_args(args)
            run_config_path = save_dir / "iter_0000123" / "run_config.yaml"

            save_checkpoint(123, [model], None, None, 456)

            if async_save:
                assert not run_config_path.exists()
                # Finalization must use this save's output settings.
                args.ckpt_convert_format = None
                args.ckpt_format = "torch"
                args.save = "later-checkpoints"
                maybe_finalize_async_save(blocking=True)

            saved_config = read_run_config(str(run_config_path))
            assert saved_config["checkpoint"]["ckpt_format"] == (
                ckpt_format if conversion else source_checkpoint.ckpt_format
            )
            assert saved_config["checkpoint"]["save"] == (
                str(save_dir) if conversion else source_checkpoint.save
            )
            assert saved_config["model"]["vocab_size"] == 256
            assert saved_config["profiling"] == source_config_dict["profiling"]
            restored_config = PretrainConfigContainer.from_yaml(str(run_config_path))
            assert (
                restored_config.checkpoint.ckpt_format == saved_config["checkpoint"]["ckpt_format"]
            )
            assert restored_config.checkpoint.save == saved_config["checkpoint"]["save"]
            assert restored_config.checkpoint.verify_integrity == (
                False
                if conversion and ckpt_format != "torch_dist"
                else source_checkpoint.verify_integrity
            )
            assert get_run_config() is run_config
            assert run_config.checkpoint is source_checkpoint
            assert run_config.to_dict() == source_config_dict

            with open(save_dir / "latest_checkpointed_iteration.txt") as f:
                assert int(f.read()) == 123
            if ckpt_format == "torch":
                assert (save_dir / "iter_0000123" / "mp_rank_00" / "model_optim_rng.pt").exists()
            else:
                assert (save_dir / "iter_0000123" / ".metadata").exists()


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
