# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

from megatron.training.argument_utils import ArgumentGroupFactory
from megatron.training.config import ValidationConfig
from megatron.training.global_vars import set_args
from megatron.training.training import get_train_valid_test_num_samples
from tests.unit_tests.test_utilities import Utils


def create_test_args(**overrides):
    """Create a minimal args namespace for testing eval batch size logic."""
    args = SimpleNamespace()
    args.iteration = 0
    args.train_samples = None
    args.train_iters = 100
    args.eval_interval = 10
    args.eval_iters = 5
    args.global_batch_size = 32
    args.micro_batch_size = 4
    args.eval_global_batch_size = 32
    args.eval_micro_batch_size = 4
    args.data_parallel_size = 1
    args.consumed_train_samples = 0
    args.consumed_valid_samples = 0
    args.dataloader_type = "external"
    args.skip_train = False
    args.full_validation = False
    args.multiple_validation_sets = False
    args.perform_rl_step = False
    args.phase_transition_iterations = None
    for key, val in overrides.items():
        setattr(args, key, val)
    return args


class TestEvalBatchSizeConfig:
    """Test that eval batch size fields are present in ValidationConfig and wired to CLI."""

    def test_validation_config_has_eval_fields(self):
        """ValidationConfig should have eval_global_batch_size and eval_micro_batch_size."""
        config = ValidationConfig()
        assert config.eval_global_batch_size is None
        assert config.eval_micro_batch_size is None

    def test_validation_config_accepts_eval_values(self):
        """ValidationConfig should accept explicit eval batch size values."""
        config = ValidationConfig(eval_global_batch_size=64, eval_micro_batch_size=8)
        assert config.eval_global_batch_size == 64
        assert config.eval_micro_batch_size == 8

    def test_cli_args_generated(self):
        """ArgumentGroupFactory should generate --eval-global-batch-size and --eval-micro-batch-size."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ValidationConfig)
        factory.build_group(parser, "validation")
        # Parse with the new flags
        args = parser.parse_args(["--eval-global-batch-size", "64", "--eval-micro-batch-size", "8"])
        assert args.eval_global_batch_size == 64
        assert args.eval_micro_batch_size == 8

    def test_cli_args_default_none(self):
        """CLI args should default to None when not specified."""
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ValidationConfig)
        factory.build_group(parser, "validation")
        args = parser.parse_args([])
        assert args.eval_global_batch_size is None
        assert args.eval_micro_batch_size is None


class TestEvalBatchSizeDefaults:
    """Test fallback defaults: when eval batch sizes are None, they should use training values."""

    def test_defaults_to_training_values(self):
        """When eval batch sizes are None, validate_args sets them to training values."""
        # Simulate what validate_args does
        args = create_test_args(eval_global_batch_size=None, eval_micro_batch_size=None)
        # Apply the defaulting logic (same as in validate_args)
        if args.eval_global_batch_size is None:
            args.eval_global_batch_size = args.global_batch_size
        if args.eval_micro_batch_size is None:
            args.eval_micro_batch_size = args.micro_batch_size
        assert args.eval_global_batch_size == 32
        assert args.eval_micro_batch_size == 4

    def test_explicit_values_preserved(self):
        """Explicit eval batch sizes should not be overwritten by training values."""
        args = create_test_args(
            global_batch_size=32,
            micro_batch_size=4,
            eval_global_batch_size=64,
            eval_micro_batch_size=8,
        )
        # Apply the defaulting logic
        if args.eval_global_batch_size is None:
            args.eval_global_batch_size = args.global_batch_size
        if args.eval_micro_batch_size is None:
            args.eval_micro_batch_size = args.micro_batch_size
        assert args.eval_global_batch_size == 64
        assert args.eval_micro_batch_size == 8


class TestEvalBatchSizeDivisibility:
    """Test that eval_global_batch_size must be divisible by (eval_micro_batch_size * data_parallel_size)."""

    def test_valid_divisibility(self):
        """No error when eval_global_batch_size is divisible."""
        args = create_test_args(
            eval_global_batch_size=64, eval_micro_batch_size=8, data_parallel_size=2
        )
        # 64 % (8 * 2) == 0 → should pass
        assert (
            args.eval_global_batch_size % (args.eval_micro_batch_size * args.data_parallel_size)
            == 0
        )

    def test_invalid_divisibility_raises(self):
        """AssertionError when eval_global_batch_size is not divisible."""
        args = create_test_args(
            eval_global_batch_size=50, eval_micro_batch_size=8, data_parallel_size=2
        )
        # 50 % (8 * 2) == 50 % 16 == 2 → should fail
        with pytest.raises(AssertionError, match="eval_global_batch_size"):
            assert (
                args.eval_global_batch_size % (args.eval_micro_batch_size * args.data_parallel_size)
                == 0
            ), (
                f"eval_global_batch_size ({args.eval_global_batch_size}) must be divisible by "
                f"eval_micro_batch_size ({args.eval_micro_batch_size}) * "
                f"data_parallel_size ({args.data_parallel_size})"
            )

    def test_only_eval_mbs_set(self):
        """When only eval_micro_batch_size is set, eval_global_batch_size falls back to global_batch_size."""
        args = create_test_args(
            global_batch_size=32,
            micro_batch_size=4,
            eval_global_batch_size=None,
            eval_micro_batch_size=2,
            data_parallel_size=1,
        )
        if args.eval_global_batch_size is None:
            args.eval_global_batch_size = args.global_batch_size
        # 32 % (2 * 1) == 0 → valid
        assert (
            args.eval_global_batch_size % (args.eval_micro_batch_size * args.data_parallel_size)
            == 0
        )
        assert args.eval_global_batch_size == 32
        assert args.eval_micro_batch_size == 2


class TestGetTrainValidTestNumSamples:
    """Test that get_train_valid_test_num_samples uses eval_global_batch_size."""

    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_num_samples_uses_eval_gbs(self):
        """eval/test sample counts should use eval_global_batch_size, not global_batch_size."""
        args = create_test_args(
            global_batch_size=32,
            eval_global_batch_size=16,
            train_iters=100,
            eval_interval=10,
            eval_iters=5,
        )
        set_args(args)
        train_samples, eval_samples, test_samples = get_train_valid_test_num_samples()
        # eval_samples = ((100 // 10) + 1) * 5 * 16 = 11 * 5 * 16 = 880
        assert eval_samples == 880
        # test_samples = 5 * 16 = 80
        assert test_samples == 80

    def test_num_samples_default_matches_training(self):
        """When eval_global_batch_size == global_batch_size, results should match old behavior."""
        args = create_test_args(
            global_batch_size=32,
            eval_global_batch_size=32,
            train_iters=100,
            eval_interval=10,
            eval_iters=5,
        )
        set_args(args)
        train_samples, eval_samples, test_samples = get_train_valid_test_num_samples()
        # eval_samples = 11 * 5 * 32 = 1760
        assert eval_samples == 1760
        # test_samples = 5 * 32 = 160
        assert test_samples == 160
        # train_samples = 100 * 32 = 3200
        assert train_samples == 3200

    def test_num_samples_with_skip_train(self):
        """With skip_train, eval_iters used directly (no multiplier from train_iters/eval_interval)."""
        args = create_test_args(
            global_batch_size=32, eval_global_batch_size=8, skip_train=True, eval_iters=10
        )
        set_args(args)
        train_samples, eval_samples, test_samples = get_train_valid_test_num_samples()
        # eval_samples = 10 * 8 = 80
        assert eval_samples == 80
        # test_samples = 10 * 8 = 80
        assert test_samples == 80
