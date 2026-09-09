# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from argparse import ArgumentParser
from types import SimpleNamespace

from megatron.training.argument_utils import ArgumentGroupFactory
from megatron.training.config import ValidationConfig


def create_test_args(**overrides):
    """Create a minimal args namespace for testing start_eval_at_iter logic."""
    args = SimpleNamespace()
    args.iteration = 0
    args.eval_interval = 10
    args.eval_iters = 5
    args.do_valid = True
    args.start_eval_at_iter = None
    for key, val in overrides.items():
        setattr(args, key, val)
    return args


def should_run_eval(args, iteration):
    """Replicate the eval condition from megatron/training/training.py."""
    return (
        args.eval_interval
        and iteration % args.eval_interval == 0
        and args.do_valid
        and (args.start_eval_at_iter is None or iteration >= args.start_eval_at_iter)
    )


class TestStartEvalAtIterConfig:
    """Test that start_eval_at_iter is present in ValidationConfig and wired to CLI."""

    def test_validation_config_has_field(self):
        config = ValidationConfig()
        assert config.start_eval_at_iter is None

    def test_validation_config_accepts_value(self):
        config = ValidationConfig(start_eval_at_iter=50)
        assert config.start_eval_at_iter == 50

    def test_cli_arg_generated(self):
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ValidationConfig)
        factory.build_group(parser, "validation")
        args = parser.parse_args(["--start-eval-at-iter", "100"])
        assert args.start_eval_at_iter == 100

    def test_cli_arg_default_none(self):
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(ValidationConfig)
        factory.build_group(parser, "validation")
        args = parser.parse_args([])
        assert args.start_eval_at_iter is None


class TestStartEvalAtIterLogic:
    """Test that start_eval_at_iter correctly gates evaluation in the training loop."""

    def test_eval_runs_without_start_eval_at_iter(self):
        """When start_eval_at_iter is None, eval runs at every eval_interval as before."""
        args = create_test_args()
        assert should_run_eval(args, 10)
        assert should_run_eval(args, 20)
        assert should_run_eval(args, 0)

    def test_eval_skipped_before_start_iter(self):
        """Eval should not run before start_eval_at_iter."""
        args = create_test_args(start_eval_at_iter=50)
        assert not should_run_eval(args, 10)
        assert not should_run_eval(args, 20)
        assert not should_run_eval(args, 30)
        assert not should_run_eval(args, 40)

    def test_eval_runs_at_start_iter(self):
        """Eval should run at exactly start_eval_at_iter if it aligns with eval_interval."""
        args = create_test_args(start_eval_at_iter=50)
        assert should_run_eval(args, 50)

    def test_eval_runs_after_start_iter(self):
        """Eval should run normally after start_eval_at_iter."""
        args = create_test_args(start_eval_at_iter=50)
        assert should_run_eval(args, 60)
        assert should_run_eval(args, 100)

    def test_non_interval_iterations_still_skipped(self):
        """Iterations not aligned with eval_interval should still be skipped."""
        args = create_test_args(start_eval_at_iter=50)
        assert not should_run_eval(args, 55)
        assert not should_run_eval(args, 73)

    def test_start_eval_at_iter_misaligned_with_interval(self):
        """When start_eval_at_iter doesn't align with eval_interval, first eval is at next aligned iteration."""
        args = create_test_args(start_eval_at_iter=25)
        # 25 is not divisible by 10, so iteration 25 won't trigger eval
        assert not should_run_eval(args, 25)
        # But 30 >= 25 and is divisible by 10
        assert should_run_eval(args, 30)

    def test_start_eval_at_iter_zero(self):
        """start_eval_at_iter=0 should behave like None (eval from the start)."""
        args = create_test_args(start_eval_at_iter=0)
        assert should_run_eval(args, 0)
        assert should_run_eval(args, 10)

    def test_do_valid_false_still_prevents_eval(self):
        """Even with start_eval_at_iter satisfied, do_valid=False should prevent eval."""
        args = create_test_args(start_eval_at_iter=10, do_valid=False)
        assert not should_run_eval(args, 20)

    def test_eval_interval_none_prevents_eval(self):
        """eval_interval=None should prevent eval regardless of start_eval_at_iter."""
        args = create_test_args(start_eval_at_iter=10, eval_interval=None)
        assert not should_run_eval(args, 20)
