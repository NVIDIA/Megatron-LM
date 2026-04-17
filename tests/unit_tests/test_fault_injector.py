# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import math
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest

import megatron.core.fault_injector as fault_injector
from megatron.training.argument_utils import ArgumentGroupFactory
from megatron.training.config import FaultInjectorConfig


def create_test_args(**overrides):
    args = SimpleNamespace(
        fault_injector_fault_delay=None,
        fault_injector_delay_start_iteration=None,
        fault_injector_mtti_seconds=None,
        fault_injector_offset_seconds=None,
    )
    for key, val in overrides.items():
        setattr(args, key, val)
    return args


class TestFaultInjectorConfig:
    def test_fault_injector_config_has_delay_start_iteration(self):
        config = FaultInjectorConfig()
        assert config.fault_injector_delay_start_iteration is None

    def test_cli_arg_generated(self):
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(FaultInjectorConfig)
        factory.build_group(parser, "fault injector")
        args = parser.parse_args(["--fault-injector-delay-start-iteration", "100"])
        assert args.fault_injector_delay_start_iteration == 100

    def test_old_cli_arg_removed(self):
        parser = ArgumentParser()
        factory = ArgumentGroupFactory(FaultInjectorConfig)
        factory.build_group(parser, "fault injector")
        with pytest.raises(SystemExit):
            parser.parse_args(["--fault-injector-fault-iteration", "100"])


class TestFaultInjectorScheduling:
    def test_should_setup_fault_injection_at_start_without_anchor(self):
        args = create_test_args()
        assert fault_injector.should_setup_fault_injection_at_start(args)
        assert not fault_injector.should_setup_fault_injection_at_iteration(args, 0)

    def test_should_setup_fault_injection_at_matching_iteration(self):
        args = create_test_args(fault_injector_delay_start_iteration=12)
        assert not fault_injector.should_setup_fault_injection_at_start(args)
        assert not fault_injector.should_setup_fault_injection_at_iteration(args, 11)
        assert fault_injector.should_setup_fault_injection_at_iteration(args, 12)

    def test_get_fault_delay_returns_explicit_delay(self):
        args = create_test_args(
            fault_injector_fault_delay=7.5, fault_injector_delay_start_iteration=100
        )
        assert fault_injector.get_fault_delay(args) == 7.5

    def test_get_fault_delay_samples_mtti_with_offset(self, monkeypatch):
        args = create_test_args(
            fault_injector_delay_start_iteration=50,
            fault_injector_mtti_seconds=10.0,
            fault_injector_offset_seconds=2.0,
        )
        monkeypatch.setattr(fault_injector, "rng", SimpleNamespace(random=lambda: 0.5))

        fault_delay = fault_injector.get_fault_delay(args)

        assert math.isclose(fault_delay, 2.0 + (math.log(2.0) * 10.0))

    def test_get_fault_delay_requires_time_based_configuration(self):
        args = create_test_args(fault_injector_delay_start_iteration=25)
        with pytest.raises(
            AssertionError,
            match="fault_injector_fault_delay or fault_injector_mtti_seconds must be specified",
        ):
            fault_injector.get_fault_delay(args)
