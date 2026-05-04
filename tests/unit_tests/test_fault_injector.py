# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import enum
import math
from argparse import ArgumentParser
from types import SimpleNamespace

import pytest
import torch

import megatron.core.fault_injector as fault_injector
from megatron.training.argument_utils import ArgumentGroupFactory
from megatron.training.config import FaultInjectorConfig


def create_test_config(**overrides):
    config = FaultInjectorConfig(**overrides)
    return config


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
    def test_get_fault_ranks_parses_explicit_rank_list(self, monkeypatch):
        config = create_test_config(fault_injector_ranks="0,3,7", fault_injector_num_ranks=None)
        monkeypatch.setattr(fault_injector.dist, "get_world_size", lambda: 8)

        assert fault_injector.get_fault_ranks(config) == [0, 3, 7]

    def test_get_fault_ranks_samples_requested_num_ranks(self, monkeypatch):
        sampled = []
        config = create_test_config(fault_injector_ranks=None, fault_injector_num_ranks=2)

        def fake_sample(population, k):
            sampled.append((list(population), k))
            return [2, 5]

        monkeypatch.setattr(fault_injector.dist, "get_world_size", lambda: 8)
        monkeypatch.setattr(fault_injector, "rng", SimpleNamespace(sample=fake_sample))

        assert fault_injector.get_fault_ranks(config) == [2, 5]
        assert sampled == [([1, 2, 3, 4, 5, 6, 7], 2)]

    def test_get_fault_requires_fault_types(self, monkeypatch):
        config = create_test_config()
        monkeypatch.setattr(fault_injector, "has_nvidia_resiliency_ext", True)

        with pytest.raises(AssertionError, match="fault_injector_fault_types must be specified"):
            fault_injector.get_fault(config)

    def test_get_fault_parses_types_and_normalizes_probabilities(self, monkeypatch):
        class FakeFault(enum.IntEnum):
            HANG = 1
            CRASH = 2

        captured = []
        config = create_test_config(
            fault_injector_fault_types="hang,crash", fault_injector_fault_probabilities="2,1"
        )

        def fake_choices(fault_types, fault_probabilities, k):
            captured.append((fault_types, fault_probabilities, k))
            return [fault_types[1]]

        monkeypatch.setattr(fault_injector, "has_nvidia_resiliency_ext", True)
        monkeypatch.setattr(fault_injector, "Fault", FakeFault, raising=False)
        monkeypatch.setattr(fault_injector, "rng", SimpleNamespace(choices=fake_choices))

        fault = fault_injector.get_fault(config)

        assert fault == FakeFault.CRASH
        assert captured[0][0] == [FakeFault.HANG, FakeFault.CRASH]
        assert captured[0][2] == 1
        assert captured[0][1] == pytest.approx([2 / 3, 1 / 3])

    def test_should_setup_fault_injection_at_start_without_anchor(self):
        config = create_test_config()
        assert fault_injector.should_setup_fault_injection_at_start(config)
        assert not fault_injector.should_setup_fault_injection_at_iteration(config, 0)

    def test_should_setup_fault_injection_at_matching_iteration(self):
        config = create_test_config(fault_injector_delay_start_iteration=12)
        assert not fault_injector.should_setup_fault_injection_at_start(config)
        assert not fault_injector.should_setup_fault_injection_at_iteration(config, 11)
        assert fault_injector.should_setup_fault_injection_at_iteration(config, 12)

    def test_get_fault_delay_returns_explicit_delay(self):
        config = create_test_config(
            fault_injector_fault_delay=7.5, fault_injector_delay_start_iteration=100
        )
        assert fault_injector.get_fault_delay(config) == 7.5

    def test_get_fault_delay_samples_mtti_with_offset(self, monkeypatch):
        config = create_test_config(
            fault_injector_delay_start_iteration=50,
            fault_injector_mtti_seconds=10.0,
            fault_injector_offset_seconds=2.0,
        )
        monkeypatch.setattr(fault_injector, "rng", SimpleNamespace(random=lambda: 0.5))

        fault_delay = fault_injector.get_fault_delay(config)

        assert math.isclose(fault_delay, 2.0 + (math.log(2.0) * 10.0))

    def test_get_fault_delay_requires_time_based_configuration(self):
        config = create_test_config(fault_injector_delay_start_iteration=25)
        with pytest.raises(
            AssertionError,
            match="fault_injector_fault_delay or fault_injector_mtti_seconds must be specified",
        ):
            fault_injector.get_fault_delay(config)


class TestFaultInjectorSetup:
    def test_setup_fault_injection_uses_single_plan_broadcast_for_zero_valued_fault(
        self, monkeypatch
    ):
        class ZeroFault(enum.IntEnum):
            ZERO_FAULT = 0

        broadcasts = []
        dispatched = []
        config = create_test_config(fault_injector_seed=123)
        fake_torch = SimpleNamespace(
            device=lambda *_args, **_kwargs: "cpu",
            full=torch.full,
            float64=torch.float64,
            cuda=SimpleNamespace(current_device=lambda: 0),
        )

        monkeypatch.setattr(fault_injector, "has_nvidia_resiliency_ext", True)
        monkeypatch.setattr(fault_injector, "Fault", ZeroFault, raising=False)
        monkeypatch.setattr(fault_injector, "torch", fake_torch)
        monkeypatch.setattr(fault_injector.dist, "get_rank", lambda: 0)
        monkeypatch.setattr(fault_injector.dist, "get_world_size", lambda: 4)
        monkeypatch.setattr(
            fault_injector.dist,
            "broadcast",
            lambda tensor, src: broadcasts.append((tensor.clone(), src)),
        )
        monkeypatch.setattr(fault_injector, "clear_workload_exception", lambda: None, raising=False)
        monkeypatch.setattr(fault_injector, "get_fault_ranks", lambda _config: [0])
        monkeypatch.setattr(fault_injector, "get_fault", lambda _config: ZeroFault.ZERO_FAULT)
        monkeypatch.setattr(fault_injector, "get_fault_delay", lambda _config: 3.5)
        monkeypatch.setattr(
            fault_injector,
            "dispatch_fault_injection",
            lambda fault, delay, callback: dispatched.append((fault, delay, callback)),
            raising=False,
        )
        monkeypatch.setattr(fault_injector, "rng", None)

        fault_injector.setup_fault_injection(config)

        assert len(broadcasts) == 1
        assert dispatched == [(ZeroFault.ZERO_FAULT, 3.5, None)]
