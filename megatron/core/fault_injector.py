# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import datetime
import logging
import math
import random
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, TypeVar, cast

import torch
import torch.distributed as dist

try:
    from nvidia_resiliency_ext.shared_utils.inject_fault import (  # type: ignore[import-untyped]
        Fault,
        clear_workload_exception,
        dispatch_fault_injection,
        maybe_raise_workload_exception,
    )

    has_nvidia_resiliency_ext = True
except ModuleNotFoundError:
    has_nvidia_resiliency_ext = False

    def maybe_raise_workload_exception():  # pylint: disable=missing-function-docstring
        raise ModuleNotFoundError(
            "nvidia_resiliency_ext is required for fault injection. "
            "Please install it or disable fault injection."
        )


__all__ = ["FaultInjectorConfig", "setup_fault_injection", "maybe_raise_workload_exception"]


def _require_nvidia_resiliency_ext():
    if not has_nvidia_resiliency_ext:
        raise ModuleNotFoundError(
            "nvidia_resiliency_ext is required for fault injection. "
            "Please install it or disable fault injection."
        )


logger = logging.getLogger(__name__)

_T = TypeVar("_T")


@dataclass(kw_only=True)
class FaultInjectorConfig:
    """Configuration for fault injection testing via nvidia_resiliency_ext."""

    fault_injector_ranks: Optional[str] = None
    """Comma-separated list of ranks to inject faults on."""

    fault_injector_num_ranks: Optional[int] = None
    """Number of ranks to inject faults on (random selection)."""

    fault_injector_fault_types: Optional[str] = None
    """Comma-separated list of fault types to inject (e.g. 'hang,crash')."""

    fault_injector_fault_probabilities: Optional[str] = None
    """Comma-separated list of fault probabilities (normalized at runtime)."""

    fault_injector_fault_delay: Optional[float] = None
    """Force a specific fault delay in seconds from training start or delay_start_iteration."""

    fault_injector_delay_start_iteration: Optional[int] = None
    """Start the fault delay timer after iteration N completes.
    If unset, fault delay timing starts from the beginning of training."""

    fault_injector_mtti_seconds: Optional[float] = None
    """Mean time to inject (MTTI) in seconds; used when fault_delay is None."""

    fault_injector_offset_seconds: Optional[float] = None
    """Offset seconds added to the sampled fault delay."""

    fault_injector_seed: Optional[int] = None
    """RNG seed for the fault injector."""


class _FaultInjectorRNG(Protocol):
    """Minimal RNG interface used by fault injector helper functions."""

    def sample(self, population: Sequence[int], k: int) -> list[int]:
        """Return ``k`` sampled items from the given population."""
        ...

    def choices(self, population: Sequence[_T], weights: Sequence[float], k: int) -> list[_T]:
        """Return ``k`` weighted samples from the given population."""
        ...

    def random(self) -> float:
        """Return a floating-point value in the half-open interval [0.0, 1.0)."""
        ...


rng: _FaultInjectorRNG | None = None


def _require_rng() -> _FaultInjectorRNG:
    assert rng is not None, "fault injector rng must be initialized"
    return rng


def get_fault_ranks(config: FaultInjectorConfig):
    """Return list of ranks to inject faults on, from explicit list or random sample."""
    global rng

    force_ranks = config.fault_injector_ranks
    world_size = dist.get_world_size()

    if force_ranks is not None:
        assert (
            config.fault_injector_num_ranks is None
        ), "Cannot specify both force_ranks and num_ranks"
        if ',' in force_ranks:
            fault_ranks = [int(r) for r in force_ranks.split(",")]
        else:
            fault_ranks = [int(force_ranks)]
        assert all(
            0 <= r < world_size for r in fault_ranks
        ), f"Fault ranks must be between 0 and {world_size - 1}"
        assert len(fault_ranks) > 0, "Must specify at least one fault rank"
    else:
        assert (
            config.fault_injector_num_ranks is not None
        ), "Must specify either force_ranks or num_ranks"
        fault_ranks = _require_rng().sample(range(1, world_size), k=config.fault_injector_num_ranks)

    return fault_ranks


def get_fault(config: FaultInjectorConfig):
    """Sample a fault type according to the configured types and probabilities."""
    _require_nvidia_resiliency_ext()
    global rng

    fault_types_config = config.fault_injector_fault_types
    fault_probabilities_config = config.fault_injector_fault_probabilities
    assert fault_types_config is not None, "fault_injector_fault_types must be specified"

    if ',' in fault_types_config:
        fault_types = [Fault[t.upper()] for t in fault_types_config.split(",")]
    else:
        fault_types = [Fault[fault_types_config.upper()]]

    if fault_probabilities_config is not None:
        if ',' in fault_probabilities_config:
            fault_probabilities = [float(p) for p in fault_probabilities_config.split(",")]
        else:
            fault_probabilities = [float(fault_probabilities_config)]
        fault_probabilities = [p / sum(fault_probabilities) for p in fault_probabilities]
    else:
        fault_probabilities = [1 / len(fault_types) for _ in fault_types]

    assert len(fault_types) > 0, "Must specify at least one fault type"
    assert len(fault_types) == len(
        fault_probabilities
    ), "Number of fault types and fault probabilities must match"

    return _require_rng().choices(fault_types, fault_probabilities, k=1)[0]


def should_setup_fault_injection_at_start(config: FaultInjectorConfig):
    """Return True when fault timing is anchored to training start."""
    return config.fault_injector_delay_start_iteration is None


def should_setup_fault_injection_at_iteration(config: FaultInjectorConfig, iteration):
    """Return True when fault timing should start from the given iteration."""
    delay_start_iteration = config.fault_injector_delay_start_iteration
    return delay_start_iteration is not None and delay_start_iteration == iteration


def get_fault_delay(config: FaultInjectorConfig):
    """Return fault delay in seconds from the configured scheduling anchor."""
    global rng

    fault_delay = config.fault_injector_fault_delay
    assert (
        fault_delay is not None or config.fault_injector_mtti_seconds is not None
    ), "fault_injector_fault_delay or fault_injector_mtti_seconds must be specified"
    if fault_delay is None:
        mtti_seconds = config.fault_injector_mtti_seconds
        assert mtti_seconds is not None, "fault_injector_mtti_seconds must be specified"
        offset_seconds = config.fault_injector_offset_seconds or 0.0
        lambda_inj = 1.0 / mtti_seconds
        fault_delay = offset_seconds + (-math.log(1.0 - _require_rng().random()) / lambda_inj)

    return fault_delay


def setup_fault_injection(config: FaultInjectorConfig):
    """Broadcast fault plan across ranks and dispatch injection on target ranks."""
    _require_nvidia_resiliency_ext()
    global rng

    my_rank = dist.get_rank()
    world_size = dist.get_world_size()

    device = torch.device("cuda", torch.cuda.current_device())
    plan_tensor = torch.full((world_size + 1,), float("nan"), dtype=torch.float64, device=device)

    clear_workload_exception()

    if my_rank == 0:
        if rng is None:
            rng = cast(_FaultInjectorRNG, random.Random(config.fault_injector_seed))

        fault_ranks = get_fault_ranks(config)
        fault = get_fault(config)
        fault_delay = get_fault_delay(config)

        for rank in fault_ranks:
            plan_tensor[rank] = float(fault.value)
        plan_tensor[world_size] = fault_delay

    dist.broadcast(plan_tensor, src=0)

    planned_fault = float(plan_tensor[my_rank].item())
    is_target_rank = not math.isnan(planned_fault)

    if is_target_rank:
        fault = Fault(int(planned_fault))
        fault_delay = float(plan_tensor[world_size].item())
        current_time = datetime.datetime.now()
        fault_time = current_time + datetime.timedelta(seconds=fault_delay)
        timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        fault_timestamp = fault_time.strftime("%Y-%m-%d %H:%M:%S.%f")
        logger.warning(
            f"[{timestamp}] FAULT INJECTION: Rank {my_rank} will inject fault "
            f"{fault.name} at {fault_timestamp}"
        )
        dispatch_fault_injection(fault=fault, delay=fault_delay, callback=None)
