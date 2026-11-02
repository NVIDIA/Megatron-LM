# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Count the calls of the operations that HyperConnectionModule instances have bound.

A bound partial captures its function when the module is built, so patching the kernel
module afterwards does not reach the module. The spy therefore wraps the modules' instance
attributes after the model is built, and restores them on exit.

Counts grow only while Python runs: in eager iterations, and once while a CUDA graph is
captured, never while it is replayed. Modules built after the spy was entered are not spied.

Usage::

    with spy_fused_mhc_ops(model) as spy:
        run_iteration(model)
    spy.assert_ran("_sinkhorn_op", expected_function)
"""

import functools
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List

import torch

from megatron.core.transformer.hyper_connection import HyperConnectionModule

SPIED_ATTRIBUTES = (
    "_glm_proj_rms_op",
    "_proj_rms_compute_h_op",
    "_sinkhorn_op",
    "_h_aggregate_op",
    "_h_aggregate_into_op",
    "_h_post_bda_op",
)


@dataclass
class OpRecord:
    """Calls of one bound operation of one module."""

    module: str  # qualified module name, or "#<index>" for a bare list
    attribute: str
    target: object  # the object that was bound when the spy was installed
    calls: int = 0
    unsupported: int = 0  # target.supported(...) returned False: the caller fell back


class _SupportedOpProxy:
    """Counting stand-in for an operation object with a ``supported`` predicate.

    The caller must reach ``supported`` through the module attribute, so that its fallbacks
    are counted too.
    """

    def __init__(self, record: OpRecord) -> None:
        self._record = record

    def supported(self, *args, **kwargs):
        """Forward to the target's predicate and count the inputs it refuses."""
        result = self._record.target.supported(*args, **kwargs)
        if not result:
            self._record.unsupported += 1
        return result

    def __call__(self, *args, **kwargs):
        self._record.calls += 1
        return self._record.target(*args, **kwargs)

    def __getattr__(self, name):
        # Only names the proxy lacks reach here. copy and pickle probe dunders on an instance
        # whose _record is not set yet, so never look those up through _record.
        if name == "_record" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self._record.target, name)


def _counting(record: OpRecord):
    if callable(getattr(record.target, "supported", None)):
        return _SupportedOpProxy(record)

    @functools.wraps(record.target, updated=())
    def counting(*args, **kwargs):
        record.calls += 1
        return record.target(*args, **kwargs)

    return counting


def _binds(bound, target) -> bool:
    return bound is target or (isinstance(bound, functools.partial) and bound.func is target)


class FusedMHCOpSpy:
    """Call counts of the spied operations, per module and summed over the modules."""

    def __init__(self, records: List[OpRecord]) -> None:
        self.records = records

    def _records(self, attribute: str) -> List[OpRecord]:
        if attribute not in SPIED_ATTRIBUTES:
            raise KeyError(
                f"{attribute!r} is not spied; the spied attributes are {SPIED_ATTRIBUTES}"
            )
        return [record for record in self.records if record.attribute == attribute]

    def calls(self, attribute: str) -> int:
        """Return the calls of ``attribute``, summed over the modules."""
        return sum(record.calls for record in self._records(attribute))

    def unsupported(self, attribute: str) -> int:
        """Return how often ``attribute``'s ``supported`` refused its inputs, over the modules."""
        return sum(record.unsupported for record in self._records(attribute))

    def targets(self, attribute: str) -> set:
        """Return the objects bound to ``attribute`` when the spy was installed."""
        return {record.target for record in self._records(attribute)}

    def assert_ran(self, attribute: str, target=None, min_calls: int = 1) -> None:
        """Assert that ``attribute`` ran at least ``min_calls`` times over all modules.

        With ``target``, every spied module must also have bound ``target`` itself or a
        ``functools.partial`` of it.
        """
        records = self._records(attribute)
        calls = sum(record.calls for record in records)
        assert calls >= min_calls, (
            f"{attribute} ran {calls} time(s), expected at least {min_calls}; per module: "
            f"{[(record.module, record.calls, record.target) for record in records]}"
        )
        if target is not None:
            unbound = [
                (record.module, record.target)
                for record in records
                if not _binds(record.target, target)
            ]
            assert not unbound, f"{attribute} is not bound to {target!r} in {unbound}"

    def assert_no_unsupported(self, attribute: str = "_glm_proj_rms_op") -> None:
        """Assert that ``attribute``'s ``supported`` never refused its inputs."""
        unsupported = self.unsupported(attribute)
        assert unsupported == 0, (
            f"{attribute}.supported() refused {unsupported} call(s), so the module fell back "
            "to its unfused path"
        )


@contextmanager
def spy_fused_mhc_ops(root) -> Iterator[FusedMHCOpSpy]:
    """Spy on the bound mHC operations under ``root`` until the context exits.

    Args:
        root: an ``nn.Module``, whose ``HyperConnectionModule`` instances (``root`` included)
            are spied, or an iterable of modules, which are spied as given.
    """
    if isinstance(root, torch.nn.Module):
        modules = [
            (name or "<root>", module)
            for name, module in root.named_modules()
            if isinstance(module, HyperConnectionModule)
        ]
    else:
        modules = [(f"#{index}", module) for index, module in enumerate(root)]
    if not modules:
        raise ValueError("found no mHC module to spy on")

    records, installed = [], []
    try:
        for name, module in modules:
            for attribute in SPIED_ATTRIBUTES:
                record = OpRecord(name, attribute, getattr(module, attribute))
                records.append(record)
                # None stays None: the module tests the attribute to choose its path.
                if record.target is not None:
                    setattr(module, attribute, _counting(record))
                    installed.append((module, attribute, record.target))
        yield FusedMHCOpSpy(records)
    finally:
        for module, attribute, target in reversed(installed):
            setattr(module, attribute, target)
