# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Opt-in gate and shared fixtures for the reverse-converter end-to-end suite.

The gate keeps this expensive GPU suite invisible to a default ``pytest tests``
run and to CI: unless opted in (``MCORE_CHECKPOINT_E2E=1`` or ``--run-e2e``), the
``test_*.py`` modules here are never even imported. A second guard skips
everything if pytest was launched under ``torch.distributed.run`` (WORLD_SIZE>1),
because this suite is a single controller that spawns its OWN torchrun children.

Training is shared via a memoizing session-scoped factory so each family trains
exactly once, no matter how the four test modules parametrize — the resume,
bit-exact and reshard checks all reuse the same converted checkpoints.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pytest

from tests.integration_tests.tools.checkpoint.fsdp_dtensor_to_torch_dist import config, harness


# --------------------------------------------------------------------------
# Opt-in gate
# --------------------------------------------------------------------------
def _e2e_enabled(pytest_config) -> bool:
    if os.environ.get("MCORE_CHECKPOINT_E2E") == "1":
        return True
    try:
        return bool(pytest_config.getoption("--run-e2e"))
    except (ValueError, KeyError):  # option not registered in this invocation
        return False


def pytest_ignore_collect(collection_path, config):  # noqa: A002 (pytest hook name)
    """Do not even import the test modules unless the suite is opted in.

    Returning True for a ``test_*.py`` path means it is not collected, not
    imported, and never touches CUDA — so a bare ``pytest tests`` stays green and
    fast. The conftest itself still loads (it must, to register this hook).
    """
    if _e2e_enabled(config):
        return None
    return Path(str(collection_path)).name.startswith("test_")


def pytest_collection_modifyitems(config, items):
    """Never run under torch.distributed.run — this suite spawns its own torchrun."""
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        skip = pytest.mark.skip(
            reason="Run as plain pytest (single controller); this suite spawns its own "
            "torchrun children. Do not launch it under torch.distributed.run."
        )
        for item in items:
            item.add_marker(skip)


def pytest_configure(config):
    """Seed the deterministic env for the torchrun children (they inherit it)."""
    for key, value in config_env().items():
        os.environ.setdefault(key, value)


def config_env():
    return dict(config.DETERMINISTIC_ENV)


# --------------------------------------------------------------------------
# Shared training (one real FSDP run per family, reused across checks)
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class FamilyRun:
    """Products of one real Megatron-FSDP training run + conversion for a family."""

    family: object
    root: Path
    fsdp_dir: Path
    td: Dict[int, Path]  # {60: <root>/td60, 80: <root>/td80}
    fsdp_metrics: Dict[int, harness.IterMetrics]  # per-iter (lm loss, lr) reference
    train_log: Path


@pytest.fixture(scope="session")
def results_root(tmp_path_factory) -> Path:
    """Where checkpoints + logs land. Honors RESULTS_DIR; else an auto-cleaned tmp dir."""
    override = config.results_root_override()
    if override is not None:
        override.mkdir(parents=True, exist_ok=True)
        return override
    return tmp_path_factory.mktemp("ckpt_e2e")


def _build_family_run(fam, out_dir: Path, *, nproc: int = 1, src_parallel=()) -> FamilyRun:
    text = harness.run_training(fam, out_dir, nproc=nproc, src_parallel=src_parallel)
    td = {
        it: harness.convert(out_dir / "fsdp" / f"iter_{it:07d}", out_dir / f"td{it}", it)
        for it in config.CONVERT_ITERS
    }
    return FamilyRun(
        family=fam,
        root=out_dir,
        fsdp_dir=out_dir / "fsdp",
        td=td,
        fsdp_metrics=harness.parse_iter_metrics(text),
        train_log=out_dir / "train_fsdp.log",
    )


def _cached(cache, key, build):
    """Memoize ``build()`` under ``key`` — caching failures too, so a family that
    fails to train once fails fast for all its tests instead of retraining each time.
    """
    if key not in cache:
        try:
            cache[key] = build()
        except Exception as exc:  # noqa: BLE001 — deliberately cache the failure
            cache[key] = exc
    result = cache[key]
    if isinstance(result, Exception):
        raise result
    return result


@pytest.fixture(scope="session")
def family_runs(results_root):
    """Memoizing factory: ``family_runs(fam) -> FamilyRun``, training each family once."""
    cache: Dict[str, object] = {}

    def get(fam) -> FamilyRun:
        return _cached(cache, fam.name, lambda: _build_family_run(fam, results_root / fam.name))

    return get


@pytest.fixture(scope="session")
def sharded_family_runs(results_root):
    """Memoizing factory for source-side sharding: ``sharded_family_runs(fam, layout)``."""
    cache: Dict[tuple, object] = {}

    def get(fam, layout: str) -> FamilyRun:
        def build():
            out_dir = results_root / f"{fam.name}__src_{layout}"
            src_parallel = config.source_parallel_flags(layout)
            return _build_family_run(fam, out_dir, nproc=2, src_parallel=src_parallel)

        return _cached(cache, (fam.name, layout), build)

    return get
