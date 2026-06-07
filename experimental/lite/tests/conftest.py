from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


LITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
for root in (REPO_ROOT, LITE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_addoption(parser):
    parser.addoption(
        "--mlite-smoke",
        action="store_true",
        default=False,
        help="run Megatron Lite smoke tests",
    )


def pytest_collection_modifyitems(config, items):
    run_smoke = config.getoption("--mlite-smoke") or os.getenv("MLITE_RUN_SMOKE") == "1"
    if run_smoke:
        return
    skip_smoke = pytest.mark.skip(reason="set --mlite-smoke or MLITE_RUN_SMOKE=1 to run")
    for item in items:
        if "smoke" in item.keywords:
            item.add_marker(skip_smoke)
