from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import pytest


LITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
VERL_EXAMPLE_ROOT = LITE_ROOT / "examples" / "verl"
for root in (REPO_ROOT, LITE_ROOT, VERL_EXAMPLE_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pytest_configure(config):
    config.addinivalue_line("markers", "mlite: mark a test as Megatron Lite validation coverage")
    config.addinivalue_line(
        "markers",
        "smoke: mark a Megatron Lite smoke test; skipped unless --mlite-smoke or MLITE_RUN_SMOKE=1 is set",
    )
    config.addinivalue_line("markers", "gpu: mark a test as requiring CUDA")
    config.addinivalue_line("markers", "distributed: mark a test as requiring torch.distributed")


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


@pytest.fixture
def transformer_engine_import_stub(monkeypatch):
    def install() -> None:
        try:
            import transformer_engine.pytorch  # noqa: F401

            return
        except ModuleNotFoundError as exc:
            if exc.name not in {"transformer_engine", "transformer_engine.pytorch"}:
                raise

        class _UnavailableTE:
            def __init__(self, *args, **kwargs):
                raise RuntimeError(
                    "Transformer Engine is not installed in this test environment."
                )

        root = types.ModuleType("transformer_engine")
        root.__version__ = "0.0.0"
        pytorch = types.ModuleType("transformer_engine.pytorch")
        pytorch.DotProductAttention = _UnavailableTE
        pytorch.LayerNormLinear = _UnavailableTE
        pytorch.Linear = _UnavailableTE
        pytorch.RMSNorm = _UnavailableTE
        root.pytorch = pytorch
        monkeypatch.setitem(sys.modules, "transformer_engine", root)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)

    return install
