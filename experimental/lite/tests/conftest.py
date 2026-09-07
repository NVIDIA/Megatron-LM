# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

LITE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
HARNESS_ROOT = Path(__file__).resolve().parent / "_test_harness"
for root in (REPO_ROOT, LITE_ROOT, HARNESS_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from markers import MarkerError, execution_for_item, register  # noqa: E402


def pytest_configure(config):
    register(config)


def pytest_collection_modifyitems(config, items):
    plan_path = os.getenv("MLITE_TEST_PLAN_PATH")
    tests = []
    for item in items:
        try:
            execution = execution_for_item(item)
        except MarkerError as exc:
            raise pytest.UsageError(f"{item.nodeid}: {exc}") from exc
        if execution.gpus and os.getenv("MLITE_TEST_HARNESS") != "1":
            item.add_marker(
                pytest.mark.skip(reason="run GPU tests through tests/run_tests.sh")
            )
        if plan_path:
            tests.append({"nodeid": item.nodeid, **execution.as_dict()})

    if plan_path:
        destination = Path(plan_path)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps({"tests": tests}, sort_keys=True), encoding="utf-8"
        )
        temporary.replace(destination)


@pytest.fixture
def transformer_engine_import_stub(monkeypatch):
    def install() -> None:
        try:
            import transformer_engine.pytorch  # noqa: F401

            return
        except (ModuleNotFoundError, OSError) as exc:
            if isinstance(exc, ModuleNotFoundError) and exc.name not in {
                "transformer_engine",
                "transformer_engine.pytorch",
            }:
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
        permutation = types.ModuleType("transformer_engine.pytorch.permutation")
        router = types.ModuleType("transformer_engine.pytorch.router")
        cpp_extensions = types.ModuleType("transformer_engine.pytorch.cpp_extensions")
        module = types.ModuleType("transformer_engine.pytorch.module")
        module_base = types.ModuleType("transformer_engine.pytorch.module.base")

        def unavailable_kernel(*args, **kwargs):
            raise RuntimeError("Transformer Engine fused kernel is not installed.")

        permutation.moe_permute = unavailable_kernel
        permutation.moe_permute_and_pad_with_probs = unavailable_kernel
        permutation.moe_permute_with_probs = unavailable_kernel
        permutation.moe_unpermute = unavailable_kernel
        router.fused_compute_score_for_moe_aux_loss = unavailable_kernel
        router.fused_moe_aux_loss = unavailable_kernel
        router.fused_topk_with_score_function = unavailable_kernel
        cpp_extensions.general_gemm = lambda *args, **kwargs: None
        module_base.get_workspace = lambda: None
        module.base = module_base
        pytorch.permutation = permutation
        pytorch.router = router
        pytorch.cpp_extensions = cpp_extensions
        pytorch.module = module
        root.pytorch = pytorch
        monkeypatch.setitem(sys.modules, "transformer_engine", root)
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch", pytorch)
        monkeypatch.setitem(
            sys.modules, "transformer_engine.pytorch.permutation", permutation
        )
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.router", router)
        monkeypatch.setitem(
            sys.modules, "transformer_engine.pytorch.cpp_extensions", cpp_extensions
        )
        monkeypatch.setitem(sys.modules, "transformer_engine.pytorch.module", module)
        monkeypatch.setitem(
            sys.modules, "transformer_engine.pytorch.module.base", module_base
        )

    return install
