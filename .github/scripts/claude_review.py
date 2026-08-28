#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Compatibility launcher for Megatron-LM's local isolated review package."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_PACKAGE_DIR = Path(__file__).with_suffix("")
_SPEC = importlib.util.spec_from_file_location(
    "megatron_claude_review", _PACKAGE_DIR / "__init__.py", submodule_search_locations=[str(_PACKAGE_DIR)]
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError("isolated review package is unavailable")
_package = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _package
_SPEC.loader.exec_module(_package)

for _name in _package.__all__:
    globals()[_name] = getattr(_package, _name)

main = __import__("megatron_claude_review.cli", fromlist=["main"]).main

if __name__ == "__main__":
    raise SystemExit(main())
