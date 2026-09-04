# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch a VERL module after registering the MLite engine."""

from __future__ import annotations

import runpy
import sys

def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m verl_mlite.launch <module> [args...]")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    # Import the engine so its EngineRegistry.register decorator runs before the
    # verl trainer resolves the "mlite" backend.
    import verl_mlite.engine  # noqa: F401

    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
