# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Launch a VERL module after applying MLite compatibility patches."""

from __future__ import annotations

import runpy
import sys

from verl_mlite.compat import apply_runtime_patches


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m verl_mlite.launch <module> [args...]")
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    apply_runtime_patches()
    # Import the engine so its EngineRegistry.register decorator runs before the
    # verl trainer resolves the "mlite" backend.
    import verl_mlite.engine  # noqa: F401

    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
