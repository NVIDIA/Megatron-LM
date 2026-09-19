# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Run a script or module after explicitly enabling the early process policy."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from . import configure_determinism


def main(argv: list[str] | None = None) -> None:
    """Launch with Python's script/module argument semantics and strict policy.

    The target must still enable deterministic_mode on its model configuration;
    Core training and Bridge validate that configuration before initialization.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--module", action="store_true", help="Run target as a module")
    parser.add_argument("target", help="Python script path or module name")
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments forwarded to the target")
    args = parser.parse_args(argv)
    configure_determinism({"deterministic_mode": True})
    previous_argv, previous_path = sys.argv, sys.path.copy()
    try:
        sys.argv = [args.target, *args.args]
        if args.module:
            runpy.run_module(args.target, run_name="__main__", alter_sys=True)
        else:
            path = Path(args.target).resolve()
            sys.path.insert(0, str(path.parent))
            runpy.run_path(str(path), run_name="__main__")
    finally:
        sys.argv, sys.path[:] = previous_argv, previous_path


if __name__ == "__main__":
    main()
