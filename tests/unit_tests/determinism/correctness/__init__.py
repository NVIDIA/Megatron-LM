# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact correctness tests for ``--deterministic-mode``.

Two-run comparison + parametrize over preset × parallelism. The package
import sets the determinism env vars eagerly so cuBLAS / TE / NCCL capture
them at their respective first-use sites. ``apply_determinism_env`` uses
``setdefault`` — if pytest's collection has already touched CUDA via
another module before this package imports, the writes silently no-op and
the launcher's shell-side exports are what actually take effect (the CI
recipe relies on this defense-in-depth).
"""

import os

from megatron.training.determinism import apply_determinism_env

apply_determinism_env(os.environ)
