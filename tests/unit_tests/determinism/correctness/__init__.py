# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact correctness tests for ``--deterministic-mode``.

Two-run comparison + parametrize over preset × parallelism. The package
import sets the determinism env vars eagerly so cuBLAS / TE / NCCL capture
them at their respective first-use sites. ``set_determinism_env_vars``
uses ``setdefault`` — if pytest's collection has already touched CUDA via
another module before this package imports, the writes silently no-op and
the launcher's shell-side exports are what actually take effect (the CI
recipe relies on this defense-in-depth).
"""

from megatron.training.determinism import set_determinism_env_vars

set_determinism_env_vars()
