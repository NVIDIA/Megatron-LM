# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Mirrors the split that ``megatron-bridge`` uses for the same purpose:

* :func:`set_determinism_env_var_defaults` — env-var setdefaults that must happen
  BEFORE the first cuBLAS / Transformer Engine kernel invocation in the
  process. Equivalent to bridge's
  ``PerfEnvPlugin._set_determinism_env_vars`` (``scripts/performance/perf_plugins.py``).
* :func:`apply_determinism_to_args` — config-level overrides applied to a
  parsed ``args`` Namespace. Equivalent to bridge's
  ``apply_determinism_overrides`` (``recipes/utils/determinism_utils.py``)
  but works on the ``args`` produced by Megatron-LM's argparser.

Callers:

* CLI training (``pretrain_*.py``) goes through ``validate_args`` which
  calls :func:`apply_determinism_to_args` when ``--deterministic-mode`` is
  passed.
* Test suite (``tests/unit_tests/determinism/``) and standalone profiling
  scripts call :func:`set_determinism_env_var_defaults` directly at import time —
  they don't have an ``args`` Namespace.
"""

from __future__ import annotations

import os

import torch

# Maps each arg name to the value it must hold for bit-exact execution;
# verified by :func:`apply_determinism_to_args`.
ARG_VALUES_REQUIRED_FOR_DETERMINISM = {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}

# Env-var defaults required for bit-exact reproducibility.
DETERMINISM_ENV_VAR_DEFAULTS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Accepted NCCL_ALGO tokens under --deterministic-mode. Comma-separated lists
# are valid NCCL syntax; every token in a launcher-supplied ``NCCL_ALGO`` must
# be in this set.
#   - ``Ring``          the default; bit-exact by construction, fully verified.
#   - ``CollnetDirect``,
#     ``CollnetChain``  verified bit-exact at smaller scale with SHARP
#                       in-network reduction (AllReduce and an end-to-end run).
#   - ``^NVLS``         excludes NVLS rather than selecting an algo, so NCCL
#                       falls back to whichever algo fits the hardware.
#                       Verified bit-exact in our setup; some risk remains
#                       because determinism then depends on that fallback algo.
#
# ``Tree`` is intentionally NOT accepted: its intra-node chain reduction
# order is not user-controllable and its multi-node inter-tree topology can
# vary across runs without a pinned topology file, so we cannot vouch for it.
#
# Exposed as ``frozenset`` so external pre-submit validators (e.g. a Slurm
# launcher that pre-validates a user-supplied ``nccl_algo`` string) can reuse
# it without duplicating the list.
ACCEPTED_NCCL_ALGO_TOKENS: frozenset[str] = frozenset({"Ring", "CollnetDirect", "CollnetChain", "^NVLS"})


def set_determinism_env_var_defaults() -> None:
    """Set defaults for the env vars required for bit-exact reproducibility.

    Only fills in values the launcher has not already exported (``setdefault``);
    an env var that is already set is left untouched.

    These env vars are captured by their respective libraries at first use
    (NCCL at communicator init, cuBLAS at handle creation, TE at first
    attention forward), so the call must happen BEFORE any of those events.
    Uses ``setdefault`` so any value the launcher has already exported
    wins — defense in depth: in the test process pytest may import another
    module that triggers CUDA-context creation before this package loads,
    in which case the Python-side setdefault is too late and the launcher's
    shell-side export is what actually takes effect.
    """
    for k, v in DETERMINISM_ENV_VAR_DEFAULTS.items():
        os.environ.setdefault(k, v)


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode requirements to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Asserts every option in ``ARG_VALUES_REQUIRED_FOR_DETERMINISM`` holds
       its required value. This is a verification-only check — it never
       mutates ``args``.
    2. Sets env-var defaults via :func:`set_determinism_env_var_defaults` — a user-supplied
       value (e.g. an ``NCCL_ALGO`` exported by the launcher) survives the
       setdefault, so this step does not second-guess the user's choice of
       deterministic algo.
    3. Calls ``torch.use_deterministic_algorithms(True)``.

    Incompatible options are rejected with an explicit error rather than
    silently overridden: the user must turn them off themselves so the
    deterministic run matches the config they asked for. The assertion runs
    FIRST so a malformed or incompatible args Namespace fails fast without
    leaving the process in a half-deterministic state (env vars set but
    torch global state untouched).
    """
    # Verification only — read each option's effective value and never flip it,
    # so a default that drifts to a bad value breaks the run instead of silently
    # running non-deterministically.
    mismatched = [
        f"{name}={required!r} (got {actual!r})"
        for name, required in ARG_VALUES_REQUIRED_FOR_DETERMINISM.items()
        if (actual := getattr(args, name)) != required
    ]
    assert (
        not mismatched
    ), f"--deterministic-mode requires: {', '.join(mismatched)}. Adjust these options to continue."

    # NB: ``--use-flash-attn`` is intentionally NOT rejected under
    # --deterministic-mode. FlashAttention is deterministic on supported
    # configs (modern TE + Hopper/Blackwell); the bit-exact correctness
    # suite covers it across recipes (see
    # ``tests/unit_tests/determinism/correctness/test_fp8_determinism.py``
    # and the bf16 parallelism matrix). Backend choice is left to TE's
    # default selection or to the launcher via ``NVTE_*_ATTN`` env vars.

    # NCCL_ALGO sanity check on a launcher-supplied value. See
    # :data:`ACCEPTED_NCCL_ALGO_TOKENS` for the accepted set and the rationale
    # for each entry (and why ``Tree`` is excluded).
    nccl_algo = os.environ.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in ACCEPTED_NCCL_ALGO_TOKENS for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r} must be a comma-separated subset of "
            f"{sorted(ACCEPTED_NCCL_ALGO_TOKENS)}."
        )

    # Apply env vars. ``setdefault`` preserves any launcher-set value that just
    # passed validation; the default of ``Ring`` only kicks in when nothing was set.
    set_determinism_env_var_defaults()

    # Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
