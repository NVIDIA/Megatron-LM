# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Two entry points:

* :func:`apply_determinism_env` — validate env-var settings and setdefault
  the canonical values. Must run BEFORE the first cuBLAS / Transformer
  Engine kernel invocation.
* :func:`apply_determinism_to_args` — validate a parsed ``args`` Namespace,
  call :func:`apply_determinism_env` on ``os.environ``, and flip
  ``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import os
from typing import MutableMapping

import torch

# Maps each arg name to the value it must hold for bit-exact execution;
# verified by :func:`apply_determinism_to_args`.
ARG_VALUES_REQUIRED_FOR_DETERMINISM = {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}

# Env-var defaults required for bit-exact reproducibility.
DETERMINISM_ENV_VAR_DEFAULTS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # TRITON_CACHE_AUTOTUNING is deliberately absent: unset is already deterministic, so
    # turning caching on is the operator's call. See apply_determinism_env().
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
ACCEPTED_NCCL_ALGO_TOKENS: frozenset[str] = frozenset({"Ring", "CollnetDirect", "CollnetChain", "^NVLS"})

# Env vars whose valid deterministic values are a small fixed exact-match set
# (unlike NCCL_ALGO which accepts comma-separated subsets of tokens). An unset
# value is fine -- apply_determinism_env() fills the canonical default. A
# set-but-invalid value fails hard.
#   - ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``: TE reads it as ``int(value)``; only
#     ``"0"`` means deterministic (any nonzero int enables non-deterministic
#     algos). See ``megatron/core/extensions/transformer_engine.py``.
#   - ``CUBLAS_WORKSPACE_CONFIG``: NVIDIA docs list ``:4096:8`` (4x4MiB) and
#     ``:16:8`` (8x16KiB) as the two deterministic workspace configurations;
#     any other value breaks reproducibility.
#   - ``TRITON_CACHE_AUTOTUNING``: the consumer tests it as ``== "1"``, so any
#     other truthy spelling ("true", "yes") would silently read as opted out.
#     Both settings are deterministic, so both are accepted and neither is
#     defaulted -- see :func:`apply_determinism_env` for the pairing rule.
ACCEPTED_ENV_VAR_VALUES: dict[str, frozenset[str]] = {
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": frozenset({"0"}),
    "CUBLAS_WORKSPACE_CONFIG": frozenset({":4096:8", ":16:8"}),
    "TRITON_CACHE_AUTOTUNING": frozenset({"0", "1"}),
}


def apply_determinism_env(env: MutableMapping[str, str]) -> None:
    """Validate every determinism env var in ``env``, then setdefault the canonical values.

    Semantics per key:

    * ``NCCL_ALGO`` — if set, each comma-separated token must be in
      :data:`ACCEPTED_NCCL_ALGO_TOKENS`.
    * ``NVTE_ALLOW_NONDETERMINISTIC_ALGO`` / ``CUBLAS_WORKSPACE_CONFIG`` —
      if set, must be in :data:`ACCEPTED_ENV_VAR_VALUES`.
    * ``MAMBA_DETERMINISTIC`` / ``CAUSAL_CONV1D_DETERMINISTIC`` — if set
      (non-empty), must start with ``'1'``; unset auto-follows
      :func:`torch.are_deterministic_algorithms_enabled`.
    * ``TRITON_CACHE_AUTOTUNING`` — opt-in; if set to ``'1'``, requires
      ``TRITON_CACHE_DIR``. Unset, Triton autotuning falls back to a pinned
      cheapest config, which is deterministic without any cache.

    After validation, ``setdefault`` fills every key in
    :data:`DETERMINISM_ENV_VAR_DEFAULTS` that has not been set — a value the
    caller has already set wins.

    These env vars are captured by their respective libraries at first use
    (NCCL at communicator init, cuBLAS at handle creation, TE at first
    attention forward), so the call must happen BEFORE any of those events.
    """
    # NCCL_ALGO subset check.
    nccl_algo = env.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in ACCEPTED_NCCL_ALGO_TOKENS for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r}: each token must be in "
            f"{sorted(ACCEPTED_NCCL_ALGO_TOKENS)}."
        )

    # Exact-match env vars: reject only if the caller supplied a value we
    # haven't validated as deterministic; unset is fine.
    for name, accepted in ACCEPTED_ENV_VAR_VALUES.items():
        val = env.get(name)
        assert val is None or val in accepted, (
            f"{name}={val!r} is not a deterministic setting. Accepted: {sorted(accepted)}."
        )

    # Mamba SSM and causal_conv1d auto-follow torch when unset; only reject an
    # explicit non-deterministic override.
    for name in ("MAMBA_DETERMINISTIC", "CAUSAL_CONV1D_DETERMINISTIC"):
        value = env.get(name)
        if value:
            assert value[0] == "1", (
                f"{name}={value!r} disables SSM determinism under "
                "--deterministic-mode. Unset it or set to '1'."
            )

    # Cross-field rule, so it cannot go in ACCEPTED_ENV_VAR_VALUES: caching only makes ranks
    # agree if they share one cache, and unset TRITON_CACHE_DIR means a node-local one.
    if env.get("TRITON_CACHE_AUTOTUNING") == "1":
        assert env.get("TRITON_CACHE_DIR"), (
            "TRITON_CACHE_AUTOTUNING=1 under --deterministic-mode requires TRITON_CACHE_DIR "
            "(a shared-filesystem path); unset TRITON_CACHE_AUTOTUNING to use the "
            "deterministic pinned-config fallback instead."
        )

        # Recommended, not required: changes no numerics, only visibility. print() because this
        # runs from validate_args, before logging is configured.
        if not env.get("TRITON_PRINT_AUTOTUNING"):
            print(
                "Deterministic mode: set TRITON_PRINT_AUTOTUNING=1 to log the kernel config "
                "each rank selects. A cache miss re-times the selection on that rank alone, "
                "which is how ranks come to disagree; without this the miss leaves no record.",
                flush=True,
            )

    # setdefault preserves any launcher-set value that just passed validation.
    for k, v in DETERMINISM_ENV_VAR_DEFAULTS.items():
        env.setdefault(k, v)


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode requirements to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Asserts every option in ``ARG_VALUES_REQUIRED_FOR_DETERMINISM`` holds
       its required value. This is a verification-only check — it never
       mutates ``args``.
    2. Calls :func:`apply_determinism_env` on ``os.environ`` — validates
       every determinism-relevant env var (``NCCL_ALGO``,
       ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``, ``CUBLAS_WORKSPACE_CONFIG``,
       ``MAMBA_DETERMINISTIC``, ``CAUSAL_CONV1D_DETERMINISTIC``,
       ``TRITON_CACHE_AUTOTUNING`` and its required ``TRITON_CACHE_DIR``) and
       setdefaults the canonical values.
    3. Calls ``torch.use_deterministic_algorithms(True)``.

    Incompatible options are rejected with an explicit error rather than
    silently overridden: the user must turn them off themselves so the
    deterministic run matches the config they asked for.
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

    # --use-flash-attn is intentionally NOT rejected: TE's FlashAttention is
    # deterministic on supported configs and is covered by the bit-exact
    # correctness suite.

    # Delegate env-var validation + setdefault to the single-source helper.
    apply_determinism_env(os.environ)

    # Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
