# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Mirrors the split that ``megatron-bridge`` uses for the same purpose:

* :func:`set_determinism_env_vars` — env-var setdefaults that must happen
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
  scripts call :func:`set_determinism_env_vars` directly at import time —
  they don't have an ``args`` Namespace.
"""

from __future__ import annotations

import os

import torch


def set_determinism_env_vars() -> None:
    """Populate env vars required for bit-exact reproducibility.

    These env vars are captured by their respective libraries at first use
    (NCCL at communicator init, cuBLAS at handle creation, TE at first
    attention forward), so the call must happen BEFORE any of those events.
    Uses ``setdefault`` so any value the launcher has already exported
    wins — defense in depth: in the test process pytest may import another
    module that triggers CUDA-context creation before this package loads,
    in which case the Python-side setdefault is too late and the launcher's
    shell-side export is what actually takes effect.
    """
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode overrides to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Asserts ``cross_entropy_loss_fusion`` is off (fused CE is non-deterministic).
    2. Sets env vars via :func:`set_determinism_env_vars` — a user-supplied
       value (e.g. an ``NCCL_ALGO`` exported by the launcher) survives the
       setdefault, so this step does not second-guess the user's choice of
       deterministic algo.
    3. Forces ``tp_comm_overlap=False`` (non-deterministic NCCL collectives).
    4. Calls ``torch.use_deterministic_algorithms(True)``.

    The argument assertion runs FIRST so a malformed args Namespace fails
    fast without leaving the process in a half-deterministic state (env
    vars set but torch global state untouched).
    """
    # 1. Validate args first — direct attribute access so a malformed
    #    Namespace (missing the field entirely) fails loudly rather than
    #    silently passing the check.
    assert (
        not args.cross_entropy_loss_fusion
    ), "Cross Entropy Fusion is currently not deterministic."

    # NB: ``--use-flash-attn`` is intentionally NOT rejected under
    # --deterministic-mode. FlashAttention is deterministic on supported
    # configs (modern TE + Hopper/Blackwell); the bit-exact correctness
    # suite covers it across recipes (see
    # ``tests/unit_tests/determinism/correctness/test_fp8_determinism.py``
    # and the bf16 parallelism matrix). Backend choice is left to TE's
    # default selection or to the launcher via ``NVTE_*_ATTN`` env vars.

    # 2. NCCL_ALGO sanity check on a launcher-supplied value. Tree is
    #    intentionally excluded: its intra-node chain reduction order is not
    #    user-controllable and multi-node inter-tree topology can vary
    #    across runs without a pinned topology file, so we cannot vouch
    #    for it as deterministic. ``^NVLS`` is kept because banning NVLS
    #    on hardware that exposes it is a legitimate user choice; the user
    #    is responsible for picking a fallback that's deterministic on
    #    their stack. Comma-separated lists are valid NCCL syntax.
    accepted_tokens = {"Ring", "CollnetDirect", "CollnetChain", "^NVLS"}
    nccl_algo = os.environ.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in accepted_tokens for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r} must be a comma-separated subset of "
            f"{sorted(accepted_tokens)}."
        )

    # 3. Apply env vars. ``setdefault`` preserves any launcher-set value
    #    that just passed validation; the default of ``Ring`` only kicks
    #    in when nothing was set.
    set_determinism_env_vars()

    # 4. Override tp_comm_overlap. ``warn_rank_0`` (not print_rank_0) so the
    #    override is capturable via ``pytest.warns`` / ``-W error``.
    if args.tp_comm_overlap:
        # Lazy import — warn_rank_0 lives in training.utils which has heavier
        # dependencies than this module.
        from megatron.training.utils import warn_rank_0

        warn_rank_0("Disabling tp_comm_overlap for deterministic mode.")
        args.tp_comm_overlap = False

    # 5. Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
