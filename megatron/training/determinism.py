# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Three entry points:

* :func:`apply_determinism_env` — validate env-var settings and setdefault
  the canonical values. Must run BEFORE the first cuBLAS / Transformer
  Engine kernel invocation.
* :func:`apply_determinism_to_args` — validate a parsed ``args`` Namespace,
  call :func:`apply_determinism_env` on ``os.environ``, and flip
  ``torch.use_deterministic_algorithms(True)``.
* :func:`assert_nccl_algo_still_empty` — after the first NCCL process group,
  check that PyTorch left an empty ``NCCL_ALGO`` alone.
"""

from __future__ import annotations

import ctypes
import os
from typing import MutableMapping

import torch

# Explicit: `import torch` does not reliably bind the `torch.utils.deterministic`
# submodule, and this module writes to one of its attributes.
import torch.utils.deterministic

# Maps each arg name to the value it must hold for bit-exact execution;
# verified by :func:`apply_determinism_to_args`.
ARG_VALUES_REQUIRED_FOR_DETERMINISM = {
    "cross_entropy_loss_fusion": False,
    "tp_comm_overlap": False,
}

# Not in the dict above because it inherits: unset means "follow moe_router_fusion".
# TE's fused aux-loss kernel is non-deterministic: on identical input it returns a
# different aux loss run to run, while the unfused path is bit-identical. The fused TopK
# routing has no such report against it, so it is not required off.
AUX_LOSS_FUSION_ARG = "moe_router_aux_loss_fusion"

# Env-var defaults required for bit-exact reproducibility.
DETERMINISM_ENV_VAR_DEFAULTS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    # NCCL EP's count-exchange path assigns expert slots with atomic counters.
    # Scan mode preserves token order, and therefore expert wgrad reduction order.
    # NCCL EP snapshots this setting when its group is created.
    "NCCL_EP_HT_EM_AG_SCAN_MODE": "1",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    # TRITON_CACHE_AUTOTUNING is deliberately absent: deterministic mode pins every
    # autotuner in the megatron.core.tuning scope without timing, and the variable only
    # changes Triton's behaviour for kernels outside it. See apply_determinism_env().
}

# Accepted NCCL_ALGO tokens under deterministic mode. Comma-separated lists
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
# An empty ``NCCL_ALGO`` is accepted as well and lets NCCL choose; see
# :func:`apply_determinism_env`.
#
# ``NVLS`` is accepted only together with ``Ring`` and never on Hopper; see
# :func:`apply_determinism_env` and :func:`apply_determinism_to_args`.
ACCEPTED_NCCL_ALGO_TOKENS: frozenset[str] = frozenset(
    {"Ring", "CollnetDirect", "CollnetChain", "^NVLS", "NVLS"}
)

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
#   - ``TRITON_CACHE_AUTOTUNING``: Triton tests it as ``== "1"``, so any other
#     truthy spelling ("true", "yes") would silently read as opted out. It only
#     affects autotuners outside the pinned ``megatron.core.tuning`` scope, so
#     neither setting is defaulted -- see :func:`apply_determinism_env`.
#   - ``NCCL_EP_HT_EM_AG_SCAN_MODE``: require the canonical enabled value to
#     preserve expert token order instead of using atomic slot assignment.
ACCEPTED_ENV_VAR_VALUES: dict[str, frozenset[str]] = {
    "NCCL_EP_HT_EM_AG_SCAN_MODE": frozenset({"1"}),
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": frozenset({"0"}),
    "CUBLAS_WORKSPACE_CONFIG": frozenset({":4096:8", ":16:8"}),
    "TRITON_CACHE_AUTOTUNING": frozenset({"0", "1"}),
}


def apply_determinism_env(env: MutableMapping[str, str]) -> None:
    """Validate every determinism env var in ``env``, then setdefault the canonical values.

    Semantics per key:

    * ``NCCL_ALGO`` — if set, each comma-separated token must be in
      :data:`ACCEPTED_NCCL_ALGO_TOKENS`, and ``NVLS`` needs ``Ring`` in the same
      list. The empty string is accepted and kept: NCCL (and a tuner plugin, if
      one is loaded) then chooses the algorithms. A value of only whitespace is
      rejected.
    * ``NCCL_EP_HT_EM_AG_SCAN_MODE`` / ``NVTE_ALLOW_NONDETERMINISTIC_ALGO`` /
      ``CUBLAS_WORKSPACE_CONFIG`` —
      if set, must be in :data:`ACCEPTED_ENV_VAR_VALUES`.
    * ``MAMBA_DETERMINISTIC`` / ``CAUSAL_CONV1D_DETERMINISTIC`` — if set
      (non-empty), must start with ``'1'``; unset, the external ``mamba_ssm`` and
      ``causal_conv1d`` packages auto-follow
      :func:`torch.are_deterministic_algorithms_enabled`. Megatron's own Triton
      kernels do not read them; they follow deterministic mode.
    * ``TRITON_CACHE_AUTOTUNING`` — opt-in; if set to ``'1'``, requires
      ``TRITON_CACHE_DIR``. Autotuners inside the ``--triton-autotune-modules``
      scope are pinned without timing either way; the cache only applies to
      autotuners outside that scope.

    After validation, ``setdefault`` fills every key in
    :data:`DETERMINISM_ENV_VAR_DEFAULTS` that has not been set — a value the
    caller has already set wins.

    These env vars are captured by their respective libraries at first use
    (NCCL at communicator init, cuBLAS at handle creation, TE at first
    attention forward), so the call must happen BEFORE any of those events.
    """
    # NCCL_ALGO subset check. The empty string is an explicit request to let NCCL choose:
    # PyTorch counts it as set and so does not write NCCL_ALGO=^NVLS under
    # torch.use_deterministic_algorithms(True), and NCCL 2.24 and later exclude no algorithm
    # for the empty string.
    nccl_algo = env.get("NCCL_ALGO")
    if nccl_algo not in (None, ""):
        assert nccl_algo.strip(), (
            f"NCCL_ALGO={nccl_algo!r} contains only whitespace, which NCCL 2.24 and later fail "
            "to parse. Set NCCL_ALGO to the empty string to let NCCL choose the algorithms, or "
            f"to a subset of {sorted(ACCEPTED_NCCL_ALGO_TOKENS)}, or disable deterministic mode "
            "with --disable-deterministic-mode."
        )
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert "NVLS" not in tokens or "Ring" in tokens, (
            f"NCCL_ALGO={nccl_algo!r}: NVLS needs Ring in the list, because broadcast and "
            "reduce run only with Ring. Use 'Ring,NVLS', or disable deterministic mode with "
            "--disable-deterministic-mode."
        )
        assert tokens and all(t in ACCEPTED_NCCL_ALGO_TOKENS for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r}: each token must be in "
            f"{sorted(ACCEPTED_NCCL_ALGO_TOKENS)}. Or disable deterministic mode with "
            "--disable-deterministic-mode."
        )

    # Exact-match env vars: reject only if the caller supplied a value we
    # haven't validated as deterministic; unset is fine.
    for name, accepted in ACCEPTED_ENV_VAR_VALUES.items():
        val = env.get(name)
        assert val is None or val in accepted, (
            f"{name}={val!r} is not a deterministic setting. Accepted: {sorted(accepted)}. "
            "Or disable deterministic mode with --disable-deterministic-mode."
        )

    # Mamba SSM and causal_conv1d auto-follow torch when unset; only reject an
    # explicit non-deterministic override.
    for name in ("MAMBA_DETERMINISTIC", "CAUSAL_CONV1D_DETERMINISTIC"):
        value = env.get(name)
        if value:
            assert value[0] == "1", (
                f"{name}={value!r} disables SSM determinism, which deterministic mode requires. "
                "Unset it or set to '1', or disable deterministic mode with "
                "--disable-deterministic-mode."
            )

    # Cross-field rule, so it cannot go in ACCEPTED_ENV_VAR_VALUES: caching only makes ranks
    # agree if they share one cache, and unset TRITON_CACHE_DIR means a node-local one.
    if env.get("TRITON_CACHE_AUTOTUNING") == "1":
        assert env.get("TRITON_CACHE_DIR"), (
            "TRITON_CACHE_AUTOTUNING=1 in deterministic mode requires TRITON_CACHE_DIR "
            "(a shared-filesystem path). It only affects autotuners outside the pinned "
            "--triton-autotune-modules scope; add their modules to that scope to pin them "
            "instead, unset TRITON_CACHE_AUTOTUNING, or disable deterministic mode with "
            "--disable-deterministic-mode."
        )

        # print() because this runs from validate_args, before logging is configured.
        print(
            "Deterministic mode: TRITON_CACHE_AUTOTUNING=1 only applies to autotuners outside "
            "the pinned scope; pinned kernels ignore the cache and TRITON_PRINT_AUTOTUNING. "
            "Pass --triton-autotune-enumerate-autotuners to list which autotuners are pinned and "
            "--triton-autotune-verify-every N to compare the choices ranks made.",
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
       ``NCCL_EP_HT_EM_AG_SCAN_MODE``, ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``,
       ``CUBLAS_WORKSPACE_CONFIG``,
       ``MAMBA_DETERMINISTIC``, ``CAUSAL_CONV1D_DETERMINISTIC``,
       ``TRITON_CACHE_AUTOTUNING`` and its required ``TRITON_CACHE_DIR``) and
       setdefaults the canonical values. If ``NCCL_ALGO`` is empty or has
       ``NVLS`` among its tokens, the run is rejected on Hopper; elsewhere, for
       an empty value, rank 0 warns that NCCL chooses the algorithms.
    3. Calls ``torch.use_deterministic_algorithms(True)``, then clears
       ``torch.utils.deterministic.fill_uninitialized_memory``, which that call
       turns on and which reproducibility does not need.

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

    # Mirrors the TransformerConfig.__post_init__ fallback; no config exists yet here.
    aux_loss_fusion = getattr(args, AUX_LOSS_FUSION_ARG, None)
    if aux_loss_fusion is None:
        aux_loss_fusion = getattr(args, "moe_router_fusion", False)
    if aux_loss_fusion:
        mismatched.append(f"{AUX_LOSS_FUSION_ARG}=False (got {aux_loss_fusion!r})")

    assert (
        not mismatched
    ), (
        f"Deterministic mode requires: {', '.join(mismatched)}. Adjust these options to "
        "continue, or disable deterministic mode with --disable-deterministic-mode."
    )

    # --use-flash-attn is intentionally NOT rejected: TE's FlashAttention is
    # deterministic on supported configs and is covered by the bit-exact
    # correctness suite.

    # Delegate env-var validation + setdefault to the single-source helper.
    apply_determinism_env(os.environ)
    nccl_algo = os.environ.get("NCCL_ALGO")
    if nccl_algo == "" or "NVLS" in [t.strip() for t in (nccl_algo or "").split(",")]:
        from megatron.training.utils import get_device_arch_version

        assert get_device_arch_version() != 9, (
            f"NCCL_ALGO={nccl_algo!r} in deterministic mode can let NCCL use NVLS, which "
            "may not be reproducible on Hopper unless the kernel driver and Fabric Manager are "
            "550.144.03 or later in the 550 branch, or 570 or later (the 560 branch has no fix); "
            "their versions are not checked here (NVIDIA/nccl#2360). Unset NCCL_ALGO to use the "
            "Ring default, or disable deterministic mode with --disable-deterministic-mode."
        )
    if nccl_algo == "":
        from megatron.training.utils import warn_rank_0

        warn_rank_0(
            "Deterministic mode: NCCL_ALGO is empty, so NCCL chooses the algorithm for each "
            "collective. With NCCL >= 2.24 the candidates include NVLS, where available, and "
            "Tree, which is not an accepted NCCL_ALGO token. Bit-exact reproducibility "
            'then rests on what NCCL selects; see "Letting NCCL choose the algorithm" in '
            "docs/user-guide/deterministic-training.md."
        )

    # Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)

    # Reproducibility comes from the independent output buffer that replaces an
    # unordered atomic accumulation and fixes the summation order. That stays.
    #
    # The line above also fills every uninitialized allocation (torch.empty,
    # empty_like, empty_strided, Tensor.resize_) with NaN/MAX_INT, pinning what a
    # kernel would read from memory it never wrote. Reproducibility does not need
    # that: with no padding feeding the computation, results are bit-identical
    # either way.
    #
    # It costs a kernel launch per empty allocation, serialized between real work
    # — roughly 15% TFLOP/s on large configs. Set back to True only to debug a
    # suspected uninitialized-memory read.
    torch.utils.deterministic.fill_uninitialized_memory = False


def assert_nccl_algo_still_empty() -> None:
    """No-op unless NCCL_ALGO was set empty. Call after the first NCCL process group.

    PyTorch writes NCCL_ALGO from C++, which ``os.environ`` does not see, so the value is
    read back through libc.
    """
    if os.environ.get("NCCL_ALGO") != "":
        return
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    libc.getenv.argtypes = [ctypes.c_char_p]
    actual = libc.getenv(b"NCCL_ALGO")
    assert actual == b"", (
        f'NCCL_ALGO was "" but is {actual!r} after process-group creation, so '
        "PyTorch's NVLS exclusion applied anyway. Unset NCCL_ALGO to use the Ring default, "
        "or disable deterministic mode with --disable-deterministic-mode."
    )
