# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared startup policy for Megatron Core and training-library callers.

Call :func:`configure_determinism` before CUDA initialization, process-group
creation, or backend first use. Configuration alone is not replay evidence.
Seeding, data order, checkpoint state and topology remain caller responsibilities.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, MutableMapping

import torch
import torch.utils.deterministic

logger = logging.getLogger(__name__)
_configured_pid: int | None = None
_configured_environment: dict[str, str | None] | None = None

# Maps each arg name to the value it must hold for bit-exact execution;
# verified by :func:`validate_determinism_config`.
ARG_VALUES_REQUIRED_FOR_DETERMINISM = {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}

# Not in the dict above because it inherits: unset means "follow moe_router_fusion".
# TE's fused aux-loss kernel is non-deterministic: on identical input it returns a
# different aux loss run to run, while the unfused path is bit-identical. The fused TopK
# routing has no such report against it, so it is not required off.
AUX_LOSS_FUSION_ARG = "moe_router_aux_loss_fusion"

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
#   - ``Ring``          the existing default; the physical ring still depends on topology.
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
ACCEPTED_NCCL_ALGO_TOKENS: frozenset[str] = frozenset(
    {"Ring", "CollnetDirect", "CollnetChain", "^NVLS"}
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
        if not tokens or not all(t in ACCEPTED_NCCL_ALGO_TOKENS for t in tokens):
            raise AssertionError(
                f"NCCL_ALGO={nccl_algo!r}: each token must be in "
                f"{sorted(ACCEPTED_NCCL_ALGO_TOKENS)}."
            )

    # Exact-match env vars: reject only if the caller supplied a value we
    # haven't validated as deterministic; unset is fine.
    for name, accepted in ACCEPTED_ENV_VAR_VALUES.items():
        val = env.get(name)
        if val is not None and val not in accepted:
            raise AssertionError(
                f"{name}={val!r} is not a deterministic setting. Accepted: {sorted(accepted)}."
            )

    # Mamba SSM and causal_conv1d auto-follow torch when unset; only reject an
    # explicit non-deterministic override.
    for name in ("MAMBA_DETERMINISTIC", "CAUSAL_CONV1D_DETERMINISTIC"):
        value = env.get(name)
        if value and value[0] != "1":
            raise AssertionError(
                f"{name}={value!r} disables SSM determinism under "
                "--deterministic-mode. Unset it or set to '1'."
            )

    # Cross-field rule, so it cannot go in ACCEPTED_ENV_VAR_VALUES: caching only makes ranks
    # agree if they share one cache, and unset TRITON_CACHE_DIR means a node-local one.
    if env.get("TRITON_CACHE_AUTOTUNING") == "1":
        if not env.get("TRITON_CACHE_DIR"):
            raise AssertionError(
                "TRITON_CACHE_AUTOTUNING=1 under --deterministic-mode requires TRITON_CACHE_DIR "
                "(a shared-filesystem path); unset TRITON_CACHE_AUTOTUNING to use the "
                "deterministic pinned-config fallback instead."
            )

        # Recommended, not required: changes no numerics, only visibility.
        if not env.get("TRITON_PRINT_AUTOTUNING"):
            logger.info(
                "Deterministic mode: set TRITON_PRINT_AUTOTUNING=1 to log the kernel config "
                "each rank selects. A cache miss re-times the selection on that rank alone, "
                "which is how ranks come to disagree; without this the miss leaves no record."
            )

    # setdefault preserves any launcher-set value that just passed validation.
    for k, v in DETERMINISM_ENV_VAR_DEFAULTS.items():
        env.setdefault(k, v)


def validate_determinism_config(config: object) -> dict:
    """Validate a model config, argparse Namespace or mapping without changing it.

    ``deterministic_mode`` must be true. Missing optional fusion settings use
    their MCore defaults. Validation uses explicit exceptions so ``python -O``
    cannot disable it. The returned dictionary contains the checked options.
    """
    read = (
        config.get
        if isinstance(config, Mapping)
        else lambda key, default: getattr(config, key, default)
    )
    if read("deterministic_mode", False) is not True:
        raise AssertionError(
            "configure_determinism requires deterministic_mode=True in the model config"
        )
    # Verification only — read each option's effective value and never flip it,
    # so a default that drifts to a bad value breaks the run instead of silently
    # running non-deterministically.
    mismatched = [
        f"{name}={required!r} (got {actual!r})"
        for name, required in ARG_VALUES_REQUIRED_FOR_DETERMINISM.items()
        if (actual := read(name, required)) != required
    ]

    # Mirrors the TransformerConfig.__post_init__ fallback; no config exists yet here.
    aux_loss_fusion = read(AUX_LOSS_FUSION_ARG, None)
    if aux_loss_fusion is None:
        aux_loss_fusion = read("moe_router_fusion", False)
    if aux_loss_fusion:
        mismatched.append(f"{AUX_LOSS_FUSION_ARG}=False (got {aux_loss_fusion!r})")

    if mismatched:
        raise AssertionError(
            f"--deterministic-mode requires: {', '.join(mismatched)}. "
            "Adjust these options to continue."
        )

    # --use-flash-attn is intentionally NOT rejected: TE's FlashAttention is
    # deterministic on supported configs and is covered by the bit-exact
    # correctness suite.

    return {
        "deterministic_mode": True,
        **ARG_VALUES_REQUIRED_FOR_DETERMINISM,
        AUX_LOSS_FUSION_ARG: False,
    }


def _environment_signature(env: Mapping[str, str]) -> dict[str, str | None]:
    keys = set(DETERMINISM_ENV_VAR_DEFAULTS) | {
        "MAMBA_DETERMINISTIC",
        "CAUSAL_CONV1D_DETERMINISTIC",
        "TRITON_CACHE_AUTOTUNING",
        "TRITON_CACHE_DIR",
        "CUDA_DEVICE_MAX_CONNECTIONS",
        "NCCL_PROTO",
    }
    keys.update(key for key in env if key.startswith("TRITON_AUTOTUNE_BLOCK_"))
    return {key: env.get(key) for key in sorted(keys)}


def _torch_policy_is_active() -> bool:
    return (
        torch.are_deterministic_algorithms_enabled()
        and not torch.is_deterministic_algorithms_warn_only_enabled()
        and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.benchmark
    )


def configure_determinism(config: object) -> dict:
    """Apply process-wide policy before CUDA or distributed initialization.

    Args:
        config: A ModelParallelConfig/TransformerConfig, Namespace, or mapping
            with ``deterministic_mode=True`` and the effective fusion/overlap
            settings. A mapping lets callers apply policy before constructing a
            config whose validation itself queries CUDA.

    Returns:
        A JSON-serializable effective policy, also logged at INFO. This is a
        settings record, not a determinism certificate. It does not seed RNGs
        or verify caches, hardware topology, or the caller's training state.

    Raises:
        AssertionError: An incompatible config/environment setting was supplied.
        RuntimeError: First setup is late, or policy changed after CUDA or a
            process group was initialized. Repeat calls are allowed only when
            this process applied the same policy before initialization.
    """
    global _configured_pid, _configured_environment
    options = validate_determinism_config(config)
    proposed = dict(os.environ)
    apply_determinism_env(proposed)
    # Pin external SSM libraries before they can cache an environment lookup.
    for name in ("MAMBA_DETERMINISTIC", "CAUSAL_CONV1D_DETERMINISTIC"):
        proposed.setdefault(name, "1")
    environment = _environment_signature(proposed)
    initialized = torch.cuda.is_initialized() or (
        torch.distributed.is_available() and torch.distributed.is_initialized()
    )
    if initialized:
        if (
            _configured_pid != os.getpid()
            or _configured_environment != environment
            or _environment_signature(os.environ) != environment
            or not _torch_policy_is_active()
        ):
            raise RuntimeError(
                "Determinism policy must be configured before CUDA or process-group "
                "initialization; start a fresh process and call configure_determinism first."
            )
    else:
        for key, value in environment.items():
            if value is not None:
                os.environ[key] = value
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        _configured_pid = os.getpid()
        _configured_environment = environment.copy()

    # Preserve the training allocation policy when sharing startup with libraries.
    # This diagnostic fill is separate from deterministic algorithm selection.
    # Callers can re-enable it after setup to investigate uninitialized reads.
    torch.utils.deterministic.fill_uninitialized_memory = False

    policy = {
        "schema_version": 1,
        "environment": environment,
        "options": options,
        "torch": {
            "version": str(torch.__version__),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
        },
        "scope": (
            "Startup settings only; caller owns seeding, data order, state and replay validation"
        ),
    }
    logger.info("Determinism policy: %s", json.dumps(policy, sort_keys=True))
    return policy
