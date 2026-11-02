# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Per-operation selection table for fused mHC (``mhc_fused_ops``).

Pure Python without megatron imports at run time, so that TransformerConfig validation and
the HyperConnectionModule guard share one table without an import cycle. The helpers return
reasons; each caller writes its own message around them.
"""

from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from megatron.core.transformer.transformer_config import TransformerConfig

# The operations mhc_fused_ops can select, in the order they are validated, bound and logged.
MHC_FUSED_OPS = ("proj_rms", "sinkhorn", "h_aggregate", "h_post_bda")

# The mHC precision flags. The all-operations use_fused_mhc mode rejects every one of them.
MHC_PRECISION_FLAGS = (
    "mhc_norm_eps_inside_sqrt",
    "mhc_normalize_before_projection",
    "mhc_keep_mappings_in_fp32",
    "mhc_keep_pre_in_fp32",
    "mhc_preserve_activation_dtype_rounding",
)

# Flags an operation's per-op implementation does not implement. proj_rms here is the
# projection-first fused op (fused_proj_rms_compute_h), which implements neither GLM norm
# flag. With mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection together (HF
# GLM5Next's order), proj_rms selects the Triton GLM projection pair instead, which only
# replaces the FP32 projection and RMS normalization and so honours every flag; either GLM
# norm flag alone has no per-op implementation. h_post_bda rejects the rounding flag because
# the fused post computes in FP32 and rounds once.
_UNSUPPORTED_FLAGS = {
    "proj_rms": (
        "mhc_norm_eps_inside_sqrt",
        "mhc_normalize_before_projection",
        "mhc_keep_mappings_in_fp32",
        "mhc_keep_pre_in_fp32",
    ),
    "sinkhorn": (),
    "h_aggregate": ("mhc_keep_mappings_in_fp32", "mhc_keep_pre_in_fp32"),
    "h_post_bda": ("mhc_keep_mappings_in_fp32", "mhc_preserve_activation_dtype_rounding"),
}


def glm_projection_order(config: "TransformerConfig") -> bool:
    """Return True for HF GLM5Next's projection: eps inside the sqrt, normalize first."""
    return bool(config.mhc_norm_eps_inside_sqrt and config.mhc_normalize_before_projection)


def triton_only_ops(config: "TransformerConfig") -> Tuple[str, ...]:
    """Return the operations whose per-op implementation is Triton-only.

    The fixed-launch Sinkhorn always is; so is the GLM projection pair that proj_rms selects
    with mhc_norm_eps_inside_sqrt and mhc_normalize_before_projection.
    """
    return ("proj_rms", "sinkhorn") if glm_projection_order(config) else ("sinkhorn",)


def selects_glm_projection_pair(config: "TransformerConfig") -> bool:
    """Return True when per-op proj_rms selects the Triton GLM projection pair."""
    return bool(
        config.use_fused_mhc
        and config.mhc_fused_ops is not None
        and "proj_rms" in config.mhc_fused_ops
        and glm_projection_order(config)
    )


def mhc_fused_ops_field_errors(config: "TransformerConfig") -> List[str]:
    """Return the shape errors of mhc_fused_ops and mhc_fused_tf32_rounding, in any mode.

    Callers check these before ``mhc_fused_ops_errors``, which needs a well-formed list: a
    string would match substrings there, and a non-list would raise TypeError.
    """
    errors = []
    ops = config.mhc_fused_ops
    if ops is not None:
        if not config.use_fused_mhc:
            errors.append("mhc_fused_ops requires use_fused_mhc=True")
        # A plain string would otherwise be read as a sequence of one-letter entries.
        if not isinstance(ops, list):
            errors.append(
                f"mhc_fused_ops must be a list of operation names, got {type(ops).__name__}"
            )
        elif not ops:
            errors.append(
                "mhc_fused_ops must name at least one operation; "
                "set use_fused_mhc=False to fuse none"
            )
        else:
            # The dataclass Literal is not enforced at run time.
            unknown = [op for op in ops if op not in MHC_FUSED_OPS]
            if unknown:
                errors.append(
                    f"unknown mhc_fused_ops entries {unknown}; expected entries of {MHC_FUSED_OPS}"
                )
            repeated = []
            for index, op in enumerate(ops):
                if op in ops[:index] and op not in repeated:
                    repeated.append(op)
            if repeated:
                errors.append(f"mhc_fused_ops repeats {repeated}")

    rounding = config.mhc_fused_tf32_rounding
    if rounding not in ("rn", "rz"):
        errors.append(f"unknown mhc_fused_tf32_rounding {rounding!r}; expected 'rn' or 'rz'")
    elif rounding == "rz" and not errors and not selects_glm_projection_pair(config):
        # Reached only for a well-formed selection: the pair check reads mhc_fused_ops.
        errors.append(
            "mhc_fused_tf32_rounding='rz' requires use_fused_mhc=True and 'proj_rms' in "
            "mhc_fused_ops with mhc_norm_eps_inside_sqrt=True and "
            "mhc_normalize_before_projection=True"
        )
    return errors


def mhc_fused_ops_errors(config: "TransformerConfig") -> List[str]:
    """Return the per-op table's errors and the v1 limits of per-op mode.

    Call only in per-op mode (``use_fused_mhc`` with ``mhc_fused_ops`` set), when
    ``mhc_fused_ops_field_errors`` finds nothing, and after ``cuda_graph_impl`` has absorbed
    the deprecated enable_cuda_graph and external_cuda_graph flags.
    """
    errors = []
    triton_only = triton_only_ops(config)
    for op in MHC_FUSED_OPS:
        if op not in config.mhc_fused_ops:
            continue
        if op in triton_only:
            if config.mhc_fused_backend == "cutile":
                errors.append(
                    f"{op} runs a Triton-only per-op implementation, which "
                    "mhc_fused_backend='cutile' excludes"
                )
            continue
        errors.extend(
            f"{op} does not implement {flag}"
            for flag in _UNSUPPORTED_FLAGS[op]
            if getattr(config, flag)
        )

    # Per-op mode is validated in this scope only. Anything outside it fails closed rather
    # than silently running a different configuration.
    if config.num_residual_streams != 4:
        errors.append(
            f"num_residual_streams={config.num_residual_streams}: "
            "per-op mode is validated at 4 only"
        )
    if config.mhc_recompute_attn_cuda_graph_split:
        errors.append("mhc_recompute_attn_cuda_graph_split is untested with per-op mode")
    if config.cuda_graph_impl == "transformer_engine":
        errors.append("Transformer Engine per-layer CUDA graphs are untested with per-op mode")
    return errors
