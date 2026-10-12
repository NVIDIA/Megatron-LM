# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Sequence, Set, Tuple, Union

from megatron.core.transformer.enums import CudaGraphModule, InferenceCudaGraphScope

# Maps deprecated scope strings to the (attr_name, new_value) they should set.
# new_value is the actual value to assign: a str for cuda_graph_impl (which is a
# Literal string type) or an InferenceCudaGraphScope enum for inference_cuda_graph_scope.
CUDA_GRAPH_MODULES_DEPRECATIONS = {
    'full_iteration': ('cuda_graph_impl', 'full_iteration'),
    'full_iteration_inference': ('inference_cuda_graph_scope', InferenceCudaGraphScope.block),
}

# Canonical mapping from cuda_graph_impl to the set of allowed inference granularities.
# Shared by transformer_config.__post_init__ and validate_args to avoid duplication.
ALLOWED_INFERENCE_SCOPES: dict[str, Set[InferenceCudaGraphScope]] = {
    "none": {InferenceCudaGraphScope.none},
    "local": {InferenceCudaGraphScope.layer, InferenceCudaGraphScope.block},
    "transformer_engine": {InferenceCudaGraphScope.none},
    "full_iteration": {InferenceCudaGraphScope.none},
}


def is_whole_moe_cuda_graph_scope(cuda_graph_modules: Sequence[CudaGraphModule]) -> bool:
    """Whether a per-layer CUDA graph scope captures the complete MoE module.

    An empty normalized scope represents whole-layer capture, while an explicit
    ``CudaGraphModule.moe`` captures the complete MoE submodule.
    """

    return not cuda_graph_modules or CudaGraphModule.moe in cuda_graph_modules


def validate_moe_cuda_graph_support(config) -> None:
    """Validate backend support when the capture includes a whole-MoE module."""

    if (
        config.num_moe_experts is None
        or config.num_moe_experts <= 1
        or not is_whole_moe_cuda_graph_scope(config.cuda_graph_modules)
        or (
            config.moe_expert_capacity_factor is not None
            and config.moe_pad_expert_input_to_capacity
        )
    ):
        return

    assert (
        config.cuda_graph_impl == "transformer_engine"
        and config.moe_token_dispatcher_type == "flex"
        and config.moe_flex_dispatcher_backend == "hybridep"
        and config.moe_expert_rank_capacity_factor is not None
        and config.moe_paged_stash
        and config.use_transformer_engine_op_fuser
    ), (
        "moe cuda graph is only supported with drop-padding MoE or transformer_engine "
        "sync-free HybridEP with rank capacity and paged stash."
    )


def cuda_graph_captures_attention(config) -> bool:
    """Return whether the normalized training graph scope includes attention."""
    impl = getattr(config, "cuda_graph_impl", "none")
    modules = getattr(config, "cuda_graph_modules", None)
    return impl == "full_iteration" or (
        impl in ("local", "transformer_engine") and (not modules or CudaGraphModule.attn in modules)
    )


def normalize_cuda_graph_modules(
    scopes: Optional[Union[str, CudaGraphModule, List[Union[str, CudaGraphModule]]]],
) -> Tuple[List[CudaGraphModule], List[Tuple[str, str, object]], bool]:
    """Normalize mixed CUDA graph scope inputs into enum values plus deprecation metadata."""

    if scopes is None:
        raw_scopes = []
    elif isinstance(scopes, list):
        raw_scopes = list(scopes)
    elif isinstance(scopes, str):
        raw_scopes = scopes.split(',') if scopes else []
    else:
        raw_scopes = [scopes]

    if "full" in raw_scopes:
        assert raw_scopes == ["full"], "full scope cannot be used with other scopes."
        return [], [], True

    normalized_scopes: List[CudaGraphModule] = []
    deprecated_scopes: List[Tuple[str, str, object]] = []
    for scope in raw_scopes:
        if isinstance(scope, CudaGraphModule):
            normalized_scopes.append(scope)
        else:
            assert isinstance(scope, str), (
                "cuda_graph_modules values must be strings or CudaGraphModule enums, "
                f"got {scope!r}."
            )
            if scope in CUDA_GRAPH_MODULES_DEPRECATIONS:
                attr, value = CUDA_GRAPH_MODULES_DEPRECATIONS[scope]
                deprecated_scopes.append((scope, attr, value))
            else:
                normalized_scopes.append(CudaGraphModule[scope])

    return normalized_scopes, deprecated_scopes, False


def normalize_inference_cuda_graph_scope(
    scope: Optional[Union[str, InferenceCudaGraphScope]], cuda_graph_impl: str
) -> InferenceCudaGraphScope:
    """Normalize inference CUDA graph scope and apply the impl-derived default."""

    if scope is None:
        if cuda_graph_impl == "local":
            return InferenceCudaGraphScope.layer
        return InferenceCudaGraphScope.none

    if isinstance(scope, InferenceCudaGraphScope):
        return scope

    assert isinstance(scope, str), (
        "inference_cuda_graph_scope must be a string or "
        f"InferenceCudaGraphScope enum, got {scope!r}."
    )
    return InferenceCudaGraphScope[scope]


def validate_deprecated_cuda_graph_modules_migration_inputs(
    deprecated_scopes: List[Tuple[str, str, object]],
    cuda_graph_impl: str,
    inference_cuda_graph_scope: Optional[Union[str, InferenceCudaGraphScope]],
) -> None:
    """Reject ambiguous mixed old/new CUDA graph inputs before applying migration.

    Deprecated scope strings are still accepted for compatibility, but only when they are not
    combined with conflicting new-style fields.
    """

    deprecated_scope_names = [scope for scope, _, _ in deprecated_scopes]
    if not deprecated_scope_names:
        return

    if len(set(deprecated_scope_names)) > 1:
        raise AssertionError(
            "cuda_graph_modules cannot contain multiple deprecated values at the same time: "
            f"{deprecated_scope_names!r}."
        )

    scope = deprecated_scope_names[0]
    if isinstance(inference_cuda_graph_scope, str):
        inference_cuda_graph_scope = InferenceCudaGraphScope[inference_cuda_graph_scope]

    if scope == "full_iteration":
        assert cuda_graph_impl in ("none", "local", "full_iteration"), (
            "cuda_graph_modules='full_iteration' cannot be combined with "
            f"cuda_graph_impl={cuda_graph_impl!r}."
        )
        assert inference_cuda_graph_scope in (None, InferenceCudaGraphScope.none), (
            "cuda_graph_modules='full_iteration' cannot be combined with "
            "inference_cuda_graph_scope="
            f"{getattr(inference_cuda_graph_scope, 'name', inference_cuda_graph_scope)!r}."
        )
    elif scope == "full_iteration_inference":
        assert cuda_graph_impl in ("none", "local"), (
            "cuda_graph_modules='full_iteration_inference' cannot be combined with "
            f"cuda_graph_impl={cuda_graph_impl!r}."
        )
        assert inference_cuda_graph_scope in (None, InferenceCudaGraphScope.block), (
            "cuda_graph_modules='full_iteration_inference' cannot be combined with "
            "inference_cuda_graph_scope="
            f"{getattr(inference_cuda_graph_scope, 'name', inference_cuda_graph_scope)!r}."
        )


def get_deprecated_cuda_graph_modules_migration(
    scope: str, attr: str, value: object, cuda_graph_impl: str
) -> Optional[Tuple[str, object]]:
    """Return the effective new-style migration for a deprecated cuda_graph_modules value."""

    if scope == "full_iteration_inference" and cuda_graph_impl == "none":
        return None
    return attr, value
