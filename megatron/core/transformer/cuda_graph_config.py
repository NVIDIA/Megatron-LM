# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, NamedTuple, Optional, Set, Tuple, Union

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


class NormalizedCudaGraphModules(NamedTuple):
    """Result of normalize_cuda_graph_modules."""

    scopes: List[CudaGraphModule]
    """Fully resolved CudaGraphModule enum values."""
    deprecated: List[Tuple[str, str, object]]
    """List of (scope_name, attr_name, new_value) for each deprecated scope that was found."""
    used_full_scope: bool
    """True if the input contained the deprecated 'full' shorthand."""


def normalize_cuda_graph_modules(
    scopes: Optional[Union[str, CudaGraphModule, List[Union[str, CudaGraphModule]]]]
) -> NormalizedCudaGraphModules:
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
        return NormalizedCudaGraphModules([], [], True)

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

    return NormalizedCudaGraphModules(normalized_scopes, deprecated_scopes, False)


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
