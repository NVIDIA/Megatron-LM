# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import List, Optional, Set, Tuple, Union

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


PACKED_DSA_CP_CUDA_GRAPH_ERROR = (
    "CUDA graph capture is not supported for this packed DSA context-parallel "
    "configuration: the requested capture path cannot preserve or reconstruct the "
    "per-sequence CP layout during warmup and replay. Full CUDA Graph support for "
    "packed DSA+CP is deferred; set cuda_graph_impl='none'. Other unguarded capture "
    "combinations remain experimental unless separately validated."
)


def is_packed_dsa_cp_cuda_graph_capture_unsupported(
    *,
    experimental_attention_variant: Optional[str],
    dsa_kernel_backend: str,
    sequence_packing_scheduler: Optional[str],
    dynamic_context_parallel: bool,
    context_parallel_size: int,
    cuda_graph_impl: str,
    cuda_graph_modules: Optional[List[CudaGraphModule]],
    inference_cuda_graph_scope: InferenceCudaGraphScope,
) -> bool:
    """Return whether a packed DSA+CP graph combination is proven unsupported.

    Keep this predicate limited to configurations covered by GPU evidence. The
    static validation matrix exercised configured CP=2 and CP=4; the dynamic
    matrix exercised configured CP=2 only. Combined local attention+MLP,
    attention+Mamba, and TE attention+MLP scopes were measured only at static
    CP=2. CP=3 failed during model construction before eager preflight or graph
    capture, so it is not graph-specific evidence and remains outside this
    guard. Dynamic CP with configured CP=1 is likewise not inferred from the
    DP x CP pool because that neighboring topology did not reach capture.
    Unmeasured MoE/Mamba partial scopes and full-iteration capture also remain
    available until they are measured.
    """

    static_packing = sequence_packing_scheduler == "dp_balanced" and not dynamic_context_parallel
    dynamic_packing = (
        sequence_packing_scheduler == "default_dynamic_cp" and dynamic_context_parallel
    )
    if experimental_attention_variant != "dsa":
        return False
    if not (static_packing or dynamic_packing):
        return False

    # Capture scope execution uses membership checks, so repeated spellings such
    # as ``attn,attn`` are the same effective scope as ``attn``.
    modules = frozenset(cuda_graph_modules or [])
    if dsa_kernel_backend == "cudnn":
        if cuda_graph_impl == "local":
            measured_scopes = set()
            if static_packing and context_parallel_size in (2, 4):
                measured_scopes.update(
                    {
                        frozenset(),
                        frozenset({CudaGraphModule.attn}),
                        frozenset({CudaGraphModule.mlp}),
                    }
                )
                if context_parallel_size == 2:
                    measured_scopes.update(
                        {
                            frozenset({CudaGraphModule.attn, CudaGraphModule.mlp}),
                            frozenset({CudaGraphModule.attn, CudaGraphModule.mamba}),
                        }
                    )
            elif dynamic_packing and context_parallel_size == 2:
                measured_scopes.update(
                    {
                        frozenset(),
                        frozenset({CudaGraphModule.attn}),
                        frozenset({CudaGraphModule.mlp}),
                    }
                )

            if modules not in measured_scopes:
                return False
            if not modules:
                # ``block`` owns an inference-only TransformerBlock graph and
                # skips the per-layer whole-scope manager exercised by the
                # training validation matrix.
                return inference_cuda_graph_scope == InferenceCudaGraphScope.layer
            return True
        if cuda_graph_impl == "transformer_engine":
            measured_scopes = set()
            if static_packing and context_parallel_size in (2, 4):
                measured_scopes.update({frozenset(), frozenset({CudaGraphModule.attn})})
                if context_parallel_size == 2:
                    measured_scopes.add(frozenset({CudaGraphModule.attn, CudaGraphModule.mlp}))
            elif dynamic_packing and context_parallel_size == 2:
                measured_scopes.update({frozenset(), frozenset({CudaGraphModule.attn})})
            return modules in measured_scopes

    # The unfused scorer was measured only for static attention-only local
    # capture. TileLang was unavailable in the validation image, and the
    # remaining backend-none scopes were not measured, so keep them available.
    if (
        dsa_kernel_backend == "none"
        and static_packing
        and context_parallel_size == 2
        and cuda_graph_impl == "local"
    ):
        return modules == {CudaGraphModule.attn}
    return False


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
