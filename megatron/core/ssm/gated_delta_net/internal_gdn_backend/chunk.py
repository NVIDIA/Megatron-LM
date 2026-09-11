# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLA-compatible public interface for the internal chunked GDR backend."""

from functools import lru_cache
from typing import Callable

import torch

from megatron.core.transformer.enums import CudaGraphModule


@lru_cache(maxsize=1)
def _load_internal_chunk_gated_delta_rule() -> (
    Callable[..., tuple[torch.Tensor, torch.Tensor | None]]
):
    """Load FLA and the in-tree orchestration only when this backend is used."""
    try:
        from .implementation import chunk_gated_delta_rule
    except ImportError as exc:
        raise RuntimeError(
            "The internal GDR backend requires the flash-linear-attention dependency "
            "from the Megatron Core development environment."
        ) from exc
    return chunk_gated_delta_rule


@lru_cache(maxsize=1)
def _load_internal_prepare_validated_chunk_metadata() -> (
    Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]
):
    """Load the metadata helper only when the internal backend needs it."""
    try:
        from .implementation import prepare_validated_chunk_metadata
    except ImportError as exc:
        raise RuntimeError(
            "The internal GDR backend requires the flash-linear-attention dependency "
            "from the Megatron Core development environment."
        ) from exc
    return prepare_validated_chunk_metadata


def _gdn_is_cuda_graphed(config) -> bool:
    """Return whether this GDN attention region is included in CUDA-graph capture."""
    impl = getattr(config, "cuda_graph_impl", "none")
    if impl == "none":
        return False
    if impl == "full_iteration":
        return True
    modules = getattr(config, "cuda_graph_modules", None)
    return not modules or CudaGraphModule.attn in modules


_GLOBAL_CU_SEQLENS_UNSET = object()


def _logical_global_cu_seqlens_cpu(cu_seqlens: torch.Tensor | None) -> torch.Tensor | None:
    """Return a detached CPU copy of logical global packed offsets."""
    if cu_seqlens is None:
        return None
    if cu_seqlens.dim() != 1:
        raise ValueError("Internal GDN CP global cu_seqlens must be one-dimensional.")

    cpu = cu_seqlens.detach()
    if cpu.device.type != "cpu":
        cpu = cpu.to(device="cpu")
    if cpu.dtype != torch.long:
        cpu = cpu.to(dtype=torch.long)
    if cpu.numel() < 2:
        raise ValueError("Internal GDN CP global cu_seqlens requires at least two entries.")

    logical_entries = cpu.numel()
    final_offset = int(cpu[-1])
    while logical_entries > 1 and int(cpu[logical_entries - 2]) == final_offset:
        logical_entries -= 1
    cpu = cpu[:logical_entries].clone()

    if int(cpu[0]) != 0:
        raise ValueError("Internal GDN CP global cu_seqlens must start at zero.")
    if bool(((cpu[1:] - cpu[:-1]) <= 0).any()):
        raise ValueError("Internal GDN CP global cu_seqlens must be strictly increasing.")
    return cpu


def _same_cpu_tensor_values(lhs: torch.Tensor | None, rhs: torch.Tensor | None) -> bool:
    if lhs is None or rhs is None:
        return lhs is rhs
    return lhs.shape == rhs.shape and lhs.dtype == rhs.dtype and torch.equal(lhs, rhs)


def prepare_cp_context_metadata(
    context: object,
    *,
    config: object | None = None,
    global_num_sequences: int | None = None,
    global_cu_seqlens: object = _GLOBAL_CU_SEQLENS_UNSET,
) -> object:
    """Attach internal CP preprocessing hints to an FLA CP context."""
    if global_cu_seqlens is _GLOBAL_CU_SEQLENS_UNSET:
        global_cu_seqlens_cpu = getattr(context, "global_cu_seqlens_cpu", None)
    else:
        global_cu_seqlens_cpu = _logical_global_cu_seqlens_cpu(global_cu_seqlens)

    if global_cu_seqlens_cpu is not None:
        inferred_num_sequences = int(global_cu_seqlens_cpu.numel() - 1)
        if global_num_sequences is None:
            global_num_sequences = inferred_num_sequences
        elif int(global_num_sequences) != inferred_num_sequences:
            raise ValueError(
                "Internal GDN CP global_num_sequences does not match global cu_seqlens."
            )

    if global_num_sequences is not None:
        global_num_sequences = int(global_num_sequences)
        if global_num_sequences < 1:
            raise ValueError("Internal GDN CP metadata requires at least one sequence.")

    metadata_changed = (
        getattr(context, "global_num_seqs", None) != global_num_sequences
        or not _same_cpu_tensor_values(
            getattr(context, "global_cu_seqlens_cpu", None), global_cu_seqlens_cpu
        )
    )
    if metadata_changed:
        context.global_num_seqs = global_num_sequences
        context.global_cu_seqlens_cpu = global_cu_seqlens_cpu
        context._cutedsl_metadata_generation = (
            getattr(context, "_cutedsl_metadata_generation", 0) + 1
        )
        context._cutedsl_chain_memo = {}
        context._cutedsl_window_memo = {}
    context._cutedsl_cuda_graph_enabled = _gdn_is_cuda_graphed(config)
    return context


def prepare_validated_chunk_metadata(
    cu_seqlens: torch.Tensor | None, *, include_chunk_indices: bool = True
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Prepare trusted packed metadata before entering the profiled GDR region."""
    helper = _load_internal_prepare_validated_chunk_metadata()
    return helper(cu_seqlens, include_chunk_indices=include_chunk_indices)


def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    state_v_first: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    validated_chunk_indices: torch.LongTensor | None = None,
    validated_chunk_offsets: torch.LongTensor | None = None,
    cp_context: object | None = None,
    recompute_h: bool = False,
    **kwargs: object,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run the in-tree CuTe DSL GDR path with FLA fallback."""
    if use_beta_sigmoid_in_kernel:
        raise ValueError("The internal GDR backend does not support in-kernel beta sigmoid.")
    if allow_neg_eigval:
        raise ValueError("The internal GDR backend does not support negative eigenvalues.")

    transpose_state_layout = kwargs.pop("transpose_state_layout", None)
    if transpose_state_layout is not None:
        if state_v_first and not bool(transpose_state_layout):
            raise ValueError("state_v_first conflicts with transpose_state_layout=False.")
        state_v_first = bool(transpose_state_layout)

    implementation = _load_internal_chunk_gated_delta_rule()
    return implementation(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        validated_chunk_indices=validated_chunk_indices,
        validated_chunk_offsets=validated_chunk_offsets,
        cp_context=cp_context,
        recompute_h=recompute_h,
        **kwargs,
    )
