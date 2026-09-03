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



def make_dense_cp_cu_seqlens_cpu(batch: int, *, total_tokens_per_sequence: int) -> torch.Tensor:
    """Return synthetic dense-batch host offsets for internal CP preprocessing."""
    return torch.arange(batch + 1, dtype=torch.long) * total_tokens_per_sequence


def get_trusted_cp_cu_seqlens_cpu(
    packed_seq_params: object, cu_seqlens: torch.Tensor, *, total_tokens: int
) -> torch.Tensor:
    """Return the prebuilt THD CP host offsets needed by internal CP preprocessing."""
    from megatron.core.context_parallel_layout.routes import _get_trusted_thd_cp_cu_seqlens_cpu

    cpu_offsets = _get_trusted_thd_cp_cu_seqlens_cpu(packed_seq_params, cu_seqlens)
    if int(cpu_offsets[-1]) != total_tokens:
        raise ValueError(
            "GDN: trusted cu_seqlens_q endpoint does not match "
            f"total_sequence_length={total_tokens}."
        )
    return cpu_offsets


def prepare_cp_context_metadata(
    context: object, global_cu_seqlens_cpu: torch.Tensor, *, config: object | None = None
) -> object:
    """Attach internal CP preprocessing metadata to an FLA CP context."""
    if global_cu_seqlens_cpu.device.type != "cpu":
        raise ValueError("Internal GDN CP context metadata requires CPU cu_seqlens offsets.")
    source_version = global_cu_seqlens_cpu._version
    if (
        getattr(context, "_cutedsl_global_offsets_owner", None) is global_cu_seqlens_cpu
        and getattr(context, "_cutedsl_global_offsets_owner_version", None) == source_version
    ):
        context._cutedsl_cuda_graph_enabled = _gdn_is_cuda_graphed(config)
        return context

    global_offsets = global_cu_seqlens_cpu.to(device="cpu", dtype=torch.long).clone()
    context.global_num_seqs = len(global_offsets) - 1
    context.global_cu_seqlens_cpu = global_offsets
    context._cutedsl_global_offsets_owner = global_cu_seqlens_cpu
    context._cutedsl_global_offsets_owner_version = source_version
    context._cutedsl_metadata_generation = getattr(context, "_cutedsl_metadata_generation", 0) + 1
    context._cutedsl_chain_memo = {}
    context._cutedsl_window_memo = {}
    context._cutedsl_cuda_graph_enabled = _gdn_is_cuda_graphed(config)
    return context


def prepare_validated_chunk_metadata(
    cu_seqlens: torch.Tensor | None,
    *,
    include_chunk_indices: bool = True,
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
