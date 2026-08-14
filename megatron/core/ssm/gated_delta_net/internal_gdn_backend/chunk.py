# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLA-compatible public interface for the internal chunked GDR backend."""

from functools import lru_cache
from typing import Callable

import torch


@lru_cache(maxsize=1)
def _load_internal_chunk_gated_delta_rule() -> Callable[
    ..., tuple[torch.Tensor, torch.Tensor | None]
]:
    """Load FLA and the in-tree orchestration only when this backend is used."""
    try:
        from .implementation import chunk_gated_delta_rule
    except ImportError as exc:
        raise RuntimeError(
            "The internal GDR backend requires the flash-linear-attention dependency "
            "from the Megatron Core development environment."
        ) from exc
    return chunk_gated_delta_rule


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
    cu_seqlens_cpu: torch.LongTensor | None = None,
    cp_context: object | None = None,
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
        cu_seqlens_cpu=cu_seqlens_cpu,
        cp_context=cp_context,
        **kwargs,
    )
