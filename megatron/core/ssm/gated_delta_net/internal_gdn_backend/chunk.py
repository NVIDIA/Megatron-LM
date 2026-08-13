# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLA-compatible public interface for the internal chunked GDR backend."""

import torch


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
    """Run the internal chunked gated delta rule implementation.

    The signature intentionally matches FLA's chunk_gated_delta_rule API so callers can switch
    backends without reshaping inputs or translating recurrence options. Internal low-level kernels
    should live in the sibling kernels package and be composed behind this stable entry point.

    Raises:
        NotImplementedError: Until the internal kernel implementation is added.
    """
    raise NotImplementedError(
        "The internal chunked gated delta rule API is available, but its kernels have not been "
        "added yet."
    )
