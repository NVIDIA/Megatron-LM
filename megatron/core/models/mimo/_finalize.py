# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Gradient finalization for colocated MimoModel training.

A colocated :class:`MimoModel` is a ``MegatronModule`` whose language model and
modality submodules are *separately* wrapped in ``DistributedDataParallel`` over
*different* data-parallel groups (the encoder and LLM can run different DP
factorizations under fan-in / fan-out). The stock
:func:`megatron.core.distributed.finalize_model_grads.finalize_model_grads`
expects a list of DDP chunks and a single ``pg_collection``; called on a bare
``MimoModel`` it raises ``AttributeError`` (``MimoModel`` has no
``finish_grad_sync`` / ``scale_gradients``) and it cannot reduce two submodules
over two different DP groups.

:func:`finalize_mimo_grads` dispatches finalization to each DDP submodule with
its own ``pg_collection``. Wire it into ``config.finalize_model_grads_func`` for
colocated runs (see :func:`colocated_forward_backward_with_pp`).
"""

from typing import Optional

import torch
import torch.distributed as dist

from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection


def finalize_mimo_grads(
    mimo_model,
    num_tokens: Optional[torch.Tensor],
    language_pg: ProcessGroupCollection,
    vision_pg: ProcessGroupCollection,
    force_all_reduce: bool = False,
) -> None:
    """Finalize gradients for a colocated MimoModel's two DDP submodules.

    Mirrors :func:`finalize_model_grads` but dispatches per submodule with its
    own process-group collection, so the encoder (over ``vision_pg``) and the
    LLM (over ``language_pg``) are each reduced over the correct DP group.

    When ``num_tokens`` is provided (``calculate_per_token_loss=True``), it is the
    LLM-local valid-token sum (the loss runs LLM-side). It is reduced to a single
    global divisor ``N_global`` over the LLM PP+DP groups and applied *uniformly*
    to both submodules, so the encoder and LLM see the same normalization. When
    ``num_tokens`` is ``None`` (the per-microbatch-mean default), each submodule's
    DDP applies its own ``1/dp_size`` scaling and no extra divide is performed.

    Args:
        mimo_model: The colocated ``MimoModel`` (not DDP-wrapped itself).
        num_tokens: LLM-local token-count tensor, or ``None``.
        language_pg: Process-group collection for the language model's DDP.
        vision_pg: Process-group collection for the modality submodules' DDP.
        force_all_reduce: Forwarded to per-submodule ``finalize_model_grads``.
    """
    submodules = []  # list[(ddp_chunk, pg_collection)]
    if mimo_model.language_model is not None:
        submodules.append((mimo_model.language_model, language_pg))
    for submodule in mimo_model.modality_submodules.values():
        if submodule is not None:
            submodules.append((submodule, vision_pg))

    # Reduce the LLM-local token count to a single global divisor (over the LLM
    # PP+DP groups) so both submodules normalize identically. The encoder's own
    # DP group would yield a different count under DP mismatch.
    n_global = None
    if num_tokens is not None:
        pp = getattr(language_pg, 'pp', None)
        if pp is not None and pp.size() > 1:
            last_rank = dist.get_global_rank(pp, pp.size() - 1)
            dist.broadcast(num_tokens, src=last_rank, group=pp)
        dp = (
            language_pg.dp_cp if getattr(language_pg, 'dp_cp', None) is not None else language_pg.dp
        )
        dist.all_reduce(num_tokens, group=dp, op=dist.ReduceOp.SUM)
        n_global = num_tokens.item()

    # Per-side DDP finish with no built-in num_tokens scaling.
    for chunk, pg in submodules:
        finalize_model_grads(
            [chunk], num_tokens=None, pg_collection=pg, force_all_reduce=force_all_reduce
        )

    # Uniform divide by the global token count (guard the fully-masked batch).
    if n_global is not None and n_global > 0:
        inv = 1.0 / n_global
        for chunk, _ in submodules:
            chunk.scale_gradients(inv)
