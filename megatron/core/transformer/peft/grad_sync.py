# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from NVIDIA-NeMo/Megatron-Bridge (Apache-2.0).
# Source: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/
# c031e61d82dafbe4655a2e2d7e742b88f7e81c3a/src/megatron/bridge/peft/utils.py

"""Gradient synchronization for expert-replicated adapter parameters."""

import logging
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from megatron.core.distributed import finalize_model_grads
from megatron.core.process_groups_config import ProcessGroupCollection

from .process_groups import _get_process_group, _process_group_size

logger = logging.getLogger(__name__)
_EXPERT_PARALLEL_REPLICATED_ATTR = "expert_parallel_replicated"
_EXPERT_PARALLEL_GRAD_HOOK_ATTR = "_expert_parallel_grad_hook"


def _broadcast_shared_expert_weights(weights: list[torch.Tensor], ep_group: object | None) -> None:
    """Broadcast shared expert adapter weights from EP rank zero so every EP rank starts bitwise identical."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if not weights or ep_group is None or _process_group_size(ep_group) <= 1:
        return

    src_rank = torch.distributed.get_global_rank(ep_group, 0)
    with torch.no_grad():
        for weight in weights:
            if weight.is_meta:
                raise RuntimeError(
                    "Shared expert adapter parameters must be materialized before EP synchronization"
                )
            if weight.is_cuda or torch.distributed.get_backend(ep_group) != "nccl":
                torch.distributed.broadcast(weight, src=src_rank, group=ep_group)
                continue
            # NCCL cannot broadcast CPU tensors; stage through the current device.
            staged_weight = weight.to(torch.device("cuda", torch.cuda.current_device()))
            torch.distributed.broadcast(staged_weight, src=src_rank, group=ep_group)
            weight.copy_(staged_weight.cpu())


def mark_expert_parallel_replicated(weight: torch.Tensor, *, ep_group: object | None) -> None:
    """Tag ``weight`` as replicated across expert parallelism so its gradient is summed over EP every step.

    MCore's DDP reduces ``is_expert`` parameters over expert-DP only;
    :func:`allreduce_expert_parallel_replicated_grads` adds the EP sum once per step. Until
    :func:`enable_expert_parallel_grad_sync_in_finalize` removes it, a per-microbatch fallback hook sums eagerly
    computed gradients across EP so training loops that never install the finalize wrapper keep working.

    Args:
        weight: Parameter replicated across the EP group.
        ep_group: Expert-model-parallel process group, or ``None`` when expert parallelism is not set up.
    """

    setattr(weight, _EXPERT_PARALLEL_REPLICATED_ATTR, True)
    if ep_group is None or _process_group_size(ep_group) <= 1 or not weight.requires_grad:
        return

    def _allreduce_grad_across_ep(grad: torch.Tensor) -> torch.Tensor:
        if getattr(weight, "grad_added_to_main_grad", False):
            # MCore accumulated the real gradient into ``main_grad`` and handed autograd a dummy tensor.
            logger.warning(
                "Fused weight-gradient accumulation put an expert-parallel replicated adapter gradient into "
                "main_grad, which the per-layer fallback hook cannot synchronize. Install "
                "finalize_model_grads_with_expert_adapter_sync as finalize_model_grads_func; otherwise the shared "
                "adapter weights drift apart across EP ranks."
            )
            return grad
        # Sum across EP first; MCore's expert DDP then reduces across expert-DP and scales expert buffers by
        # 1 / dp_cp_group.size(), i.e. the full EP x expert-DP data-parallel world.
        grad = grad.contiguous()
        torch.distributed.all_reduce(grad, op=torch.distributed.ReduceOp.SUM, group=ep_group)
        return grad

    setattr(
        weight, _EXPERT_PARALLEL_GRAD_HOOK_ATTR, weight.register_hook(_allreduce_grad_across_ep)
    )


def _expert_parallel_replicated_params(
    model: nn.Module | list[nn.Module],
) -> Iterator[nn.Parameter]:
    """Yield the parameters tagged by :func:`mark_expert_parallel_replicated`."""

    for model_chunk in model if isinstance(model, list) else [model]:
        if isinstance(model_chunk, nn.Module):
            yield from (
                p
                for p in model_chunk.parameters()
                if getattr(p, _EXPERT_PARALLEL_REPLICATED_ATTR, False)
            )


def _remove_fallback_hook(param: nn.Parameter) -> bool:
    """Remove the per-layer fallback hook from ``param``; return whether one was still installed."""

    handle = getattr(param, _EXPERT_PARALLEL_GRAD_HOOK_ATTR, None)
    if handle is None:
        return False
    handle.remove()
    delattr(param, _EXPERT_PARALLEL_GRAD_HOOK_ATTR)
    return True


def enable_expert_parallel_grad_sync_in_finalize(model: nn.Module | list[nn.Module]) -> int:
    """Remove the per-layer fallback hooks so :func:`allreduce_expert_parallel_replicated_grads` owns the EP sum.

    Bridge's training setup calls this once the model is wrapped.

    Args:
        model: Model chunk or list of model chunks, DDP-wrapped or bare.

    Returns:
        Number of tagged parameters.
    """

    params = list(_expert_parallel_replicated_params(model))
    for param in params:
        _remove_fallback_hook(param)
    return len(params)


def allreduce_expert_parallel_replicated_grads(
    model: nn.Module | list[nn.Module], *, ep_group: object | None = None
) -> None:
    """Sum the tagged parameters' gradients across the EP group once per step, coalesced per dtype.

    Call it after the data-parallel gradient sync (after Megatron-Core's ``finalize_model_grads``), the point
    where MCore also sums sequence-parallel layernorm gradients across TP. It reads ``main_grad`` under DDP and
    ``.grad`` otherwise, so fused and eager weight-gradient accumulation give the same result.

    Args:
        model: Model chunk or list of model chunks, DDP-wrapped or bare.
        ep_group: Explicit expert-model-parallel group of the adapted model.
    """

    params = list(_expert_parallel_replicated_params(model))
    if not params or not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return
    if ep_group is None:
        raise ValueError("Expert adapter gradient synchronization requires ep_group")
    if ep_group is None or _process_group_size(ep_group) <= 1:
        return

    grads_by_dtype: dict[tuple[torch.dtype, torch.device], list[torch.Tensor]] = {}
    for param in params:
        hook_was_active = _remove_fallback_hook(param)
        if not param.requires_grad:
            continue
        if hook_was_active and not getattr(param, "grad_added_to_main_grad", False):
            # The fallback hook already summed this step's eagerly computed gradient.
            continue
        grad = getattr(param, "main_grad", None)
        if grad is None:
            grad = param.grad
        if grad is None:
            continue
        # Megatron-FSDP keeps gradients as DTensors; reduce the local shard like MCore's finalize step does.
        grad = getattr(grad, "_local_tensor", grad).data
        grads_by_dtype.setdefault((grad.dtype, grad.device), []).append(grad)

    for grads in grads_by_dtype.values():
        if len(grads) == 1:
            torch.distributed.all_reduce(
                grads[0], op=torch.distributed.ReduceOp.SUM, group=ep_group
            )
            continue
        coalesced = _flatten_dense_tensors(grads)
        torch.distributed.all_reduce(coalesced, op=torch.distributed.ReduceOp.SUM, group=ep_group)
        for grad, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads), strict=True):
            grad.copy_(synced)


def finalize_model_grads_with_expert_adapter_sync(
    model: list[nn.Module],
    num_tokens: torch.Tensor | None = None,
    pg_collection: ProcessGroupCollection | None = None,
    **kwargs: object,
) -> None:
    """Run Megatron-Core's ``finalize_model_grads``, then :func:`allreduce_expert_parallel_replicated_grads`.

    Drop-in replacement for ``config.finalize_model_grads_func``. Bridge's training setup installs it; training
    loops that call ``finalize_model_grads`` themselves should call this instead.

    Args:
        model: List of model chunks.
        num_tokens: Optional global token count used by MCore for per-token loss scaling.
        pg_collection: Process groups for MCore's finalize step; its ``ep`` group drives the EP sum.
        **kwargs: Forwarded to ``finalize_model_grads`` (for example ``force_all_reduce``).
    """

    finalize_model_grads(model, num_tokens, pg_collection=pg_collection, **kwargs)
    allreduce_expert_parallel_replicated_grads(
        model, ep_group=_get_process_group(pg_collection, "ep")
    )
