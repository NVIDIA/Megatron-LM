# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Prepare full THD packs and modality routes using the existing DCP scheduler."""

from dataclasses import dataclass
from typing import Any, Hashable, Mapping, Optional, Sequence

import torch
import torch.distributed as dist

from megatron.core.datasets.data_schedule import (
    BasePackingScheduler,
    _build_thd_padding_mask,
    _sanitize_thd_padding_values,
)
from megatron.core.datasets.data_schedule_utils import (
    build_packed_microbatches,
    pad_packed_batch_before_cp_slice,
)
from megatron.core.model_parallel_config import ModelParallelConfig
from megatron.core.models.mimo.comm.pack_bridge import FeatureSlice, PackFeaturePlan
from megatron.core.models.mimo.planning import build_round_plans


@dataclass(frozen=True)
class MimoRoundPlan:
    """One local full pack and the transport plan shared by all decoder ranks."""

    sample_ids: tuple[Hashable, ...]
    feature_plans: tuple[PackFeaturePlan, ...]
    cp_group: dist.ProcessGroup
    model_inputs: dict[str, Any]


@dataclass(frozen=True)
class MimoStepPlan:
    """Prepared rounds and source counts, independent of host input-bin accounting."""

    rounds: tuple[MimoRoundPlan, ...]
    num_source_samples: int
    num_supervised_tokens: int
    source_tokens: int
    source_tokens_squared: int


def prepare_mimo_step(
    samples: Mapping[Hashable, Mapping[str, Any]],
    feature_slices: Mapping[Hashable, FeatureSlice],
    *,
    scheduler: BasePackingScheduler,
    domain_ranks: Sequence[int],
    cp_groups: Mapping[int, dist.ProcessGroup],
    config: ModelParallelConfig,
    device: torch.device,
    tp_cp_groups: Optional[Mapping[int, dist.ProcessGroup]] = None,
) -> MimoStepPlan:
    """Plan colocated TP1/PP1 multimodal packs without executing either model.

    Source ownership is **replicated**: every transport rank must have the same
    ordered mapping of complete source samples and media descriptors. Rank-local
    datasets must materialize those inputs before calling this function; the
    text-only scheduler reroute cannot transport media metadata. The function
    validates the local contract and selected groups, but does not communicate
    or compare source tensors across ranks.

    Each sample provides one-dimensional ``tokens``, ``labels``, ``loss_mask``
    of ``padded_seq_len`` elements, ``original_seq_len``, true ``position_ids``
    of shape ``[S]`` or ``[axes, S]``, and ordered ``media_ids``. Positions may
    include physical padding, which is discarded. Stable sample/media IDs need
    not be integers; scheduler-local indices are private to this preparation.

    ``cp_groups`` maps each allowed CP size to this rank's pre-created group,
    including a singleton for CP1. For TP1, TP x CP defaults to that same group.
    Returned inputs are full packs: splice features before calling the model's
    unified CP partition. No encoder, processor, dataset, process-group creation,
    or training configuration mutation occurs here.
    """
    if config.tensor_model_parallel_size != 1 or config.pipeline_model_parallel_size != 1:
        raise ValueError("Colocated MIMO preparation currently requires TP=PP=1")
    domain_ranks = tuple(domain_ranks)
    rank = dist.get_rank()
    if len(set(domain_ranks)) != len(domain_ranks) or rank not in domain_ranks:
        raise ValueError("The decoder domain must contain this rank exactly once")
    if not samples:
        raise ValueError("A source step requires at least one sample")
    ids = tuple(samples)
    text_samples, lengths = {}, []
    position_shape = None
    supervised = 0
    for index, sid in enumerate(ids):
        sample = samples[sid]
        logical, physical = int(sample["original_seq_len"]), int(sample["padded_seq_len"])
        if not 0 < logical <= physical:
            raise ValueError("Source lengths must satisfy 0 < original_seq_len <= padded_seq_len")
        tensors = {}
        for name in ("tokens", "labels", "loss_mask"):
            tensor = sample[name]
            if tensor.ndim != 1 or tensor.numel() != physical:
                raise ValueError(f"{name} must be one-dimensional with padded_seq_len elements")
            tensors[name] = tensor.to(device=device)
        positions = sample["position_ids"]
        if positions.ndim not in (1, 2) or positions.shape[-1] not in (logical, physical):
            raise ValueError("position_ids must have shape [S] or [axes, S] for this sample")
        axes = positions.shape[:-1]
        if position_shape is not None and axes != position_shape:
            raise ValueError("All source samples must use the same position axes")
        position_shape = axes
        # The shared builder handles one-dimensional token fields. Restore true
        # multi-axis positions below after physical padding has reached its final offsets.
        tensors["position_ids"] = torch.zeros(physical, device=device, dtype=torch.long)
        tensors["original_seq_len"] = torch.tensor(logical, device=device, dtype=torch.int32)
        tensors["padded_seq_len"] = torch.tensor(physical, device=device, dtype=torch.int32)
        text_samples[index] = tensors
        lengths.append((index, physical))
        supervised += int(sample["loss_mask"][:logical].sum().item())

    assignments = scheduler.get_groups_and_subsamples(lengths)
    scheduled = [
        index
        for assignment in assignments
        for pack in dict.fromkeys(tuple(indices) for indices in assignment)
        for index in pack
    ]
    if sorted(scheduled) != list(range(len(ids))):
        raise ValueError("The schedule must contain each source sample exactly once")
    local_rank = domain_ranks.index(rank)
    # Validate schedules and groups before building any GPU packs.
    routes = []
    for assignment in assignments:
        stable_assignment = [[ids[index] for index in pack] for pack in assignment]
        plans = build_round_plans(stable_assignment, samples, feature_slices, domain_ranks)
        local = next(plan for plan in plans if rank in plan.cp_ranks)
        group = cp_groups.get(len(local.cp_ranks))
        if group is None or tuple(dist.get_process_group_ranks(group)) != local.cp_ranks:
            raise ValueError("The selected runtime CP group must match the feature-plan ranks")
        tp_cp = group if tp_cp_groups is None else tp_cp_groups.get(len(local.cp_ranks))
        if tp_cp is None or tuple(dist.get_process_group_ranks(tp_cp)) != local.cp_ranks:
            raise ValueError("TP1 requires a matching runtime TP x CP group")
        routes.append((plans, group, tp_cp))

    batches = build_packed_microbatches(
        text_samples, assignments, local_rank, device, scheduler.is_dynamic_cp
    )
    rounds = []
    for assignment, packed, (plans, group, tp_cp) in zip(assignments, batches, routes):
        packed["padding_mask"] = _build_thd_padding_mask(
            packed["cu_seqlens"], packed["cu_seqlens_padded"]
        )
        _sanitize_thd_padding_values(packed, packed["padding_mask"])
        pad_packed_batch_before_cp_slice(packed, config, group.size(), 1)
        positions = torch.zeros(
            (*position_shape, packed["tokens"].numel()), device=device, dtype=torch.long
        )
        sample_ids = tuple(ids[index] for index in assignment[local_rank])
        for slot, sid in enumerate(sample_ids):
            start = int(packed["cu_seqlens_padded"][slot])
            length = int(samples[sid]["original_seq_len"])
            positions[..., start : start + length] = samples[sid]["position_ids"][..., :length].to(
                device=device
            )
        inputs = {
            "input_ids": packed["tokens"].unsqueeze(0),
            "labels": packed["labels"].unsqueeze(0),
            "loss_mask": packed["loss_mask"].unsqueeze(0),
            "position_ids": positions.unsqueeze(-2),
            "padding_mask": packed["padding_mask"].unsqueeze(0),
            "cu_seqlens": packed["cu_seqlens"],
            "cu_seqlens_padded": packed["cu_seqlens_padded"],
            "max_seqlen": packed["max_seqlen"],
            "cp_group": group,
            "tp_cp_group": tp_cp,
        }
        rounds.append(MimoRoundPlan(sample_ids, plans, group, inputs))
    return MimoStepPlan(
        tuple(rounds),
        len(samples),
        supervised,
        sum(length for _, length in lengths),
        sum(length * length for _, length in lengths),
    )
