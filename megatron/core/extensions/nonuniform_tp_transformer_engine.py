# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Transformer Engine adapter helpers for nonuniform tensor parallelism."""

from contextlib import contextmanager
from typing import Iterable, Optional, Sequence, Tuple

TPDomains = Tuple[Tuple[int, ...], ...]


def normalize_tp_domains(tp_domains: Sequence[Sequence[int]]) -> TPDomains:
    """Return deterministic, validated TP domains for all ranks.

    Transformer Engine userbuffer initialization creates one process group per TP domain.
    Every rank must create those groups in the same order, so callers may pass domains in
    any order and this helper normalizes them by first rank.
    """
    normalized = []
    seen_ranks = set()

    for domain in tp_domains:
        domain_tuple = tuple(int(rank) for rank in domain)
        if not domain_tuple:
            raise ValueError("NTP TP domains must not be empty")
        if len(set(domain_tuple)) != len(domain_tuple):
            raise ValueError(f"NTP TP domain contains duplicate ranks: {domain_tuple}")
        overlap = seen_ranks.intersection(domain_tuple)
        if overlap:
            raise ValueError(f"NTP TP domains overlap on ranks: {sorted(overlap)}")
        seen_ranks.update(domain_tuple)
        normalized.append(domain_tuple)

    if not normalized:
        raise ValueError("At least one NTP TP domain is required")

    return tuple(sorted(normalized, key=lambda domain: (domain[0], len(domain), domain)))


def _subgroup_arg(
    args: Tuple[object, ...], kwargs: dict, index: int, name: str, default: object = None
) -> object:
    if name in kwargs:
        return kwargs[name]
    if len(args) > index:
        return args[index]
    return default


def _new_group_kwargs(args: Tuple[object, ...], kwargs: dict, domain_index: int) -> dict:
    timeout = _subgroup_arg(args, kwargs, 0, "timeout")
    backend = _subgroup_arg(args, kwargs, 1, "backend")
    pg_options = _subgroup_arg(args, kwargs, 2, "pg_options")
    use_local_synchronization = _subgroup_arg(args, kwargs, 3, "use_local_synchronization", False)
    group_desc = _subgroup_arg(args, kwargs, 4, "group_desc")

    new_group_kwargs = {}
    if timeout is not None:
        new_group_kwargs["timeout"] = timeout
    if backend is not None:
        new_group_kwargs["backend"] = backend
    if pg_options is not None:
        new_group_kwargs["pg_options"] = pg_options
    if use_local_synchronization:
        new_group_kwargs["use_local_synchronization"] = use_local_synchronization
    if group_desc is not None:
        new_group_kwargs["group_desc"] = f"{group_desc}_ntp_{domain_index}"
    return new_group_kwargs


@contextmanager
def transformer_engine_userbuffer_tp_domains(
    tp_domains: Sequence[Sequence[int]],
    *,
    distributed: Optional[object] = None,
    tp_group: Optional[object] = None,
):
    """Use explicit TP domains while Transformer Engine initializes userbuffers.

    Transformer Engine currently accepts a scalar ``tp_size`` and derives TP domains by
    chunking its bootstrap process group. NTP needs mixed-size TP domains, so this context
    manager redirects TE's no-ranks bootstrap group to the current NTP TP group and keeps
    subgroup enumeration on the caller-provided domains for TE versions or paths that need it.
    """
    if distributed is None:
        import torch.distributed as distributed  # type: ignore[no-redef]

    domains = normalize_tp_domains(tp_domains)
    rank = distributed.get_rank()
    if not any(rank in domain for domain in domains):
        raise RuntimeError(f"Rank {rank} is not present in any NTP TP domain: {domains}")

    original_new_group = distributed.new_group
    original_new_subgroups = distributed.new_subgroups_by_enumeration

    def ntp_new_group(*args, **kwargs):
        ranks = kwargs.get("ranks")
        if ranks is None and args:
            ranks = args[0]
        if ranks is None and tp_group is not None:
            return tp_group
        return original_new_group(*args, **kwargs)

    def ntp_new_subgroups_by_enumeration(
        _ranks_per_subgroup_list: Iterable[Iterable[int]], *args, **kwargs
    ):
        current_group = None
        groups = []
        for domain_index, domain in enumerate(domains):
            group = distributed.new_group(
                ranks=list(domain), **_new_group_kwargs(args, kwargs, domain_index)
            )
            groups.append(group)
            if rank in domain:
                current_group = group

        if current_group is None:
            raise RuntimeError(f"Rank {rank} did not get an NTP TP domain")
        return current_group, groups

    distributed.new_group = ntp_new_group
    distributed.new_subgroups_by_enumeration = ntp_new_subgroups_by_enumeration
    try:
        yield domains
    finally:
        distributed.new_group = original_new_group
        distributed.new_subgroups_by_enumeration = original_new_subgroups


def initialize_transformer_engine_userbuffers_for_nonuniform_tp(
    *,
    shape: Sequence[int],
    tp_size: int,
    tp_domains: Sequence[Sequence[int]],
    bootstrap_backend: str,
    tp_group: Optional[object] = None,
    **initialize_ub_kwargs,
) -> TPDomains:
    """Initialize Transformer Engine userbuffers on explicit NTP TP domains."""
    try:
        from transformer_engine.pytorch import module as te_module
    except ImportError as exc:
        raise RuntimeError("NTP TP communication overlap requires Transformer Engine") from exc

    if tp_group is None:
        from megatron.core import parallel_state

        tp_group = parallel_state.get_tensor_model_parallel_group()

    with transformer_engine_userbuffer_tp_domains(
        tp_domains, tp_group=tp_group
    ) as normalized_domains:
        te_module.base.initialize_ub(
            shape=list(shape),
            tp_size=tp_size,
            bootstrap_backend=bootstrap_backend,
            **initialize_ub_kwargs,
        )
    return normalized_domains
