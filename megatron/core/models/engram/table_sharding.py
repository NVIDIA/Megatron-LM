# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Process groups of the Engram tables.

The rows of every Engram table are sharded over a row group and replicated over a replica
group, in one of two layouts:

* EP (``EngramConfig.parallel_size`` unset): the MoE expert-parallel group and expert-DP. The
  tables are expert parameters (``allreduce=False``), reduced and optimized with the experts.
* Sharded (``parallel_size = P``): each pipeline stage's dp_cp group, read in rank order as a
  [replica, row shard] mesh (``create_row_parallel_groups``). Rows are sharded over ``P``
  consecutive dp_cp ranks and replicated over the ranks at the same position of the other
  shards. The tables form the ``'engram'`` parameter domain: DDP reduces their gradients over
  the replica group, and a DistributedOptimizer of their own shards their state over it.

With TP = expert TP = 1 and no GTP, a stage's dp_cp group holds its EP x expert-DP grid, and
every EP group is a contiguous block of it. ``P = size(dp_cp)`` then gives every row a single
owner, and ``P = size(EP)`` reproduces the EP layout's row ranges, initialization seeds and
checkpoint shards.

The EP layout looks rows up with NCCL all-to-alls over the row group. The sharded layout uses
the symmetric device transport by default when ``parallel_size`` is above 1
(``EngramConfig.communication_backend``, see ``symmetric_lookup``), or the same NCCL
all-to-alls as a numerical reference.
:func:`initialize_engram_parallel_state` runs on every rank before any Engram layer is built;
the Engram module then takes its groups from :func:`resolve_engram_groups`.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.distributed import ProcessGroup

from megatron.core.process_groups_config import ProcessGroupCollection, create_row_parallel_groups
from megatron.core.utils import get_pg_size, log_single_rank

from . import symmetric_lookup
from .config import COMMUNICATION_BACKENDS, EngramConfig
from .distributed_embedding import EPShardedMultiTableEmbedding

logger = logging.getLogger(__name__)

ENGRAM_PARAMETER_DOMAIN = "engram"
"""Parameter domain of the sharded tables (see ``get_parameter_parallel_domain``)."""

# Domain of the tables in the EP layout, where they are expert parameters.
_EP_LAYOUT_DOMAIN = "expert"

# Prefix of the group descriptions, ENGRAM_SHARD and ENGRAM_REPLICA.
_GROUP_NAME = "ENGRAM"

# Row and replica groups by (parent dp_cp group, parallel size). Every rank creates them in the
# same call, so every rank holds the same keys and reuses them in the same later calls.
_ROW_GROUPS: Dict[Tuple[ProcessGroup, int], Tuple[ProcessGroup, ProcessGroup]] = {}


def initialize_engram_parallel_state(
    pg_collection: ProcessGroupCollection,
    engram_config: Optional[EngramConfig],
    *,
    timeout: Optional[timedelta] = None,
) -> None:
    """Prepare the process groups of the Engram tables before the model is built.

    Collective over the default group, except for the EP layout with the NCCL lookup, which
    returns at once: every rank calls it at the same point, before it builds any Engram layer,
    including pipeline stages without one. Calling it again, with the same collection or
    another one over the same dp_cp groups, creates no new group.

    With the symmetric backend (``engram_config.communication_backend``), every rank first
    rejects a fake default process group and checks the transport's libraries (CuTe DSL,
    NCCL4Py, the NCCL runtime and its device bitcode), and one gather over the default group
    makes a failure on any rank raise on every rank, before any group or table exists.

    Sharded layout (``engram_config.parallel_size`` set): splits the dp_cp group of each stage
    into row groups of ``parallel_size`` ranks and the replica groups across them (a size that
    does not divide dp_cp raises on every rank), sets ``pg_collection.engram`` and
    ``pg_collection.engram_dp``, and registers the replica group as the ``'engram'`` parameter
    domain in ``pg_collection.parameter_replica_groups``.

    No collective runs on a group this creates. NCCL creates a communicator at the first
    collective of a group, and DDP and the distributed optimizer skip the collectives of a
    single-rank replica group, so that group never gets one. The symmetric transport creates
    its own communicator once the model is wrapped (``initialize_engram_transports``).

    Args:
        pg_collection (ProcessGroupCollection): The groups the model and DDP are built with;
            updated in place.
        engram_config (EngramConfig, optional): The Engram configuration; None (no Engram)
            returns at once.
        timeout (timedelta, optional): Timeout of the groups this creates. None gives them the
            timeout of the dp_cp group they split (``initialize_model_parallel`` sets it from
            ``distributed_timeout_minutes``), or the backend's default timeout when this torch
            does not expose it. Groups already created for the same dp_cp group and size are
            reused as they are.
    """
    if engram_config is None:
        return
    if _uses_symmetric_lookup(engram_config):
        _reject_fake_process_group()
        _require_symmetric_transport_on_every_rank()
    if engram_config.parallel_size is not None:
        _initialize_row_groups(pg_collection, engram_config.parallel_size, timeout)


def resolve_engram_groups(
    pg_collection: ProcessGroupCollection, engram_config: EngramConfig
) -> Tuple[Optional[ProcessGroup], Optional[ProcessGroup], str]:
    """Return the row group, replica group and parameter domain of the Engram tables.

    The EP layout returns ``(pg_collection.ep, pg_collection.expt_dp, 'expert')``; the sharded
    layout returns ``(pg_collection.engram, pg_collection.engram_dp, 'engram')``, which
    :func:`initialize_engram_parallel_state` must have set up. Never communicates.

    Args:
        pg_collection (ProcessGroupCollection): The model's process groups.
        engram_config (EngramConfig): The Engram configuration.

    Returns:
        Tuple of the row group, the replica group and the parameter domain name.
    """
    if engram_config.parallel_size is None:
        return (
            getattr(pg_collection, "ep", None),
            getattr(pg_collection, "expt_dp", None),
            _EP_LAYOUT_DOMAIN,
        )
    row_group = getattr(pg_collection, "engram", None)
    replica_group = getattr(pg_collection, "engram_dp", None)
    replica_groups = getattr(pg_collection, "parameter_replica_groups", None) or {}
    if (
        row_group is None
        or replica_group is None
        or replica_groups.get(ENGRAM_PARAMETER_DOMAIN) is not replica_group
    ):
        # Creating the groups here instead would deadlock: it is a collective over every rank,
        # and pipeline stages without Engram layers never get here.
        raise RuntimeError(
            "Engram parallel_size is set, but the process groups hold no registered Engram row "
            "and replica groups: call initialize_engram_parallel_state(pg_collection, "
            "engram_config) on every rank before building the model, as "
            "HybridModelBuilder.build_distributed_models does."
        )
    if get_pg_size(row_group) != engram_config.parallel_size:
        raise ValueError(
            f"The Engram row group has {get_pg_size(row_group)} ranks, but parallel_size is "
            f"{engram_config.parallel_size}."
        )
    return row_group, replica_group, ENGRAM_PARAMETER_DOMAIN


def tag_engram_tables(embedding: EPShardedMultiTableEmbedding) -> None:
    """Put every table shard of ``embedding`` in the ``'engram'`` parameter domain.

    DDP then keeps the shards in buffers of their own, reduced over the domain's replica group,
    and the optimizer gives them a DistributedOptimizer of their own. The shards keep
    ``allreduce=False``; their domain takes precedence over it.
    """
    for table in embedding.tables:
        table.weight.parallel_domain = ENGRAM_PARAMETER_DOMAIN


def clear_engram_parallel_state() -> None:
    """Forget the cached Engram groups.

    ``parallel_state.destroy_model_parallel`` calls it, so the next model-parallel setup creates
    new groups. Collections that already hold the groups keep their references.
    """
    _ROW_GROUPS.clear()


def _uses_symmetric_lookup(engram_config: EngramConfig) -> bool:
    """Whether the tables use the symmetric transport.

    EngramConfig resolves and checks the backend when it is built; this repeats the checks
    for a configuration changed afterwards, before any rank creates a group.
    """
    backend = engram_config.communication_backend
    if backend not in COMMUNICATION_BACKENDS:
        raise ValueError(
            f"Engram communication_backend must be resolved to one of {COMMUNICATION_BACKENDS}, "
            f"got {backend!r}."
        )
    if backend == "symmetric" and engram_config.parallel_size is None:
        raise ValueError(
            "The Engram symmetric backend requires parallel_size: the expert-parallel table "
            "layout looks its rows up with NCCL all-to-alls only."
        )
    return backend == "symmetric"


def _reject_fake_process_group() -> None:
    """Reject the symmetric backend on a fake process group (memory estimation runs).

    Its transport creates real NCCL communicators over the row group, whose ranks a fake group
    never starts, so the first one would wait forever. Every rank has the same default backend
    and raises alike.
    """
    if torch.distributed.is_initialized() and torch.distributed.get_backend() == "fake":
        raise ValueError(
            "The Engram symmetric lookup creates NCCL communicators of its own and cannot run on "
            "a fake process group; use communication_backend='nccl'."
        )


def _require_symmetric_transport_on_every_rank() -> None:
    """Check the symmetric transport's libraries on every rank; a failure raises on all ranks.

    The check is local, so the ranks of one job could disagree (nodes with different software,
    say). One object gather over the default group shares every outcome before any rank enters
    a collective that a failed rank would never join.
    """
    failure = None
    try:
        symmetric_lookup.require_symmetric_transport()
    except Exception as exc:  # pylint: disable=broad-except
        failure = exc
    problem = None if failure is None else f"{type(failure).__name__}: {failure}"
    problems = [problem]
    if torch.distributed.is_initialized():
        problems = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(problems, problem)
    failed = [rank for rank, reported in enumerate(problems) if reported is not None]
    if not failed:
        return
    reason = f"; on this rank: {problem}" if problem is not None else ""
    raise ImportError(
        f"The Engram symmetric lookup is unavailable on ranks {_format_ranks(failed)}{reason}. "
        "communication_backend='nccl' looks the rows up with NCCL all-to-alls instead."
    ) from failure


def _initialize_row_groups(
    pg_collection: ProcessGroupCollection, parallel_size: int, timeout: Optional[timedelta]
) -> ProcessGroup:
    """Set up the sharded layout's groups on ``pg_collection``; return the row group."""
    _check_sharded_layout(pg_collection)
    if getattr(pg_collection, "engram", None) is None:
        key = (pg_collection.dp_cp, parallel_size)
        if key not in _ROW_GROUPS:
            row_group, replica_group, _ = create_row_parallel_groups(
                pg_collection.dp_cp,
                parallel_size,
                world_group=torch.distributed.group.WORLD,
                name=_GROUP_NAME,
                timeout=timeout if timeout is not None else _group_timeout(pg_collection.dp_cp),
            )
            _ROW_GROUPS[key] = (row_group, replica_group)
            log_single_rank(
                logger,
                logging.INFO,
                "Engram table rows are sharded over row groups of %d ranks and replicated over "
                "replica groups of %d ranks",
                parallel_size,
                get_pg_size(replica_group),
            )
        pg_collection.engram, pg_collection.engram_dp = _ROW_GROUPS[key]
    elif get_pg_size(pg_collection.engram) != parallel_size:
        raise ValueError(
            f"pg_collection.engram has {get_pg_size(pg_collection.engram)} ranks, but "
            f"parallel_size is {parallel_size}."
        )
    elif getattr(pg_collection, "engram_dp", None) is None:
        raise ValueError("pg_collection.engram is set, but pg_collection.engram_dp is not.")
    replica_groups = dict(getattr(pg_collection, "parameter_replica_groups", None) or {})
    registered = replica_groups.setdefault(ENGRAM_PARAMETER_DOMAIN, pg_collection.engram_dp)
    if registered is not pg_collection.engram_dp:
        raise ValueError(
            "pg_collection.parameter_replica_groups registers another group than "
            f"pg_collection.engram_dp for the {ENGRAM_PARAMETER_DOMAIN!r} domain."
        )
    pg_collection.parameter_replica_groups = replica_groups
    return pg_collection.engram


def _group_timeout(group: ProcessGroup) -> Optional[timedelta]:
    """The timeout ``group`` was created with, or None when this torch does not expose it.

    The new groups take the timeout of the dp_cp group they split, so they follow
    ``distributed_timeout_minutes`` like the groups of ``initialize_model_parallel``; torch's
    ``new_group`` would otherwise give them its backend default (10 minutes for NCCL).
    """
    try:
        device_type = "cuda" if torch.distributed.get_backend(group) == "nccl" else "cpu"
        # Private torch API: Backend.options holds the timeout the group was created with.
        return group._get_backend(torch.device(device_type)).options._timeout
    except Exception:  # pylint: disable=broad-except
        return None


def _check_sharded_layout(pg_collection: ProcessGroupCollection) -> None:
    """Reject process groups that the sharded layout does not support.

    The checks depend on the parallel configuration only, so every rank takes the same decision
    before the collective group creation.
    """
    if not torch.distributed.is_initialized():
        raise RuntimeError("Engram parallel_size needs torch.distributed: it shards over dp_cp.")
    if getattr(pg_collection, "dp_cp", None) is None:
        raise ValueError("Engram parallel_size shards the tables over pg_collection.dp_cp, unset.")
    # Only without tensor, expert-tensor and GTP sharding does dp_cp hold the expert grid, and do
    # the gradient and parameter norms count each row shard of the tables exactly once.
    sizes = {
        name: get_pg_size(getattr(pg_collection, name, None))
        for name in ("tp", "expt_tp", "gtp_remat", "expt_gtp_remat")
    }
    unsupported = [f"{name} has {size} ranks" for name, size in sizes.items() if size > 1]
    if unsupported:
        raise ValueError(
            "Engram parallel_size requires single-rank tp, expt_tp, gtp_remat and "
            f"expt_gtp_remat groups; {', '.join(unsupported)}."
        )


def _format_ranks(ranks: Sequence[int]) -> str:
    if len(ranks) <= 8:
        return str(list(ranks))
    return f"[{', '.join(map(str, ranks[:4]))}, ..., {ranks[-1]}] ({len(ranks)} ranks)"
