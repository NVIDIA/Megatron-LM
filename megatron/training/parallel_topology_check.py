# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Preflight check that reports how process groups are laid out across nodes.

Each rank contributes its hostname and the global ranks of its selected process
groups through a single ``all_gather_object`` on the world group. Every rank then
holds the same view, so rank 0 can log a compact summary and every rank reaches
the same decision when the check is configured to fail.
"""

import socket
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from megatron.core import parallel_state as mpu
from megatron.training.utils import print_rank_0, warn_rank_0

# Short names accepted on the command line, mapped to parallel_state getters.
PROCESS_GROUP_GETTERS: Dict[str, Callable[[], Optional[torch.distributed.ProcessGroup]]] = {
    "tp": lambda: mpu.get_tensor_model_parallel_group(),
    "pp": lambda: mpu.get_pipeline_model_parallel_group(),
    "cp": lambda: mpu.get_context_parallel_group(),
    "dp": lambda: mpu.get_data_parallel_group(),
    "dp_cp": lambda: mpu.get_data_parallel_group(with_context_parallel=True),
    "dp_replica": lambda: mpu.get_data_parallel_group(with_gtp_remat=False),
    "mp": lambda: mpu.get_model_parallel_group(),
    "tp_cp": lambda: mpu.get_tensor_and_context_parallel_group(),
    "tp_dp": lambda: mpu.get_tensor_and_data_parallel_group(),
    "tp_dp_cp": lambda: mpu.get_tensor_and_data_parallel_group(with_context_parallel=True),
    "gtp_remat": lambda: mpu.get_gtp_weight_remat_group(),
    "ep": lambda: mpu.get_expert_model_parallel_group(),
    "etp": lambda: mpu.get_expert_tensor_parallel_group(),
    "tp_ep": lambda: mpu.get_expert_tensor_and_model_parallel_group(),
    "ep_dp": lambda: mpu.get_expert_data_parallel_group(),
    "embd": lambda: mpu.get_embedding_group(),
    "pos_embd": lambda: mpu.get_position_embedding_group(),
    "intra_dist_opt": lambda: mpu.get_intra_distributed_optimizer_instance_group(),
    "inter_dist_opt": lambda: mpu.get_inter_distributed_optimizer_instance_group(),
}

MAX_LOGGED_SPANNING_GROUPS = 4


def parse_process_group_names(names: Optional[Iterable[str]]) -> List[str]:
    """Normalize a list of group names, allowing comma separated entries.

    Args:
        names: Names such as ``["dp,tp", "cp"]``.

    Returns:
        The deduplicated names in their original order.

    Raises:
        ValueError: If a name has no known getter.
    """
    parsed: List[str] = []
    for entry in names or []:
        for name in entry.split(","):
            name = name.strip().lower()
            if not name or name in parsed:
                continue
            if name not in PROCESS_GROUP_GETTERS:
                raise ValueError(
                    f"Unknown process group '{name}' for the parallel topology check. "
                    f"Valid names: {', '.join(sorted(PROCESS_GROUP_GETTERS))}."
                )
            parsed.append(name)
    return parsed


def _local_group_ranks(names: Sequence[str]) -> Dict[str, Optional[Tuple[int, ...]]]:
    """Return the global ranks of this rank's group for each name, or None if unavailable."""
    local: Dict[str, Optional[Tuple[int, ...]]] = {}
    for name in names:
        try:
            group = PROCESS_GROUP_GETTERS[name]()
        except (AssertionError, RuntimeError):
            group = None
        if group is None:
            local[name] = None
            continue
        local[name] = tuple(torch.distributed.get_process_group_ranks(group))
    return local


def summarize_topology(
    hostnames: Sequence[str],
    gathered_groups: Sequence[Dict[str, Optional[Tuple[int, ...]]]],
    names: Sequence[str],
) -> Dict[str, Dict]:
    """Build a per-name summary from the gathered hostnames and group ranks.

    Args:
        hostnames: Hostname of every global rank, indexed by rank.
        gathered_groups: For every global rank, its group ranks keyed by name.
        names: Group names to summarize.

    Returns:
        A dict keyed by group name with the unique groups, their hosts and the
        groups that span more than one node.
    """
    summary: Dict[str, Dict] = {}
    for name in names:
        unique = sorted({g[name] for g in gathered_groups if g.get(name) is not None})
        groups = []
        for ranks in unique:
            hosts = sorted({hostnames[r] for r in ranks})
            groups.append({"ranks": list(ranks), "hosts": hosts, "spans_nodes": len(hosts) > 1})
        summary[name] = {"groups": groups, "spanning": [g for g in groups if g["spans_nodes"]]}
    return summary


def format_summary(summary: Dict[str, Dict], num_nodes: int, world_size: int) -> str:
    """Render a compact, human readable report of the summary."""
    lines = [f"parallel topology check: world_size={world_size}, nodes={num_nodes}"]
    for name, info in summary.items():
        groups = info["groups"]
        if not groups:
            lines.append(f"  {name}: not initialized, skipped")
            continue
        sizes = sorted({len(g["ranks"]) for g in groups})
        size_str = "/".join(str(s) for s in sizes)
        lines.append(
            f"  {name}: {len(groups)} group(s) of size {size_str}, "
            f"{len(info['spanning'])} span nodes"
        )
        for g in info["spanning"][:MAX_LOGGED_SPANNING_GROUPS]:
            lines.append(f"    ranks {g['ranks']} on hosts {g['hosts']}")
        extra = len(info["spanning"]) - MAX_LOGGED_SPANNING_GROUPS
        if extra > 0:
            lines.append(f"    ... {extra} more spanning group(s)")
    return "\n".join(lines)


def check_parallel_topology(
    mode: str = "off",
    process_groups: Optional[Iterable[str]] = None,
    intra_node_process_groups: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Dict]]:
    """Report which process groups span nodes, optionally warning or failing.

    The summary for ``process_groups`` is logged from rank 0. Groups listed in
    ``intra_node_process_groups`` are expected to stay on one node; if any of
    them spans nodes, ``mode="warn"`` logs a warning and ``mode="error"`` raises
    on every rank.

    Args:
        mode: One of ``off``, ``warn`` or ``error``.
        process_groups: Group names to report, comma separated entries allowed.
        intra_node_process_groups: Group names expected to stay within a node.

    Returns:
        The summary dict, or None when the check is off.
    """
    if mode == "off":
        return None
    if mode not in ("warn", "error"):
        raise ValueError(f"Unknown parallel topology check mode '{mode}'.")

    report_names = parse_process_group_names(process_groups)
    intra_names = parse_process_group_names(intra_node_process_groups)
    names = report_names + [n for n in intra_names if n not in report_names]
    if not names:
        return None

    payload = (socket.gethostname(), _local_group_ranks(names))
    world_size = torch.distributed.get_world_size()
    gathered: List = [None] * world_size
    torch.distributed.all_gather_object(gathered, payload)
    hostnames = [p[0] for p in gathered]
    summary = summarize_topology(hostnames, [p[1] for p in gathered], names)

    if report_names:
        report = {n: summary[n] for n in report_names}
        print_rank_0(format_summary(report, len(set(hostnames)), world_size))

    offending = [n for n in intra_names if summary[n]["spanning"]]
    if offending:
        msg = (
            "parallel topology check: process group(s) "
            f"{', '.join(offending)} span multiple nodes, which usually means "
            "intra-node communication is going over the inter-node network. "
            + "; ".join(
                f"{n}: ranks {summary[n]['spanning'][0]['ranks']} on hosts "
                f"{summary[n]['spanning'][0]['hosts']}"
                for n in offending
            )
        )
        if mode == "error":
            raise RuntimeError(msg)
        warn_rank_0(msg)
    return summary
