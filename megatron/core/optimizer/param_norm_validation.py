# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check embedding table selection and data-parallel reduction order at optimizer setup."""

from collections import defaultdict

import torch.distributed as dist


class ParamNormAgreement:
    """Record the ordered tables seen by each group that holds their shards or replicas."""

    def __init__(self):
        self.tables = defaultdict(list)

    def add(self, constraint, owner, dp_group, *, num_rows, pg_collection):
        full_dp = getattr(pg_collection, "dp_cp", None) or getattr(pg_collection, "dp", None)
        tp = getattr(pg_collection, "tp", None)
        gtp = getattr(pg_collection, "gtp_remat", None)
        if getattr(constraint.parameter, "is_gtp_weight_remat", False):
            gtp = constraint.parameter.group
        for group in dict.fromkeys((dp_group, full_dp, tp, gtp)):
            if group is not None and group.size() > 1:
                # GTP padding can leave different row counts on different model-parallel tiles.
                rows = num_rows if group is dp_group else None
                self.tables[group].append(
                    (constraint.name, owner, constraint.rule.validation_key(), rows)
                )

    def validate(self, process_group, error, *, tied_group, tied_rule):
        """All construction peers exchange their tables, including local errors and empty plans."""
        if not dist.is_initialized():
            if error is not None:
                raise ValueError(error)
            return

        schedules = {}
        try:
            domain = None
            if process_group is not None and process_group is not dist.group.WORLD:
                domain = set(dist.get_process_group_ranks(process_group))
            groups = [("plan", group, tables) for group, tables in self.tables.items()]
            if tied_group is not None and tied_group.size() > 1:
                # Only the physical pipeline copies participate; their parameter names may differ.
                groups.append(("shared word embedding", tied_group, tied_rule))
            for kind, group, tables in groups:
                if (
                    domain is not None
                    and group is not process_group
                    and not domain.issuperset(dist.get_process_group_ranks(group))
                ):
                    raise ValueError(
                        f"{kind} group {group.group_name} extends outside the construction group"
                    )
                schedules[kind, group.group_name, group.size()] = tables
        except Exception as exc:
            error = str(exc)

        report = (dist.get_rank(), schedules, error)
        reports = [report] * dist.get_world_size(process_group)
        if len(reports) > 1:
            dist.all_gather_object(reports, report, group=process_group)

        by_group = defaultdict(list)
        for rank, schedules, error in reports:
            if error is not None:
                raise ValueError(f"Parameter norm setup failed on rank {rank}: {error}")
            for key, tables in schedules.items():
                by_group[key].append((rank, tables))
        for (kind, name, size), participants in by_group.items():
            rank, expected = participants[0]
            label = f"Parameter norm {kind} (group {name})"
            if kind == "plan" and len(participants) != size:
                raise ValueError(
                    f"{label} mismatch: {len(participants)} of {size} peers declared {expected}"
                )
            for peer, actual in participants[1:]:
                if actual != expected:
                    raise ValueError(
                        f"{label} mismatch: rank {rank} has {expected}, rank {peer} has {actual}"
                    )
