# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Per-module ``HyperCommGrid`` topology and process-group ownership for hetero MIMO training."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch.distributed as dist

from megatron.core.hyper_comm_grid import HyperCommGrid
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY, ModuleLayout, RankRole
from megatron.core.parallel_state import default_embedding_ranks, default_position_embedding_ranks
from megatron.core.process_groups_config import (
    MultiModuleProcessGroupCollection,
    ProcessGroupCollection,
)

_EXPERT_VIEW = "expert"


@dataclass
class ModuleGridSpec:
    """One module's grid factorization and placement.

    ``num_ranks`` is the ground truth; ``dp`` and ``expt_dp`` are derived in ``__post_init__``.
    """

    name: str
    num_ranks: int
    tp: int = 1
    cp: int = 1
    pp: int = 1
    ep: int = 1
    rank_offset: int = 0
    # Experts default to TP=1 (set explicitly for MoE); intentionally not Megatron's etp=tp default.
    expt_tp: int = 1
    dp: int = field(init=False)
    expt_dp: int = field(init=False)

    def __post_init__(self) -> None:
        dense = self.tp * self.cp * self.pp
        if self.num_ranks % dense != 0:
            raise ValueError(
                f"num_ranks ({self.num_ranks}) must be divisible by tp*cp*pp ({dense})"
            )
        self.dp = self.num_ranks // dense
        expert = self.expt_tp * self.ep * self.pp
        if self.num_ranks % expert != 0:
            raise ValueError(
                f"num_ranks ({self.num_ranks}) must be divisible by expt_tp*ep*pp ({expert})"
            )
        self.expt_dp = self.num_ranks // expert

    @property
    def size(self) -> int:
        """Total ranks spanned by this module's grid."""
        return self.num_ranks


@dataclass
class HeteroTopology:
    """Process groups and rank topology for one hetero MIMO run."""

    grids: dict[str, HyperCommGrid]
    module_pgs: dict[str, ProcessGroupCollection]
    schedule_pg_collection: MultiModuleProcessGroupCollection

    def destroy(self) -> None:
        """Destroy every process group owned by this topology."""
        destroyed: set[int] = set()
        for pgc in self.module_pgs.values():
            for pg in (pgc.embd, pgc.pos_embd):
                if pg is None or id(pg) in destroyed or not _is_process_group_member(pg):
                    continue
                dist.destroy_process_group(pg)
                destroyed.add(id(pg))
        for grid in self.grids.values():
            grid.destroy()


def create_topology(specs: list[ModuleGridSpec]) -> HeteroTopology:
    """Build every module's grid, PGC, and embedding groups.

    Exactly one spec must be named ``MIMO_LANGUAGE_MODULE_KEY`` (the language module); specs must
    tile ``[0, world_size)`` with no gaps (validated by :func:`_validate_grid_layout`).
    """
    if not specs:
        raise ValueError("create_topology requires at least one ModuleGridSpec")
    language_specs = [spec for spec in specs if spec.name == MIMO_LANGUAGE_MODULE_KEY]
    if len(language_specs) != 1:
        raise ValueError(
            f"create_topology requires exactly one spec named {MIMO_LANGUAGE_MODULE_KEY!r} "
            f"(the language module), got {len(language_specs)}"
        )
    language_name = MIMO_LANGUAGE_MODULE_KEY

    grids: dict[str, HyperCommGrid] = {}
    module_pgs: dict[str, ProcessGroupCollection] = {}
    try:
        for spec in specs:
            grids[spec.name] = _build_grid(spec)
        _validate_grid_layout(grids)

        for name, grid in grids.items():
            module_pgs[name] = pg_collection_from_grid(grid, is_language=(name == language_name))

        schedule_pg_collection = build_schedule_pg_collection(grids, module_pgs, language_name)
        return HeteroTopology(
            grids=grids, module_pgs=module_pgs, schedule_pg_collection=schedule_pg_collection
        )
    except Exception:
        HeteroTopology(grids=grids, module_pgs=module_pgs, schedule_pg_collection=None).destroy()
        raise


def _build_grid(spec: ModuleGridSpec) -> HyperCommGrid:
    """Create a dense grid plus its expert view and the process groups MIMO needs."""
    grid = HyperCommGrid(
        shape=[spec.tp, spec.cp, spec.dp, spec.pp],
        dim_names=["tp", "cp", "dp", "pp"],
        rank_offset=spec.rank_offset,
        backend="nccl",
    )
    # Expert factorization over the same rank span; pp is shared with the base view.
    grid.register_view(
        _EXPERT_VIEW,
        shape=[spec.expt_tp, spec.ep, spec.expt_dp, spec.pp],
        dim_names=["expt_tp", "ep", "expt_dp", "pp"],
        shared_dims=["pp"],
    )

    try:
        for dims in (["tp"], ["cp"], ["pp"], ["dp"], ["dp", "cp"], ["tp", "cp"], ["tp", "pp"]):
            grid.create_pg(dims)
        for dims in (["ep"], ["expt_tp"], ["expt_dp"], ["expt_tp", "ep"], ["expt_tp", "ep", "pp"]):
            grid.create_pg(dims, view=_EXPERT_VIEW)
    except Exception:
        grid.destroy()
        raise
    return grid


def _validate_grid_layout(grids: dict[str, HyperCommGrid]) -> None:
    """Assert grids tile the world disjointly (non-colocated) XOR fully share ranks (colocated),
    with no gaps. Colocated-vs-not is decided via the core ``RankRole.build`` path.
    """
    spans = {name: (g.rank_offset, g.rank_offset + g.size) for name, g in grids.items()}
    names = list(spans)
    all_same = all(spans[n] == spans[names[0]] for n in names)
    pairwise_disjoint = all(
        spans[a][1] <= spans[b][0] or spans[b][1] <= spans[a][0]
        for i, a in enumerate(names)
        for b in names[i + 1 :]
    )
    if not (all_same or pairwise_disjoint):
        raise ValueError(
            f"Module grids must either fully share ranks or be pairwise disjoint, got {spans}"
        )

    # Disjoint spans must also leave no rank uncovered (their union == [0, world_size)).
    world_size = dist.get_world_size()
    covered_ranks: set[int] = set()
    for start, end in spans.values():
        covered_ranks.update(range(start, end))
    if covered_ranks != set(range(world_size)):
        raise ValueError(
            f"Module grids must partition the world [0, {world_size}) with no gaps, got {spans}"
        )

    modality_names = [n for n in names if n != MIMO_LANGUAGE_MODULE_KEY]
    role = RankRole.build(modality_names, grids)
    expected = ModuleLayout.COLOCATED if all_same else ModuleLayout.NON_COLOCATED
    if role.mode is not expected:
        raise ValueError(f"RankRole reported {role.mode} but rank spans imply {expected}: {spans}")


def pg_collection_from_grid(
    grid: HyperCommGrid, is_language: bool = False
) -> ProcessGroupCollection:
    """Adapt a populated ``HyperCommGrid`` into a ``ProcessGroupCollection``.

    Only the language module gets embedding groups; others leave ``embd``/``pos_embd`` as ``None``.
    """
    pgc = ProcessGroupCollection()
    pgc.tp = grid.get_pg("tp")
    pgc.cp = grid.get_pg("cp")
    pgc.pp = grid.get_pg("pp")
    pgc.dp = grid.get_pg("dp")
    pgc.dp_cp = grid.get_pg(["dp", "cp"])
    pgc.intra_dp_cp = pgc.dp_cp
    pgc.tp_cp = grid.get_pg(["tp", "cp"])
    pgc.mp = grid.get_pg(["tp", "pp"])
    pgc.ep = grid.get_pg("ep", view=_EXPERT_VIEW)
    pgc.expt_tp = grid.get_pg("expt_tp", view=_EXPERT_VIEW)
    pgc.expt_dp = grid.get_pg("expt_dp", view=_EXPERT_VIEW)
    pgc.intra_expt_dp = pgc.expt_dp
    pgc.tp_ep = grid.get_pg(["expt_tp", "ep"], view=_EXPERT_VIEW)
    pgc.tp_ep_pp = grid.get_pg(["expt_tp", "ep", "pp"], view=_EXPERT_VIEW)
    pgc.embd = None
    pgc.pos_embd = None
    if is_language:
        _build_language_embedding_groups(grid, pgc)
    return pgc


def _build_language_embedding_groups(grid: HyperCommGrid, pgc: ProcessGroupCollection) -> None:
    """Set this rank's word/position embedding groups, mirroring parallel_state.

    Creation is collective: every grid rank calls ``new_group`` for each PP tuple.
    """
    own_pp_ranks = tuple(dist.get_process_group_ranks(pgc.pp)) if pgc.pp is not None else ()
    for pp_ranks in grid.get_rank_enum("pp"):
        emb_group = dist.new_group(ranks=default_embedding_ranks(list(pp_ranks)))
        pos_group = dist.new_group(ranks=default_position_embedding_ranks(list(pp_ranks)))
        if tuple(pp_ranks) == own_pp_ranks:
            if _is_process_group_member(emb_group):
                pgc.embd = emb_group
            if _is_process_group_member(pos_group):
                pgc.pos_embd = pos_group


def build_schedule_pg_collection(
    grids: dict[str, HyperCommGrid],
    module_pgs: dict[str, ProcessGroupCollection],
    language_name: str,
) -> MultiModuleProcessGroupCollection:
    """Build the schedule-facing collection of the modules this rank participates in."""
    rank_modules = {}
    rank_language_name = None
    for name, grid in grids.items():
        # Include only modules this rank belongs to (colocated -> all; non-colocated -> its own),
        # so language_model_module_name is set only when this rank is in the language module.
        if not grid.is_current_rank_in_grid():
            continue
        rank_modules[name] = module_pgs[name]
        if name == language_name:
            rank_language_name = name
    return MultiModuleProcessGroupCollection(
        module_pgs=rank_modules, language_model_module_name=rank_language_name
    )


def _is_process_group_member(pg: Optional[dist.ProcessGroup]) -> bool:
    """Whether the current rank belongs to ``pg`` (get_rank returns -1 for non-members, no raise)."""
    return pg is not None and dist.get_rank(group=pg) >= 0
