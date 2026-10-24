"""ParallelState and process group initialization."""

from __future__ import annotations

from dataclasses import dataclass

import torch.distributed as dist  # pyright: ignore[reportMissingImports]

from megatron.lite.primitive.utils import ensure_divisible


@dataclass
class ParallelState:
    tp_group: dist.ProcessGroup | None = None
    ep_group: dist.ProcessGroup | None = None
    etp_group: dist.ProcessGroup | None = None
    cp_group: dist.ProcessGroup | None = None
    pp_group: dist.ProcessGroup | None = None
    pp_cpu_group: dist.ProcessGroup | None = None
    dp_group: dist.ProcessGroup | None = None
    dp_cp_group: dist.ProcessGroup | None = None
    tp_ep_group: dist.ProcessGroup | None = None
    ep_dp_group: dist.ProcessGroup | None = None
    cp_global_ranks: list[int] | None = None
    pp_global_ranks: list[int] | None = None

    tp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    dp_cp_size: int = 1
    expert_dp_size: int = 1

    tp_rank: int = 0
    ep_rank: int = 0
    etp_rank: int = 0
    cp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    dp_cp_rank: int = 0
    expert_dp_rank: int = 0

    pp_is_first: bool = True
    pp_is_last: bool = True
    pp_next_rank: int = -1
    pp_prev_rank: int = -1


def init_parallel(config) -> ParallelState:
    """
    Initialize all process groups using dual rank decomposition.

    Dense layers: world = TP × CP × PP × DP
    Expert layers: world = ETP × EP × PP × expert_DP
    """
    assert dist.is_initialized(), "Call torch.distributed.init_process_group first"

    world = dist.get_world_size()
    rank = dist.get_rank()
    tp, ep, etp, cp, pp = config.tp, config.ep, config.etp, config.cp, config.pp
    assert etp == 1, "ETP>1 is temporarily disabled pending fix"

    dense_dp = ensure_divisible(world, tp * cp * pp)
    expert_dp = ensure_divisible(world, etp * ep * pp)

    ps = ParallelState()
    ps.tp_size, ps.ep_size, ps.etp_size = tp, ep, etp
    ps.cp_size, ps.pp_size, ps.dp_size = cp, pp, dense_dp
    ps.expert_dp_size = expert_dp
    ps.dp_cp_size = dense_dp * cp

    def _d(tp_i, cp_i, dp_i, pp_i):
        return ((pp_i * dense_dp + dp_i) * cp + cp_i) * tp + tp_i

    def _e(etp_i, ep_i, edp_i, pp_i):
        return ((pp_i * expert_dp + edp_i) * ep + ep_i) * etp + etp_i

    t = rank
    my_tp = t % tp
    t //= tp
    my_cp = t % cp
    t //= cp
    my_ddp = t % dense_dp
    t //= dense_dp
    my_pp = t

    t = rank
    my_etp = t % etp
    t //= etp
    my_ep = t % ep
    t //= ep
    my_edp = t % expert_dp
    t //= expert_dp
    assert t == my_pp, "PP rank must agree between dense and expert decompositions"

    ps.tp_rank, ps.cp_rank, ps.dp_rank, ps.pp_rank = my_tp, my_cp, my_ddp, my_pp
    ps.dp_cp_rank = my_ddp * cp + my_cp
    ps.ep_rank, ps.etp_rank, ps.expert_dp_rank = my_ep, my_etp, my_edp
    ps.pp_is_first = my_pp == 0
    ps.pp_is_last = my_pp == pp - 1

    for d in range(dense_dp):
        for p in range(pp):
            for c in range(cp):
                ranks = [_d(t, c, d, p) for t in range(tp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ps.tp_group = g

    for d in range(dense_dp):
        for p in range(pp):
            for t in range(tp):
                ranks = [_d(t, c, d, p) for c in range(cp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ps.cp_group = g
                    ps.cp_global_ranks = ranks

    for d in range(dense_dp):
        for c in range(cp):
            for t in range(tp):
                ranks = [_d(t, c, d, p) for p in range(pp)]
                g = dist.new_group(ranks)
                try:
                    cpu_g = dist.new_group(ranks, backend="gloo")
                except (RuntimeError, ValueError):
                    cpu_g = None
                if rank in ranks:
                    ps.pp_group = g
                    ps.pp_cpu_group = cpu_g
                    ps.pp_global_ranks = ranks

    for p in range(pp):
        for c in range(cp):
            for t in range(tp):
                ranks = [_d(t, c, d, p) for d in range(dense_dp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ps.dp_group = g

    for p in range(pp):
        for t in range(tp):
            ranks = [_d(t, c, d, p) for d in range(dense_dp) for c in range(cp)]
            g = dist.new_group(ranks)
            if rank in ranks:
                ps.dp_cp_group = g

    for d in range(expert_dp):
        for p in range(pp):
            for t in range(etp):
                ranks = [_e(t, e, d, p) for e in range(ep)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ps.ep_group = g

    if etp > 1:
        for d in range(expert_dp):
            for p in range(pp):
                for e in range(ep):
                    ranks = [_e(t, e, d, p) for t in range(etp)]
                    g = dist.new_group(ranks)
                    if rank in ranks:
                        ps.etp_group = g

    for d in range(expert_dp):
        for p in range(pp):
            ranks = [_e(t, e, d, p) for e in range(ep) for t in range(etp)]
            g = dist.new_group(ranks)
            if rank in ranks:
                ps.tp_ep_group = g

    for p in range(pp):
        for e in range(ep):
            for t in range(etp):
                ranks = [_e(t, e, d, p) for d in range(expert_dp)]
                g = dist.new_group(ranks)
                if rank in ranks:
                    ps.ep_dp_group = g

    pp_ranks = ps.pp_global_ranks
    if pp_ranks is None:
        raise RuntimeError("Pipeline ranks were not initialized.")
    ps.pp_next_rank = pp_ranks[(my_pp + 1) % pp] if pp > 1 else rank
    ps.pp_prev_rank = pp_ranks[(my_pp - 1) % pp] if pp > 1 else rank

    return ps


__all__ = ["ParallelState", "init_parallel"]
