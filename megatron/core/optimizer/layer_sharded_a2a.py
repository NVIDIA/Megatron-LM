# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""all_to_all routing utilities for layer-sharded Muon.

Layer sharding assigns every 2D weight one Newton-Schulz "home" rank inside the
(GTP_remat x TP) weight-shard domain — i.e. the GTP domain (GTP = TP x
GTP_remat). The forward exchanges route each rank's local momentum shards so
the home assembles the complete matrix; the backward exchanges scatter the
orthogonalized result back to the original shards.

The exchange runs in two stages (``route_to_ns_home`` / ``route_from_ns_home``):
one all_to_all over the GTP_remat group (dim 0), then one over the TP group
(along ``partition_dim``), reusing the process groups Megatron already has.
These helpers are axis-generic — the caller invokes them once with the GTP_remat
group and once with the TP group — so their arguments are named for the role
(``group``, ``shard_dim``), not for a specific axis. A trivial group (None or
size 1) performs no communication, so a 1-D domain costs a single all_to_all
per direction.

All functions support heterogeneous parameter shapes and uneven home
assignments (ranks may own zero matrices in a given exchange, receiving
zero-size all_to_all splits).
"""

import torch


def _group_rank_and_size(group: "torch.distributed.ProcessGroup | None") -> tuple[int, int]:
    """(rank, size) of ``group``, honouring the module convention that ``None``
    means "no group / size 1" — NOT torch's "the default group"."""
    if group is None:
        return 0, 1
    return group.rank(), group.size()


def _cat_or_empty(parts: list[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    """Concatenate ``parts`` into one flat send buffer, or an empty buffer
    matching ``ref``'s dtype/device when this rank has nothing to send (all of
    its all_to_all input splits are zero)."""
    if parts:
        return torch.cat(parts)
    return torch.empty(0, dtype=ref.dtype, device=ref.device)


def _group_by_home(num_params: int, param_to_home_rank: dict, size: int) -> list[list[int]]:
    """Group param indices by their NS home rank in the group.

    The wired path (LPT bin-packing in LayerWiseDistributedOptimizer) always
    supplies every entry; the ``i % size`` default is a round-robin FALLBACK
    for direct-API callers or missing entries only. Assignments may be uneven —
    with fewer params than ranks, the unassigned ranks simply get empty lists
    and receive zero-size all_to_all splits.
    """
    send_idx: list[list[int]] = [[] for _ in range(size)]
    for param_idx in range(num_params):
        home = param_to_home_rank.get(param_idx, param_idx % size)
        send_idx[home].append(param_idx)
    return send_idx


def route_to_ns_home(
    momentum_list: list[torch.Tensor],
    param_to_home_rank: dict,
    group: "torch.distributed.ProcessGroup | None",
    shard_dim: int = 0,
    plan: "dict | None" = None,
) -> tuple[list[torch.Tensor], list[int]]:
    """Forward all_to_all for layer sharding: redistribute momentum shards.

    Each rank holds a (P/S, Q) momentum shard of every param. This
    redistributes them so each rank ends up with the complete (P, Q) momentum
    for its assigned subset.

    Args:
        momentum_list: List of momentum tensors, one per param. Each has shape
            (P/S, Q) where S = group size (this rank's shard along
            ``shard_dim``).
        param_to_home_rank: Dict mapping param index -> NS home rank in
            ``group``.
        group: The process group to communicate within (the GTP_remat group in
            stage 1, the TP group in stage 2). None means size 1: no exchange.
        shard_dim: Dimension the shards split (0 for the GTP_remat stage; the
            param's partition_dim for the TP stage).
        plan: Optional mutable dict caching the routing metadata (index
            groupings, split sizes, unpack offsets), which is a pure function
            of shapes, homes and group size — all static across steps. Pass an
            empty dict on the first call (it is filled) and the same dict on
            later calls (the metadata rebuild is skipped; only the data
            movement runs). The CALLER owns validity: reuse a plan only while
            the participating params, their shapes and their homes are
            unchanged. None (default) rebuilds every call.

    Returns:
        Tuple of:
            - complete_momentums: List of complete (P, Q) tensors for params
              assigned to this rank, in the order they appear in momentum_list.
            - my_param_indices: Indices into momentum_list for params assigned
              to this rank.
    """
    if not momentum_list:
        # Nothing to exchange; avoid sending an empty buffer into the collective.
        return [], []

    rank, size = _group_rank_and_size(group)
    if size <= 1:
        # Trivial group: every param is homed locally and the shard IS the
        # complete matrix. Also keeps a None group away from
        # all_to_all_single, where None would mean the WORLD group.
        return list(momentum_list), list(range(len(momentum_list)))

    # Routing metadata: a pure function of shapes, homes and group size — built
    # once and reused via ``plan`` (indices and sizes only, never tensors).
    if plan is None:
        plan = {}
    if not plan:
        send_idx = _group_by_home(len(momentum_list), param_to_home_rank, size)
        my_param_indices = send_idx[rank]
        my_param_numel = sum(momentum_list[i].numel() for i in my_param_indices)
        # Prefix offsets of each of my params within one source block.
        # Precomputed: deriving them inline is O(n^2 * size), and n reaches the
        # hundreds when a home owns many same-shape expert weights.
        param_offsets = [0]
        for i in my_param_indices:
            param_offsets.append(param_offsets[-1] + momentum_list[i].numel())
        plan.update(
            send_idx=send_idx,
            input_split_sizes=[
                sum(momentum_list[i].numel() for i in send_idx[r]) for r in range(size)
            ],
            my_param_indices=my_param_indices,
            my_param_numel=my_param_numel,
            output_split_sizes=[my_param_numel] * size,
            param_offsets=param_offsets,
        )
    send_idx = plan['send_idx']
    my_param_indices = plan['my_param_indices']
    my_param_numel = plan['my_param_numel']
    param_offsets = plan['param_offsets']

    # Build flat send buffer: [data_for_rank_0 | data_for_rank_1 | ...]
    # For each destination r', send my momentum shards for params assigned to r'
    send_parts = [
        torch.cat([momentum_list[i].contiguous().flatten() for i in send_idx[r]])
        for r in range(size)
        if send_idx[r]
    ]

    send_buf = _cat_or_empty(send_parts, momentum_list[0])

    recv_buf = torch.empty(my_param_numel * size, dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=plan['output_split_sizes'],
        input_split_sizes=plan['input_split_sizes'],
        group=group,
    )

    # Unpack: for each of my assigned params, concatenate the shards from all
    # sources. recv_buf layout: [from_r0 | from_r1 | ... | from_r(S-1)], where
    # each from_rk block contains that source's shards of my params, in order.
    complete_momentums = []
    for pos, i in enumerate(my_param_indices):
        m_template = momentum_list[i]
        numel = m_template.numel()
        offset = param_offsets[pos]
        # Slices of a 1-D contiguous buffer are already contiguous, so cat can
        # take the views directly.
        shards = [
            recv_buf[
                r_prime * my_param_numel + offset : r_prime * my_param_numel + offset + numel
            ].view(m_template.shape)
            for r_prime in range(size)
        ]
        complete_momentums.append(torch.cat(shards, dim=shard_dim))  # (P, Q)

    return complete_momentums, list(my_param_indices)


def route_from_ns_home(
    ns_results: list[torch.Tensor],
    my_param_indices: list[int],
    momentum_list: list[torch.Tensor],
    param_to_home_rank: dict,
    group: "torch.distributed.ProcessGroup | None",
    shard_dim: int = 0,
    plan: "dict | None" = None,
) -> list["torch.Tensor | None"]:
    """Backward all_to_all for layer sharding: distribute NS results as shards.

    Each NS-home rank has complete (P, Q) NS results for its assigned params.
    This redistributes them so every rank gets its (P/S, Q) shard for every
    param.

    Args:
        ns_results: Complete (P, Q) NS result tensors, one per assigned param,
            in the order of my_param_indices.
        my_param_indices: Indices into momentum_list for params assigned to
            this rank.
        momentum_list: List of original momentum tensors (provides shapes).
        param_to_home_rank: Dict mapping param index -> NS home rank in
            ``group``.
        group: The process group to communicate within (see the fwd docstring).
            None means size 1: no exchange.
        shard_dim: Dimension the shards split (see the fwd docstring).
        plan: Optional routing-metadata cache (see the fwd docstring; same
            ownership rules). The shape-invariant precondition check also runs
            only when the plan is built.

    Returns:
        List of (P/S, Q) NS update shards, one per param in momentum_list
        order. None for params that did not participate (should not occur in
        normal usage).
    """
    if not momentum_list:
        return []

    rank, size = _group_rank_and_size(group)
    if size <= 1:
        update_shards: list["torch.Tensor | None"] = [None] * len(momentum_list)
        for ns_r, idx in zip(ns_results, my_param_indices):
            update_shards[idx] = ns_r
        return update_shards

    if plan is None:
        plan = {}
    if not plan:
        send_idx = _group_by_home(len(momentum_list), param_to_home_rank, size)

        # Precondition: ns_r must span exactly ``size`` equal-sized shards so the
        # uniform-stride narrow below is correct.  A violated invariant produces
        # silent corruption (narrow is in-bounds but slices the wrong rows).
        # Shape-only, so checking once at plan-build time covers every reuse.
        for ns_r, idx in zip(ns_results, my_param_indices):
            expected = momentum_list[idx].shape[shard_dim] * size
            assert ns_r.shape[shard_dim] == expected, (
                f"route_from_ns_home: full-matrix dim[{shard_dim}]="
                f"{ns_r.shape[shard_dim]} != shard_size="
                f"{momentum_list[idx].shape[shard_dim]} × group size={size}; "
                "all shards must be equal-sized (divisibility/padding invariant violated)."
            )

        my_numel = sum(momentum_list[i].numel() for i in my_param_indices)
        plan.update(
            send_idx=send_idx,
            input_split_sizes=[my_numel if ns_results else 0] * size,
            output_split_sizes=[
                sum(momentum_list[i].numel() for i in send_idx[r]) for r in range(size)
            ],
        )
    send_idx = plan['send_idx']
    output_split_sizes = plan['output_split_sizes']

    # Build send buffer: for each destination r', send that rank's shard of
    # each of MY ns_results. Shard size is per-param (heterogeneous shapes).
    send_parts = []
    for r_prime in range(size):
        if ns_results:
            send_parts.append(
                torch.cat(
                    [
                        ns_r.narrow(
                            shard_dim,
                            r_prime * momentum_list[idx].shape[shard_dim],
                            momentum_list[idx].shape[shard_dim],
                        )
                        .contiguous()
                        .flatten()
                        for ns_r, idx in zip(ns_results, my_param_indices)
                    ]
                )
            )

    send_buf = _cat_or_empty(send_parts, momentum_list[0])

    recv_buf = torch.empty(sum(output_split_sizes), dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=plan['input_split_sizes'],
        group=group,
    )

    # Unpack into per-param update shards
    update_shards: list["torch.Tensor | None"] = [None] * len(momentum_list)
    offset = 0
    for r_prime in range(size):
        for i in send_idx[r_prime]:
            m_template = momentum_list[i]
            shard = recv_buf[offset : offset + m_template.numel()].view(m_template.shape)
            update_shards[i] = shard.contiguous()
            offset += m_template.numel()

    return update_shards
