# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""all_to_all routing utilities for layer-sharded Muon.

Layer sharding assigns every 2D weight one Newton-Schulz "home" rank inside the
(GTP_remat x TP) weight-shard domain — i.e. the GTP domain (GTP = TP x
GTP_remat). The forward exchanges route each rank's local momentum shards so
the home assembles the complete matrix; the backward exchanges scatter the
orthogonalized result back to the original shards.

Two equivalent exchange strategies are provided:

- Two-stage (``layer_sharded_all_to_all_{fwd,bwd}``): one all_to_all over the
  GTP_remat group (dim 0), then one over the TP group (along
  ``partition_dim``), reusing the existing process groups. These helpers are
  axis-generic — the caller invokes them once with the GTP_remat group and once
  with the TP group — so their arguments are named for the role (``group``,
  ``shard_dim``), not for a specific axis.
- Fused (``layer_sharded_fused_{fwd,bwd}``): a single all_to_all per direction
  over the flattened (GTP_remat x TP) domain group. It moves the exact same
  shard blocks and assembles them in the exact same order, so the NS input is
  bit-identical to the two-stage path.

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


def layer_sharded_all_to_all_fwd(
    momentum_list: list[torch.Tensor],
    param_to_home_rank: dict,
    group: "torch.distributed.ProcessGroup | None",
    shard_dim: int = 0,
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

    Returns:
        Tuple of:
            - complete_momentums: List of complete (P, Q) tensors for params
              assigned to this rank, in the order they appear in momentum_list.
            - my_param_indices: Indices into momentum_list for params assigned
              to this rank.
    """
    rank, size = _group_rank_and_size(group)
    if size <= 1:
        # Trivial group: every param is homed locally and the shard IS the
        # complete matrix. Also keeps a None group away from
        # all_to_all_single, where None would mean the WORLD group.
        return list(momentum_list), list(range(len(momentum_list)))

    # Group params by their NS home rank. The wired path (LPT bin-packing in
    # LayerWiseDistributedOptimizer) always supplies every entry; the
    # ``i % size`` default is a round-robin FALLBACK for direct-API callers or
    # missing entries only. Assignments may be uneven — with fewer params than
    # ranks, the unassigned ranks simply receive zero-size splits below.
    rank_to_param_mapping = [[] for _ in range(size)]
    for param_idx, param_momentum in enumerate(momentum_list):
        home = param_to_home_rank.get(param_idx, param_idx % size)
        rank_to_param_mapping[home].append((param_idx, param_momentum))

    # Build flat send buffer: [data_for_rank_0 | data_for_rank_1 | ...]
    # For each destination r', send my momentum shards for params assigned to r'
    send_parts = []
    input_split_sizes = []
    for r_prime in range(size):
        if rank_to_param_mapping[r_prime]:
            chunk = torch.cat(
                [m.contiguous().flatten() for _, m in rank_to_param_mapping[r_prime]]
            )
            send_parts.append(chunk)
            input_split_sizes.append(chunk.numel())
        else:
            input_split_sizes.append(0)

    _ref = momentum_list[0] if momentum_list else None
    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(
            0,
            dtype=_ref.dtype if _ref is not None else torch.float32,
            device=_ref.device if _ref is not None else torch.device('cpu'),
        )
    )

    # Compute recv split sizes: from each source r', we receive momentum shards
    # for all params assigned to us. Each shard has the same numel as the param.
    my_params = rank_to_param_mapping[rank]
    my_param_numel = sum(m.numel() for _, m in my_params)
    output_split_sizes = [my_param_numel] * size

    recv_buf = torch.empty(my_param_numel * size, dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    # Unpack: for each of my assigned params, concatenate the shards from all
    # sources. recv_buf layout: [from_r0 | from_r1 | ... | from_r(S-1)], where
    # each from_rk block contains that source's shards of my params, in order.
    my_param_indices = [i for i, _ in my_params]

    # Prefix offsets of each of my params within one source block. Precomputed:
    # deriving them inline is O(n^2 * size), and n reaches the hundreds when a
    # home owns many same-shape expert weights.
    param_offsets = [0]
    for _, m in my_params:
        param_offsets.append(param_offsets[-1] + m.numel())

    complete_momentums = []
    for pos, (_, m_template) in enumerate(my_params):
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

    return complete_momentums, my_param_indices


def layer_sharded_all_to_all_bwd(
    ns_results: list[torch.Tensor],
    my_param_indices: list[int],
    momentum_list: list[torch.Tensor],
    param_to_home_rank: dict,
    group: "torch.distributed.ProcessGroup | None",
    shard_dim: int = 0,
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

    # Group params by their NS home rank (same grouping and same round-robin
    # fallback as fwd — see the comment there).
    rank_to_param_mapping = [[] for _ in range(size)]
    for param_idx, param_momentum in enumerate(momentum_list):
        home = param_to_home_rank.get(param_idx, param_idx % size)
        rank_to_param_mapping[home].append((param_idx, param_momentum))

    # Precondition: ns_r must span exactly ``size`` equal-sized shards so the
    # uniform-stride narrow below is correct.  A violated invariant produces silent
    # corruption (narrow is in-bounds but slices the wrong rows).
    if ns_results:
        for ns_r, idx in zip(ns_results, my_param_indices):
            expected = momentum_list[idx].shape[shard_dim] * size
            assert ns_r.shape[shard_dim] == expected, (
                f"layer_sharded_all_to_all_bwd: full-matrix dim[{shard_dim}]="
                f"{ns_r.shape[shard_dim]} != shard_size="
                f"{momentum_list[idx].shape[shard_dim]} × group size={size}; "
                "all shards must be equal-sized (divisibility/padding invariant violated)."
            )

    # Build send buffer: for each destination r', send that rank's shard of
    # each of MY ns_results. Shard size is per-param (heterogeneous shapes).
    send_parts = []
    input_split_sizes = []
    for r_prime in range(size):
        if ns_results:
            chunk = torch.cat(
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
            send_parts.append(chunk)
            input_split_sizes.append(chunk.numel())
        else:
            input_split_sizes.append(0)

    _ref_bwd = momentum_list[0] if momentum_list else None
    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(
            0,
            dtype=_ref_bwd.dtype if _ref_bwd is not None else torch.float32,
            device=_ref_bwd.device if _ref_bwd is not None else torch.device('cpu'),
        )
    )

    # Recv: from each source r', receive the shard for MY rank of their NS results
    output_split_sizes = []
    for r_prime in range(size):
        # Number of params assigned to r', each contributing shard_size * Q elements
        r_prime_params = rank_to_param_mapping[r_prime]
        total_numel = sum(m.numel() for _, m in r_prime_params)  # P/S * Q * n_params
        output_split_sizes.append(total_numel)

    recv_buf = torch.empty(sum(output_split_sizes), dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
    )

    # Unpack into per-param update shards
    update_shards: list["torch.Tensor | None"] = [None] * len(momentum_list)
    offset = 0
    for r_prime in range(size):
        for i, m_template in rank_to_param_mapping[r_prime]:
            shard = recv_buf[offset : offset + m_template.numel()].view(m_template.shape)
            update_shards[i] = shard.contiguous()
            offset += m_template.numel()

    return update_shards


def layer_sharded_fused_fwd(
    momentum_list: list[torch.Tensor],
    param_homes: list[tuple[int, int]],
    partition_dims: list["int | None"],
    gtp_remat_rank: int,
    tp_rank: int,
    gtp_remat_size: int,
    tp_size: int,
    fused_group: "torch.distributed.ProcessGroup",
) -> tuple[list[torch.Tensor], list[int]]:
    """Single fused all_to_all over the flattened (GTP_remat x TP) domain (forward).

    Functionally identical to the two-stage ``layer_sharded_all_to_all_fwd``
    (over GTP_remat) followed by a second stage over TP: the exact same shard
    blocks travel to the same NS home and are concatenated in the exact same
    order, so the assembled full matrix is bit-identical. One collective
    replaces up to three (GTP_remat, then TP once per non-empty partition_dim).

    Rank convention: the caller must construct ``fused_group`` so that its
    group rank ``g * tp_size + t`` is the process with coordinates ``(g, t)``
    in (gtp_remat_group, tp_group) — i.e. TP innermost, matching Megatron's
    ``tp-gtp_remat-...`` order.

    Sharding model per full matrix ``(P, Q)`` (mirrors the two-stage path):
      - ``partition_dim == 0``: TP shards dim 0, then GTP_remat shards the
        TP-local rows. Source ``(g, t)`` holds full rows
        ``[t*P/T + g*P/(T*G), ...)`` — block index ``t*G + g`` along dim 0.
      - ``partition_dim == 1``: GTP_remat shards dim 0, TP shards dim 1. Source
        ``(g, t)`` holds the 2-D block ``[g*P/G:(g+1)*P/G, t*Q/T:(t+1)*Q/T]``.
      - ``partition_dim is None``: not TP-sharded; every TP peer holds the
        identical ``(P/G, Q)`` shard, so only sources with ``t == t_home``
        contribute.

    Args:
        momentum_list: Local momentum shard per param.
        param_homes: ``(g_home, t_home)`` per param.
        partition_dims: TP partition dim per param (0, 1, or None).
        gtp_remat_rank / tp_rank: This rank's coordinates. Not derivable from
            ``fused_group`` alone, so they stay explicit parameters (unlike the
            two-stage helpers).
        gtp_remat_size / tp_size: Domain extents.
        fused_group: Flattened process group of size
            ``gtp_remat_size * tp_size``.

    Returns:
        ``(full_mats, my_param_indices)`` — complete matrices for params homed
        on this rank, and their indices into ``momentum_list``.
    """
    G, T = gtp_remat_size, tp_size
    S = G * T
    my_flat = gtp_remat_rank * T + tp_rank
    n = len(momentum_list)

    dest = [gh * T + th for gh, th in param_homes]

    # --- send: my local shard of param i goes to its home, except that for
    # non-TP-sharded params only the t == t_home column contributes (all TP peers
    # hold identical data; sending T copies would be pure waste).
    send_lists: list[list[int]] = [[] for _ in range(S)]
    for i in range(n):
        if partition_dims[i] is None and tp_rank != param_homes[i][1]:
            continue
        send_lists[dest[i]].append(i)

    send_parts = []
    input_split_sizes = []
    for d in range(S):
        if send_lists[d]:
            chunk = torch.cat([momentum_list[i].contiguous().flatten() for i in send_lists[d]])
            send_parts.append(chunk)
            input_split_sizes.append(chunk.numel())
        else:
            input_split_sizes.append(0)

    _ref = momentum_list[0] if momentum_list else None
    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(
            0,
            dtype=_ref.dtype if _ref is not None else torch.float32,
            device=_ref.device if _ref is not None else torch.device('cpu'),
        )
    )

    # --- recv: from source (g_s, t_s) I receive shards of my params, except the
    # non-TP-sharded ones arrive only from the t_s == tp_rank column.
    my_param_indices = [i for i in range(n) if dest[i] == my_flat]
    contrib: list[list[int]] = []
    output_split_sizes = []
    for s in range(S):
        t_s = s % T
        lst = [i for i in my_param_indices if partition_dims[i] is not None or t_s == tp_rank]
        contrib.append(lst)
        output_split_sizes.append(sum(momentum_list[i].numel() for i in lst))

    recv_buf = torch.empty(sum(output_split_sizes), dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=fused_group,
    )

    # --- unpack: index every (source, param) piece via running offsets.
    piece: dict[tuple[int, int], torch.Tensor] = {}
    offset = 0
    for s in range(S):
        for i in contrib[s]:
            numel = momentum_list[i].numel()
            piece[(s, i)] = recv_buf[offset : offset + numel].view(momentum_list[i].shape)
            offset += numel

    # --- reassemble in the same block order the two-stage path produces.
    full_mats = []
    for i in my_param_indices:
        pd = partition_dims[i]
        if pd == 0:
            # dim-0 blocks ordered (t outer, g inner)
            blocks = [piece[(g * T + t, i)] for t in range(T) for g in range(G)]
            full_mats.append(torch.cat(blocks, dim=0))
        elif pd == 1:
            rows = [torch.cat([piece[(g * T + t, i)] for t in range(T)], dim=1) for g in range(G)]
            full_mats.append(torch.cat(rows, dim=0))
        else:
            blocks = [piece[(g * T + tp_rank, i)] for g in range(G)]
            full_mats.append(torch.cat(blocks, dim=0))

    return full_mats, my_param_indices


def layer_sharded_fused_bwd(
    ns_results: list[torch.Tensor],
    my_param_indices: list[int],
    momentum_list: list[torch.Tensor],
    param_homes: list[tuple[int, int]],
    partition_dims: list["int | None"],
    gtp_remat_rank: int,
    tp_rank: int,
    gtp_remat_size: int,
    tp_size: int,
    fused_group: "torch.distributed.ProcessGroup",
) -> list["torch.Tensor | None"]:
    """Single fused all_to_all over the flattened (GTP_remat x TP) domain (backward).

    Inverse of :func:`layer_sharded_fused_fwd`: each NS home slices its full-matrix
    results into the per-source blocks defined there and scatters them back. Every
    rank receives exactly one update shard per param — including non-TP-sharded
    params, whose ``(P/G, Q)`` shard is sent to all T TP peers of each GTP_remat row.

    Args / conventions: see :func:`layer_sharded_fused_fwd`.

    Returns:
        Update shards in ``momentum_list`` order (same shapes as the local shards).
    """
    G, T = gtp_remat_size, tp_size
    S = G * T
    n = len(momentum_list)
    if n == 0:
        return []

    dest = [gh * T + th for gh, th in param_homes]

    # Precondition: each ns_r must span exactly G×T (pd=0) or G (pd=1/None) equal-sized
    # blocks so the uniform-stride slicing below is correct.  Silent corruption otherwise.
    for ns_r, i in zip(ns_results, my_param_indices):
        shape = momentum_list[i].shape
        pd = partition_dims[i]
        if pd == 0:
            assert ns_r.shape[0] == shape[0] * G * T, (
                f"layer_sharded_fused_bwd pd=0: ns_r.shape[0]={ns_r.shape[0]} != "
                f"shard_rows={shape[0]} × G={G} × T={T}"
            )
        elif pd == 1:
            assert ns_r.shape[0] == shape[0] * G and ns_r.shape[1] == shape[1] * T, (
                f"layer_sharded_fused_bwd pd=1: ns_r shape {tuple(ns_r.shape)} != "
                f"({shape[0]}×{G}, {shape[1]}×{T})"
            )
        else:
            assert ns_r.shape[0] == shape[0] * G, (
                f"layer_sharded_fused_bwd pd=None: ns_r.shape[0]={ns_r.shape[0]} != "
                f"shard_rows={shape[0]} × G={G}"
            )

    # --- send: one piece per (my param, destination rank).
    send_parts = []
    input_split_sizes = []
    for s in range(S):
        g_d, t_d = divmod(s, T)
        pieces = []
        for ns_r, i in zip(ns_results, my_param_indices):
            shape = momentum_list[i].shape
            pd = partition_dims[i]
            if pd == 0:
                rows = shape[0]
                pieces.append(ns_r.narrow(0, (t_d * G + g_d) * rows, rows).contiguous().flatten())
            elif pd == 1:
                rows, cols = shape
                pieces.append(
                    ns_r[g_d * rows : (g_d + 1) * rows, t_d * cols : (t_d + 1) * cols]
                    .contiguous()
                    .flatten()
                )
            else:
                rows = shape[0]
                pieces.append(ns_r.narrow(0, g_d * rows, rows).contiguous().flatten())
        if pieces:
            chunk = torch.cat(pieces)
            send_parts.append(chunk)
            input_split_sizes.append(chunk.numel())
        else:
            input_split_sizes.append(0)

    _ref = momentum_list[0]
    send_buf = (
        torch.cat(send_parts)
        if send_parts
        else torch.empty(0, dtype=_ref.dtype, device=_ref.device)
    )

    # --- recv: from each home, its params' shards (ordered by ascending param index,
    # matching that home's my_param_indices construction).
    params_of_home: list[list[int]] = [[] for _ in range(S)]
    for i in range(n):
        params_of_home[dest[i]].append(i)
    output_split_sizes = [
        sum(momentum_list[i].numel() for i in params_of_home[s]) for s in range(S)
    ]

    recv_buf = torch.empty(sum(output_split_sizes), dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=fused_group,
    )

    update_shards: list["torch.Tensor | None"] = [None] * n
    offset = 0
    for s in range(S):
        for i in params_of_home[s]:
            numel = momentum_list[i].numel()
            update_shards[i] = recv_buf[offset : offset + numel].view(momentum_list[i].shape)
            offset += numel

    return update_shards
