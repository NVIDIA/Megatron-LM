# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""all_to_all routing utilities for layer-sharded Muon.

Layer sharding assigns every 2D weight one Newton-Schulz "home" rank inside the
(GTP x TP) weight-shard domain. The forward exchanges route each rank's local
momentum shards so the home assembles the complete matrix; the backward
exchanges scatter the orthogonalized result back to the original shards.

Two equivalent exchange strategies are provided:

- Two-stage (``layer_sharded_all_to_all_{fwd,bwd}``): one all_to_all over the
  GTP group (dim 0), then one over the TP group (along ``partition_dim``),
  reusing the existing process groups.
- Fused (``layer_sharded_fused_{fwd,bwd}``): a single all_to_all per direction
  over the flattened (GTP x TP) domain group. It moves the exact same shard
  blocks and assembles them in the exact same order, so the NS input is
  bit-identical to the two-stage path.

All functions support heterogeneous parameter shapes and uneven home
assignments (ranks may own zero matrices in a given exchange).
"""

import torch


def layer_sharded_all_to_all_fwd(
    momentum_list: list[torch.Tensor],
    param_to_gtp_rank: dict,
    gtp_rank: int,
    gtp_size: int,
    gtp_group: "torch.distributed.ProcessGroup",
    gtp_dim: int = 0,
) -> tuple[list[torch.Tensor], list[int]]:
    """Forward all_to_all for layer sharding: redistribute GTP momentum shards.

    Each GPU holds (P/S, Q) momentum shards for all L/DP params. This redistributes
    them so each GPU ends up with complete (P, Q) momentum for its GTP-assigned subset.

    Args:
        momentum_list: List of momentum tensors, one per param. Each has shape (P/S, Q)
            where S = gtp_size (the GTP row shard on this GPU).
        param_to_gtp_rank: Dict mapping param index -> NS home GTP rank.
        gtp_rank: This GPU's rank within gtp_group.
        gtp_size: Size of gtp_group.
        gtp_group: The GTP process group to communicate within.
        gtp_dim: Dimension sharded by GTP (0 for column-parallel weights).

    Returns:
        Tuple of:
            - complete_momentums: List of complete (P, Q) tensors for params assigned
              to this GPU (ns_home == gtp_rank), in the order they appear in momentum_list.
            - my_param_indices: Indices into momentum_list for params assigned to this GPU.
    """
    # Group params by their NS home GTP rank
    params_for_rank = [[] for _ in range(gtp_size)]
    for i, m in enumerate(momentum_list):
        home = param_to_gtp_rank.get(i, i % gtp_size)
        params_for_rank[home].append((i, m))

    # Build flat send buffer: [data_for_rank_0 | data_for_rank_1 | ...]
    # For each destination g', send my momentum shards for params assigned to g'
    send_parts = []
    input_split_sizes = []
    for g_prime in range(gtp_size):
        if params_for_rank[g_prime]:
            chunk = torch.cat([m.contiguous().flatten() for _, m in params_for_rank[g_prime]])
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

    # Compute recv split sizes: from each source g', we receive momentum shards
    # for all params assigned to us (gtp_rank). Each shard has the same numel as the param.
    my_params = params_for_rank[gtp_rank]
    my_param_numel = sum(m.numel() for _, m in my_params)
    output_split_sizes = [my_param_numel] * gtp_size

    recv_buf = torch.empty(my_param_numel * gtp_size, dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=gtp_group,
    )

    # Unpack: for each of my assigned params, concatenate GTP shards from all sources.
    # recv_buf layout: [from_g0_for_my_params | from_g1_for_my_params | ... | from_g(S-1)_for_my_params]
    # Each from_gk block contains shards for my params in order.
    my_param_indices = [i for i, _ in my_params]

    # Prefix offsets of each of my params within one source block. Precomputed:
    # deriving them inline is O(n^2 * gtp_size), and n reaches the hundreds when a
    # home owns many same-shape expert weights.
    param_offsets = [0]
    for _, m in my_params:
        param_offsets.append(param_offsets[-1] + m.numel())

    complete_momentums = []
    for param_idx, (_, m_template) in enumerate(my_params):
        numel = m_template.numel()
        offset = param_offsets[param_idx]
        # Slices of a 1-D contiguous buffer are already contiguous, so cat can take
        # the views directly.
        shards = [
            recv_buf[
                g_prime * my_param_numel + offset : g_prime * my_param_numel + offset + numel
            ].view(m_template.shape)
            for g_prime in range(gtp_size)
        ]
        complete_momentums.append(torch.cat(shards, dim=gtp_dim))  # (P, Q)

    return complete_momentums, my_param_indices


def layer_sharded_all_to_all_bwd(
    ns_results: list[torch.Tensor],
    my_param_indices: list[int],
    momentum_list: list[torch.Tensor],
    param_to_gtp_rank: dict,
    gtp_rank: int,
    gtp_size: int,
    gtp_group: "torch.distributed.ProcessGroup",
    gtp_dim: int = 0,
) -> list[torch.Tensor | None]:
    """Backward all_to_all for layer sharding: distribute NS results as GTP shards.

    Each NS-home GPU has complete (P, Q) NS results for its assigned params. This
    redistributes them so every GPU gets the (P/S, Q) GTP row shard for all L/DP params.

    Args:
        ns_results: Complete (P, Q) NS result tensors, one per assigned param, in order
            corresponding to my_param_indices.
        my_param_indices: Indices into momentum_list for params assigned to this GPU.
        momentum_list: List of original momentum tensors (provides shapes).
        param_to_gtp_rank: Dict mapping param index -> NS home GTP rank.
        gtp_rank: This GPU's rank within gtp_group.
        gtp_size: Size of gtp_group.
        gtp_group: The GTP process group to communicate within.
        gtp_dim: Dimension sharded by GTP (0 for column-parallel weights).

    Returns:
        List of (P/S, Q) NS update shards, one per param in momentum_list order.
        None for params that did not participate (should not occur in normal usage).
    """
    if not momentum_list:
        return []

    # Group params by their NS home GTP rank (same grouping as fwd)
    params_for_rank = [[] for _ in range(gtp_size)]
    for i, m in enumerate(momentum_list):
        home = param_to_gtp_rank.get(i, i % gtp_size)
        params_for_rank[home].append((i, m))

    # Precondition: ns_r must span exactly gtp_size equal-sized shards so the
    # uniform-stride narrow below is correct.  A violated invariant produces silent
    # corruption (narrow is in-bounds but slices the wrong rows).
    if ns_results:
        for ns_r, idx in zip(ns_results, my_param_indices):
            expected = momentum_list[idx].shape[gtp_dim] * gtp_size
            assert ns_r.shape[gtp_dim] == expected, (
                f"layer_sharded_all_to_all_bwd: full-matrix dim[{gtp_dim}]="
                f"{ns_r.shape[gtp_dim]} != shard_size="
                f"{momentum_list[idx].shape[gtp_dim]} × gtp_size={gtp_size}; "
                "all shards must be equal-sized (divisibility/padding invariant violated)."
            )

    # Build send buffer: for each destination g', send that rank's GTP row shard
    # of each of MY ns_results. Shard size is per-param (heterogeneous shapes).
    send_parts = []
    input_split_sizes = []
    for g_prime in range(gtp_size):
        if ns_results:
            chunk = torch.cat(
                [
                    ns_r.narrow(
                        gtp_dim,
                        g_prime * momentum_list[idx].shape[gtp_dim],
                        momentum_list[idx].shape[gtp_dim],
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

    # Recv: from each source g', receive the GTP row shard for MY rank of their NS results
    output_split_sizes = []
    for g_prime in range(gtp_size):
        # Number of params assigned to g', each contributing shard_size * Q elements
        g_prime_params = params_for_rank[g_prime]
        total_numel = sum(m.numel() for _, m in g_prime_params)  # P/S * Q * n_params
        output_split_sizes.append(total_numel)

    recv_buf = torch.empty(sum(output_split_sizes), dtype=send_buf.dtype, device=send_buf.device)

    torch.distributed.all_to_all_single(
        recv_buf,
        send_buf,
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=gtp_group,
    )

    # Unpack into per-param update shards
    update_shards: list[torch.Tensor | None] = [None] * len(momentum_list)
    offset = 0
    for g_prime in range(gtp_size):
        for i, m_template in params_for_rank[g_prime]:
            shard = recv_buf[offset : offset + m_template.numel()].view(m_template.shape)
            update_shards[i] = shard.contiguous()
            offset += m_template.numel()

    return update_shards


def layer_sharded_fused_fwd(
    momentum_list: list[torch.Tensor],
    param_homes: list[tuple[int, int]],
    partition_dims: list["int | None"],
    g_rank: int,
    t_rank: int,
    gtp_size: int,
    tp_size: int,
    fused_group: "torch.distributed.ProcessGroup",
) -> tuple[list[torch.Tensor], list[int]]:
    """Single fused all_to_all over the flattened (GTP x TP) domain (forward).

    Functionally identical to the two-stage ``layer_sharded_all_to_all_fwd`` (over GTP)
    followed by a second stage over TP: the exact same shard blocks travel to the same
    NS home and are concatenated in the exact same order, so the assembled full matrix
    is bit-identical. One collective replaces up to four.

    Rank convention: the caller must construct ``fused_group`` so that its group rank
    ``g * tp_size + t`` is the process with coordinates ``(g, t)`` in (gtp_group,
    tp_group) — i.e. TP innermost, matching Megatron's ``tp-gtp_remat-...`` order.

    Sharding model per full matrix ``(P, Q)`` (mirrors the two-stage path):
      - ``partition_dim == 0``: TP shards dim 0, then GTP shards the TP-local rows.
        Source ``(g, t)`` holds full rows ``[t*P/T + g*P/(T*G), ...)`` — block index
        ``t*G + g`` along dim 0.
      - ``partition_dim == 1``: GTP shards dim 0, TP shards dim 1. Source ``(g, t)``
        holds the 2-D block ``[g*P/G:(g+1)*P/G, t*Q/T:(t+1)*Q/T]``.
      - ``partition_dim is None``: not TP-sharded; every TP peer holds the identical
        ``(P/G, Q)`` shard, so only sources with ``t == t_home`` contribute.

    Args:
        momentum_list: Local momentum shard per param.
        param_homes: ``(g_home, t_home)`` per param.
        partition_dims: TP partition dim per param (0, 1, or None).
        g_rank / t_rank: This rank's coordinates.
        gtp_size / tp_size: Domain extents.
        fused_group: Flattened process group of size ``gtp_size * tp_size``.

    Returns:
        ``(full_mats, my_param_indices)`` — complete matrices for params homed on this
        rank, and their indices into ``momentum_list``.
    """
    G, T = gtp_size, tp_size
    S = G * T
    my_flat = g_rank * T + t_rank
    n = len(momentum_list)

    dest = [gh * T + th for gh, th in param_homes]

    # --- send: my local shard of param i goes to its home, except that for
    # non-TP-sharded params only the t == t_home column contributes (all TP peers
    # hold identical data; sending T copies would be pure waste).
    send_lists: list[list[int]] = [[] for _ in range(S)]
    for i in range(n):
        if partition_dims[i] is None and t_rank != param_homes[i][1]:
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
    # non-TP-sharded ones arrive only from the t_s == t_rank column.
    my_param_indices = [i for i in range(n) if dest[i] == my_flat]
    contrib: list[list[int]] = []
    output_split_sizes = []
    for s in range(S):
        t_s = s % T
        lst = [i for i in my_param_indices if partition_dims[i] is not None or t_s == t_rank]
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
            blocks = [piece[(g * T + t_rank, i)] for g in range(G)]
            full_mats.append(torch.cat(blocks, dim=0))

    return full_mats, my_param_indices


def layer_sharded_fused_bwd(
    ns_results: list[torch.Tensor],
    my_param_indices: list[int],
    momentum_list: list[torch.Tensor],
    param_homes: list[tuple[int, int]],
    partition_dims: list["int | None"],
    g_rank: int,
    t_rank: int,
    gtp_size: int,
    tp_size: int,
    fused_group: "torch.distributed.ProcessGroup",
) -> list["torch.Tensor | None"]:
    """Single fused all_to_all over the flattened (GTP x TP) domain (backward).

    Inverse of :func:`layer_sharded_fused_fwd`: each NS home slices its full-matrix
    results into the per-source blocks defined there and scatters them back. Every
    rank receives exactly one update shard per param — including non-TP-sharded
    params, whose ``(P/G, Q)`` shard is sent to all T TP peers of each GTP row.

    Args / conventions: see :func:`layer_sharded_fused_fwd`.

    Returns:
        Update shards in ``momentum_list`` order (same shapes as the local shards).
    """
    G, T = gtp_size, tp_size
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
