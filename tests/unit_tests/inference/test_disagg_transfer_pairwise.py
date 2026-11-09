# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Real NCCL integrity tests for heterogeneous KV and SSM transfers."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.ssm_reshard import SSMShardLayout, SSMStateDims
from megatron.core.inference.disaggregation.transfer_backends.nccl import NcclTransferBackend

LAYERS, HEADS = 6, 4
TOKENS, HEAD_DIM, POOL = 3, 2, 7
SRC_BLOCKS, DST_BLOCKS = [5, 1], [3, 0]
SRC_SLOTS, DST_SLOTS = [4, 1], [2, 5]
SENTINEL = -19.0
SSM_DIMS = SSMStateDims(nheads=4, headdim=2, d_state=2, ngroups=2, d_conv=3)

TOPOLOGIES = [
    pytest.param((2, 1), (1, 1), [4, 1], [3], id="tp2-to-tp1"),
    pytest.param((1, 1), (2, 1), [5], [2, 0], id="tp1-to-tp2"),
    pytest.param((1, 2), (1, 1), [3, 0], [5], id="pp2-to-pp1"),
    pytest.param((2, 2), (1, 2), [5, 1, 4, 0], [3, 2], id="tp2pp2-to-tp1pp2"),
]


def _global_states(device):
    """Canonical request tensors with every coordinate uniquely valued."""
    count = len(SRC_BLOCKS)
    kv = torch.arange(2 * LAYERS * count * TOKENS * HEADS * HEAD_DIM, device=device)
    kv = kv.reshape(2, LAYERS, count, TOKENS, HEADS, HEAD_DIM).float() + 100
    conv_dim = SSM_DIMS.nheads * SSM_DIMS.headdim + 2 * SSM_DIMS.ngroups * SSM_DIMS.d_state
    conv = torch.arange(count * LAYERS * conv_dim * SSM_DIMS.d_conv, device=device)
    conv = conv.reshape(count, LAYERS, conv_dim, SSM_DIMS.d_conv).float() + 10_000
    recurrent = torch.arange(
        count * LAYERS * SSM_DIMS.nheads * SSM_DIMS.headdim * SSM_DIMS.d_state, device=device
    )
    recurrent = (
        recurrent.reshape(
            count, LAYERS, SSM_DIMS.nheads, SSM_DIMS.headdim, SSM_DIMS.d_state
        ).float()
        + 20_000
    )
    return kv, conv, recurrent


def _role_coordinates(rank, ranks, topology):
    tp, pp = topology
    logical_rank = ranks.index(rank)
    return logical_rank % tp, logical_rank // tp, LAYERS // pp


def _ssm_shard(global_state, layout, kind):
    """Direct canonical SSM shard for one exported layout."""
    layer_lo, layer_hi = layout.layer_range()
    tp_rank, tp_size = layout.tp_rank, layout.tp_size
    if kind == "recurrent":
        width = SSM_DIMS.nheads // tp_size
        return global_state[:, layer_lo:layer_hi, tp_rank * width : (tp_rank + 1) * width]

    inner = SSM_DIMS.nheads * SSM_DIMS.headdim
    group = SSM_DIMS.ngroups * SSM_DIMS.d_state
    inner_local, group_local = inner // tp_size, group // tp_size
    x = global_state[:, layer_lo:layer_hi, tp_rank * inner_local : (tp_rank + 1) * inner_local]
    b = global_state[
        :, layer_lo:layer_hi, inner + tp_rank * group_local : inner + (tp_rank + 1) * group_local
    ]
    c = global_state[
        :,
        layer_lo:layer_hi,
        inner + group + tp_rank * group_local : inner + group + (tp_rank + 1) * group_local,
    ]
    return torch.cat((x, b, c), dim=2)


def _make_role_backends(rank, ranks, topology, device, kv, conv, recurrent, source):
    """Construct rank-owned backends and materialize canonical source slots."""
    tp_rank, pp_rank, local_layers = _role_coordinates(rank, ranks, topology)
    tp, pp = topology
    layer_start = pp_rank * local_layers
    head_start = tp_rank * (HEADS // tp)
    kv_buf = torch.full(
        (2, local_layers, POOL, TOKENS, HEADS // tp, HEAD_DIM), SENTINEL, device=device
    )
    kv_backend = NcclTransferBackend(
        agent_name=f"pairwise-kv-r{rank}",
        memory_buffer=kv_buf,
        expected_num_blocks=POOL,
        tp_size=tp,
        tp_rank=tp_rank,
        num_kv_heads_global=HEADS,
        heads_per_partition=HEADS // tp,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS,
        global_rank=rank,
        pp_size=pp,
        pp_rank=pp_rank,
        num_layers_global=LAYERS,
        layer_start=layer_start,
        layer_end=layer_start + local_layers,
    )
    layout = SSMShardLayout(rank, tp, tp_rank, layer_start, local_layers, SSM_DIMS)
    conv_buf = torch.full(
        (local_layers, POOL, layout.conv_dim_local, SSM_DIMS.d_conv), SENTINEL, device=device
    )
    recurrent_buf = torch.full(
        (local_layers, POOL, layout.nheads_local, SSM_DIMS.headdim, SSM_DIMS.d_state),
        SENTINEL,
        device=device,
    )
    conv_backend = NcclTransferBackend(
        agent_name=f"pairwise-conv-r{rank}",
        memory_buffer=conv_buf,
        expected_num_blocks=POOL,
        ssm_layout=layout,
        ssm_state_kind="conv",
    )
    recurrent_backend = NcclTransferBackend(
        agent_name=f"pairwise-recurrent-r{rank}",
        memory_buffer=recurrent_buf,
        expected_num_blocks=POOL,
        ssm_layout=layout,
        ssm_state_kind="recurrent",
    )
    if source:
        layer_slice = slice(layer_start, layer_start + local_layers)
        head_slice = slice(head_start, head_start + HEADS // tp)
        for request, block in enumerate(SRC_BLOCKS):
            kv_buf[:, :, block] = kv[:, layer_slice, request, :, head_slice]
        for request, slot in enumerate(SRC_SLOTS):
            conv_buf[:, slot] = _ssm_shard(conv, layout, "conv")[request]
            recurrent_buf[:, slot] = _ssm_shard(recurrent, layout, "recurrent")[request]
    return {
        "backends": {"kv": kv_backend, "conv": conv_backend, "recurrent": recurrent_backend},
        "buffers": {"kv": kv_buf, "conv": conv_buf, "recurrent": recurrent_buf},
        "layout": layout,
    }


def _wire_meta(all_meta, ranks, topology, kind, ids):
    """Rebuild the TP-within-PP metadata envelope used by handoff messages."""
    tp, pp = topology
    if pp == 1:
        return {"tp_metas": [all_meta[rank][kind] for rank in ranks]}
    return {
        "pp_metas": [
            {
                "tp_metas": [all_meta[rank][kind] for rank in ranks[p * tp : (p + 1) * tp]],
                "block_ids": ids,
            }
            for p in range(pp)
        ]
    }


def _settle(handle):
    """Exercise both completion APIs repeatedly, including after scatter."""
    handle.poll()
    handle.wait()
    assert handle.poll()
    handle.wait()
    assert handle.poll()


def _expected(role, kv, conv, recurrent):
    layout = role["layout"]
    layer_lo, layer_hi = layout.layer_range()
    head_lo = layout.tp_rank * (HEADS // layout.tp_size)
    head_hi = head_lo + HEADS // layout.tp_size
    expected = {name: torch.full_like(buf, SENTINEL) for name, buf in role["buffers"].items()}
    for request, block in enumerate(DST_BLOCKS):
        expected["kv"][:, :, block] = kv[:, layer_lo:layer_hi, request, :, head_lo:head_hi]
    conv_shard = _ssm_shard(conv, layout, "conv")
    recurrent_shard = _ssm_shard(recurrent, layout, "recurrent")
    for request, slot in enumerate(DST_SLOTS):
        expected["conv"][:, slot] = conv_shard[request]
        expected["recurrent"][:, slot] = recurrent_shard[request]
    return expected


@pytest.fixture(scope="module")
def distributed_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    control_group = dist.new_group(backend="gloo")
    dist.barrier(device_ids=[local_rank])  # Collectively initialize the common NCCL world.
    return dist.get_rank(), torch.device("cuda", local_rank), control_group


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 6
        and int(os.environ.get("WORLD_SIZE", "1")) == 6
    ),
    reason="requires torchrun with exactly six CUDA ranks",
)
@pytest.mark.parametrize("src_topology,dst_topology,src_ranks,dst_ranks", TOPOLOGIES)
def test_pairwise_kv_and_ssm_transfer_integrity(
    distributed_world, src_topology, dst_topology, src_ranks, dst_ranks
):
    rank, device, control_group = distributed_world
    kv, conv, recurrent = _global_states(device)
    role = None
    if rank in src_ranks:
        role = _make_role_backends(rank, src_ranks, src_topology, device, kv, conv, recurrent, True)
    elif rank in dst_ranks:
        role = _make_role_backends(
            rank, dst_ranks, dst_topology, device, kv, conv, recurrent, False
        )

    local_meta = (
        {}
        if role is None
        else {kind: backend.export_meta() for kind, backend in role["backends"].items()}
    )
    all_meta = [None] * dist.get_world_size()
    dist.all_gather_object(all_meta, local_meta, group=control_group)

    for kind, src_ids, dst_ids in (
        ("kv", SRC_BLOCKS, DST_BLOCKS),
        ("conv", SRC_SLOTS, DST_SLOTS),
        ("recurrent", SRC_SLOTS, DST_SLOTS),
    ):
        if rank in src_ranks:
            peer = _wire_meta(all_meta, dst_ranks, dst_topology, kind, dst_ids)
            _settle(role["backends"][kind].begin_push_blocks(peer, src_ids))
        elif rank in dst_ranks:
            peer = _wire_meta(all_meta, src_ranks, src_topology, kind, src_ids)
            _settle(role["backends"][kind].begin_pull_blocks(peer, src_ids, dst_ids))
        dist.barrier(group=control_group)

    local_ok = True
    witnesses = {}
    if role is not None:
        witnesses = {
            kind: float(role["buffers"][kind][role["buffers"][kind] != SENTINEL].sum())
            for kind in role["buffers"]
        }
    if rank in dst_ranks:
        expected = _expected(role, kv, conv, recurrent)
        local_ok = all(torch.equal(role["buffers"][kind], expected[kind]) for kind in expected)
    status = torch.tensor(int(local_ok))
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=control_group)
    role_name = "src" if rank in src_ranks else "dst" if rank in dst_ranks else "unused"
    kv_meta = local_meta.get("kv", {})
    print(
        f"PAIRWISE_WITNESS rank={rank} role={role_name} src={src_topology} dst={dst_topology} "
        f"transport={kv_meta.get('transport')} tp_rank={kv_meta.get('tp_rank')} "
        f"pp_rank={kv_meta.get('pp_rank')} layers={kv_meta.get('layer_start'), kv_meta.get('layer_end')} "
        f"checksums={witnesses} ok={local_ok}",
        flush=True,
    )
    assert status.item() == 1
