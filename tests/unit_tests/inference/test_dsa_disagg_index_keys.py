# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""DSA index-key transfer for disaggregated prefill -> decode handoffs."""

import os

import pytest
import torch
import torch.distributed as dist

from megatron.core.inference.disaggregation.transfer_backends.nccl import NcclTransferBackend
from megatron.core.inference.disaggregation.utils import (
    drop_transfer_prefix_blocks,
    dsa_index_key_transfer_view,
    dsa_transfer_meta,
    transfer_peer_records,
)

LAYERS, HEADS, TOKENS, HEAD_DIM, POOL = 4, 2, 128, 2, 9
INDEX_KEYS = {"bf16": (torch.bfloat16, TOKENS, 128)}
SRC_BLOCKS, DST_BLOCKS = [6, 2, 4], [1, 7, 3]
CACHED = 1  # The decode already holds the first block (prefix-cache hit), keys included.
SENTINEL = 11

TOPOLOGIES = [
    pytest.param((2, 1), (1, 1), [0, 1], [2], id="tp2-to-tp1"),
    pytest.param((1, 1), (2, 1), [0], [2, 3], id="tp1-to-tp2"),
    pytest.param((1, 2), (1, 1), [0, 1], [3], id="pp2-to-pp1"),
]


def _rank_meta(kv_meta, dsa_meta, dtype):
    return dict(kv_meta, dsa=dict(dsa_meta, index_key_dtype=dtype))


def test_index_key_view_is_one_plane_of_one_head():
    for dtype, (torch_dtype, tokens, dim) in INDEX_KEYS.items():
        cache = torch.zeros(LAYERS, POOL, tokens, dim, dtype=torch_dtype)
        view = dsa_index_key_transfer_view(cache)
        assert view.shape == (1, LAYERS, POOL, tokens, 1, dim), dtype
        assert view.data_ptr() == cache.data_ptr() and view.is_contiguous()
    with pytest.raises(ValueError, match="must be 4-D"):
        dsa_index_key_transfer_view(torch.zeros(2, LAYERS, POOL, TOKENS, 1, 1))


def test_transfer_meta_keeps_nesting_and_trimmed_block_ids():
    local = {"index_key_dtype": "bf16", "head_dim": 128, "tokens_per_block": 2, "element_size": 2}
    rank = lambda name: {"agent": f"kv-{name}", "dsa": dict(local, agent=f"dsa-{name}")}
    meta = {
        "resume_tokens": [5],
        "pp_metas": [
            {"tp_metas": [rank("a"), rank("b")], "block_ids": [6, 2, 4]},
            {"tp_metas": rank("c"), "block_ids": [8, 0, 5]},
        ],
    }
    trimmed, src = drop_transfer_prefix_blocks(meta, [6, 2, 4], CACHED)
    converted = dsa_transfer_meta(trimmed, local)
    records = transfer_peer_records(converted, src)
    assert [(peer["agent"], blocks) for peer, blocks in records] == [
        ("dsa-a", [2, 4]),
        ("dsa-b", [2, 4]),
        ("dsa-c", [0, 5]),
    ]
    flat = dsa_transfer_meta(dict(rank("d"), block_ids=[3]), local)
    assert flat["agent"] == "dsa-d" and flat["block_ids"] == [3]
    assert dsa_transfer_meta({"tp_metas": [rank("e")]}, local) == {
        "tp_metas": [dict(local, agent="dsa-e")]
    }


@pytest.mark.parametrize(
    "peer, error",
    [
        ({"agent": "kv"}, "does not export DSA index keys"),
        ({"dsa": {"index_key_dtype": "bf16", "head_dim": 64}}, "head_dim: peer=64 local=128"),
        (
            {"dsa": {"index_key_dtype": "bf16", "head_dim": 128, "tokens_per_block": 4}},
            "tokens_per_block: peer=4 local=2",
        ),
    ],
)
def test_transfer_meta_rejects_incompatible_peers(peer, error):
    local = {"index_key_dtype": "bf16", "head_dim": 128, "tokens_per_block": 2, "element_size": 2}
    with pytest.raises(ValueError, match=error):
        dsa_transfer_meta({"tp_metas": [peer]}, local)


def _canonical(device):
    """One request's KV ([2, L, blocks, T, H, d]) and index keys ([L, blocks, T', d'])."""
    generator = torch.Generator().manual_seed(3)
    count = len(SRC_BLOCKS)
    kv = torch.randn(2, LAYERS, count, TOKENS, HEADS, HEAD_DIM, generator=generator)
    keys = {}
    for dtype, (torch_dtype, tokens, dim) in INDEX_KEYS.items():
        shape = (LAYERS, count, tokens, dim)
        keys[dtype] = torch.randn(shape, generator=generator).to(torch_dtype)
    return kv.to(device), {dtype: value.to(device) for dtype, value in keys.items()}


def _make_rank(rank, ranks, topology, device, dtype, kv, keys, source):
    tp, pp = topology
    logical = ranks.index(rank)
    tp_rank, pp_rank = logical % tp, logical // tp
    local_layers = LAYERS // pp
    layer_start = pp_rank * local_layers
    layers = slice(layer_start, layer_start + local_layers)
    heads = slice(tp_rank * (HEADS // tp), (tp_rank + 1) * (HEADS // tp))
    torch_dtype, tokens, dim = INDEX_KEYS[dtype]
    kv_buf = torch.full(
        (2, local_layers, POOL, TOKENS, HEADS // tp, HEAD_DIM), SENTINEL, device=device
    )
    key_buf = torch.full((local_layers, POOL, tokens, dim), SENTINEL, dtype=torch_dtype)
    key_buf = key_buf.to(device)
    common = dict(
        expected_num_blocks=POOL,
        tp_size=tp,
        tp_rank=tp_rank,
        global_rank=rank,
        pp_size=pp,
        pp_rank=pp_rank,
        num_layers_global=LAYERS,
        layer_start=layer_start,
        layer_end=layer_start + local_layers,
    )
    kv_backend = NcclTransferBackend(
        agent_name=f"kv-r{rank}",
        memory_buffer=kv_buf,
        num_kv_heads_global=HEADS,
        heads_per_partition=HEADS // tp,
        head_dim=HEAD_DIM,
        tokens_per_block=TOKENS,
        **common,
    )
    key_view = dsa_index_key_transfer_view(key_buf)
    key_backend = kv_backend.new_registered_buffer(
        agent_name=f"dsa-r{rank}",
        memory_buffer=key_view,
        num_kv_heads_global=1,
        heads_per_partition=1,
        head_dim=dim,
        tokens_per_block=tokens,
        **common,
    )
    blocks = SRC_BLOCKS if source else DST_BLOCKS
    for index, block in enumerate(blocks):
        if source or index < CACHED:
            # Keys are replicated on every TP rank, like the duplicated indexer projection.
            kv_buf[:, :, block] = kv[:, layers, index, :, heads]
            key_buf[:, block] = keys[dtype][layers, index]
    return {
        "kv": kv_backend,
        "dsa": key_backend,
        "kv_buf": kv_buf,
        "key_buf": key_buf,
        "layers": layers,
        "heads": heads,
    }


def _wire(all_meta, ranks, topology, ids):
    tp, pp = topology
    if pp == 1:
        return {"tp_metas": [all_meta[rank] for rank in ranks]}
    return {
        "pp_metas": [
            {
                "tp_metas": [all_meta[rank] for rank in ranks[p * tp : (p + 1) * tp]],
                "block_ids": ids,
            }
            for p in range(pp)
        ]
    }


@pytest.fixture(scope="module")
def distributed_world():
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    control_group = dist.new_group(backend="gloo")
    dist.barrier(device_ids=[local_rank])
    return dist.get_rank(), torch.device("cuda", local_rank), control_group


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 4
        and int(os.environ.get("WORLD_SIZE", "1")) == 4
    ),
    reason="requires torchrun with exactly four CUDA ranks",
)
@pytest.mark.parametrize("dtype", ["bf16"])
@pytest.mark.parametrize("src_topology,dst_topology,src_ranks,dst_ranks", TOPOLOGIES)
def test_nccl_index_keys_follow_kv_blocks(
    distributed_world, dtype, src_topology, dst_topology, src_ranks, dst_ranks
):
    """KV and index keys move in one NCCL batch, remapped to the decode's blocks, skipping the
    decode's cached prefix and resharding KV heads while keys come from one TP replica."""
    rank, device, control_group = distributed_world
    kv, keys = _canonical(device)
    role = None
    if rank in src_ranks:
        role = _make_rank(rank, src_ranks, src_topology, device, dtype, kv, keys, True)
    elif rank in dst_ranks:
        role = _make_rank(rank, dst_ranks, dst_topology, device, dtype, kv, keys, False)
    local = None
    if role is not None:
        local = _rank_meta(role["kv"].export_meta(), role["dsa"].export_meta(), dtype)
    all_meta = [None] * dist.get_world_size()
    dist.all_gather_object(all_meta, local, group=control_group)

    if rank in src_ranks:
        # SEND_KV: the decode's metadata with its uncached destination blocks.
        peer = _wire(all_meta, dst_ranks, dst_topology, DST_BLOCKS[CACHED:])
        blocks = SRC_BLOCKS[CACHED:]
        plans = [
            role["kv"].prepare_push_blocks(peer, blocks),
            role["dsa"].prepare_push_blocks(dsa_transfer_meta(peer, local["dsa"]), blocks),
        ]
        role["kv"].start_prepared(plans).wait()
    elif rank in dst_ranks:
        handoff = _wire(all_meta, src_ranks, src_topology, SRC_BLOCKS)
        peer, blocks = drop_transfer_prefix_blocks(handoff, SRC_BLOCKS, CACHED)
        plans = [
            role["kv"].prepare_pull_blocks(peer, blocks, DST_BLOCKS[CACHED:]),
            role["dsa"].prepare_pull_blocks(
                dsa_transfer_meta(peer, local["dsa"]), blocks, DST_BLOCKS[CACHED:]
            ),
        ]
        handle = role["kv"].start_prepared(plans)
        handle.wait()
        assert handle.poll()
    dist.barrier(group=control_group)

    ok = True
    if rank in dst_ranks:
        expected_kv = torch.full_like(role["kv_buf"], SENTINEL)
        expected_keys = torch.full_like(role["key_buf"], SENTINEL)
        for index, block in enumerate(DST_BLOCKS):
            expected_kv[:, :, block] = kv[:, role["layers"], index, :, role["heads"]]
            expected_keys[:, block] = keys[dtype][role["layers"], index]
        ok = torch.equal(role["kv_buf"], expected_kv) and torch.equal(
            role["key_buf"], expected_keys
        )
    status = torch.tensor(int(ok))
    dist.all_reduce(status, op=dist.ReduceOp.MIN, group=control_group)
    assert status.item() == 1
