# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Distributed unit test of the two-sided NCCL transfer backend.

Prefill TP2 -> decode TP1 and prefill TP1 -> decode TP2 on real GPUs: decode posts
begin_pull_blocks, prefill posts the matching begin_push_blocks, and the
decode's paged buffer must end up byte-identical to a direct shard of a known
global KV, including remapped destination blocks. Exercises head merging and
splitting through the same reshard plan the
NIXL backend uses. The test uses the process group provided by the unit-test
runner instead of spawning a nested distributed job.
"""

import os

import pytest
import torch

L, H, HD, T, NB = 4, 8, 16, 8, 6  # layers, kv heads, head dim, tokens/block, pool blocks
BLOCKS = [1, 3]  # The request's source blocks.


def _global_blocks():
    """Global KV for the request's blocks: (block, kv, layer, token, head, dim)
    with a distinct value per (block, kv, layer, head)."""
    g = torch.zeros(len(BLOCKS), 2, L, T, H, HD)
    for b in range(len(BLOCKS)):
        for kv in range(2):
            for l in range(L):
                for h in range(H):
                    g[b, kv, l, :, h, :] = ((b * 2 + kv) * L + l) * 100 + h
    return g


def _backend(rank, tp_size, tp_rank, device):
    from megatron.core.inference.disaggregation.transfer_backends.nccl import NcclTransferBackend

    heads_local = H // tp_size
    buf = torch.zeros(2, L, NB, T, heads_local, HD, device=device)
    backend = NcclTransferBackend(
        agent_name=f"test-rank{rank}",
        memory_buffer=buf,
        expected_num_blocks=NB,
        tp_size=tp_size,
        tp_rank=tp_rank,
        num_kv_heads_global=H,
        heads_per_partition=heads_local,
        head_dim=HD,
        tokens_per_block=T,
        global_rank=rank,
        pp_size=1,
        pp_rank=0,
        num_layers_global=L,
        layer_start=0,
        layer_end=L,
    )
    return backend, buf


def _meta_stub(rank, tp_size, tp_rank):
    """A rank's export_meta, built without its backend; the address fields are
    unused by NCCL and the geometry is deterministic."""
    heads_local = H // tp_size
    return {
        "transport": "nccl",
        "nccl_rank": rank,
        "num_blocks": NB,
        "blocks_axis": 2,
        "num_outer": 2 * L,
        "heads_per_partition": heads_local,
        "head_dim": HD,
        "tokens_per_block": T,
        "element_size": 4,
        "bytes_per_slice": T * heads_local * HD * 4,
        "outer_stride_bytes": NB * T * heads_local * HD * 4,
        "base_addr": 0,
        "device_id": rank,
        "global_rank": rank,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
        "pp_size": 1,
        "pp_rank": 0,
        "num_layers_global": L,
        "num_kv_heads_global": H,
        "layer_start": 0,
        "layer_end": L,
    }


@pytest.mark.skipif(
    not (
        torch.cuda.is_available()
        and torch.cuda.device_count() >= 3
        and int(os.environ.get("WORLD_SIZE", "1")) >= 3
    ),
    reason="requires torchrun with >=3 CUDA ranks for prefill/decode TP2/TP1 layouts",
)
@pytest.mark.parametrize("prefill_tp,decode_tp", [(2, 1), (1, 2)])
@pytest.mark.parametrize(
    "destination_blocks", [BLOCKS, [4, 0]], ids=["same-blocks", "remapped-blocks"]
)
def test_nccl_push_pull_tp_layouts(prefill_tp, decode_tp, destination_blocks):
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    if not dist.is_initialized():
        dist.init_process_group("nccl")

    rank = dist.get_rank()
    control_group = dist.new_group(backend="gloo")

    # Initialize the default NCCL communicator collectively before ranks 0–2
    # use it for point-to-point transfers. Extra CI ranks synchronize through
    # the Gloo control group and do not issue conflicting NCCL collectives.
    dist.barrier()

    transfer_ok = True
    g = _global_blocks().to(device)
    if rank < prefill_tp:
        backend, buf = _backend(rank, prefill_tp, rank, device)
        heads = slice(rank * (H // prefill_tp), (rank + 1) * (H // prefill_tp))
        for i, block in enumerate(BLOCKS):
            # buffer layout [2, L, B, T, h, d]
            buf[:, :, block] = g[i, :, :, :, heads, :]
        # In production the decode's metas arrive in SEND_KV.
        metas = [
            _meta_stub(prefill_tp + tp_rank, decode_tp, tp_rank) for tp_rank in range(decode_tp)
        ]
        handle = backend.begin_push_blocks({"tp_metas": metas}, BLOCKS)
        handle.wait()
    elif rank < prefill_tp + decode_tp:
        tp_rank = rank - prefill_tp
        backend, buf = _backend(rank, decode_tp, tp_rank, device)
        # In production the prefills' metas arrive in the hand-off kv_meta.
        metas = [_meta_stub(src_rank, prefill_tp, src_rank) for src_rank in range(prefill_tp)]
        handle = backend.begin_pull_blocks({"tp_metas": metas}, BLOCKS, destination_blocks)
        handle.wait()
        expected = torch.zeros_like(buf)
        heads = slice(tp_rank * (H // decode_tp), (tp_rank + 1) * (H // decode_tp))
        for i, block in enumerate(destination_blocks):
            expected[:, :, block] = g[i, :, :, :, heads, :]
        transfer_ok = torch.equal(buf, expected)

    result = torch.tensor(int(transfer_ok))
    dist.all_reduce(result, op=dist.ReduceOp.MIN, group=control_group)
    assert result.item() == 1
