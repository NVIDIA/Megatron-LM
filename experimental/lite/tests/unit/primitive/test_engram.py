# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from megatron.lite.primitive.modules.engram import Engram, ShardedFrozenTable


def test_engram_masked_rows_pass_the_residual_through():
    engram = Engram(4, 2, torch.nn.Embedding(6, 4), torch.nn.Linear(8, 12, bias=False))
    hidden, ids = torch.randn(3, 1, 2, 4), torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]]])
    out = engram(hidden, ids, torch.tensor([[True], [False], [True]]))
    assert torch.equal(out[1], hidden[1]) and not torch.equal(out[0], hidden[0])


def test_release_load_table_allocates_exact_host_bytes_without_random_init(monkeypatch):
    monkeypatch.setattr(torch, "randn", None)  # no random FP32 table when the release is loaded
    table = ShardedFrozenTable(5, 64, SimpleNamespace(size=lambda: 2), initialize=False)
    weight, scale = table.embedding.weight, table.embedding.scale  # 3 rows per rank
    assert (weight.shape, scale.shape, weight.untyped_storage().nbytes()) == ((3, 64), (3, 2), 192)
    assert (weight.dtype, scale.dtype) == (torch.float8_e4m3fn, torch.float8_e8m0fnu)


def _worker(rank, root):
    from megatron.core import dist_checkpointing
    from megatron.lite.primitive.ckpt.row_state import attach_checkpoint_metadata

    torch.cuda.set_device(rank)
    store = f"file://{root}/store"
    dist.init_process_group("cpu:gloo,cuda:nccl", init_method=store, rank=rank, world_size=2)
    group, torch_rng = dist.group.WORLD, torch.manual_seed(rank)
    before = torch.cuda.memory_allocated()
    table = ShardedFrozenTable(5000, 256, group)  # 2500 rows of FP8 + E8M0 scale per rank
    weight, scale = table.embedding.weight, table.embedding.scale
    assert weight.device.type == "cpu" and weight.is_pinned() and scale.is_pinned()
    assert torch.cuda.memory_allocated() == before and not list(table.parameters())
    table.cuda()  # device moves leave the host FP8 bytes in place
    assert table.embedding.weight is weight and weight.dtype == torch.float8_e4m3fn
    parts = [None, None]  # as bytes: Core's object unpickler has no FP8 storage type
    dist.all_gather_object(parts, [t.view(torch.uint8).clone() for t in (weight, scale)])
    full, scales = (torch.cat(p).cuda().view(t.dtype) for p, t in zip(zip(*parts), (weight, scale)))
    ids = torch.tensor([[0, 4999], [2500, 3], [1, 2499]], device="cuda")[rank:]  # 3 and 2 rows
    # The official decode: FP32 product per 32-wide block, then BF16.
    expected = full[ids].float().unflatten(-1, (-1, 32)) * scales[ids].float().unsqueeze(-1)
    assert torch.equal(table(ids), expected.flatten(-2).to(torch.bfloat16))
    module = torch.nn.Module()
    module.embed, ps = table, SimpleNamespace(dp_cp_rank=rank, dp_cp_group=group)
    attach_checkpoint_metadata([module], ps)
    Path(root, "new").mkdir(exist_ok=True)  # both ranks
    dist_checkpointing.save(module.sharded_state_dict(), f"{root}/new")
    weight.view(torch.uint8).zero_(), scale.view(torch.uint8).zero_()
    dist_checkpointing.load(module.sharded_state_dict(), f"{root}/new")
    assert all(
        torch.equal(a.view(torch.uint8), b.view(torch.uint8))
        for a, b in zip((weight, scale), parts[rank])  # pre-save clones
    )
    dist.destroy_process_group()


@pytest.mark.gpus(2, min_architecture="blackwell")
def test_host_fp8_table_matches_the_official_decode_and_checkpoints_its_bytes(tmp_path):
    torch.multiprocessing.spawn(_worker, args=(str(tmp_path),), nprocs=2)


def _ragged_cpu(rank, root):
    options = dict(rank=rank, world_size=2, timeout=timedelta(seconds=10))
    dist.init_process_group("gloo", init_method=f"file://{root}/store", **options)
    try:
        table = ShardedFrozenTable(6, 32, dist.group.WORLD, initialize=False)
        full = torch.arange(192).reshape(6, 32).remainder(8).to(torch.float8_e4m3fn)
        table.embedding.weight.copy_(full.chunk(2)[rank])
        table.embedding.scale.view(torch.uint8).fill_(127)
        for counts in ((3, 1), (2, 0), (0, 2), (0, 0)):
            ids = (torch.arange(counts[rank] * 24) + rank).remainder(6).reshape(-1, 1, 24)
            actual, expected = table(ids), full.float()[ids].bfloat16()
            assert actual.shape == expected.shape and actual.dtype == expected.dtype
            assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    finally:
        dist.destroy_process_group()


def test_host_table_ragged_3d_ids_including_empty_ranks(tmp_path):
    torch.multiprocessing.spawn(_ragged_cpu, args=(str(tmp_path),), nprocs=2)
