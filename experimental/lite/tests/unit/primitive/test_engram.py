# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Generic gated table masking and real two-device row-sharded lookup."""
import pytest
import torch
import torch.distributed as dist

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.lite.primitive.modules.engram import Engram, ShardedFrozenTable


def test_engram_masked_rows_pass_the_residual_through():
    engram = Engram(4, 2, torch.nn.Embedding(6, 4), torch.nn.Linear(8, 12, bias=False))
    hidden, ids = torch.randn(3, 1, 2, 4), torch.tensor([[[0, 1]], [[2, 3]], [[4, 5]]])
    out = engram(hidden, ids, torch.tensor([[True], [False], [True]]))
    assert torch.equal(out[1], hidden[1]) and not torch.equal(out[0], hidden[0])


def _gather_worker(rank, store):
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"file://{store}", rank=rank, world_size=2)
    ranks = dict(tp_rank=0, ep_rank=0, etp_rank=0, gtp_remat_rank=0, egtp_remat_rank=0)
    model_parallel_cuda_manual_seed(915, **ranks, gtp_remat_world_size=1, egtp_remat_world_size=1)
    config = TransformerConfig(num_layers=1, hidden_size=8, num_attention_heads=1)
    table = ShardedFrozenTable(5, 8, config, dist.group.WORLD)  # 6 padded rows, 3 per rank
    full = torch.arange(48.0, device="cuda").reshape(6, 8).to(table.weight.dtype)
    table.weight.copy_(full[3 * rank : 3 * rank + 3])
    ids = torch.tensor([[0, 4], [2, 3], [1, 0]], device="cuda")[rank:]  # 3 and 2 query rows
    assert torch.equal(table(ids), full[ids])
    dist.destroy_process_group()


@pytest.mark.gpus(2, min_architecture="blackwell")
def test_sharded_engram_table_matches_a_single_table_lookup(tmp_path):
    torch.multiprocessing.spawn(_gather_worker, args=(f"{tmp_path}/store",), nprocs=2)
