# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hash, gate and real two-rank communication checks, runnable without CUDA fixtures."""

import tempfile
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.optimizer import OptimizerConfig, get_standard_config_overrides
from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded


def _hasher():
    args = SimpleNamespace(
        engram_layer_ids=[0],
        engram_num_embeddings=[0],
        engram_max_ngram_size=4,
        engram_vocab_size=11,
        engram_n_heads=2,
        engram_head_dim=3,
    )
    layout = EngramLayout.from_args(args)
    rows = sum(sum(primes) for primes in layout.primes[0])
    options = EngramConfig([0], [rows], 4, 11, 2, 3, 0, 20)
    return EngramHasher(options, torch.arange(20))


def _config():
    return SimpleNamespace(
        use_cpu_initialization=True,
        params_dtype=torch.float32,
        perform_initialization=True,
        sequence_parallel=False,
        deterministic_mode=True,
        init_method=lambda t: torch.nn.init.normal_(t, std=0.1),
        mhc_num_residual_streams=2,
        hidden_size=4,
        layernorm_epsilon=1e-6,
    )


def test_hash_resets_and_memory_gradients():
    hasher = _hasher()
    ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    sequence = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1]])
    hashes = hasher(ids, sequence_ids=sequence)
    torch.testing.assert_close(hashes[:, 4:], hasher(ids[:, 4:]))
    mask = torch.ones_like(ids, dtype=torch.bool)
    mask[:, 3] = False
    torch.testing.assert_close(hasher(ids, mask)[:, 4:], hasher(ids[:, 4:]))
    model = Engram(_config(), hasher.layout, 0, SimpleNamespace(ep=None, tp=None, expt_dp=None))
    overrides = get_standard_config_overrides(OptimizerConfig())
    for name in ("q_weight", "k_weight"):
        parameter = getattr(model, name)
        assert _is_muon_excluded(parameter)
        assert any(
            key.matches(parameter, name) and override.get("wd_mult") == 0.0
            for key, override in overrides.items()
        )
    hidden = torch.randn(8, 1, 8, requires_grad=True)
    out = model(hidden, hashes, mask)
    torch.testing.assert_close(out[3], hidden[3])
    out.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def _distributed_worker(rank, path):
    dist.init_process_group('gloo', init_method='file://' + path, rank=rank, world_size=2)
    group = dist.group.WORLD
    torch.manual_seed(123)
    before = torch.random.get_rng_state()
    model = EPShardedMultiTableEmbedding(_config(), (11, 13), 3, _config().init_method, group)
    assert torch.equal(before, torch.random.get_rng_state())
    full = [torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) for n in (11, 13)]
    with torch.no_grad():
        for table, ref in zip(model.tables, full):
            table.weight.copy_(ref[table.row_start : table.row_end])
    ids = torch.tensor([[0, 12], [10, 0], [0, 12]])[: rank + 2]
    out = model(ids)
    expected = torch.stack([ref[ids[:, i]] for i, ref in enumerate(full)], 1)
    torch.testing.assert_close(out, expected)
    out.sum().backward()
    for i, table in enumerate(model.tables):
        counts = torch.bincount(ids[:, i], minlength=full[i].shape[0]).float()
        dist.all_reduce(counts)
        torch.testing.assert_close(
            table.weight.grad, counts[table.row_start : table.row_end, None].expand_as(table.weight)
        )
        shard = table.sharded_state_dict()['weight']
        assert shard.global_shape == tuple(full[i].shape)
    hasher = _hasher()
    # Check both normal halo exchange and tiny shards spanning multiple neighbors.
    for local_length in (4, 1):
        tokens = torch.arange(1, 2 * local_length + 1).unsqueeze(0)
        sequences = torch.arange(2 * local_length).unsqueeze(0) // 3
        local = slice(rank * local_length, (rank + 1) * local_length)
        actual = hasher(tokens[:, local], sequence_ids=sequences[:, local], cp_group=group)
        expected = hasher(tokens, sequence_ids=sequences)[:, local]
        torch.testing.assert_close(actual, expected)
    dist.destroy_process_group()


def test_two_rank_lookup_gradients_and_cp_hash_history():
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_worker, args=(directory + '/rendezvous',), nprocs=2, join=True)
