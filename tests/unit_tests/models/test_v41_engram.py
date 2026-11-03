# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Hash, gate and real two-rank communication checks, runnable without CUDA fixtures."""

import copy
import tempfile
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import Engram, EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from megatron.core.models.engram.distributed_embedding import EPShardedMultiTableEmbedding
from megatron.core.optimizer import (
    OptimizerConfig,
    _get_param_groups,
    get_standard_config_overrides,
)
from megatron.core.optimizer.emerging_optimizers import (
    Sinkhorn,
    _is_muon_excluded,
    _muon_param_overrides_factory,
)
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    tag_params_for_buffer_routing,
)
from megatron.core.optimizer.optimizer import copy_optimizer_param_metadata
from megatron.core.optimizer.param_layout import BufferKey, FullParamLayout
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler


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


@pytest.mark.parametrize('lr_multiplier', [1.0, 5.0])
@pytest.mark.parametrize('norm_decay', [False, True])
def test_engram_optimizer_groups_and_schedule(monkeypatch, lr_multiplier, norm_decay):
    """Only lookup tables get the Engram LR; fused norm vectors follow the norm policy."""
    monkeypatch.setattr(dist, 'get_world_size', lambda **kwargs: 1)
    monkeypatch.setattr(
        dist, 'all_gather_object', lambda out, value, **kwargs: out.__setitem__(0, value)
    )
    config = _config()
    config.engram_config = SimpleNamespace(embedding_lr_multiplier=lr_multiplier)
    model = torch.nn.ModuleDict(
        {
            'engram': Engram(
                config, _hasher().layout, 0, SimpleNamespace(ep=None, tp=None, expt_dp=None)
            ),
            'embedding': torch.nn.Embedding(20, 4),
        }
    )
    model['embedding'].weight.is_embedding_or_output_parameter = True
    tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
    optimizer_config = OptimizerConfig(
        optimizer='muon', lr=0.01, min_lr=0.001, apply_wd_to_norm=norm_decay
    )
    overrides = get_standard_config_overrides(optimizer_config)
    overrides.update(_muon_param_overrides_factory())
    groups = _get_param_groups([model], optimizer_config, overrides)
    scheduler = OptimizerParamScheduler(
        SimpleNamespace(param_groups=groups),
        init_lr=0.002,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=10,
        lr_decay_steps=100,
        lr_decay_style='cosine',
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=100,
        wd_incr_style='constant',
    )
    by_param = {id(p): group for group in groups for p in group['params']}
    table = model['engram'].embed.tables[0].weight
    for step in (0, 5, 10, 50, 100):
        scheduler.step(step - scheduler.num_steps)
        base_lr = by_param[id(model['embedding'].weight)]['lr']
        assert by_param[id(table)]['lr'] == pytest.approx(base_lr * lr_multiplier)
        assert by_param[id(table)]['optimizer'] == 'sinkhorn'
        assert by_param[id(table)]['weight_decay'] == 0
        for param in (model['engram'].q_weight, model['engram'].k_weight):
            assert by_param[id(param)]['optimizer'] == 'adam'
            assert by_param[id(param)]['lr'] == pytest.approx(base_lr)
            assert by_param[id(param)]['weight_decay'] == pytest.approx(0.1 if norm_decay else 0)
        assert by_param[id(model['engram'].wkv.weight)]['lr'] == pytest.approx(base_lr)
        assert by_param[id(model['engram'].wkv.weight)]['weight_decay'] == pytest.approx(0.1)


def _sinkhorn_distributed_worker(rank, path, use_layout):
    """Compare EP2 x replica-DP2 updates/resume with complete-matrix EO updates."""
    dist.init_process_group(
        'gloo', init_method='file://' + path, rank=rank, world_size=4, timeout=timedelta(seconds=60)
    )
    ep_groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
    replica_groups = [dist.new_group(ranks) for ranks in ([0, 2], [1, 3])]
    ep = ep_groups[rank // 2]
    replica = replica_groups[rank % 2]
    try:
        torch.manual_seed(109)
        model = EPShardedMultiTableEmbedding(
            _config(), (17, 19), 5, _config().init_method, ep, expt_dp_group=replica
        )
        # Different local sizes deliberately perturb ordinary bin packing across EP peers.
        model.extra = torch.nn.Parameter(torch.zeros(8 if rank % 2 == 0 else 11, 5))
        model.extra.allreduce = False
        tag_params_for_buffer_routing([model], use_sinkhorn_for_embeddings=True)
        full_weights = [torch.nn.Parameter(torch.randn(rows, 5)) for rows in (17, 19)]
        references = [Sinkhorn([param], lr=0.02) for param in full_weights]
        for table, full in zip(model.tables, full_weights):
            table.weight.data.copy_(full[table.row_start : table.row_end])
        optimizer = Sinkhorn(
            [{'params': [t.weight for t in model.tables], 'is_expert_parallel': True}], lr=0.02
        )
        extra_optimizer = torch.optim.SGD(
            [{'params': [model.extra], 'is_expert_parallel': True}], lr=0.02
        )
        owner = object.__new__(LayerWiseDistributedOptimizer)
        owner.dp_cp, owner.expt_dp = dist.group.WORLD, replica
        layouts = None
        if use_layout:
            layout = owner._compute_per_buffer_param_layout(
                list(model.parameters()),
                50,
                2,
                SimpleNamespace(pad_buckets_for_high_nccl_busbw=False),
            )
            layouts = [
                FullParamLayout({BufferKey(torch.float32, torch.float32, True, True): layout})
            ]
        owner.shard_params([optimizer, extra_optimizer], layouts)
        local_params = optimizer.param_groups[0]['params']
        assert len(local_params) == 1
        assert local_params[0] is model.tables[replica.rank()].weight
        restored = None
        generator = torch.Generator().manual_seed(311)
        for iteration in range(5):
            gradients = [torch.randn(p.shape, generator=generator) for p in full_weights]
            for table, grad, reference in zip(model.tables, gradients, references):
                grad[: table.global_num_embeddings // 2] *= torch.tensor([100, 1, 0.1, 0.01, 2])
                grad[0].zero_()
                grad[1].mul_(1e-7)
                if iteration == 1:
                    # A missing local gradient must still update old momentum globally.
                    grad[(table.global_num_embeddings + 1) // 2 :].zero_()
                absent = iteration == 3 or (iteration == 1 and ep.rank() == 1)
                table.weight.grad = (
                    None if absent else grad[table.row_start : table.row_end].clone()
                )
                reference.param_groups[0]['params'][0].grad = None if iteration == 3 else grad
                reference.step()
            optimizer.step()
            if restored is not None:
                for actual, resumed in zip(local_params, restored.param_groups[0]['params']):
                    resumed.grad = None if actual.grad is None else actual.grad.clone()
                restored.step()
                torch.testing.assert_close(
                    restored.param_groups[0]['params'][0], local_params[0], rtol=0, atol=0
                )
            for index, (table, full) in enumerate(zip(model.tables, full_weights)):
                # Mirror LayerWise's replica parameter synchronization after each update.
                dist.broadcast(
                    table.weight.data, dist.get_global_rank(replica, index), group=replica
                )
                torch.testing.assert_close(
                    table.weight, full[table.row_start : table.row_end], rtol=3e-6, atol=3e-7
                )
            if iteration == 2:
                resumed_param = torch.nn.Parameter(local_params[0].detach().clone())
                copy_optimizer_param_metadata(resumed_param, local_params[0])
                assert resumed_param.row_shard_group is ep
                restored = Sinkhorn([{'params': [resumed_param], 'is_expert_parallel': True}])
                restored.load_state_dict(copy.deepcopy(optimizer.state_dict()))

        # A valid logical matrix may include an empty row shard.
        empty = torch.nn.Parameter(torch.ones(4 if ep.rank() == 0 else 0, 4))
        empty.row_shard_group = ep
        empty.row_shard_global_rows = 4
        empty.row_shard_start = 0 if ep.rank() == 0 else 4
        empty_optimizer = Sinkhorn([empty])
        empty.grad = torch.ones_like(empty) if ep.rank() == 0 else None
        empty_optimizer.step()
        torch.testing.assert_close(empty, torch.full_like(empty, 1 - 0.02 * 0.18))
        if ep.rank() == 1:
            assert empty.grad is None
        empty.row_shard_global_rows = 4 + ep.rank()
        with pytest.raises(ValueError, match='logical matrix metadata'):
            Sinkhorn([empty])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize('use_layout', [False, True])
def test_sinkhorn_ep_and_replica_owners_match_complete_update(use_layout):
    pytest.importorskip('emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils')
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(
            _sinkhorn_distributed_worker,
            args=(directory + '/rendezvous', use_layout),
            nprocs=4,
            join=True,
        )


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
