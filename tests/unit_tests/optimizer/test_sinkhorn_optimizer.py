# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Reference updates, semantic routing, and distributed logical-table statistics."""

import copy
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from emerging_optimizers.orthogonalized_optimizers.sinkhorn_utils import sinkhorn_balance

from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.emerging_optimizers import TensorParallelMuon, _eopt_init_state_fn
from megatron.core.optimizer.sinkhorn import SinkhornOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.layers import copy_tensor_model_parallel_attributes
from megatron.core.transformer.attention import QKVLayout
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def _parameter(data, ranges, start=0, group=None):
    p = torch.nn.Parameter(data.clone())
    p.sinkhorn_table_ranges = tuple(ranges)
    p.sinkhorn_row_start = start
    p.sinkhorn_group = group
    return p


def _serial(update):
    # Independent Algorithm 1 oracle, including locally wide/empty intersections.
    values = update.float().clone()
    rho = torch.linalg.vector_norm(values, dim=1, keepdim=True)
    mean = rho.sum() / max(values.shape[0], 1)
    values.masked_fill_(rho <= 1e-3 * mean, 0)
    for k in range(11):
        values /= torch.linalg.vector_norm(values, dim=1 if k % 2 == 0 else 0, keepdim=True) + 1e-20
    return values * update.shape[1] ** 0.5


def test_multi_step_momentum_and_independent_tables():
    torch.manual_seed(4)
    data = torch.randn(17, 4)
    p = _parameter(data, ((0, 8), (8, 17)))
    opt = SinkhornOptimizer([p], lr=0.01)
    expected, momentum = data.clone(), torch.zeros_like(data)
    for step in range(3):
        grad = torch.randn_like(data)
        grad[:8] *= 100
        grad[0] = 0
        p.grad = grad.clone()
        momentum.lerp_(grad, 0.05)
        nesterov = grad.lerp(momentum, 0.95)
        direction = torch.cat([sinkhorn_balance(nesterov[:8]), sinkhorn_balance(nesterov[8:])])
        expected.add_(direction, alpha=-0.01 * 0.18)
        opt.step()
        torch.testing.assert_close(p, expected)
        torch.testing.assert_close(opt.state[p]["momentum_buffer"], momentum)
        assert set(opt.state[p]) == {"momentum_buffer"}
    assert not torch.allclose(direction, sinkhorn_balance(nesterov))


def test_checkpoint_initialization_and_resume():
    p = _parameter(torch.randn(8, 4), ((0, 8),))
    opt = SinkhornOptimizer([p])
    _eopt_init_state_fn(opt)
    assert opt.state[p]["momentum_buffer"].dtype == torch.float32
    p.grad = torch.randn_like(p)
    opt.step()
    restored = _parameter(p.detach(), ((0, 8),))
    resumed = SinkhornOptimizer([restored])
    resumed.load_state_dict(copy.deepcopy(opt.state_dict()))
    restored.grad = p.grad.clone()
    opt.step()
    resumed.step()
    torch.testing.assert_close(restored, p)
    local = SinkhornOptimizer([restored], balance_mode="local")
    with pytest.raises(ValueError, match="changed on resume"):
        local.load_state_dict(opt.state_dict())


def test_zero_gradient_and_metadata_copy():
    p = _parameter(torch.zeros(8, 4), ((0, 8),))
    p.optimizer_role = "engram"
    master = torch.nn.Parameter(p.detach().clone())
    copy_tensor_model_parallel_attributes(master, p)
    assert master.sinkhorn_table_ranges == ((0, 8),)
    assert master.optimizer_role == "engram"
    master.grad = torch.zeros_like(master)
    opt = SinkhornOptimizer([master])
    opt.step()
    assert torch.isfinite(master).all() and master.count_nonzero() == 0
    opt.param_groups[0]["weight_decay"] = 0.1
    with pytest.raises(ValueError, match="weight decay"):
        opt.step()


@pytest.mark.parametrize(
    "kwargs",
    [{"num_normalization_steps": 2}, {"eps": 0}, {"momentum": 1}, {"balance_mode": "invalid"}],
)
def test_invalid_sinkhorn_settings(kwargs):
    p = _parameter(torch.zeros(8, 4), ((0, 8),))
    with pytest.raises(ValueError):
        SinkhornOptimizer([p], **kwargs)


def test_scheduler_applies_table_lr_and_zero_decay_once():
    groups = [
        dict(params=[], max_lr=0.01, min_lr=0.001, wd_mult=0.0, schedule_lr_multiplier=5.0),
        dict(params=[], max_lr=0.01, min_lr=0.001, wd_mult=1.0),
    ]
    scheduler = OptimizerParamScheduler(
        SimpleNamespace(param_groups=groups),
        init_lr=0.0001,
        max_lr=0.01,
        min_lr=0.001,
        lr_warmup_steps=2,
        lr_decay_steps=10,
        lr_decay_style="cosine",
        start_wd=0.1,
        end_wd=0.1,
        wd_incr_steps=10,
        wd_incr_style="constant",
    )
    for increment in (0, 1, 1, 3, 10):
        scheduler.step(increment)
        assert groups[0]["lr"] == pytest.approx(5 * groups[1]["lr"])
        assert groups[0]["weight_decay"] == 0
        assert groups[1]["weight_decay"] == 0.1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"use_layer_wise_distributed_optimizer": True},
        {"use_distributed_optimizer": True},
        {"optimizer_cpu_offload": True},
        {"optimizer_cuda_graph": True},
    ],
)
def test_unsupported_global_backends_fail_early(kwargs):
    with pytest.raises(ValueError):
        OptimizerConfig(optimizer="muon", engram_optimizer="sinkhorn", **kwargs)


def _distributed_worker(rank, path, mode):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://" + path, rank=rank, world_size=4)
    ep_groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    dp_groups = [dist.new_group([0, 2]), dist.new_group([1, 3])]
    ep = ep_groups[rank // 2]
    dp = dp_groups[rank % 2]
    # EP shards split two logical tables. The second EP rank has no first-table
    # rows; the first has only one row from the second table (a locally wide slice).
    start = (rank % 2) * 8
    data = torch.arange(64, dtype=torch.float32).view(16, 4) / 100
    p = _parameter(data[start : start + 8], ((0, 7), (7, 16)), start, ep)
    opt = SinkhornOptimizer([p], lr=0.01, balance_mode=mode)
    expected = data.clone()
    momentum = torch.zeros_like(data)
    for step in range(3):
        grad = torch.arange(64, dtype=torch.float32).view(16, 4) / 20 + step
        grad[:7, 0] *= 20
        if step == 1:
            grad[8:] = 0
        # Different DP contributions, finalized to the same gradient per EP shard.
        p.grad = grad[start : start + 8].clone() * (rank // 2 + 1)
        dist.all_reduce(p.grad, group=dp)
        p.grad /= 3
        if step == 1 and rank % 2 == 1:
            p.grad = None
        if mode == "local" and p.grad is None:
            # Local no-gradient semantics differ from a dense zero gradient.
            p.grad = torch.zeros_like(p)
        momentum.lerp_(grad, 0.05)
        direction = grad.lerp(momentum, 0.95)
        if mode == "global":
            delta = torch.cat([_serial(direction[:7]), _serial(direction[7:])])
        else:
            delta = torch.cat(
                [_serial(direction[:7]), _serial(direction[7:8]), _serial(direction[8:])]
            )
        expected.add_(delta, alpha=-0.01 * 0.18)
        opt.step()
        torch.testing.assert_close(p, expected[start : start + 8], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(opt.state[p]["momentum_buffer"], momentum[start : start + 8])
    p.grad = None
    before = p.detach().clone()
    state = opt.state[p]["momentum_buffer"].clone()
    opt.step()
    torch.testing.assert_close(p, before)
    torch.testing.assert_close(opt.state[p]["momentum_buffer"], state)
    # Reused PR #6326 public path: unequal heads must not make collective
    # participation depend on rank-local completeness. Also remove GTP padding.
    full = torch.arange(32, dtype=torch.float32).view(8, 4)
    reference = TensorParallelMuon([torch.nn.Parameter(full.clone())], update_rms=0.18)
    expected = torch.cat(
        [
            reference.scaled_orthogonalize_fn(full[:6], None),
            reference.scaled_orthogonalize_fn(full[6:], None),
        ]
    )
    for gtp in (False, True):
        shard = (
            torch.nn.functional.pad(full, (0, 0, 0, 4), value=10000).chunk(4)[rank]
            if gtp
            else full.chunk(4)[rank]
        )
        q = torch.nn.Parameter(torch.zeros_like(shard))
        q.is_qkv = True
        q.partition_dim = 0
        q.qkv_split_shapes_global = [6, 2]
        q.qkv_split_shapes = [2]
        q.qkv_split_heads_are_complete = rank == 3 and not gtp
        q.is_gtp_weight_remat = gtp
        q.qkv_gtp_pad_length = 4 if gtp else 0
        groups = ProcessGroupCollection(
            tp=None if gtp else dist.group.WORLD, gtp_remat=dist.group.WORLD if gtp else None
        )
        muon = TensorParallelMuon(
            [q],
            split_qkv=True,
            split_qkv_per_head=True,
            is_qkv_fn=lambda p: p.is_qkv,
            pg_collection=groups,
            update_rms=0.18,
        )
        actual = muon.orthogonalize(q, shard)
        wanted = (
            torch.nn.functional.pad(expected, (0, 0, 0, 4)).chunk(4)[rank]
            if gtp
            else expected.chunk(4)[rank]
        )
        torch.testing.assert_close(actual, wanted, rtol=1e-5, atol=1e-6)
    dist.destroy_process_group()


@pytest.mark.parametrize("mode", ["global", "local"])
def test_ep_statistics_exclude_dp_replicas(mode):
    with tempfile.TemporaryDirectory() as directory:
        mp.spawn(_distributed_worker, args=(directory + "/init", mode), nprocs=4, join=True)


class _MixedModel(torch.nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.config = TransformerConfig(
            num_layers=1, hidden_size=4, num_attention_heads=2, kv_channels=2
        )
        self.linear = torch.nn.Linear(4, 4, dtype=dtype, device="cuda")
        self.query = torch.nn.Linear(4, 8, bias=False, dtype=dtype, device="cuda")
        self.query.weight.qkv_layout = QKVLayout.from_mla_head_components(2, (4,))
        self.norm = torch.nn.LayerNorm(4, dtype=dtype, device="cuda")
        self.embedding = torch.nn.Embedding(17, 4, dtype=dtype, device="cuda")
        self.embedding.weight.is_embedding_or_output_parameter = True
        self.output_layer = torch.nn.Linear(4, 17, bias=False, dtype=dtype, device="cuda")
        self.output_layer.weight.is_embedding_or_output_parameter = True
        self.engram = torch.nn.Module()
        self.engram.weight = _parameter(
            torch.randn(23, 4, device="cuda", dtype=dtype), ((0, 11), (11, 23))
        )
        self.engram.weight.is_engram_embedding = True
        self.engram.weight.allreduce = False


@pytest.mark.parametrize("bf16", [False, True])
def test_public_mixed_factory_and_master_metadata(bf16):
    Utils.initialize_model_parallel()
    try:
        model = _MixedModel(torch.bfloat16 if bf16 else torch.float32)
        config = OptimizerConfig(
            optimizer="muon",
            lr=0.001,
            min_lr=0.0001,
            bf16=bf16,
            clip_grad=0,
            embedding_optimizer="sinkhorn",
            output_layer_optimizer="sinkhorn",
            engram_optimizer="sinkhorn",
            engram_lr_multiplier=5,
            apply_wd_to_norm=True,
            muon_split_qkv_per_head=True,
            muon_update_rms=0.18,
        )
        opt = get_megatron_optimizer(
            config,
            [model],
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            use_gloo_process_groups=False,
        )
        for p in model.parameters():
            p.grad = torch.randn_like(p)
        groups = opt.param_groups
        norm_group = (
            next(g for g in groups if any(p is model.norm.weight for p in g["params"]))
            if not bf16
            else None
        )
        if norm_group:
            assert norm_group["wd_mult"] == 1
        sinkhorn_groups = [g for g in groups if g.get("optimizer") == "sinkhorn"]
        assert sinkhorn_groups and all(g["wd_mult"] == 0 for g in sinkhorn_groups)
        assert any(g.get("schedule_lr_multiplier") == 5 for g in sinkhorn_groups)
        assert opt.step()[0]
        for child in opt.chained_optimizers:
            if isinstance(child.optimizer, SinkhornOptimizer):
                for p in child.optimizer.state:
                    assert p.dtype == torch.float32
                    assert p.sinkhorn_table_ranges
                    assert child.optimizer.state[p]["momentum_buffer"].dtype == torch.float32
        state = copy.deepcopy(opt.state_dict())
        opt.load_state_dict(state)
    finally:
        Utils.destroy_model_parallel()


def test_local_sinkhorn_with_legacy_layerwise_buffers():
    from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig

    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        model = _MixedModel(torch.bfloat16)
        wrapped = DistributedDataParallel(
            model.config, DistributedDataParallelConfig(), model, pg_collection=groups
        )
        config = OptimizerConfig(
            optimizer="muon",
            # A visible BF16 model delta, not just a sub-ULP FP32 master update.
            lr=0.1,
            bf16=True,
            clip_grad=0,
            embedding_optimizer="sinkhorn",
            output_layer_optimizer="sinkhorn",
            engram_optimizer="sinkhorn",
            sinkhorn_balance_mode="local",
            use_layer_wise_distributed_optimizer=True,
        )
        opt = get_megatron_optimizer(
            config, [wrapped], pg_collection=groups, use_gloo_process_groups=False
        )
        before = model.engram.weight.detach().clone()
        sum(p.float().square().sum() for p in model.parameters()).backward()
        wrapped.finish_grad_sync()
        assert opt.step()[0]
        assert not torch.equal(model.engram.weight, before)
    finally:
        Utils.destroy_model_parallel()


def test_norm_decay_option_with_standard_adam():
    from megatron.core.distributed import DistributedDataParallelConfig

    Utils.initialize_model_parallel()
    try:
        model = _MixedModel(torch.float32)
        model.ddp_config = DistributedDataParallelConfig()
        config = OptimizerConfig(optimizer="adam", lr=0.001, apply_wd_to_norm=True)
        opt = get_megatron_optimizer(
            config,
            [model],
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            use_gloo_process_groups=False,
        )
        group = next(
            g for g in opt.param_groups if any(p is model.norm.weight for p in g["params"])
        )
        assert group["wd_mult"] == 1.0
    finally:
        Utils.destroy_model_parallel()


def test_exact_muon_rms_is_stable_for_tiny_updates(monkeypatch):
    from megatron.core.optimizer import emerging_optimizers as eo

    monkeypatch.setattr(eo, "newton_schulz_tp", lambda x, **kwargs: torch.full_like(x, 1e-24))
    p = torch.nn.Parameter(torch.zeros(4, 8))
    opt = TensorParallelMuon([p], update_rms=0.18)
    torch.testing.assert_close(opt.orthogonalize(p, torch.ones_like(p)), torch.full_like(p, 0.18))


def test_layer_sharded_backend_rejects_per_head():
    from megatron.core.optimizer.layer_sharded_muon import LayerShardedMuon

    with pytest.raises(ValueError, match="split-QKV"):
        LayerShardedMuon(
            [torch.nn.Parameter(torch.zeros(4, 4))], gtp_remat_group=None, split_qkv_per_head=True
        )


def test_muon_whole_head_rms_and_zero_blocks():
    grad = torch.randn(12, 8)
    grad[:4] = 0
    p = torch.nn.Parameter(torch.zeros_like(grad))
    p.is_qkv = True
    p.qkv_split_shapes = [4, 4, 4]
    opt = TensorParallelMuon(
        [p],
        split_qkv=True,
        split_qkv_per_head=True,
        is_qkv_fn=lambda p: p.is_qkv,
        qkv_split_shapes=[4, 4, 4],
        update_rms=0.18,
    )
    actual = opt.orthogonalize(p, grad)
    assert actual[:4].count_nonzero() == 0
    torch.testing.assert_close(
        actual[4:].view(2, 4, 8).square().mean((-2, -1)).sqrt(), torch.full((2,), 0.18)
    )
