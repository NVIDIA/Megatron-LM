# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from megatron.core import parallel_state
from megatron.core.dist_checkpointing import (
    load,
    load_common_state_dict,
    load_tensors_metadata,
    save,
)
from megatron.core.dist_checkpointing.optimizer import make_sharded_optimizer_tensor
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import ChainedOptimizer, OptimizerConfig
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.engram import Engram, EngramConfig, RowShardedMultiHeadEmbedding
from megatron.core.transformer.engram.optimizer import get_engram_optimizer
from megatron.core.transformer.engram.parallel import EngramParallelGroups
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.engram.test_integration import model_parallel
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    Utils.world_size not in (2, 4, 8), reason="requires two, four, or eight ranks"
)

ROW_WIDTH = 80


def _prepare_standalone_row_grads(module):
    """Only standalone lookup tests allocate a small explicit gradient buffer."""
    for memory in module.modules():
        if isinstance(memory, RowShardedMultiHeadEmbedding):
            weight = memory.embedding.weight
            assert not hasattr(weight, "main_grad")
            weight.main_grad = torch.zeros_like(weight, dtype=torch.float32)


def _set_global_row_values(memory):
    with torch.no_grad():
        for (start, end), local_offset, global_offset in zip(
            memory.layout.local_bounds, memory.layout.local_offsets, memory.layout.global_offsets
        ):
            rows = torch.arange(start, end, device="cuda") + global_offset
            values = rows[:, None] * 10 + torch.arange(memory.embedding_dim, device="cuda")
            memory.embedding.weight[local_offset : local_offset + end - start].copy_(values)


def _config(backend="row_a2a", **kwargs):
    values = dict(
        enabled=True,
        hash_table_min_sizes=(17, 19),
        max_ngram_size=3,
        embedding_dim_per_ngram=8,
        num_hash_heads_per_ngram=2,
        layer_ids=(0,),
        pad_id=0,
        seed=7,
        kernel_size=2,
        hidden_size=8,
        table_backend=backend,
        params_dtype=torch.float32,
        use_cpu_initialization=True,
    )
    values.update(kwargs)
    return EngramConfig(**values)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_row_a2a_forward_backward_duplicates_and_uneven_splits(dtype):
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        memory = RowShardedMultiHeadEmbedding(
            [5, 7], D=4, layer_id=3, params_dtype=dtype, use_cpu_initialization=True
        ).cuda()
        _prepare_standalone_row_grads(memory)
        _set_global_row_values(memory)
        ids = torch.tensor([[[0, 0], [4, 6], [0, 2]]], device="cuda")
        output = memory(ids)
        full_weight = torch.arange(12, device="cuda")[:, None] * 10 + torch.arange(4, device="cuda")
        expected = F.embedding(ids + torch.tensor([0, 5], device="cuda"), full_weight.to(dtype))
        torch.testing.assert_close(output, expected)
        output.sum().backward()
        grad = memory.embedding.weight.main_grad
        assert memory.embedding.weight.grad is None
        assert grad.dtype == torch.float32 and not grad.is_sparse
        full_grad = torch.zeros(12, 4, device="cuda")
        full_grad.index_add_(
            0,
            (ids + torch.tensor([0, 5], device="cuda")).reshape(-1),
            torch.ones(ids.numel(), 4, device="cuda"),
        )
        expected_grad = torch.cat(
            [
                full_grad[global_offset + start : global_offset + end]
                for (start, end), global_offset in zip(
                    memory.layout.local_bounds, memory.layout.global_offsets
                )
            ]
        )
        torch.testing.assert_close(grad, expected_grad)
        # A second microbatch accumulates directly in the same FP32 storage.
        pointer = grad.data_ptr()
        memory(ids).sum().backward()
        assert memory.embedding.weight.main_grad.data_ptr() == pointer
        torch.testing.assert_close(grad, expected_grad * 2)


def test_row_a2a_empty_request_is_collective_safe():
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        memory = RowShardedMultiHeadEmbedding(
            [2, 3], D=4, layer_id=3, use_cpu_initialization=True
        ).cuda()
        _prepare_standalone_row_grads(memory)
        ids = torch.empty((0, 0, 2), dtype=torch.long, device="cuda")
        output = memory(ids)
        assert output.shape == (0, 0, 2, 4)
        output.sum().backward()
        assert memory.embedding.weight.grad is None
        assert torch.count_nonzero(memory.embedding.weight.main_grad) == 0


def test_row_a2a_full_engram_recompute_parity():
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        module = Engram(engram_config=_config(), tokenizer_lookup=torch.arange(64)).cuda()
        _prepare_standalone_row_grads(module)
        ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], device="cuda")
        compressed_input_ids = module.compress_input_ids(ids)
        hidden = torch.randn(4, 2, 8, device="cuda")

        direct_hidden = hidden.detach().clone().requires_grad_(True)
        direct = module(direct_hidden, 0, compressed_input_ids)
        direct.square().sum().backward()
        direct_hidden_grad = direct_hidden.grad.clone()
        direct_table_grad = module.layers[
            "0"
        ].multi_head_embedding.embedding.weight.main_grad.clone()
        module.zero_grad(set_to_none=True)
        module.layers["0"].multi_head_embedding.embedding.weight.main_grad.zero_()
        recompute_hidden = hidden.detach().clone().requires_grad_(True)
        recomputed = checkpoint(
            lambda value, compressed_ids: module(value, 0, compressed_ids),
            recompute_hidden,
            compressed_input_ids,
            use_reentrant=False,
        )
        recomputed.square().sum().backward()
        torch.testing.assert_close(recomputed, direct)
        torch.testing.assert_close(recompute_hidden.grad, direct_hidden_grad)
        torch.testing.assert_close(
            module.layers["0"].multi_head_embedding.embedding.weight.main_grad, direct_table_grad
        )


def test_row_a2a_dcp_roundtrip(tmp_path_dist_ckpt):
    Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
    error = None
    try:
        try:
            source = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            _set_global_row_values(source)
            expected = source.embedding.weight.detach().clone()
            checkpoint_dir = tmp_path_dist_ckpt / "row_a2a_roundtrip"
            if torch.distributed.get_rank() == 0:
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.distributed.barrier()
            prefix = "engram.layers.3.multi_head_embedding."
            save(source.sharded_state_dict(prefix=prefix), checkpoint_dir)
            restored = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            state = load(restored.sharded_state_dict(prefix=prefix), checkpoint_dir)
            restored.load_state_dict(
                {
                    "offsets": state[f"{prefix}offsets"],
                    "embedding.weight": state[f"{prefix}embedding.weight"],
                }
            )
            torch.testing.assert_close(restored.embedding.weight, expected)
        except Exception as exc:
            error = f"rank {torch.distributed.get_rank()}: {type(exc).__name__}: {exc}"
        errors = [None] * Utils.world_size
        torch.distributed.all_gather_object(errors, error)
        assert not any(errors), errors
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(
    Utils.world_size == 8,
    reason="the one-to-many checkpoint topology fixture is covered on two and four ranks",
)
def test_row_a2a_dcp_reshards_one_to_many_and_back(tmp_path_dist_ckpt):
    prefix = "engram.layers.3.multi_head_embedding."
    first_dir = tmp_path_dist_ckpt / "row_a2a_one_to_many"
    second_dir = tmp_path_dist_ckpt / "row_a2a_many_to_one"
    with model_parallel(pipeline_model_parallel_size=Utils.world_size):
        source = None
        if torch.distributed.get_rank() == 0:
            source = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            _set_global_row_values(source)
            first_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        save(source.sharded_state_dict(prefix=prefix) if source else {}, first_dir)

        Utils.initialize_model_parallel(tensor_model_parallel_size=Utils.world_size)
        sharded = RowShardedMultiHeadEmbedding(
            [5, 7], D=4, layer_id=3, use_cpu_initialization=True
        ).cuda()
        state = load(sharded.sharded_state_dict(prefix=prefix), first_dir)
        sharded.load_state_dict(
            {
                "offsets": state[f"{prefix}offsets"],
                "embedding.weight": state[f"{prefix}embedding.weight"],
            }
        )
        expected = sharded.embedding.weight.detach().clone()
        _set_global_row_values(sharded)
        torch.testing.assert_close(sharded.embedding.weight, expected)

        if torch.distributed.get_rank() == 0:
            second_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        save(sharded.sharded_state_dict(prefix=prefix), second_dir)

        Utils.initialize_model_parallel(pipeline_model_parallel_size=Utils.world_size)
        if torch.distributed.get_rank() == 0:
            restored = RowShardedMultiHeadEmbedding(
                [5, 7], D=4, layer_id=3, use_cpu_initialization=True
            ).cuda()
            state = load(restored.sharded_state_dict(prefix=prefix), second_dir)
            restored.load_state_dict(
                {
                    "offsets": state[f"{prefix}offsets"],
                    "embedding.weight": state[f"{prefix}embedding.weight"],
                }
            )
            _set_global_row_values(restored)
            torch.testing.assert_close(
                state[f"{prefix}embedding.weight"], restored.embedding.weight
            )
        else:
            load({}, second_dir)


class _IntegrationModel(torch.nn.Module):
    def __init__(self, parallel_groups=None):
        super().__init__()
        if parallel_groups is None:
            pg = ProcessGroupCollection.use_mpu_process_groups()
            parallel_groups = EngramParallelGroups.create(pg.tp_dp_cp)
        self.memory = RowShardedMultiHeadEmbedding(
            [5, 7],
            D=ROW_WIDTH,
            layer_id=3,
            use_cpu_initialization=True,
            parallel_groups=parallel_groups,
        ).cuda()
        # Keep at least one dense optimizer shard on every supported test rank.
        self.dense = torch.nn.Parameter(torch.ones(8, device="cuda"))
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False, use_megatron_fsdp=False)

    def forward(self, ids):
        mask = torch.zeros(ROW_WIDTH, device="cuda")
        mask[0] = 1.0
        return (self.memory(ids) * mask).sum() + self.dense.sum()


class _ThreeOptimizerIntegrationModel(_IntegrationModel):
    def __init__(self, parallel_groups=None):
        super().__init__(parallel_groups=parallel_groups)
        self.expert = torch.nn.Parameter(torch.ones(8, device="cuda"))
        self.expert.allreduce = False


def _transformer_config():
    return TransformerConfig(
        num_layers=1, hidden_size=ROW_WIDTH, num_attention_heads=1, use_cpu_initialization=True
    )


def test_ddp_groups_row_table_by_optimizer_replicas():
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        model = _IntegrationModel()
        ddp = DistributedDataParallel(
            _transformer_config(), DistributedDataParallelConfig(overlap_grad_reduce=False), model
        )
        row_weight = model.memory.embedding.weight
        assert sum(row_weight is p for b in ddp.buffers for p in b.params) == 1
        assert row_weight in ddp.param_to_bucket_group
        assert ddp.param_to_bucket_group[row_weight] is not ddp.param_to_bucket_group[model.dense]
        assert row_weight.main_grad.dtype == torch.float32
        assert row_weight.main_grad.shape == row_weight.shape
        assert row_weight.main_grad.device == row_weight.device
        row_weight.main_grad.fill_(3.0)
        ddp.zero_grad_buffer()
        assert torch.count_nonzero(row_weight.main_grad) == 0
        assert hasattr(model.dense, "main_grad")
        ddp.broadcast_params()


def _owner_leaf(optimizer):
    children = getattr(optimizer, "chained_optimizers", None)
    if children is None:
        return (
            optimizer
            if any(group.get("is_engram_row_parallel") for group in optimizer.param_groups)
            else None
        )
    for child in children:
        result = _owner_leaf(child)
        if result is not None:
            return result
    return None


def _integration_groups():
    return EngramParallelGroups.create(
        ProcessGroupCollection.use_mpu_process_groups().tp_dp_cp, Utils.world_size
    )


def test_chained_distributed_optimizer_checkpoint_roundtrip(tmp_path_dist_ckpt):
    if Utils.world_size == 8:
        pytest.skip("the focused DistOpt checkpoint fixture is covered on two and four ranks")
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        transformer_config = _transformer_config()
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True, overlap_grad_reduce=False
        )
        optimizer_config = OptimizerConfig(
            optimizer="adam",
            lr=1.0e-2,
            min_lr=1.0e-4,
            use_distributed_optimizer=True,
            clip_grad=0.0,
        )

        groups = _integration_groups()

        def build():
            model = _ThreeOptimizerIntegrationModel(parallel_groups=groups)
            ddp = DistributedDataParallel(transformer_config, ddp_config, model)
            optimizer = get_engram_optimizer(optimizer_config, [ddp], parallel_groups=groups)
            return model, optimizer

        source, source_optimizer = build()
        assert len(source_optimizer.chained_optimizers) == 3
        assert all(
            isinstance(child, DistributedOptimizer) for child in source_optimizer.chained_optimizers
        )
        source.dense.main_grad.fill_(0.25)
        source.expert.main_grad.fill_(0.5)
        source.memory.embedding.weight.main_grad.zero_()
        source.memory.embedding.weight.main_grad[0].fill_(0.75)
        assert source_optimizer.step()[0]

        checkpoint_dir = tmp_path_dist_ckpt / "chained_optimizer_roundtrip"
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        metadata = {"distrib_optim_sharding_type": "dp_reshardable"}

        save(
            source_optimizer.sharded_state_dict(
                _native_checkpoint_model_state(source), metadata=metadata
            ),
            checkpoint_dir,
        )

        restored, restored_optimizer = build()
        state = load(
            restored_optimizer.sharded_state_dict(
                _native_checkpoint_model_state(restored), is_loading=True, metadata=metadata
            ),
            checkpoint_dir,
        )
        with torch.no_grad():
            restored_optimizer.load_state_dict(state)
            for source_param, restored_param in zip(source.parameters(), restored.parameters()):
                restored_param.copy_(source_param)
        for model, optimizer in ((source, source_optimizer), (restored, restored_optimizer)):
            model.dense.main_grad.fill_(0.25)
            model.expert.main_grad.fill_(0.5)
            model.memory.embedding.weight.main_grad.zero_()
            model.memory.embedding.weight.main_grad[-1].fill_(0.5)
            assert optimizer.step()[0]
        for source_param, restored_param in zip(source.parameters(), restored.parameters()):
            torch.testing.assert_close(restored_param, source_param)


@pytest.mark.parametrize("table_sizes", [(5, 7), (1, 1)])
def test_row_factory_native_optimizer_targets_are_independent(tmp_path_dist_ckpt, table_sizes):
    with model_parallel(tensor_model_parallel_size=Utils.world_size):
        memory = RowShardedMultiHeadEmbedding(
            table_sizes, D=4, layer_id=3, params_dtype=torch.bfloat16, use_cpu_initialization=True
        ).cuda()
        weight = memory.embedding.weight
        with torch.no_grad():
            weight.fill_(0.125)
        targets = {"model": weight}
        targets.update(
            {
                name: torch.full_like(weight, value, dtype=torch.float32)
                for name, value in (("master", 3.0), ("exp_avg", 5.0), ("exp_avg_sq", 7.0))
            }
        )
        expected = {name: value.detach().clone() for name, value in targets.items()}
        pointers = {name: value.data_ptr() for name, value in targets.items()}
        if weight.numel():
            assert len(set(pointers.values())) == len(targets)

        def state():
            model_factory = memory._row_sharded_factory("engram.weight", weight)
            return {
                name: (
                    model_factory
                    if name == "model"
                    else make_sharded_optimizer_tensor(model_factory, target, f"optimizer.{name}")
                )
                for name, target in targets.items()
            }

        checkpoint_dir = tmp_path_dist_ckpt / f"native_targets_{table_sizes[0]}"
        if torch.distributed.get_rank() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        torch.distributed.barrier()
        save(state(), checkpoint_dir)
        assert not any("_restore_target" in key for key in load_tensors_metadata(checkpoint_dir))
        assert "_restore_target" not in repr(load_common_state_dict(checkpoint_dir))
        with torch.no_grad():
            for target in targets.values():
                target.fill_(-9)
        restored = load(state(), checkpoint_dir)
        for name, target in targets.items():
            assert target.data_ptr() == pointers[name]
            assert restored[name].data_ptr() == pointers[name]
            assert restored[name].dtype == target.dtype
            torch.testing.assert_close(target, expected[name])


class _ReplicaModel(torch.nn.Module):
    def __init__(self, config, topology, sizes, dtype):
        super().__init__()
        self.config = config
        self.memory = RowShardedMultiHeadEmbedding(
            sizes,
            D=8,
            layer_id=0,
            params_dtype=dtype,
            parallel_groups=topology,
            calculate_per_token_loss=config.calculate_per_token_loss,
        ).cuda()
        self.dense = torch.nn.Parameter(torch.ones(256, device='cuda', dtype=dtype))
        with torch.no_grad():
            for (start, end), local, offset in zip(
                self.memory.layout.local_bounds,
                self.memory.layout.local_offsets,
                self.memory.layout.global_offsets,
            ):
                rows = torch.arange(start + offset, end + offset, device='cuda')
                self.memory.embedding.weight[local : local + end - start].copy_(
                    ((rows[:, None] * 8 + torch.arange(8, device='cuda')) % 31).float() / 64
                )

    def forward(self, ids, coefficient):
        return self.memory(ids).float().sum() * coefficient + self.dense.float().sum()


@pytest.mark.parametrize(
    'backend,per_token', [('fused', False), ('fused', True), ('adam', False), ('adamw', True)]
)
@pytest.mark.parametrize(
    'tp,cp,shards,sizes', [(1, 1, 1, (5, 7)), (2, 1, 2, (1, 1)), (1, 2, 2, (129, 131))]
)
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_actual_a2a_native_adam_math(tp, cp, shards, sizes, dtype, backend, per_token, monkeypatch):
    if backend != 'fused':
        import megatron.core.optimizer as factory
        from megatron.core.transformer.engram import optimizer as adapter

        monkeypatch.setattr(factory, 'USING_PYTORCH_OPTIMIZER', True)
        monkeypatch.setattr(factory, 'Adam', torch.optim.AdamW)
        monkeypatch.setattr(adapter, 'USING_PYTORCH_OPTIMIZER', True)
        monkeypatch.setattr('megatron.core.optimizer.distrib_optimizer.Adam', torch.optim.Adam)
        for flag in ('HAVE_APEX_OR_TE', 'USING_TE_OPTIMIZER', 'USING_APEX_OPTIMIZER'):
            monkeypatch.setattr('megatron.core.optimizer.distrib_optimizer.' + flag, False)
    if Utils.world_size not in (2, 4):
        pytest.skip('two/four-rank integration matrix')
    with model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp):
        pg = ProcessGroupCollection.use_mpu_process_groups()
        topology = EngramParallelGroups.create(pg.tp_dp_cp, shards)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            tensor_model_parallel_size=tp,
            context_parallel_size=cp,
        )
        config.calculate_per_token_loss = per_token
        raw = _ReplicaModel(config, topology, sizes, dtype)
        ddp = DistributedDataParallel(
            config, DistributedDataParallelConfig(use_distributed_optimizer=True), raw
        )
        optim_config = OptimizerConfig(
            optimizer='adam',
            decoupled_weight_decay=backend != 'adam',
            lr=0.002,
            min_lr=0.0001,
            weight_decay=0.0,
            bf16=dtype == torch.bfloat16,
            use_distributed_optimizer=True,
            clip_grad=0.7,
            log_num_zeros_in_grad=True,
            adam_beta2=0.99,
        )
        optimizer = get_engram_optimizer(
            optim_config,
            [ddp],
            pg_collection=pg,
            use_gloo_process_groups=False,
            parallel_groups=topology,
        )
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=0.0,
            max_lr=optim_config.lr,
            min_lr=optim_config.min_lr,
            lr_warmup_steps=0,
            lr_decay_steps=10,
            lr_decay_style='linear',
            start_wd=0.0,
            end_wd=0.0,
            wd_incr_steps=10,
            wd_incr_style='constant',
        )
        dense_before = raw.dense.detach().clone()
        owner = optimizer.chained_optimizers[-1]
        if backend != 'fused':
            assert type(owner.optimizer) is (
                torch.optim.AdamW if backend == 'adamw' else torch.optim.Adam
            )
        assert owner.config.use_distributed_optimizer
        assert owner.data_parallel_group_gloo is None
        reference = torch.nn.Parameter(
            (torch.arange(sum(sizes) * 8, device='cuda') % 31).float().reshape(-1, 8) / 64
        )
        reference_optimizer = torch.optim.AdamW(
            [reference], lr=0.01, betas=(0.9, 0.99), eps=optim_config.adam_eps, weight_decay=0.0
        )
        pointer = raw.memory.embedding.weight.main_grad.data_ptr()
        state_pointers = [
            (p.data_ptr(), s['exp_avg'].data_ptr(), s['exp_avg_sq'].data_ptr())
            for p, s in owner.optimizer.state.items()
        ]
        for step in range(4):
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            coefficient = 0.0 if step in (1, 2) else (Utils.rank + 1) / 4
            ids = torch.tensor([[[1, 2], [1, 2]]], device='cuda')
            if step == 3:
                ids.fill_(3)
            for table, size in enumerate(sizes):
                ids[..., table].remainder_(size)
            ddp(ids, coefficient).backward()
            ddp.finish_grad_sync()
            if per_token:
                ddp.scale_gradients(1 / pg.dp_cp.size())
            gradient = torch.zeros_like(reference)
            for rank in range(Utils.world_size):
                value = 0.0 if step in (1, 2) else (rank + 1) / 4
                for table in range(2):
                    row = (3 if step == 3 else table + 1) % sizes[table]
                    gradient[row + (0 if table == 0 else sizes[0])] += value * 2 / Utils.world_size
            expected_norm = math.sqrt(256 + gradient.square().sum().item())
            scale = 0.7 / (expected_norm + 1e-6)
            reference.grad = gradient * scale
            table_group = owner.param_groups[0]
            dense_group = next(g for g in optimizer.param_groups if g.get('default_config'))
            assert table_group['lr'] == pytest.approx(5 * dense_group['lr'])
            assert table_group['weight_decay'] == table_group['wd_mult'] == 0.0
            reference_optimizer.param_groups[0]['lr'] = table_group['lr']
            reference_optimizer.step()
            success, norm, zeros = optimizer.step()
            assert success
            scheduler.step(1)
            assert norm == pytest.approx(expected_norm, rel=1e-6)
            assert zeros == sum(sizes) * 8 - torch.count_nonzero(gradient).item()
            for (start, end), local, offset in zip(
                raw.memory.layout.local_bounds,
                raw.memory.layout.local_offsets,
                raw.memory.layout.global_offsets,
            ):
                torch.testing.assert_close(
                    raw.memory.embedding.weight[local : local + end - start],
                    reference[start + offset : end + offset].to(dtype),
                    rtol=1e-5,
                    atol=1e-6,
                )
            param = raw.memory.embedding.weight
            if param in owner.model_param_gbuf_map:
                actual = owner._get_main_param_and_optimizer_states(param)
                interval = owner._get_model_param_range_map(param)['param']
                for name in ('param', 'exp_avg', 'exp_avg_sq'):
                    full = (
                        reference if name == 'param' else reference_optimizer.state[reference][name]
                    )
                    local_reference = torch.cat(
                        [
                            full[start + offset : end + offset]
                            for (start, end), offset in zip(
                                raw.memory.layout.local_bounds, raw.memory.layout.global_offsets
                            )
                        ]
                    ).flatten()[interval.start : interval.end]
                    torch.testing.assert_close(actual[name], local_reference, rtol=1e-5, atol=1e-6)
            assert pointer == param.main_grad.data_ptr()
            assert state_pointers == [
                (p.data_ptr(), s['exp_avg'].data_ptr(), s['exp_avg_sq'].data_ptr())
                for p, s in owner.optimizer.state.items()
            ]
        assert not torch.equal(raw.dense, dense_before)
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        ddp(ids, 1.0).backward()
        if Utils.rank == Utils.world_size - 1:
            raw.memory.embedding.weight.main_grad.fill_(float('inf'))
        ddp.finish_grad_sync()
        before_params = [p.clone() for p in optimizer.get_parameters()]
        before_state = [
            (s['exp_avg'].clone(), s['exp_avg_sq'].clone()) for s in owner.optimizer.state.values()
        ]
        before_steps = [g.get('step') for g in owner.param_groups]
        success, _, _ = optimizer.step()
        assert not success
        for before, after in zip(before_params, optimizer.get_parameters()):
            assert torch.equal(before, after)
        for (m, v), state in zip(before_state, owner.optimizer.state.values()):
            assert torch.equal(m, state['exp_avg']) and torch.equal(v, state['exp_avg_sq'])
        assert before_steps == [g.get('step') for g in owner.param_groups]


def _native_checkpoint_model_state(model):
    from megatron.core.dist_checkpointing.mapping import ShardedTensor

    state = model.memory.sharded_state_dict(prefix="memory.")
    for name in ("dense", "expert"):
        if hasattr(model, name):
            # These fixture parameters are replicated, with identical assigned
            # gradients on all ranks. Only the Engram table has row ownership.
            state[name] = ShardedTensor.from_rank_offsets(
                name,
                getattr(model, name),
                replica_id=(
                    0,
                    parallel_state.get_tensor_model_parallel_rank(),
                    parallel_state.get_data_parallel_rank(),
                ),
            )
    return state
