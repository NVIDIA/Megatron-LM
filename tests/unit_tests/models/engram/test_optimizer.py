# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import copy
import dataclasses
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import megatron.core.optimizer as optimizer_factory
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.param_and_grad_buffer import partition_buckets
from megatron.core.optimizer import (
    ChainedOptimizer,
    OptimizerConfig,
    ParamKey,
    _get_megatron_optimizer_based_on_param_groups,
    _get_param_groups,
)
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.engram import optimizer as engram_optimizer
from megatron.core.transformer.engram.memory import (
    EngramTableParameterMetadata,
    mark_engram_table_parameter,
)
from megatron.core.transformer.engram.optimizer import _validate_owner_config, get_engram_optimizer
from megatron.core.transformer.engram.parallel import EngramParallelGroups
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.models.engram.test_integration import model_parallel
from tests.unit_tests.test_utilities import Utils

ROW_WIDTH = 80


class _OptimizerFixtureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.table_with_no_embedding_in_name = nn.Parameter(torch.randn(7, ROW_WIDTH))
        mark_engram_table_parameter(
            self.table_with_no_embedding_in_name, EngramTableParameterMetadata(row_parallel=True)
        )
        self.fusion = nn.Linear(ROW_WIDTH, ROW_WIDTH, bias=False)
        self.ddp_config = SimpleNamespace(use_custom_fsdp=False, use_megatron_fsdp=False)


def _groups(model, monkeypatch, config=None, overrides=None):
    """Capture the final policy at the native factory boundary."""
    monkeypatch.setattr(torch.distributed, 'get_world_size', lambda group=None: 1)
    monkeypatch.setattr(
        torch.distributed,
        'all_gather_object',
        lambda output, value, group=None: output.__setitem__(0, value),
    )
    config = config or OptimizerConfig(optimizer='adam', lr=2e-4, min_lr=2e-5)
    captured = []

    def factory(config, models, overrides, *, final_override_fn, param_group_filter, **kwargs):
        groups = _get_param_groups(models, config, overrides, final_override_fn=final_override_fn)
        captured.append(groups)
        list(filter(param_group_filter, groups))
        return SimpleNamespace(config=config)

    monkeypatch.setattr(engram_optimizer, 'get_megatron_optimizer', factory)
    monkeypatch.setattr(
        engram_optimizer,
        '_build_table_optimizer',
        lambda *args, **kwargs: SimpleNamespace(config=config),
    )
    get_engram_optimizer(
        config,
        [model],
        {} if overrides is None else overrides,
        pg_collection=SimpleNamespace(),
        parallel_groups=SimpleNamespace(stats_group=None),
    )
    assert len(captured) == 1
    return captured[0]


@pytest.mark.parametrize('optimizer', ['adam', 'muon'])
@pytest.mark.parametrize('override', [False, True])
def test_table_policy_is_final_and_preserves_order(monkeypatch, optimizer, override):
    model = _OptimizerFixtureModel()
    model.second = nn.Parameter(torch.ones(3, ROW_WIDTH))
    model.local = nn.Parameter(torch.ones(3, ROW_WIDTH))
    model.frozen = nn.Parameter(torch.ones(3, ROW_WIDTH), requires_grad=False)
    for name in ('second', 'frozen'):
        mark_engram_table_parameter(
            getattr(model, name), EngramTableParameterMetadata(row_parallel=True)
        )
    mark_engram_table_parameter(model.local)
    config = OptimizerConfig(optimizer=optimizer, lr=2e-4, min_lr=2e-5)
    overrides = (
        {
            ParamKey(attr='engram_table_metadata'): {
                'max_lr': 0.2,
                'min_lr': 0.02,
                'lr_mult': 7.0,
                'lr': 0.123,
                'wd_mult': 0.5,
                'weight_decay': 0.25,
            }
        }
        if override
        else {}
    )
    saved = copy.deepcopy(overrides)
    groups = _groups(model, monkeypatch, config, overrides)
    by_param = {id(p): group for group in groups for p in group['params']}
    assert id(model.frozen) not in by_param and overrides == saved
    row = by_param[id(model.second)]
    assert [id(p) for p in row['params']] == [
        id(model.table_with_no_embedding_in_name),
        id(model.second),
    ]
    for param in (model.second, model.local):
        group = by_param[id(param)]
        assert group['lr_mult'] == 5.0 and group['wd_mult'] == 0.0
        assert group['max_lr'] == config.lr * 5 and group['min_lr'] == config.min_lr * 5
        assert group.get('optimizer') == ('adam' if optimizer == 'muon' else None)
        assert group.get('is_engram_row_parallel', False) is (param is model.second)
    assert row['weight_decay'] == 0.0
    assert row['lr'] == (0.123 if override else config.lr * 5)
    fusion = by_param[id(model.fusion.weight)]
    assert fusion['default_config'] and fusion['lr_mult'] == fusion['wd_mult'] == 1.0
    assert 'is_engram_row_parallel' not in fusion


@pytest.fixture
def groups():
    with model_parallel(tensor_model_parallel_size=1):
        yield ProcessGroupCollection.use_mpu_process_groups()


def _table(groups, dtype=torch.bfloat16, rows=32):
    initial = torch.arange(rows * ROW_WIDTH, device="cuda", dtype=torch.float32)
    param = nn.Parameter((initial.reshape(rows, ROW_WIDTH) / 1024).to(dtype))
    mark_engram_table_parameter(
        param,
        EngramTableParameterMetadata(
            row_parallel=True, table_group=groups.tp_dp_cp, tp_group=groups.tp
        ),
    )
    return param


def _config(dtype=torch.bfloat16):
    return OptimizerConfig(
        optimizer="adam",
        lr=3.0e-3,
        min_lr=3.0e-4,
        adam_beta1=0.8,
        adam_beta2=0.95,
        adam_eps=1.0e-8,
        weight_decay=0.1,
        bf16=dtype == torch.bfloat16,
        clip_grad=0.0,
    )


def _native(param, groups, config):
    parallel_groups = EngramParallelGroups.create(groups.tp_dp_cp)
    mark_engram_table_parameter(
        param,
        dataclasses.replace(
            param.engram_table_metadata, replica_group=parallel_groups.replica_group
        ),
    )
    module = nn.Module()
    module.register_parameter("table", param)
    model = DistributedDataParallel(
        TransformerConfig(num_layers=1, num_attention_heads=1),
        DistributedDataParallelConfig(),
        module,
    )
    optimizer = get_engram_optimizer(
        config,
        [model],
        {},
        pg_collection=groups,
        parallel_groups=parallel_groups,
        use_gloo_process_groups=False,
    )
    return optimizer.chained_optimizers[-1]


@pytest.mark.parametrize("fp16_config", [False, True])
def test_owner_adam_rejects_fp16(groups, fp16_config):
    config = _config(torch.float16)
    config.fp16 = fp16_config
    param = _table(groups, dtype=torch.float16)
    with pytest.raises(ValueError, match="FP32 and BF16"):
        _native(param, groups, config)
    assert param.main_grad.dtype == torch.float32
    assert torch.count_nonzero(param.main_grad) == 0


def _chain(children, groups):
    return ChainedOptimizer(
        children, synchronize_nonfinite_grads=True, nonfinite_grad_group=groups.tp_dp_cp
    )


def _step_value(optimizer):
    group = optimizer.optimizer.param_groups[0]
    if "step" in group:
        return int(group["step"])
    return int(optimizer.optimizer.state[group["params"][0]]["step"])


def test_owner_adam_keeps_fp32_state_with_precision_aware_parent_config(groups):
    config = OptimizerConfig(
        optimizer="adam",
        lr=3.0e-3,
        min_lr=3.0e-4,
        weight_decay=0.1,
        bf16=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        main_params_dtype=torch.float16,
        exp_avg_dtype=torch.float16,
        exp_avg_sq_dtype=torch.float16,
        store_param_remainders=False,
    )
    parent_before = copy.deepcopy(vars(config))
    assert config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    param = _table(groups)
    optimizer = _native(param, groups, config)

    assert optimizer.config is not config
    assert optimizer.config.use_distributed_optimizer
    assert not optimizer.config.use_precision_aware_optimizer
    assert not optimizer.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
    main_param = optimizer.get_parameters()[0]
    state = optimizer.optimizer.state[main_param]
    for tensor in (main_param, state["exp_avg"], state["exp_avg_sq"]):
        assert tensor.dtype == torch.float32
        assert tensor.numel() == param.numel()
        assert tensor.device == param.device
    assert all(
        group["weight_decay"] == 0.0 and group["wd_mult"] == 0.0 for group in optimizer.param_groups
    )
    assert vars(config) == parent_before


def test_native_load_reuses_fp32_buffers_already_filled_by_dcp(groups):
    param = _table(groups)
    optimizer = _native(param, groups, _config())
    main = optimizer.get_parameters()[0]
    state = optimizer.optimizer.state[main]
    pointers = [state[key].data_ptr() for key in ("exp_avg", "exp_avg_sq")]
    # The DCP loading state dict points at preallocated destination tensors.
    loading_state = optimizer.state_dict()
    for key, value in (("exp_avg", 0.25), ("exp_avg_sq", 0.5)):
        optimizer._get_main_param_and_optimizer_states(param)[key].fill_(value)
    optimizer.load_state_dict(loading_state)
    assert [
        optimizer.optimizer.state[main][key].data_ptr() for key in ("exp_avg", "exp_avg_sq")
    ] == pointers
    assert torch.all(optimizer.optimizer.state[main]["exp_avg"] == 0.25)
    assert torch.all(optimizer.optimizer.state[main]["exp_avg_sq"] == 0.5)


@pytest.mark.parametrize("source", ["table", "dense", "mtp"])
@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_chain_skips_all_children_and_counters_on_one_rank_nonfinite(groups, source, nonfinite):
    config = _config()
    table = _table(groups)
    native = _native(table, groups, config)
    dense_param = nn.Parameter(torch.ones(3, device="cuda", dtype=torch.bfloat16))
    dense_param.main_grad = torch.ones_like(dense_param, dtype=torch.float32)
    if source == "mtp":
        dense_param.grad_norm_group = "mtp"
    dense = _get_megatron_optimizer_based_on_param_groups(
        config,
        [nn.Module()],
        [{"params": [dense_param]}],
        model_parallel_group=groups.mp,
        pg_collection=groups,
    )
    # LayerWise also nests its native optimizers in a ChainedOptimizer.
    optimizer = _chain([ChainedOptimizer([dense]), native], groups)
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=0.0,
        max_lr=config.lr,
        min_lr=config.min_lr,
        lr_warmup_steps=4,
        lr_decay_steps=16,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=16,
        wd_incr_style="constant",
    )

    def training_update():
        # Exercise the optimizer+scheduler contract used by training.train_step;
        # this unit test does not mock or execute the full forward/backward driver.
        successful, _, _ = optimizer.step()
        if successful:
            scheduler.step(increment=1)
        return successful

    table.main_grad.fill_(0.5)
    initial_lrs = [group["lr"] for group in optimizer.param_groups]
    assert training_update()
    assert scheduler.num_steps == 1
    before_lrs = [group["lr"] for group in optimizer.param_groups]
    assert before_lrs != initial_lrs
    before_params = [param.detach().clone() for param in optimizer.get_parameters()]
    before_states = copy.deepcopy(optimizer.state_dict())
    before_scheduler = copy.deepcopy(scheduler.state_dict())
    table.main_grad.fill_(0.25)
    dense_param.main_grad.fill_(0.25)
    if torch.distributed.get_rank() == 0:
        target = table.main_grad if source == "table" else dense_param.main_grad
        target.view(-1)[0] = nonfinite
    assert not training_update()
    assert scheduler.state_dict() == before_scheduler
    assert [group['lr'] for group in optimizer.param_groups] == before_lrs
    for param, before in zip(optimizer.get_parameters(), before_params):
        torch.testing.assert_close(param, before, rtol=0, atol=0)

    # state_dict recursively includes moments, FP32 masters, and group/per-param steps.
    def assert_same(left, right):
        if torch.is_tensor(left):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                assert_same(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                assert_same(a, b)
        else:
            assert left == right

    assert_same(optimizer.state_dict(), before_states)
    table.main_grad.fill_(0.25)
    dense_param.main_grad.fill_(0.25)
    assert training_update()
    assert scheduler.num_steps == 2
    assert [group['lr'] for group in optimizer.param_groups] != before_lrs
    assert _step_value(native) == 2
    assert _step_value(dense) == 2
    assert optimizer._synchronize_steps() in (None, 2)


def test_empty_owner_participates_in_native_statistics(groups):
    optimizer = _native(_table(groups, rows=0), groups, _config())
    chain = _chain([optimizer], groups)
    assert chain.step()[0]
    assert optimizer.get_grad_norm() == 0.0
    assert optimizer.count_zeros() == 0
    optimizer.load_state_dict(optimizer.state_dict())


@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"optimizer": "sgd"}, "tables require Adam"),
        ({"loss_scale": 128.0}, "unity loss scaling"),
        ({"fp16": True}, "FP32 and BF16"),
        ({"params_dtype": torch.float16}, "FP32 and BF16"),
        ({"optimizer_cpu_offload": True}, "optimizer offload"),
        ({"optimizer_cuda_graph": True}, "CUDA graphs"),
    ],
)
def test_engram_optimizer_rejects_unvalidated_config(overrides, error):
    model = _OptimizerFixtureModel()
    values = {"optimizer": "adam", "lr": 2.0e-4, "min_lr": 2.0e-5}
    values.update(overrides)
    config = OptimizerConfig(**values)
    with pytest.raises(ValueError, match=error):
        _validate_owner_config(config, [model])


@pytest.mark.parametrize("overlap", [False, True])
def test_owner_adam_does_not_inherit_mxfp8_buffer_operations(groups, overlap):
    config = _config()
    config.use_distributed_optimizer = True
    config.overlap_param_gather = overlap
    config.reuse_grad_buf_for_mxfp8_param_ag = True
    param = _table(groups)
    optimizer = _native(param, groups, config)
    assert optimizer.config.use_distributed_optimizer
    assert not optimizer.config.overlap_param_gather
    assert not optimizer.config.reuse_grad_buf_for_mxfp8_param_ag
    assert config.reuse_grad_buf_for_mxfp8_param_ag
    param.main_grad.fill_(0.5)
    previous = param.detach().clone()
    assert _chain([optimizer], groups).step()[0]
    assert not torch.equal(param, previous)
    torch.testing.assert_close(
        param, optimizer.get_parameters()[0].view_as(param).to(param.dtype), rtol=0, atol=0
    )


def test_owner_statistics_use_native_group_once(groups):
    rank = torch.distributed.get_rank(group=groups.tp_dp_cp)
    world_size = torch.distributed.get_world_size(group=groups.tp_dp_cp)
    param = _table(groups, rows=rank + 1)
    optimizer = _native(param, groups, _config())
    assert optimizer.get_grad_stats_parallel_group() is groups.tp_dp_cp
    assert not hasattr(optimizer, "uses_custom_grad_norm")
    param.main_grad.zero_()
    param.main_grad[:, 0] = rank + 1
    optimizer.prepare_grads()
    expected_squared_norm = sum((rank + 1) ** 3 for rank in range(world_size))
    expected_zeros = sum((rank + 1) * (ROW_WIDTH - 1) for rank in range(world_size))
    assert float(optimizer.get_grad_norm()) == pytest.approx(expected_squared_norm**0.5)
    assert optimizer.count_zeros() == expected_zeros


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf")])
def test_single_owner_chain_keeps_checkpoint_shape_and_skips(groups, nonfinite):
    param = _table(groups)
    optimizer = _native(param, groups, _config())
    chain = _chain([optimizer], groups)
    # A one-child ChainedOptimizer retains its existing unprefixed state layout.
    assert chain.state_dict().keys() == optimizer.state_dict().keys()
    param.main_grad.fill_(0.5)
    assert chain.step()[0]
    previous = optimizer.get_parameters()[0].detach().clone()
    if torch.distributed.get_rank(group=groups.tp_dp_cp) == 0:
        param.main_grad[0, 0] = nonfinite
    assert not chain.step()[0]
    torch.testing.assert_close(optimizer.get_parameters()[0], previous, rtol=0, atol=0)
    assert _step_value(optimizer) == 1


@pytest.mark.parametrize("optimizer", ["adam", "sgd"])
def test_no_table_uses_ordinary_policy_without_owner_validation(monkeypatch, optimizer):
    model = nn.Linear(3, 2)
    config = OptimizerConfig(
        optimizer=optimizer, lr=0.01, min_lr=0.001, fp16=True, loss_scale=128.0
    )
    groups = _groups(model, monkeypatch, config)
    assert sum(len(group["params"]) for group in groups) == 2
    assert all(not group.get("is_engram_row_parallel") for group in groups)
    assert all(group["max_lr"] == config.lr and group["lr_mult"] == 1.0 for group in groups)


def test_native_muon_layer_wise_state_is_initialized_without_an_update(groups):
    from megatron.core.transformer.engram.memory import RowShardedMultiHeadEmbedding

    topology = EngramParallelGroups.create(groups.tp_dp_cp, Utils.world_size)
    model = nn.Module()
    model.memory = RowShardedMultiHeadEmbedding(
        [5, 7], D=ROW_WIDTH, layer_id=0, params_dtype=torch.bfloat16, parallel_groups=topology
    ).cuda()
    model.fusion = nn.Linear(ROW_WIDTH, ROW_WIDTH, bias=False, device="cuda", dtype=torch.bfloat16)
    model_config = TransformerConfig(num_layers=1, hidden_size=ROW_WIDTH, num_attention_heads=1)
    ddp = DistributedDataParallel(
        model_config, DistributedDataParallelConfig(grad_reduce_in_fp32=True), model
    )
    expected_weights = [parameter.detach().clone() for parameter in model.parameters()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state()
    config = OptimizerConfig(
        optimizer="muon",
        use_layer_wise_distributed_optimizer=True,
        bf16=True,
        lr=0.001,
        min_lr=0.0,
        clip_grad=0.0,
    )
    optimizer = get_engram_optimizer(
        config, [ddp], pg_collection=groups, parallel_groups=topology, use_gloo_process_groups=False
    )

    def leaves(current):
        if hasattr(current, "chained_optimizers"):
            for child in current.chained_optimizers:
                yield from leaves(child)
        else:
            yield current

    state_tensors = []
    momentum_count = 0
    for child in leaves(optimizer):
        for group in child.param_groups:
            assert int(group.get("step", 0)) == 0
            for parameter in group["params"]:
                state = child.optimizer.state[parameter]
                assert state
                momentum_count += "momentum_buffer" in state
                for name, value in state.items():
                    if torch.is_tensor(value):
                        assert torch.count_nonzero(value) == 0
                        state_tensors.append((child, parameter, name, value))
    momentum_count = torch.tensor(momentum_count, device="cuda")
    torch.distributed.all_reduce(momentum_count)
    assert momentum_count.item() > 0
    pointers = [value.data_ptr() for _, _, _, value in state_tensors]
    engram_optimizer._initialize_optimizer_state(optimizer.chained_optimizers[0])
    assert pointers == [
        child.optimizer.state[parameter][name].data_ptr()
        for child, parameter, name, _ in state_tensors
    ]
    for expected, actual in zip(expected_weights, model.parameters()):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)


class _PipelineTables(nn.Module):
    """Minimal model with stable table IDs and optional table-free pipeline stages."""

    def __init__(self, pg, ids, topology):
        super().__init__()
        from megatron.core.transformer.engram.memory import RowShardedMultiHeadEmbedding

        self.config = TransformerConfig(num_layers=2, hidden_size=8, num_attention_heads=2)
        self.tables = nn.ModuleDict(
            {
                str(i): RowShardedMultiHeadEmbedding(
                    [7, 11],
                    5,
                    layer_id=i,
                    params_dtype=torch.float32,
                    parallel_groups=topology,
                    tp_group=pg.tp,
                    dp_group=pg.dp,
                    dp_cp_group=pg.dp_cp,
                ).cuda()
                for i in ids
            }
        )

    def sharded_state_dict(self):
        result = {}
        for name, table in self.tables.items():
            result.update(table.sharded_state_dict(prefix=f'table_{name}.'))
        return result


def _logical_values(state):
    from megatron.core.dist_checkpointing.mapping import (
        ShardedTensor,
        ShardedTensorFactory,
        is_main_replica,
    )

    def leaves(value):
        if isinstance(value, ShardedTensorFactory):
            yield from leaves(value.build())
        elif isinstance(value, ShardedTensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from leaves(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from leaves(item)

    local = [
        (x.key, x.global_shape, x.global_offset, x.data.detach().cpu().clone())
        for x in leaves(state)
        if is_main_replica(x.replica_id)
    ]
    gathered = [None] * Utils.world_size
    torch.distributed.all_gather_object(gathered, local)
    result, covered = {}, {}
    for records in gathered:
        for key, shape, offset, data in records:
            if key not in result:
                result[key], covered[key] = torch.empty(shape, dtype=data.dtype), torch.zeros(
                    shape, dtype=torch.bool
                )
            index = tuple(slice(start, start + size) for start, size in zip(offset, data.shape))
            assert not covered[key][index].any(), (key, offset)
            result[key][index], covered[key][index] = data, True
    assert all(mask.all() for mask in covered.values())
    return result


@pytest.mark.parametrize('empty_stage', [False, True])
def test_native_adam_moves_tables_between_pipeline_stages(tmp_path_dist_ckpt, empty_stage):
    """Reshard model and Adam moments PP -> TP -> PP, including an empty owner."""
    if Utils.world_size != 2:
        pytest.skip('requires two ranks')
    from megatron.core import parallel_state
    from megatron.core.dist_checkpointing import load, save

    previous = expected = None
    layouts = [(2, 1, None), (1, 1, 1), (1, 2, 2), (2, 1, None)]
    for phase, (pp, tp, shards) in enumerate(layouts):
        with model_parallel(tensor_model_parallel_size=tp, pipeline_model_parallel_size=pp):
            pg = ProcessGroupCollection.use_mpu_process_groups()
            topology = EngramParallelGroups.create(
                pg.tp_dp_cp, shards, stats_group=torch.distributed.group.WORLD
            )
            ids = [0] if empty_stage else [0, 1]
            if pp == 2:
                owner = parallel_state.get_pipeline_model_parallel_rank()
                ids = [i for i in ids if (i if phase == 0 else 1 - i) == owner]
            model = _PipelineTables(pg, ids, topology)
            ddp = DistributedDataParallel(
                model.config, DistributedDataParallelConfig(), model, pg_collection=pg
            )
            optimizer = get_engram_optimizer(
                OptimizerConfig(lr=0.002, min_lr=0.0001, weight_decay=0.0, clip_grad=0.5),
                [ddp],
                pg_collection=pg,
                parallel_groups=topology,
                use_gloo_process_groups=False,
            )

            def state(loading=False):
                weights = model.sharded_state_dict()
                return {
                    'model': weights,
                    'optimizer': optimizer.sharded_state_dict(
                        weights,
                        is_loading=loading,
                        metadata={'distrib_optim_sharding_type': 'fully_sharded_model_space'},
                    ),
                }

            if previous is None:
                for table in model.tables.values():
                    table.embedding.weight.main_grad.fill_(0.25)
                assert optimizer.step()[0]
                expected = _logical_values(state())
            else:
                loaded = load(state(True), previous)
                for name, table in model.tables.items():
                    prefix = f'table_{name}.'
                    table.load_state_dict(
                        {
                            k[len(prefix) :]: v
                            for k, v in loaded['model'].items()
                            if k.startswith(prefix)
                        }
                    )
                optimizer.load_state_dict(loaded['optimizer'])
                actual = _logical_values(state())
                assert actual.keys() == expected.keys()
                for key in expected:
                    torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, msg=key)
            directory = tmp_path_dist_ckpt / f'native_pp_{empty_stage}_{phase}'
            if Utils.rank == 0:
                directory.mkdir(parents=True, exist_ok=True)
            torch.distributed.barrier()
            save(state(), directory)
            previous = directory


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('precompute', [False, True])
def test_optimizer_replica_buffers_share_native_ddp_lifecycle(dtype, precompute):
    Utils.initialize_model_parallel()
    try:
        pg = ProcessGroupCollection.use_mpu_process_groups()
        model = nn.Module()
        model.dense = nn.Parameter(torch.ones(256, device='cuda', dtype=dtype))
        model.table = nn.Parameter(torch.ones(256, device='cuda', dtype=dtype))
        model.table.optimizer_sharding_group = pg.dp_cp
        config = DistributedDataParallelConfig(use_distributed_optimizer=precompute)
        layout = (
            DistributedOptimizer.compute_full_param_layout(
                list(model.parameters()), None, pg.dp_cp.size(), config
            )
            if precompute
            else None
        )
        ddp = DistributedDataParallel(
            TransformerConfig(num_layers=1, num_attention_heads=1),
            config,
            model,
            pg_collection=pg,
            full_param_layout=layout,
        )
        assert len(ddp.buffers) == 2
        assert sum(model.table is p for b in ddp.buffers for p in b.params) == 1
        table_buffer = next(b for b in ddp.buffers if any(p is model.table for p in b.params))
        assert table_buffer.data_parallel_group is pg.dp_cp
        assert table_buffer.grad_dtype == torch.float32
        assert table_buffer.ddp_config.use_distributed_optimizer
        assert not table_buffer.ddp_config.overlap_grad_reduce
        assert not table_buffer.ddp_config.overlap_param_gather
        # Disabling bucketing must not merge distinct optimizer replica policies.
        assert len(partition_buckets(ddp.buffers, force_single_bucket_group=True)) == 2
        for _ in range(2):
            ddp.zero_grad_buffer()
            assert torch.count_nonzero(model.table.main_grad) == 0
            model.table.main_grad.add_(pg.dp_cp.rank() + 1)
            model.table.main_grad.add_(pg.dp_cp.rank() + 1)
            ddp.scale_gradients(0.5)
            group = ddp.param_to_bucket_group[model.table]
            group.start_grad_sync()
            group.finish_grad_sync()
            group.finish_grad_sync()
            bucket = table_buffer.buckets[0]
            shard = bucket.grad_data.chunk(pg.dp_cp.size())[pg.dp_cp.rank()]
            expected = pg.dp_cp.size() * (pg.dp_cp.size() + 1) / 2
            torch.testing.assert_close(shard, torch.full_like(shard, expected), rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
