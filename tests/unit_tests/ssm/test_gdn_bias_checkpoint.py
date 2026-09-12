# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""GDN bias checkpoint migration and TP resharding on real CPU/Gloo process groups."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from megatron.core import dist_checkpointing
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_net.common import _GDNBase
from megatron.core.tensor_parallel.layers import ColumnParallelLinear
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.dist_checkpointing.cpu_test_utils import TorchDistCPUSaveShardedStrategy

pytestmark = pytest.mark.internal

_SECTIONS = (4, 4, 8, 8, 4, 4)
_NAMES = ['query', 'key', 'value', 'z', 'beta', 'alpha']


@pytest.fixture(scope='module')
def gdn_groups(cpu_default_process_group):
    """Every TP layout covers the world with the complementary DP replica group."""
    rank, world_size = dist.get_rank(), dist.get_world_size()
    assert world_size >= 4 and world_size % 4 == 0, 'Run with torchrun on 4 or 8 ranks'
    rank_sets = []
    for tp_size in (1, 2, 4):
        rank_sets.extend(
            tuple(range(base, base + tp_size)) for base in range(0, world_size, tp_size)
        )
        rank_sets.extend(tuple(range(i, world_size, tp_size)) for i in range(tp_size))
    groups = {
        ranks: dist.new_group(list(ranks), backend='gloo') for ranks in dict.fromkeys(rank_sets)
    }
    layouts = {}
    for tp_size in (1, 2, 4):
        base = rank // tp_size * tp_size
        pg = ProcessGroupCollection()
        pg.tp = groups[tuple(range(base, base + tp_size))]
        pg.dp_cp = groups[tuple(range(rank % tp_size, world_size, tp_size))]
        pg.cp = groups[(rank,)]
        pg.gtp_remat = groups[(rank,)]
        layouts[tp_size] = pg
    yield layouts
    for group in reversed(list(groups.values())):
        if group != dist.GroupMember.NON_GROUP_MEMBER:
            dist.destroy_process_group(group)


@pytest.fixture
def checkpoint_root(gdn_groups, tmp_path):
    """Use rank zero's temporary directory on every process."""
    paths = [str(tmp_path) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(paths, src=0)
    yield Path(paths[0])
    dist.barrier()


def _make_gdn(pg, bias=True, fill=True):
    """Build the real checkpoint-producing modules without the CUDA-only FLA forward setup."""
    tp_size, tp_rank = dist.get_world_size(pg.tp), dist.get_rank(pg.tp)
    config = TransformerConfig(
        num_layers=1,
        hidden_size=4,
        num_attention_heads=4,
        tensor_model_parallel_size=tp_size,
        use_cpu_initialization=True,
        perform_initialization=False,
        gradient_accumulation_fusion=False,
    )
    module = _GDNBase.__new__(_GDNBase)
    torch.nn.Module.__init__(module)
    module.pg_collection = pg
    module.conv_bias = False
    module.in_proj_split_sections = tuple(size // tp_size for size in _SECTIONS)
    module.in_proj_split_names = _NAMES
    module.qk_dim_local_tp = module.in_proj_split_sections[0]
    module.v_dim_local_tp = module.in_proj_split_sections[2]
    module.conv_dim_local_tp = sum(module.in_proj_split_sections[:3])
    module.in_proj = ColumnParallelLinear(
        4,
        sum(_SECTIONS),
        config=config,
        init_method=torch.nn.init.zeros_,
        bias=bias,
        gather_output=False,
        skip_bias_add=False,
        tp_group=pg.tp,
        pg_collection=pg,
    )
    conv_dim = module.conv_dim_local_tp
    module.conv1d = torch.nn.Conv1d(conv_dim, conv_dim, 1, groups=conv_dim, bias=False)
    parts = [
        (torch.arange(size, dtype=torch.float32) + 100 * i).chunk(tp_size)[tp_rank]
        for i, size in enumerate(_SECTIONS)
    ]
    local_bias = torch.cat(parts)
    with torch.no_grad():
        module.in_proj.weight.copy_(local_bias[:, None] * 10 + torch.arange(4))
        if bias:
            module.in_proj.bias.copy_(local_bias)
        module.conv1d.weight.copy_(torch.cat(parts[:3]).reshape(conv_dim, 1, 1))
        if not fill:
            for parameter in module.parameters():
                parameter.fill_(-1)
    return module


def _state(module, **metadata):
    return module.sharded_state_dict(
        metadata={
            'dp_cp_group': module.pg_collection.dp_cp,
            'gdn_in_proj_bias_split': True,
            **metadata,
        }
    )


def _save(state, directory):
    if dist.get_rank() == 0:
        directory.mkdir()
    dist.barrier()
    # Both save and load use the normal DCP access-integrity validation.
    dist_checkpointing.save(
        state, str(directory), sharded_strategy=TorchDistCPUSaveShardedStrategy()
    )


def _assert_parameters_equal(actual, expected):
    for name, parameter in actual.named_parameters():
        torch.testing.assert_close(
            parameter, dict(expected.named_parameters())[name], rtol=0, atol=0
        )


@pytest.mark.parametrize('source_tp,target_tp', [(2, 1), (1, 2), (2, 4), (4, 2)])
@pytest.mark.parametrize('bias', [False, True], ids=['without_bias', 'with_bias'])
def test_gdn_in_proj_bias_reshards(gdn_groups, checkpoint_root, source_tp, target_tp, bias):
    """The actual GDN state dict and DCP I/O preserve semantic rows when TP changes."""
    source = _make_gdn(gdn_groups[source_tp], bias=bias)
    state = _state(source)
    assert isinstance(state['in_proj.weight'], ShardedTensorFactory)
    if bias:
        assert isinstance(state['in_proj.bias'], ShardedTensorFactory)
        assert [part.key for part in state['in_proj.bias'].build()] == [
            f'in_proj.bias.{name}' for name in _NAMES
        ]
    else:
        assert 'in_proj.bias' not in state
    _save(state, checkpoint_root / 'semantic')

    target = _make_gdn(gdn_groups[target_tp], bias=bias, fill=False)
    loaded = dist_checkpointing.load(_state(target), str(checkpoint_root / 'semantic'))
    target.load_state_dict(loaded)
    _assert_parameters_equal(target, _make_gdn(gdn_groups[target_tp], bias=bias))


def test_gdn_legacy_bias_migration(gdn_groups, checkpoint_root):
    """An old unsplit TP2 bias loads at TP2, resaves with sections, and then loads at TP1."""
    source = _make_gdn(gdn_groups[2])
    legacy_state = _state(source)
    # Reproduce the old producer independently: ColumnParallelLinear's raw bias
    # metadata was left unchanged by GDN before this fix.
    legacy_state['in_proj.bias'] = source.in_proj.sharded_state_dict(
        prefix='in_proj.', metadata={'dp_cp_group': source.pg_collection.dp_cp}
    )['in_proj.bias']
    assert isinstance(legacy_state['in_proj.bias'], ShardedTensor)
    _save(legacy_state, checkpoint_root / 'legacy')

    migrated = _make_gdn(gdn_groups[2], fill=False)
    legacy_request = _state(
        migrated, gdn_in_proj_bias_split=False, gdn_legacy_in_proj_bias_tp_size=2
    )
    assert isinstance(legacy_request['in_proj.bias'], ShardedTensor)
    loaded = dist_checkpointing.load(legacy_request, str(checkpoint_root / 'legacy'))
    migrated.load_state_dict(loaded)
    _assert_parameters_equal(migrated, source)
    _save(_state(migrated), checkpoint_root / 'migrated')

    target = _make_gdn(gdn_groups[1], fill=False)
    loaded = dist_checkpointing.load(_state(target), str(checkpoint_root / 'migrated'))
    target.load_state_dict(loaded)
    _assert_parameters_equal(target, _make_gdn(gdn_groups[1]))


def test_gdn_legacy_bias_rejects_tp_change(gdn_groups):
    target = _make_gdn(gdn_groups[1])
    with pytest.raises(ValueError, match='must first be loaded at their saved TP size'):
        _state(target, gdn_in_proj_bias_split=False, gdn_legacy_in_proj_bias_tp_size=2)


@pytest.mark.parametrize('legacy_tp_size', [None, 0, -1, True, '2'])
def test_gdn_legacy_bias_rejects_invalid_tp_size(gdn_groups, legacy_tp_size):
    target = _make_gdn(gdn_groups[2])
    with pytest.raises(ValueError, match='must be a positive integer'):
        _state(target, gdn_in_proj_bias_split=False, gdn_legacy_in_proj_bias_tp_size=legacy_tp_size)


def test_gdn_bias_defaults_to_semantic_sections(gdn_groups):
    module = _make_gdn(gdn_groups[2])
    state = module.sharded_state_dict(metadata={'dp_cp_group': module.pg_collection.dp_cp})
    assert isinstance(state['in_proj.bias'], ShardedTensorFactory)


def test_gdn_without_bias_ignores_legacy_metadata(gdn_groups):
    module = _make_gdn(gdn_groups[1], bias=False)
    state = _state(module, gdn_in_proj_bias_split=False, gdn_legacy_in_proj_bias_tp_size=2)
    assert 'in_proj.bias' not in state
