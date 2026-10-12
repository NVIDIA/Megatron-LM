# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Global layer keys must distinguish FSDP pipeline stages in DCP checkpoints."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from megatron.core.dist_checkpointing.core import CheckpointingException
from megatron.core.transformer.fsdp_dtensor_checkpoint import (
    handle_pipeline_parallel_in_state_dict,
    validate_fsdp_dtensor_model_load,
)
from tests.unit_tests.test_utilities import Utils


def stage_model(numbers):
    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList()
    for number in numbers:
        layer = torch.nn.Linear(2, 2, bias=False)
        layer.layer_number = number
        model.decoder.layers.append(layer)
    return model


@pytest.mark.parametrize('wrapper', ['', 'module.', 'model.module.module.'])
@pytest.mark.parametrize('numbers', [[1, 2, 3], [4, 5, 6], [2, 7, 11]])
def test_global_keys_preserve_local_load_buffers(wrapper, numbers):
    model = stage_model(numbers)
    keys = [f'{wrapper}decoder.layers.{i}.weight' for i in range(len(numbers))]
    original = {key: torch.tensor([i + 1.0]) for i, key in enumerate(keys)}
    optimizer = {
        'state': {key: {'exp_avg': value.clone()} for key, value in original.items()},
        'param_to_group_meta': {key: {'lr': i + 0.1} for i, key in enumerate(keys)},
    }
    mapped, mapped_opt = handle_pipeline_parallel_in_state_dict(model, original, optimizer)
    expected = [f'{wrapper}decoder.layers.{number - 1}.weight' for number in numbers]
    assert list(mapped) == expected
    for local, glob in zip(keys, expected):
        assert mapped[glob] is original[local]
        assert mapped_opt['state'][glob] is optimizer['state'][local]
        assert mapped_opt['param_to_group_meta'][glob] is optimizer['param_to_group_meta'][local]
    assert list(original) == keys
    assert list(optimizer['state']) == keys


def test_mtp_and_non_layer_keys_are_not_shifted():
    model = stage_model([5])
    model.mtp = stage_model([99])
    values = {
        'module.decoder.layers.0.mlp.experts.linear_fc1.weight3': torch.tensor(1),
        'module.mtp.decoder.layers.0.weight': torch.tensor(2),
        'module.embedding.word_embeddings.weight': torch.tensor(3),
        'module.decoder.layers.01.weight': torch.tensor(4),
    }
    result, _ = handle_pipeline_parallel_in_state_dict(model, values, None)
    assert set(result) == {
        'module.decoder.layers.4.mlp.experts.linear_fc1.weight3',
        *list(values)[1:],
    }


def test_duplicate_global_layer_numbers_are_rejected():
    with pytest.raises(ValueError, match='Duplicate global FSDP checkpoint key'):
        handle_pipeline_parallel_in_state_dict(
            stage_model([2, 2]), {'decoder.layers.0.weight': 0, 'decoder.layers.1.weight': 1}, None
        )


def test_legacy_colliding_checkpoint_is_rejected_on_load():
    requested, _ = handle_pipeline_parallel_in_state_dict(
        stage_model([3, 4]),
        {'decoder.layers.0.weight': torch.zeros(2), 'decoder.layers.1.weight': torch.zeros(2)},
        None,
    )
    with pytest.raises(CheckpointingException, match='not present in the checkpoint'):
        validate_fsdp_dtensor_model_load(
            {'model.decoder.layers.0.weight': None, 'model.decoder.layers.1.weight': None},
            {'model': requested},
            'legacy-checkpoint',
        )


@pytest.mark.parametrize('pp', [1, 2, 4])
def test_distributed_checkpoint_retains_every_stage(tmp_path, pp):
    """Real DCP save/load with stage-specific weights, Adam moments and group metadata."""
    Utils.initialize_model_parallel(pipeline_model_parallel_size=pp)
    rank = dist.get_rank()
    world = dist.get_world_size()
    dp = world // pp
    stage = rank // dp
    groups = [dist.new_group(list(range(s * dp, (s + 1) * dp))) for s in range(pp)]
    mesh = DeviceMesh.from_group(groups[stage], 'cuda')
    # Uneven stages: 1, 2, ... layers, each with a globally unique index.
    first = stage * (stage + 1) // 2
    numbers = list(range(first + 1, first + stage + 2))
    model = stage_model(numbers)
    shared_path = [str(tmp_path / 'checkpoint') if rank == 0 else None]
    dist.broadcast_object_list(shared_path, src=0)

    def state(fill):
        weights, moments, meta = {}, {}, {}
        for local, number in enumerate(numbers):
            key = f'module.decoder.layers.{local}.weight'
            tensor = torch.full((2, 2), fill(number), device='cuda')
            weights[key] = DTensor.from_local(tensor, mesh, [Shard(0)])
            moments[key] = {'exp_avg': weights[key] * 2, 'exp_avg_sq': weights[key] * 3}
            meta[key] = {'lr': fill(number) * 0.001}
        raw_opt = {'state': moments, 'param_to_group_meta': meta}
        mapped, mapped_opt = handle_pipeline_parallel_in_state_dict(model, weights, raw_opt)
        return {'model': mapped, 'optimizer': mapped_opt}, weights, raw_opt

    try:
        saved, _, _ = state(float)
        dcp.save(saved, checkpoint_id=shared_path[0])
        loaded, raw_weights, raw_opt = state(lambda number: -1.0)
        dcp.load(loaded, checkpoint_id=shared_path[0])
        for local, number in enumerate(numbers):
            key = f'module.decoder.layers.{local}.weight'
            assert torch.all(raw_weights[key].to_local() == number)
            assert torch.all(raw_opt['state'][key]['exp_avg'].to_local() == number * 2)
            assert torch.all(raw_opt['state'][key]['exp_avg_sq'].to_local() == number * 3)
            assert raw_opt['param_to_group_meta'][key]['lr'] == number * 0.001
        dist.barrier()
    finally:
        for group in groups:
            if group != dist.GroupMember.NON_GROUP_MEMBER:
                dist.destroy_process_group(group)
        Utils.destroy_model_parallel()


def test_v2_checkpoint_preprocessing_preserves_global_optimizer_keys():
    """An uneven stage has overlapping local/global ranges; v2 must not remap twice."""
    from megatron.training import checkpointing

    model = stage_model([2, 3, 4])
    values = {f'decoder.layers.{i}.weight': torch.tensor(i) for i in [1, 2, 3]}
    optimizer = {'state': {key: {'exp_avg': value} for key, value in values.items()}}
    args = SimpleNamespace(swiglu=False, num_experts=None, megatron_fsdp_version=2)
    with patch.object(checkpointing, 'preprocess_state_dict_for_uneven_dtensor'):
        result = checkpointing.preprocess_fsdp_dtensor_state_dict(
            args, {'model': values, 'optimizer': optimizer}, model
        )
    assert list(result['optimizer']['state']) == list(values)
    for key in values:
        assert result['optimizer']['state'][key]['exp_avg'] is values[key]
