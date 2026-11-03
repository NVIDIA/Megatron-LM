# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from dataclasses import fields

import pytest

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import clear_nvte_env_vars
from tools.checkpoint.utils import (
    build_checkpoint_converter_process_groups,
    initialize_checkpoint_converter_fake_process_groups,
    register_checkpoint_converter_process_groups,
    set_checkpoint_converter_model_ranks,
)


@pytest.fixture
def fake_groups():
    """Register the converter's fake groups for TP=4, PP=2 and destroy them afterwards."""
    parallel_state.destroy_model_parallel()
    yield initialize_checkpoint_converter_fake_process_groups(
        parallel_state, tensor_parallel_size=4, pipeline_parallel_size=2, expert_parallel_size=1
    )
    parallel_state.destroy_model_parallel()


def test_fake_groups_support_default_process_group_collection(fake_groups):
    groups = ProcessGroupCollection.use_mpu_process_groups()

    assert groups.tp.size() == 4
    assert groups.cp.size() == 1
    assert groups.pp.size() == 2
    assert groups.ep.size() == 1
    assert groups.dp.size() == 1
    assert groups.dp_cp.size() == 1
    assert groups.dp_cp_gtp_remat.size() == 1
    assert groups.intra_dp_cp.size() == 1
    assert groups.expt_dp.size() == 1
    assert groups.expt_dp_gtp_remat.size() == 1


def test_fake_groups_satisfy_language_module(mocker, fake_groups):
    """A model built without a collection in a converter process, for example by a model
    provider that takes none, resolves the registered fake groups with a deprecation warning,
    and LanguageModule accepts them."""
    clear_nvte_env_vars()
    mocker.patch.object(process_groups_config, '_warned_global_process_group_fallbacks', set())
    mocker.patch('torch.distributed.is_initialized', return_value=True)

    with pytest.warns(FutureWarning, match='LanguageModule was called without `pg_collection`'):
        model = LanguageModule(
            TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)
        )

    assert model.tp_group.size() == 4
    assert model.cp_group.size() == 1
    assert model.pp_group.size() == 2


# The groups the converter creates, with their sizes for TP=4, PP=2, EP=2.
_CONVERTER_GROUP_SIZES = {
    'tp': 4,
    'cp': 1,
    'pp': 2,
    'mp': 8,
    'ep': 2,
    'expt_tp': 4,
    'tp_ep': 8,
    'tp_ep_pp': 16,
    'dp': 1,
    'dp_gtp_remat': 1,
    'dp_cp': 1,
    'dp_cp_gtp_remat': 1,
    'intra_dp_cp': 1,
    'expt_dp': 1,
    'expt_dp_gtp_remat': 1,
}


@pytest.fixture
def converter_groups():
    """Converter groups for TP=4, PP=2, EP=2, registered with an empty parallel_state."""
    parallel_state.destroy_model_parallel()
    pg_collection = build_checkpoint_converter_process_groups(
        tensor_parallel_size=4, pipeline_parallel_size=2, expert_parallel_size=2
    )
    register_checkpoint_converter_process_groups(parallel_state, pg_collection)
    yield pg_collection
    parallel_state.destroy_model_parallel()


def test_converter_collection_defines_every_field():
    groups = vars(
        build_checkpoint_converter_process_groups(
            tensor_parallel_size=4, pipeline_parallel_size=2, expert_parallel_size=2
        )
    )

    assert set(groups) == {field.name for field in fields(ProcessGroupCollection)}
    for name, size in _CONVERTER_GROUP_SIZES.items():
        assert groups[name].size() == size, name
        assert groups[name].rank() == 0, name
    assert groups['dynamic_cp'] == {}
    assert groups['dynamic_tp_cp'] == {}
    unset = set(groups) - set(_CONVERTER_GROUP_SIZES) - {'dynamic_cp', 'dynamic_tp_cp'}
    assert {name for name in unset if groups[name] is not None} == set()
    # GTP is inactive, so its data-parallel groups are the plain data-parallel groups.
    assert groups['dp_cp_gtp_remat'] is groups['dp']
    assert groups['expt_dp_gtp_remat'] is groups['expt_dp']


def test_registered_groups_form_the_default_collection(converter_groups):
    """Code that resolves the global groups gets exactly the collection the models get."""
    assert parallel_state.get_registered_process_groups() is converter_groups
    default_groups = vars(ProcessGroupCollection.use_mpu_process_groups())

    for name, group in vars(converter_groups).items():
        if isinstance(group, dict):
            assert default_groups[name] == group, name
        else:
            assert default_groups[name] is group, name


def test_model_ranks_select_one_model(converter_groups):
    set_checkpoint_converter_model_ranks(
        converter_groups, pipeline_rank=1, tensor_rank=3, expert_rank=1
    )

    assert converter_groups.pp.rank() == 1
    assert converter_groups.tp.rank() == 3
    assert converter_groups.expt_tp.rank() == 3
    assert converter_groups.ep.rank() == 1
    # Global accessors that ask a group for its own rank or size follow the collection.
    assert parallel_state.get_tensor_model_parallel_rank() == 3
    assert parallel_state.get_tensor_model_parallel_world_size() == 4
    assert parallel_state.get_pipeline_model_parallel_world_size() == 2


def test_registering_twice_is_an_error(converter_groups):
    """A converter does not replace process groups that are already registered."""
    with pytest.raises(RuntimeError, match='already'):
        initialize_checkpoint_converter_fake_process_groups(
            parallel_state, tensor_parallel_size=1, pipeline_parallel_size=1, expert_parallel_size=1
        )
    assert parallel_state.get_registered_process_groups() is converter_groups


def test_fake_groups_are_the_returned_collection(fake_groups):
    assert parallel_state.get_tensor_model_parallel_group() is fake_groups.tp
    assert parallel_state.get_pipeline_model_parallel_group() is fake_groups.pp
    assert parallel_state.get_context_parallel_group() is fake_groups.cp
    assert parallel_state.get_expert_model_parallel_group() is fake_groups.ep
