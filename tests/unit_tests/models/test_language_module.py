# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import clear_nvte_env_vars


@pytest.fixture
def explicit_groups(mocker):
    """A caller-owned grid; reading the global grid must never repair its fields."""
    clear_nvte_env_vars()
    mocker.patch.object(
        ProcessGroupCollection,
        'use_mpu_process_groups',
        side_effect=AssertionError('unexpected global collection access'),
    )
    mocker.patch.object(parallel_state, 'is_initialized', return_value=True)
    mocker.patch.object(
        parallel_state,
        'get_tensor_model_parallel_group',
        side_effect=AssertionError('unexpected global TP access'),
    )
    return ProcessGroupCollection(
        tp=mocker.sentinel.tp,
        cp=mocker.sentinel.cp,
        pp=mocker.sentinel.pp,
        embd=mocker.sentinel.embd,
    )


def config():
    return TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)


@pytest.mark.parametrize('pass_none', [False, True])
def test_requires_explicit_collection(explicit_groups, pass_none):
    kwargs = {'pg_collection': None} if pass_none else {}
    with pytest.raises(AssertionError, match='requires an explicit pg_collection'):
        LanguageModule(config(), **kwargs)


@pytest.mark.parametrize('field', ['tp', 'cp', 'pp', 'embd'])
def test_rejects_omitted_field(explicit_groups, field):
    delattr(explicit_groups, field)
    # Declared but unset collection fields are visible to hasattr and resolve to None.
    assert hasattr(explicit_groups, field)
    with pytest.raises(AssertionError, match=f'must explicitly define {field}'):
        LanguageModule(config(), pg_collection=explicit_groups)


@pytest.mark.parametrize('field', ['tp', 'cp', 'pp'])
def test_live_model_requires_participating_axes(mocker, explicit_groups, field):
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    setattr(explicit_groups, field, None)
    with pytest.raises(AssertionError, match=f'participating {field} group'):
        LanguageModule(config(), pg_collection=explicit_groups)


@pytest.mark.parametrize('initialized', [False, True])
@pytest.mark.parametrize('field', ['tp', 'cp', 'pp', 'embd'])
def test_rejects_nonmember_sentinel(mocker, explicit_groups, initialized, field):
    mocker.patch('torch.distributed.is_initialized', return_value=initialized)
    setattr(explicit_groups, field, torch.distributed.GroupMember.NON_GROUP_MEMBER)
    with pytest.raises(AssertionError, match=f'{field} cannot be NON_GROUP_MEMBER'):
        LanguageModule(config(), pg_collection=explicit_groups)


@pytest.mark.parametrize('initialized', [False, True])
def test_preserves_supplied_groups(mocker, explicit_groups, initialized):
    mocker.patch('torch.distributed.is_initialized', return_value=initialized)
    model = LanguageModule(config(), pg_collection=explicit_groups)
    assert model.pg_collection is explicit_groups
    for field in ('tp', 'cp', 'pp', 'embd'):
        assert getattr(model, f'{field}_group') is getattr(explicit_groups, field)


def test_offline_model_accepts_explicit_none(mocker, explicit_groups):
    mocker.patch('torch.distributed.is_initialized', return_value=False)
    for field in ('tp', 'cp', 'pp', 'embd'):
        setattr(explicit_groups, field, None)
    model = LanguageModule(config(), pg_collection=explicit_groups)
    assert model.tp_group is None
    assert model.cp_group is None
    assert model.pp_group is None
    assert model.embd_group is None


def test_nonparticipating_embedding_group_is_explicit_none(mocker, explicit_groups):
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    explicit_groups.embd = None
    model = LanguageModule(config(), pg_collection=explicit_groups)
    assert not model._is_in_embd_group()


def test_loss_uses_the_supplied_tensor_group(mocker, explicit_groups):
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    model = LanguageModule(config(), pg_collection=explicit_groups)
    model.vocab_parallel_cross_entropy = mocker.Mock(return_value=torch.zeros(2, 1))
    labels = torch.zeros(1, 2, dtype=torch.long)
    logits = torch.zeros(2, 1, 4)
    model.compute_language_model_loss(labels, logits)
    assert model.vocab_parallel_cross_entropy.call_args.args[-1] is explicit_groups.tp
