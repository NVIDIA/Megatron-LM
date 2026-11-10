# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import warnings
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
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
def test_omitted_collection_warns_and_uses_global_groups(mocker, explicit_groups, pass_none):
    """Omitting the collection is deprecated: it warns and keeps using the global grid."""
    mocker.patch.object(process_groups_config, '_warned_global_process_group_fallbacks', set())
    global_groups = mocker.patch.object(
        ProcessGroupCollection, 'use_mpu_process_groups', return_value=explicit_groups
    )
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    kwargs = {'pg_collection': None} if pass_none else {}
    with pytest.warns(FutureWarning, match='LanguageModule was called without `pg_collection`'):
        model = LanguageModule(config(), **kwargs)
    global_groups.assert_called_once_with()
    assert model.pg_collection is explicit_groups
    assert model.tp_group is explicit_groups.tp


@pytest.mark.parametrize('field', ['tp', 'cp', 'pp'])
def test_rejects_omitted_field(explicit_groups, field):
    delattr(explicit_groups, field)
    # Declared but unset collection fields are visible to hasattr and resolve to None.
    assert hasattr(explicit_groups, field)
    with pytest.raises(ValueError, match=f'must explicitly define {field}'):
        LanguageModule(config(), pg_collection=explicit_groups)


def test_omitted_embedding_group_warns_and_is_none(mocker, explicit_groups):
    """A collection built without embd still works until 0.23, as if embd were None."""
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    delattr(explicit_groups, 'embd')
    with pytest.warns(
        FutureWarning, match='LanguageModule was given a pg_collection without an embd'
    ):
        model = LanguageModule(config(), pg_collection=explicit_groups)
    assert model.embd_group is None


def test_omitted_embedding_field_is_none_for_any_collection_object(mocker):
    """Collections that do not default missing fields to None also get embd=None."""
    clear_nvte_env_vars()
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    groups = SimpleNamespace(tp=mocker.sentinel.tp, cp=mocker.sentinel.cp, pp=mocker.sentinel.pp)
    with pytest.warns(FutureWarning, match='without an embd field'):
        model = LanguageModule(config(), pg_collection=groups)
    assert model.embd_group is None


def test_fallback_collection_is_not_validated(mocker):
    """The deprecated path keeps main's behavior when the global grid lacks a group."""
    clear_nvte_env_vars()
    mocker.patch.object(process_groups_config, '_warned_global_process_group_fallbacks', set())
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    incomplete = ProcessGroupCollection(tp=None, cp=None, pp=None)
    mocker.patch.object(ProcessGroupCollection, 'use_mpu_process_groups', return_value=incomplete)
    with pytest.warns(FutureWarning, match='LanguageModule was called without `pg_collection`'):
        model = LanguageModule(config())
    assert model.pg_collection is incomplete


@pytest.mark.parametrize('field', ['tp', 'cp', 'pp'])
def test_live_model_requires_participating_axes(mocker, explicit_groups, field):
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    setattr(explicit_groups, field, None)
    with pytest.raises(ValueError, match=f'participating {field} group'):
        LanguageModule(config(), pg_collection=explicit_groups)


@pytest.mark.parametrize('initialized', [False, True])
@pytest.mark.parametrize('field', ['tp', 'cp', 'pp', 'embd'])
def test_rejects_nonmember_sentinel(mocker, explicit_groups, initialized, field):
    mocker.patch('torch.distributed.is_initialized', return_value=initialized)
    setattr(explicit_groups, field, torch.distributed.GroupMember.NON_GROUP_MEMBER)
    with pytest.raises(ValueError, match=f'{field} cannot be NON_GROUP_MEMBER') as exc_info:
        LanguageModule(config(), pg_collection=explicit_groups)
    # Only a nonparticipating embedding stage may use None; tp, cp and pp need a real group.
    assert ('use None' in str(exc_info.value)) == (field == 'embd')


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


@pytest.mark.parametrize(
    'supplied, initialized, pp_size, tied, mtp_layers, pre, post, mtp_process, embd_none, warns',
    [
        # Tied output layer on the last stage and the embedding on the first, without embd.
        (True, True, 2, True, 0, False, True, False, True, True),
        (True, True, 2, True, 0, True, False, False, True, True),
        # An MTP stage keeps an embedding copy even with untied output weights.
        (True, True, 2, False, 1, False, False, True, True, True),
        # A usable embd group, the deprecated omitted collection, offline construction, untied
        # weights, a last stage without a copy, and a single stage do not warn.
        (True, True, 2, True, 0, False, True, False, False, False),
        (False, True, 2, True, 0, False, True, False, True, False),
        (True, False, 2, True, 0, True, False, False, True, False),
        (True, True, 2, False, 0, False, True, False, True, False),
        (True, True, 2, False, 1, False, True, False, True, False),
        (True, True, 1, True, 0, True, True, False, True, False),
    ],
)
def test_warns_when_shared_embedding_copies_have_no_embd_group(
    mocker,
    explicit_groups,
    supplied,
    initialized,
    pp_size,
    tied,
    mtp_layers,
    pre,
    post,
    mtp_process,
    embd_none,
    warns,
):
    """A caller's collection without embd on a stage holding shared embeddings warns until 0.23."""
    mocker.patch('torch.distributed.is_initialized', return_value=initialized)
    pp_group = mocker.Mock()
    pp_group.rank.return_value = 0 if pre else pp_size - 1
    pp_group.size.return_value = pp_size
    explicit_groups.pp = pp_group
    explicit_groups.embd = None if embd_none else mocker.sentinel.embd
    stage_config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        pipeline_model_parallel_size=pp_size,
        pipeline_dtype=torch.float32,
    )
    if supplied:
        model = LanguageModule(stage_config, pg_collection=explicit_groups)
    else:
        mocker.patch.object(process_groups_config, '_warned_global_process_group_fallbacks', set())
        mocker.patch.object(
            ProcessGroupCollection, 'use_mpu_process_groups', return_value=explicit_groups
        )
        with pytest.warns(FutureWarning, match='without `pg_collection`'):
            model = LanguageModule(stage_config)
    model.config.mtp_num_layers = mtp_layers
    model.pre_process = pre
    model.post_process = post
    model.share_embeddings_and_output_weights = tied
    model.mtp_process = mtp_process
    mocker.patch.object(
        model, 'shared_embedding_or_output_weight', return_value=torch.nn.Parameter(torch.ones(2))
    )
    mocker.patch.object(LanguageModule, '_is_in_embd_group', return_value=False)
    mocker.patch.object(LanguageModule, 'embedding_warning_printed', True, create=True)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.setup_embeddings_and_output_layer()

    unsynchronized = [
        w
        for w in caught
        if issubclass(w.category, FutureWarning) and 'never synchronized' in str(w.message)
    ]
    assert len(unsynchronized) == int(warns)
