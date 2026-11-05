# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import clear_nvte_env_vars
from tools.checkpoint.utils import initialize_checkpoint_converter_fake_process_groups

_FAKE_GROUP_GLOBALS = (
    '_TENSOR_MODEL_PARALLEL_GROUP',
    '_CONTEXT_PARALLEL_GROUP',
    '_PIPELINE_MODEL_PARALLEL_GROUP',
    '_EXPERT_MODEL_PARALLEL_GROUP',
    '_DATA_PARALLEL_GROUP',
    '_DATA_PARALLEL_GROUP_WITH_CP',
    '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP',
    '_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT',
    '_INTRA_PARTIAL_DATA_PARALLEL_GROUP_WITH_CP_WITH_GTP_REMAT',
    '_EXPERT_DATA_PARALLEL_GROUP',
    '_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_INTRA_PARTIAL_EXPERT_DATA_PARALLEL_GROUP_WITH_GTP_REMAT',
    '_EXPERT_TENSOR_PARALLEL_GROUP',
    '_EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP',
    '_EXPERT_TENSOR_MODEL_PIPELINE_PARALLEL_GROUP',
    '_MODEL_PARALLEL_GROUP',
)


@pytest.fixture
def fake_groups():
    """Install the converter's fake groups for TP=4, PP=2 and restore the globals afterwards."""
    original_groups = {name: getattr(parallel_state, name) for name in _FAKE_GROUP_GLOBALS}
    try:
        initialize_checkpoint_converter_fake_process_groups(
            parallel_state, tensor_parallel_size=4, pipeline_parallel_size=2, expert_parallel_size=1
        )
        yield
    finally:
        for name, group in original_groups.items():
            setattr(parallel_state, name, group)


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
    """The saver starts a one-process torch.distributed world and builds its models without a
    collection, so LanguageModule validates the collection built from the fake groups."""
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
