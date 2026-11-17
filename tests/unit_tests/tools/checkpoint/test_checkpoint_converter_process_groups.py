# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import warnings

import pytest
import torch

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


@pytest.mark.parametrize('explicit_collection', [False, True])
def test_fake_groups_satisfy_language_module(mocker, fake_groups, explicit_collection):
    """The saver starts a one-process torch.distributed world. A collection resolved from the
    fake groups passes LanguageModule's checks when passed explicitly, and the deprecated
    omitted-collection path resolves the same groups."""
    clear_nvte_env_vars()
    mocker.patch.object(process_groups_config, '_warned_global_process_group_fallbacks', set())
    mocker.patch('torch.distributed.is_initialized', return_value=True)
    config = TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)

    if explicit_collection:
        model = LanguageModule(
            config, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
        )
    else:
        with pytest.warns(FutureWarning, match='LanguageModule was called without `pg_collection`'):
            model = LanguageModule(config)

    assert model.tp_group.size() == 4
    assert model.cp_group.size() == 1
    assert model.pp_group.size() == 2


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_fake_pipeline_stages_do_not_warn_about_unsynchronized_embeddings(
    mocker, fake_groups, pp_rank
):
    """The single-process saver fills both tied copies without embedding collectives."""
    clear_nvte_env_vars()
    mocker.patch("torch.distributed.is_initialized", return_value=True)
    mocker.patch("torch.distributed.get_world_size", return_value=1)
    mocker.patch.object(parallel_state, "_EMBEDDING_GROUP", None)
    groups = ProcessGroupCollection.use_mpu_process_groups()
    groups.pp.set_rank(pp_rank)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        pipeline_model_parallel_size=2,
        pipeline_dtype=torch.float32,
    )
    model = LanguageModule(config, pg_collection=groups)
    model.pre_process = pp_rank == 0
    model.post_process = pp_rank == 1
    model.share_embeddings_and_output_weights = True
    model.mtp_process = False
    mocker.patch.object(
        model, "shared_embedding_or_output_weight", return_value=torch.nn.Parameter(torch.ones(2))
    )
    all_reduce = mocker.patch("torch.distributed.all_reduce")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.setup_embeddings_and_output_layer()

    assert not [
        warning
        for warning in caught
        if issubclass(warning.category, FutureWarning)
        and "never synchronized" in str(warning.message)
    ]
    all_reduce.assert_not_called()
