# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The rank-filtered log and the parameter-hash check use the groups their caller passes.

When the caller omits them, both warn and fall back to the same groups of the global parallel
state as before.
"""

import logging
import warnings
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core import utils as util
from megatron.core.process_groups_config import ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

VERSIONS = "deprecated since Megatron Core 0.21 and will be removed in 0.23"


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _fallback_warning(record, owner, argument):
    """Return the single fallback warning of ``owner`` and check its removal schedule."""
    matches = [
        w
        for w in record
        if issubclass(w.category, ProcessGroupFallbackWarning)
        and f"{owner} was called without `{argument}`" in str(w.message)
    ]
    assert len(matches) == 1
    assert VERSIONS in str(matches[0].message)
    # The warning points at the code that omitted the groups.
    assert matches[0].filename == __file__
    return matches[0]


def _enabled_logger(enabled=True):
    logger = mock.Mock(spec=logging.Logger)
    logger.isEnabledFor.return_value = enabled
    return logger


@pytest.mark.skipif(Utils.world_size < 4 or Utils.world_size % 4, reason="needs a multiple of 4")
class TestLogOnEachPipelineStageGroups:
    def setup_method(self, method):
        # Every DP x CP group has two ranks, so a wrong group selects a different emitter.
        Utils.initialize_model_parallel(tensor_model_parallel_size=2, context_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_groups_select_the_emitter_without_global_reads(self):
        tp_group = new_group_with_same_ranks(parallel_state.get_tensor_model_parallel_group())
        dp_cp_group = new_group_with_same_ranks(
            parallel_state.get_data_parallel_group(with_context_parallel=True)
        )
        logger = _enabled_logger()

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            util.log_on_each_pipeline_stage(
                logger, logging.INFO, "message", tp_group=tp_group, dp_cp_group=dp_cp_group
            )

        assert logger.log.called == (tp_group.rank() == 0 and dp_cp_group.rank() == 0)
        torch.distributed.destroy_process_group(tp_group)
        torch.distributed.destroy_process_group(dp_cp_group)

    def test_omitted_groups_warn_and_select_the_global_emitter(self):
        logger = _enabled_logger()

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            util.log_on_each_pipeline_stage(logger, logging.INFO, "message")

        _fallback_warning(record, "log_on_each_pipeline_stage", "tp_group/dp_cp_group")
        expected = (
            parallel_state.get_tensor_model_parallel_rank() == 0
            and parallel_state.get_data_parallel_rank(with_context_parallel=True) == 0
        )
        assert logger.log.called == expected

    def test_disabled_level_reads_no_group(self):
        logger = _enabled_logger(enabled=False)

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            util.log_on_each_pipeline_stage(logger, logging.DEBUG, "message")

        logger.log.assert_not_called()


def _dense_and_expert_model():
    """A chunk with one dense and one expert (``allreduce=False``) parameter.

    Rank 1 holds a different dense parameter, so its hash mismatches across any data-parallel
    group that contains it.
    """
    model = torch.nn.ModuleDict(
        {
            "dense": torch.nn.Linear(8, 8, bias=False, device="cuda"),
            "expert": torch.nn.Linear(8, 8, bias=False, device="cuda"),
        }
    )
    model.dense.weight.data.fill_(2.0 if torch.distributed.get_rank() == 1 else 1.0)
    model.expert.weight.data.fill_(3.0)
    model.expert.weight.allreduce = False
    return model


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks")
class TestCheckParamHashesFallback:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("cross_check", [False, True])
    def test_given_groups_are_used_without_global_reads(self, cross_check):
        dp_group = new_group_with_same_ranks(
            parallel_state.get_data_parallel_group(with_gtp_remat=False)
        )
        expt_dp_group = new_group_with_same_ranks(
            parallel_state.get_expert_data_parallel_group(with_gtp_remat=False)
        )
        model = _dense_and_expert_model()

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            hashes_match = util.check_param_hashes_across_dp_replicas(
                [model], cross_check, dp_group=dp_group, expt_dp_group=expt_dp_group
            )

        expected = False if cross_check else torch.distributed.get_rank() != 1
        assert hashes_match == expected
        torch.distributed.destroy_process_group(dp_group)
        torch.distributed.destroy_process_group(expt_dp_group)

    @pytest.mark.parametrize("cross_check", [False, True])
    def test_omitted_groups_warn_and_use_the_global_replicate_groups(self, cross_check):
        model = _dense_and_expert_model()
        expected = util.check_param_hashes_across_dp_replicas(
            [model],
            cross_check,
            dp_group=parallel_state.get_data_parallel_group(with_gtp_remat=False),
            expt_dp_group=parallel_state.get_expert_data_parallel_group(with_gtp_remat=False),
        )

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            hashes_match = util.check_param_hashes_across_dp_replicas([model], cross_check)

        _fallback_warning(record, "check_param_hashes_across_dp_replicas", "dp_group/expt_dp_group")
        assert hashes_match == expected
        assert expected == (False if cross_check else torch.distributed.get_rank() != 1)
