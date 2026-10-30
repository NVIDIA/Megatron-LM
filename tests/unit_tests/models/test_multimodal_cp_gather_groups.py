# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The multimodal context-parallel gather resolves its group once, in the forward.

Its backward reduce-scatters over the group the forward gathered over, so a caller that omits
the group reads the global grid once, with a warning, and never in the backward.
"""

import warnings

import pytest
import torch

from megatron.core import parallel_state, process_groups_config
from megatron.core.models.multimodal.context_parallel import (
    GatherFromContextParallelRanks,
    gather_from_context_parallel_ranks,
)
from megatron.core.process_groups_config import ProcessGroupFallbackWarning
from tests.unit_tests.test_utilities import (
    Utils,
    forbid_global_process_groups,
    new_group_with_same_ranks,
)

OWNER_MESSAGE = "GatherFromContextParallelRanks was called without `cp_group`"


@pytest.fixture(autouse=True)
def fresh_warning_registry(monkeypatch):
    """Each test observes the first fallback warning of every owner."""
    monkeypatch.setattr(process_groups_config, "_warned_global_process_group_fallbacks", set())


def _local_tensor(cp_rank):
    # [batch, seq, hidden]; each CP rank holds a different slice of the sequence.
    return torch.full(
        (1, 4, 8), float(cp_rank + 1), dtype=torch.float32, device="cuda", requires_grad=True
    )


def _gather_and_backward(local_t, cp_group=None, *, forbid_backward_reads=False):
    gathered = gather_from_context_parallel_ranks(local_t, 0, cp_group=cp_group)
    weights = torch.arange(gathered.numel(), device="cuda", dtype=torch.float32).view_as(gathered)
    if forbid_backward_reads:
        with forbid_global_process_groups():
            (gathered * weights).sum().backward()
    else:
        (gathered * weights).sum().backward()
    return gathered.detach(), local_t.grad


@pytest.mark.skipif(Utils.world_size < 2 or Utils.world_size % 2, reason="needs an even world")
class TestGatherFromContextParallelRanksGroups:
    def setup_method(self, method):
        Utils.initialize_model_parallel(context_parallel_size=2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_given_group_is_used_in_forward_and_backward_without_global_reads(self):
        cp_rank = parallel_state.get_context_parallel_rank()
        cp_group = new_group_with_same_ranks(parallel_state.get_context_parallel_group())
        expected = _gather_and_backward(
            _local_tensor(cp_rank), parallel_state.get_context_parallel_group()
        )

        with forbid_global_process_groups(), warnings.catch_warnings():
            warnings.simplefilter("error", ProcessGroupFallbackWarning)
            gathered, grad = _gather_and_backward(_local_tensor(cp_rank), cp_group)

        torch.testing.assert_close(gathered, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(grad, expected[1], rtol=0, atol=0)
        torch.distributed.destroy_process_group(cp_group)

    def test_omitted_group_warns_once_and_backward_reads_no_global_group(self):
        cp_rank = parallel_state.get_context_parallel_rank()
        expected = _gather_and_backward(
            _local_tensor(cp_rank), parallel_state.get_context_parallel_group()
        )

        with pytest.warns(ProcessGroupFallbackWarning) as record:
            gathered, grad = _gather_and_backward(
                _local_tensor(cp_rank), forbid_backward_reads=True
            )

        assert len([w for w in record if OWNER_MESSAGE in str(w.message)]) == 1
        torch.testing.assert_close(gathered, expected[0], rtol=0, atol=0)
        torch.testing.assert_close(grad, expected[1], rtol=0, atol=0)


@pytest.mark.skipif(Utils.world_size < 2, reason="needs at least 2 ranks")
def test_omitted_group_without_a_global_grid_raises():
    Utils.initialize_distributed()
    if parallel_state.get_context_parallel_group(check_initialized=False) is not None:
        pytest.skip("an earlier test left the global grid initialized")
    local_t = _local_tensor(0)

    with pytest.warns(ProcessGroupFallbackWarning), pytest.raises(ValueError, match="cp_group"):
        GatherFromContextParallelRanks.apply(local_t, None)
