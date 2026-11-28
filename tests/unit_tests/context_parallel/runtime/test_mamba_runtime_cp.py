# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Forward-local Mamba context-parallel helper geometry."""

import pytest
import torch

from megatron.core.ssm.mamba_context_parallel import MambaContextParallel

pytestmark = [pytest.mark.launch_on_gb200, pytest.mark.internal]


def test_runtime_cp_views_do_not_mutate_base_helper():
    """Sequential CP1/2/4 geometry resolution remains forward-local."""

    class FakeGroup:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

        def rank(self):
            return 0

    groups = {size: FakeGroup(size) for size in (1, 2, 4)}
    helper = MambaContextParallel(
        cp_group=groups[1],
        d_inner_local_tp=64,
        nheads_local_tp=4,
        ngroups_local_tp=4,
        d_state=8,
        conv1d_weight_cp1=torch.randn(128, 1, 4),
        conv1d_bias_cp1=torch.randn(128),
        conv1d_padding=3,
        dt_bias_cp1=torch.randn(4),
        A_log_cp1=torch.randn(4),
        D_cp1=torch.randn(4),
        D_has_hdim=False,
    )

    for cp_size in (1, 2, 4, 2, 1):
        runtime_helper = helper.for_context_parallel_group(groups[cp_size])
        assert runtime_helper.cp_group is groups[cp_size]
        assert runtime_helper.cp_size == cp_size
        assert runtime_helper.nheads_local_tpcp == 4 // cp_size
        assert helper.cp_group is groups[1]
        assert helper.cp_size == 1
