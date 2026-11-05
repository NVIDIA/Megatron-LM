# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc

import pytest
import torch

from tests.unit_tests.test_utilities import Utils


@pytest.fixture(scope="function", autouse=True)
def collect_cuda_graphs_before_group_teardown():
    """Free CUDA graphs held only by reference cycles before retired groups are destroyed.

    NCCL will not destroy a communicator while a CUDA graph that captured its
    collectives is still alive: ncclCommDestroy waits for every such graph to be
    freed. RL tests such as test_get_logprobs_cuda_graphs leave their graphs
    reachable only through reference cycles, so the suite-wide
    destroy_retired_process_groups fixture would hang without a collection first.
    This fixture's teardown runs before that one's. It is scoped to this bucket so
    other buckets keep their existing object lifetimes.
    """
    yield
    if Utils.retired_process_groups and torch.distributed.is_initialized():
        gc.collect()
