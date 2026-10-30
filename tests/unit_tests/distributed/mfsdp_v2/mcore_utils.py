# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared cleanup helpers for MFSDP v2 MCore tests."""

import torch
from torch.distributed.distributed_c10d import _world

from tests.unit_tests.test_utilities import Utils


def destroy_model_parallel():
    """Utils.destroy_model_parallel, plus the groups it leaves behind.

    It clears Megatron's references but frees only a few of the groups; c10d holds its own
    reference to the rest, so their NCCL communicators -- and, with NVLS enabled, their
    multicast reservations -- would outlive the test. See #6897.
    """
    Utils.destroy_model_parallel()
    for group in list(_world.pg_map):
        if group is not torch.distributed.group.WORLD:
            torch.distributed.destroy_process_group(group)
