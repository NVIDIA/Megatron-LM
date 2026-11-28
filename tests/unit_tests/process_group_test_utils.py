# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Opt-in process-group cleanup for distributed tests."""

import pytest
import torch

from tests.unit_tests.test_utilities import Utils


@pytest.fixture(autouse=True)
def cleanup_process_groups():
    """Import this fixture to release each test's groups while preserving the session."""
    existing_groups = set(torch.distributed.distributed_c10d._world.pg_map)
    try:
        yield
    finally:
        try:
            Utils.destroy_model_parallel()
        finally:
            # MPU clears its references, but c10d still owns these groups, including any
            # replaced during runtime CP setup. Keep WORLD and groups owned by earlier tests.
            for group in reversed(list(torch.distributed.distributed_c10d._world.pg_map)):
                if group is not torch.distributed.group.WORLD and group not in existing_groups:
                    torch.distributed.destroy_process_group(group)
