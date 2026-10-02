# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Replay seeded CUDA draws under changing batch layouts."""

import pytest
import torch

from tests.unit_tests.inference.test_request_seed_sampling import (
    test_seeded_sequence_survives_batch_changes as _check_seeded_sequence,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_request_seed_sampling_cuda_replay():
    _check_seeded_sequence("cuda")
