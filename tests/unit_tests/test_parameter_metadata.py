# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import torch

from megatron.core.parameter_metadata import copy_parameter_metadata


def test_copy_parameter_metadata_copies_public_attributes_only():
    source = torch.nn.Parameter(torch.ones(1))
    destination = torch.nn.Parameter(torch.zeros(1))
    process_group = object()

    source.tensor_model_parallel = True
    source.group = process_group
    source.future_planner_metadata = "preserved without a name allowlist"
    source._quantizer = "tensor-subclass implementation detail"

    copy_parameter_metadata(destination, source)

    assert destination.tensor_model_parallel is True
    assert destination.group is process_group
    assert destination.future_planner_metadata == "preserved without a name allowlist"
    assert not hasattr(destination, "_quantizer")
