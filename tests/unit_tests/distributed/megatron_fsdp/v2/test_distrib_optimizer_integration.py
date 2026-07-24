# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer


def _optimizer_shell(*, use_megatron_fsdp_v2: bool):
    optimizer = object.__new__(DistributedOptimizer)
    optimizer.ddp_config = SimpleNamespace(
        use_megatron_fsdp=True, use_megatron_fsdp_v2=use_megatron_fsdp_v2
    )
    return optimizer


def test_v2_state_dict_delegates_to_inner_optimizer():
    """V2 exposes the complete inner optimizer state to its DCP adapter."""
    optimizer = _optimizer_shell(use_megatron_fsdp_v2=True)
    expected = {"state": {0: {"exp_avg": torch.ones(1)}}, "param_groups": []}
    optimizer.optimizer = Mock()
    optimizer.optimizer.state_dict.return_value = expected

    assert DistributedOptimizer.state_dict(optimizer) is expected
    optimizer.optimizer.state_dict.assert_called_once_with()


@pytest.mark.parametrize("use_megatron_fsdp_v2", [False, True])
def test_v2_defers_expert_name_remapping_to_checkpoint_adapter(use_megatron_fsdp_v2):
    """Only v1 remaps expert names inside DistributedOptimizer."""
    optimizer = _optimizer_shell(use_megatron_fsdp_v2=use_megatron_fsdp_v2)
    param = torch.nn.Parameter(torch.ones(1))
    live_name = "decoder.layers.0.mlp.experts.local_experts.0.weight"
    remapped_name = "decoder.layers.0.mlp.experts.local_experts.4.weight"
    optimizer.model_chunks = [
        SimpleNamespace(
            named_parameters=lambda: [(live_name, param)], config=SimpleNamespace(num_moe_experts=8)
        )
    ]

    with patch(
        "megatron.core.optimizer.distrib_optimizer.handle_experts_in_state_dict",
        return_value={remapped_name: param},
    ) as remap_experts:
        actual_name = DistributedOptimizer._param_name(optimizer, param)

    if use_megatron_fsdp_v2:
        assert actual_name == live_name
        remap_experts.assert_not_called()
    else:
        assert actual_name == remapped_name
        remap_experts.assert_called_once()
