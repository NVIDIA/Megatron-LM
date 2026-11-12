# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-MTP execution-order coverage for the MFSDP v2 MCore adapter."""

import torch
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter_v2 import FullyShardedDataParallelV2
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.distributed.mfsdp_v2.mcore_training_utils import (
    run_forward_backward_on_microbatches,
)


class _SharedPredictor(nn.Module):
    """Reuse a sharded predictor at two successive prediction depths."""

    def __init__(self) -> None:
        super().__init__()
        self.predictor = nn.Linear(16, 16, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return a loss through both uses of the same predictor."""
        for _ in range(2):
            value = self.predictor(value).relu()
        return value.square().mean()


def _config() -> TransformerConfig:
    return TransformerConfig(num_layers=2, hidden_size=16, num_attention_heads=4)


def _ddp_config(prefetch_order="recorded") -> DistributedDataParallelConfig:
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        fsdp_prefetch_order=prefetch_order,
        use_distributed_optimizer=False,
        data_parallel_sharding_strategy="optim_grads_params",
    )


def test_shared_mtp_matches_unsharded_training(distributed_setup):
    """Explicit recording supports reused modules independently of MTP settings."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    groups = ProcessGroupCollection(dp_cp=mesh.get_group())
    torch.manual_seed(1234)
    module = _SharedPredictor().to(device)
    torch.manual_seed(5678 + distributed_setup.rank)
    value = torch.randn(4, 16, device=device)
    expected = run_forward_backward_on_microbatches(
        module, lambda: module(value), num_microbatches=3
    )
    module.zero_grad(set_to_none=True)
    model = FullyShardedDataParallelV2(
        config=_config(),
        ddp_config=_ddp_config(),
        module=module,
        fsdp_unit_modules=[nn.Linear],
        device=device,
        pg_collection=groups,
    )
    actual = run_forward_backward_on_microbatches(model, lambda: model(value), num_microbatches=3)
    torch.testing.assert_close(actual, expected)
