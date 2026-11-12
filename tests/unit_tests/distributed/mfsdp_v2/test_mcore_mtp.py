# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-MTP execution-order coverage for the MFSDP v2 MCore adapter."""

from contextlib import nullcontext
from copy import deepcopy

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_adapter_v2 import FullyShardedDataParallelV2
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental import (
    FsdpContext,
    PrefetchOrder,
    RuntimePolicy,
    fully_shard_context,
    fully_shard_optimizer,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig


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


def _config(repeated_mtp=True) -> TransformerConfig:
    return TransformerConfig(
        num_layers=2,
        hidden_size=16,
        num_attention_heads=4,
        mtp_num_layers=2 if repeated_mtp else None,
        mtp_use_repeated_layer=repeated_mtp,
    )


def _ddp_config(prefetch_order="recorded") -> DistributedDataParallelConfig:
    return DistributedDataParallelConfig(
        use_megatron_fsdp=True,
        megatron_fsdp_version=2,
        fsdp_prefetch_order=prefetch_order,
        use_distributed_optimizer=False,
        data_parallel_sharding_strategy="optim_grads_params",
    )


@pytest.mark.parametrize(
    "ambient_context,repeated_mtp", [(False, True), (True, True), (False, False)]
)
def test_shared_mtp_matches_unsharded_training(distributed_setup, ambient_context, repeated_mtp):
    """Explicit recording supports reused modules independently of MTP settings."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    groups = ProcessGroupCollection(dp_cp=mesh.get_group())
    torch.manual_seed(1234)
    baseline = _SharedPredictor().to(device)
    module = deepcopy(baseline)
    torch.manual_seed(5678 + distributed_setup.rank)
    value = torch.randn(4, 16, device=device)
    construction = (
        fully_shard_context(
            FsdpContext(
                device=device,
                prefetch_order=PrefetchOrder.RECORDED,
                runtime_policy=RuntimePolicy(caller_managed_grad_sync=True),
            )
        )
        if ambient_context
        else nullcontext()
    )
    with construction:
        model = FullyShardedDataParallelV2(
            config=_config(repeated_mtp),
            ddp_config=_ddp_config(),
            module=module,
            fsdp_unit_modules=[nn.Linear],
            device=device,
            pg_collection=groups,
        )
    reference_optimizer = torch.optim.SGD(baseline.parameters(), lr=0.05)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    fully_shard_optimizer(optimizer)

    for _ in range(3):
        reference_optimizer.zero_grad()
        optimizer.zero_grad()
        expected = baseline(value)
        expected.backward()
        for parameter in baseline.parameters():
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.AVG, group=groups.dp_cp)
        actual = model(value)
        actual.backward()
        model.finish_grad_sync()
        torch.testing.assert_close(actual, expected)
        for parameter, reference in zip(model.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(parameter.grad.full_tensor(), reference.grad)
        reference_optimizer.step()
        optimizer.step()
        for parameter, reference in zip(model.parameters(), baseline.parameters(), strict=True):
            torch.testing.assert_close(parameter.full_tensor(), reference)
