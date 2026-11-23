# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.clip_grads import count_zeros_fp32, count_zeros_in_tensors
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _MixedDtypeNet(torch.nn.Module):
    """Linear layers; each casts its input to its own dtype."""

    def __init__(self, fp32_param: bool):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(64, 96, bias=True, dtype=torch.bfloat16, device="cuda"),
                torch.nn.Linear(96, 48, bias=False, dtype=torch.bfloat16, device="cuda"),
                torch.nn.Linear(48, 32, bias=True, dtype=torch.bfloat16, device="cuda"),
            ]
        )
        if fp32_param:
            self.layers.append(torch.nn.Linear(32, 16, dtype=torch.float32, device="cuda"))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x.to(layer.weight.dtype))
        return x


def _model_and_optimizer(fp32_param: bool, bucket_size: int, **optimizer_kwargs):
    model = _MixedDtypeNet(fp32_param)
    model = DistributedDataParallel(
        TransformerConfig(num_attention_heads=1, num_layers=1),
        DistributedDataParallelConfig(use_distributed_optimizer=True, bucket_size=bucket_size),
        model,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=0.01,
            bf16=True,
            use_distributed_optimizer=True,
            **{"log_num_zeros_in_grad": True, **optimizer_kwargs},
        ),
        [model],
    )
    return model, optimizer


def _reference_count(optimizer):
    params = optimizer.get_parameters()
    return count_zeros_fp32(
        params, grad_stats_parallel_group=optimizer.get_grad_stats_parallel_group()
    )


class TestNumZerosInGrads:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("fp32_param", [False, True])
    @pytest.mark.parametrize("bucket_size", [None, 2000])
    def test_grad_buffer_count_matches_per_param_count(self, fp32_param, bucket_size):
        torch.manual_seed(0)
        model, optimizer = _model_and_optimizer(fp32_param, bucket_size)
        x = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
        # Zero input features leave zero columns in the first weight's grad.
        x[:, ::3] = 0
        out = model(x)
        out.float().sum().backward()
        model.finish_grad_sync()
        optimizer.prepare_grads()

        expected = _reference_count(optimizer)
        assert expected > 0
        assert optimizer.count_zeros() == expected
        # Tiny chunks exercise region splitting; the count must not change.
        dist_opt = getattr(optimizer, "chained_optimizers", [optimizer])[0]
        num_counted, _, tensors = dist_opt.num_zeros_count_inputs()
        local = count_zeros_in_tensors(num_counted, tensors, max_chunk_numel=7)
        torch.distributed.all_reduce(local, group=optimizer.get_grad_stats_parallel_group())
        assert local.item() == expected
