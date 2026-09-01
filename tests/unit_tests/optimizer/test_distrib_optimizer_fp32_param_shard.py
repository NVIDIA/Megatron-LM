# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""REGRESSION: `--optimizer-cpu-offload` must work when the model has FP32 parameters.

``DistributedOptimizer._build_model_and_main_param_groups`` slices a shard view out of every
model parameter. The float16 branch does so via ``model_param.detach().view(-1)[...]``, but the
FP32 branch used a plain ``model_param.view(-1)[...]``, which is a non-leaf tensor because the
model parameter requires grad. ``HybridDeviceOptimizer`` passes those shards straight to
``torch.optim.Optimizer``, which rejects non-leaf tensors with
"can't optimize a non-leaf Tensor".

Any FP32 model parameter takes that branch: a fully FP32 model, or a parameter kept in FP32
inside an otherwise BF16 model (``mark_keep_in_fp32``).
"""

import torch
import torch.nn as nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


class _Fp32Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.proj(x)


def _build_model():
    config = TransformerConfig(
        num_layers=1, hidden_size=8, num_attention_heads=1, bf16=False, params_dtype=torch.float32
    )
    ddp_config = DistributedDataParallelConfig(
        use_distributed_optimizer=True, overlap_grad_reduce=False
    )
    return [DistributedDataParallel(config, ddp_config, _Fp32Net().cuda())]


class TestOptimizerCpuOffloadWithFp32Params:
    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_optimizer_step_with_fp32_params(self):
        """Building and stepping the offloaded optimizer must work on an FP32 model."""
        model = _build_model()
        assert all(p.dtype == torch.float32 for p in model[0].parameters())

        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=1e-4,
                params_dtype=torch.float32,
                use_distributed_optimizer=True,
                optimizer_cpu_offload=True,
                optimizer_offload_fraction=1.0,
            ),
            model,
        )

        before = [p.detach().clone() for p in model[0].parameters()]
        model[0].zero_grad_buffer()
        model[0](torch.randn(4, 8, device="cuda")).sum().backward()
        model[0].finish_grad_sync()
        optimizer.step()
        assert any(
            not torch.equal(b, p) for b, p in zip(before, model[0].parameters())
        ), "optimizer.step() did not update any parameter"
