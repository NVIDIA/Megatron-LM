# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Stateful graph parity with real fused TE wgrad, MCore DDP and sharded Adam."""

import gc

import pytest
import torch
from torch import nn

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.state_boundary import TensorField
from megatron.core.transformer.stateful_module import StatefulGraphs, StatefulModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.usefixtures("te_rng_tracker")
@pytest.mark.parametrize("backend", ["torch", "transformer_engine"])
def test_fused_te_graph_preserves_ddp_gradients_and_distributed_optimizer(backend):
    te = pytest.importorskip("transformer_engine.pytorch")
    if Utils.world_size < 2:
        pytest.skip("Requires at least two data-parallel ranks")
    Utils.initialize_model_parallel()
    groups = ProcessGroupCollection.use_mpu_process_groups()
    graphs = None
    try:
        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=1,
            gradient_accumulation_fusion=True,
            params_dtype=torch.float32,
        )
        ddp_config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            overlap_grad_reduce=True,
            overlap_param_gather=False,
            grad_reduce_in_fp32=True,
        )

        class Region(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = te.LayerNormLinear(
                    16,
                    16,
                    bias=False,
                    params_dtype=torch.float32,
                    device="cuda",
                    return_layernorm_output=True,
                    normalization="RMSNorm",
                    fuse_wgrad_accumulation=True,
                )

            def forward(self, hidden, state):
                output, normalized = self.projection(hidden)
                return output, {"memory": normalized}

        fields = (TensorField("memory", (4, 2, 16), torch.float32, "contiguous", True),)
        models = [
            DistributedDataParallel(
                config,
                ddp_config,
                StatefulModule(Region(), output_fields=fields),
                pg_collection=groups,
            )
            for _ in range(2)
        ]
        models[1].load_state_dict(models[0].state_dict())
        optimizers = [
            get_megatron_optimizer(
                OptimizerConfig(
                    lr=0.001, use_distributed_optimizer=True, weight_decay=0.0, clip_grad=0.0
                ),
                [model],
                pg_collection=groups,
                use_gloo_process_groups=False,
            )
            for model in models
        ]
        sample = torch.ones(4, 2, 16, device="cuda", requires_grad=True)
        for p in models[0].parameters():
            p.main_grad.fill_(3)
        before = [p.main_grad.clone() for p in models[0].parameters()]
        graphs = StatefulGraphs(models[0].module, sample, {}, slots=2, backend=backend)
        for p, saved in zip(models[0].parameters(), before):
            torch.testing.assert_close(p.main_grad, saved, rtol=0, atol=0)
        for _ in range(2):
            for model, optimizer in zip(models, optimizers):
                optimizer.zero_grad()
                model.zero_grad_buffer()
            for microbatch in range(2):
                source = (
                    torch.arange(128, device="cuda").float().reshape(4, 2, 16) / 128
                    + 1
                    + Utils.rank * 0.1
                    + microbatch
                )
                for i, model in enumerate(models):
                    value = source.detach().clone().requires_grad_()
                    if i == 0:
                        hidden, state = graphs.run(value, {}, slot=microbatch)
                    else:
                        hidden, state = model.module.run(value, {})
                    # DDP accumulates both microbatches, then launches one reduce-scatter.
                    if microbatch == 0:
                        with model.no_sync():
                            (hidden.square().mean() + state["memory"].square().mean()).backward()
                    else:
                        (hidden.square().mean() + state["memory"].square().mean()).backward()
            for model in models:
                model.finish_grad_sync()
            for actual, expected in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(
                    actual.main_grad, expected.main_grad, rtol=1e-5, atol=1e-6
                )
            for model, optimizer in zip(models, optimizers):
                assert optimizer.step()[0]
                model.start_param_sync(force_sync=True)
            for actual, expected in zip(models[0].parameters(), models[1].parameters()):
                torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        del hidden, state
    finally:
        if graphs is not None:
            graphs.abort()
            graphs.close()
        gc.collect()
        Utils.destroy_model_parallel()
