# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared TP clipping regression for local and Transformer Engine attention."""

import pytest
import torch

from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.optimizer import FP32Optimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.test_utilities import Utils


def assert_softmax_offset_clipping_replays(attention_cls):
    """Check the analytical TP norm and clipped gradients, then replay byte-for-byte."""
    if Utils.world_size < 2 or Utils.world_size % 2:
        pytest.skip("requires an even world size of at least two")
    Utils.initialize_model_parallel(2, 1)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"])
        config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=2,
            kv_channels=64,
            tensor_model_parallel_size=2,
            sequence_parallel=True,
            softmax_type="learnable",
            perform_initialization=False,
            params_dtype=torch.float32,
            deterministic_mode=True,
        )
        attention = attention_cls(config, 1, AttnMaskType.causal, "self", pg_collection=groups)
        param = attention.softmax_offset
        # Distinct head-shard gradients expose dropping TP rank 1 from the norm.
        gradient = 3.0 if groups.tp.rank() == 0 else 4.0
        optimizer = FP32Optimizer(
            torch.optim.SGD([param], lr=0.1), OptimizerConfig(), init_state_fn=None
        )
        optimizer.tp_group = groups.tp
        optimizer.grad_stats_parallel_group = groups.tp

        def clip(grad):
            param.grad = grad
            norm = optimizer.clip_grad_norm(1.0)
            assert float(norm) == pytest.approx(5.0), f"Expected TP norm 5, got {float(norm)}"
            torch.testing.assert_close(param.grad, torch.full_like(param, gradient / 5.0))
            return torch.as_tensor(norm, dtype=torch.float64, device=param.device), param.grad

        assert_replays_bit_exact(
            clip,
            (torch.full_like(param, gradient),),
            backward=False,
            what=f"{attention_cls.__name__} softmax_offset norm and clipping",
        )
    finally:
        Utils.destroy_model_parallel()
