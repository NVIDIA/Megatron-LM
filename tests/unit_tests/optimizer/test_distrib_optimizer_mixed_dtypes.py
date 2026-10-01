# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Parameter/state association for distributed optimizer groups with mixed dtypes."""

import io
from copy import deepcopy

import pytest
import torch

from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("precision_aware", [False, True])
@pytest.mark.parametrize(
    "dtypes",
    [
        pytest.param((torch.bfloat16, torch.float32), id="bf16-first"),
        pytest.param((torch.float32, torch.bfloat16), id="fp32-first"),
        pytest.param((torch.bfloat16, torch.bfloat16), id="bf16-only"),
        pytest.param((torch.float32, torch.float32), id="fp32-only"),
    ],
)
def test_parameter_state_round_trip(dtypes, precision_aware):
    """Checkpoint state must stay with its model parameter after shard reordering."""
    Utils.initialize_model_parallel()
    try:
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        model = torch.nn.Module()
        model.weights = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.full((256, 128 * (index + 1)), index + 1.0, dtype=dtype, device="cuda")
                )
                for index, dtype in enumerate((*dtypes, dtypes[0]))
            ]
        )
        # The vector uses a separate weight-decay group, exercising group indices too.
        model.biases = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.full((1024,), 4.0, device="cuda"))]
        )
        ddp = DistributedDataParallel(
            TransformerConfig(num_layers=1, hidden_size=128, num_attention_heads=1, bf16=True),
            DistributedDataParallelConfig(use_distributed_optimizer=True),
            model,
            pg_collection=pg_collection,
        )
        optimizer = get_megatron_optimizer(
            OptimizerConfig(
                optimizer="adam",
                lr=0.01,
                weight_decay=0.0,
                bf16=True,
                use_distributed_optimizer=True,
                use_precision_aware_optimizer=precision_aware,
                store_param_remainders=False,
                clip_grad=0.0,
            ),
            [ddp],
            use_gloo_process_groups=True,
        )
        if hasattr(optimizer, "chained_optimizers"):
            assert len(optimizer.chained_optimizers) == 1
            optimizer = optimizer.chained_optimizers[0]
        assert isinstance(optimizer, DistributedOptimizer)
        # Initialize Adam state through a real step without changing parameter values.
        for param in model.parameters():
            param.main_grad.zero_()
        assert optimizer.step()[0]

        for param in optimizer.model_param_gbuf_map:
            param_range = optimizer._get_model_param_range_map(param)["param"]
            state = optimizer._get_main_param_and_optimizer_states(param)
            expected = param.detach().view(-1)[param_range.start : param_range.end].float()
            torch.testing.assert_close(state["param"], expected, rtol=0, atol=0)

        def step():
            for index, param in enumerate(model.parameters()):
                param.main_grad.fill_(index + 1.0)
            assert optimizer.step()[0]

        def snapshot():
            return {
                param: {
                    key: value.detach().clone()
                    for key, value in optimizer._get_main_param_and_optimizer_states(param).items()
                }
                for param in optimizer.model_param_gbuf_map
            }

        step()
        first_step = snapshot()
        metadata = deepcopy(optimizer.state_dict())
        checkpoint = optimizer.get_parameter_state_dp_zero()
        if checkpoint is not None:
            buffer = io.BytesIO()
            torch.save(checkpoint, buffer)
            buffer.seek(0)
            checkpoint = torch.load(buffer, weights_only=True)
        step()
        second_step = snapshot()

        optimizer.load_state_dict(metadata)
        optimizer.load_parameter_state_from_dp_zero(checkpoint)
        for param, state in snapshot().items():
            for key, value in state.items():
                torch.testing.assert_close(value, first_step[param][key], rtol=0, atol=0)
        step()
        for param, state in snapshot().items():
            for key, value in state.items():
                torch.testing.assert_close(value, second_step[param][key], rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
