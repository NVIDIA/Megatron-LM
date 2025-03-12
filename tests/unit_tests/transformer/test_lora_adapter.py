# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

from functools import partial
from typing import Any, Callable, Generator

import pytest
import torch
from torch.optim import Adam

from megatron.core.extensions.transformer_engine import (
    TEColumnParallelLinear,
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.mappings import _gather_along_first_dim
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.lora_adapter import LoraAdapter
from megatron.core.transformer.transformer_config import TransformerConfig

from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize('pipeline_model_parallel_size', [1, 2])
@pytest.mark.parametrize(
    "expert_tensor_parallel_size, tensor_model_parallel_size",
    [
        (1, 1),
        (1, 2), 
        (2, 1),
        # EP2+TP2 requires sequence parallelism. LoraAdapter doesn't support sequence parallelism fully, thus the next line is commented out.
        # (2, 2),
    ]
)
@pytest.mark.parametrize(
    "base_layer, is_expert",
    [
        (partial(ColumnParallelLinear), False),
        (partial(ColumnParallelLinear), True),
        (partial(TEColumnParallelLinear, gather_output=False), False),
        (partial(TEColumnParallelLinear, gather_output=False), True),
        (partial(TELayerNormColumnParallelLinear, gather_output=False), False),
        # TELayerNormColumnParallelLinear layer is not used in MoE and rises 'Transformer Engine linear layers do not yet support MoE'
        # in TELayerNormColumnParallelLinear.__init__(), thus the next line is commented out.
        # (partial(TELayerNormColumnParallelLinear, gather_output=False), True),
        (partial(RowParallelLinear, input_is_parallel=True), False),
        (partial(RowParallelLinear, input_is_parallel=True), True),
        (partial(TERowParallelLinear, input_is_parallel=True), False),
        (partial(TERowParallelLinear, input_is_parallel=True), True),
    ]
)
class TestLoraAdapterWithLoraLayers:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self, expert_tensor_parallel_size: int, pipeline_model_parallel_size: int, tensor_model_parallel_size: int, base_layer: Callable, is_expert: bool) -> Generator[Any, Any, Any]:
        Utils.initialize_model_parallel(
            expert_tensor_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)
        self.is_expert = is_expert
        self.input_size = 4
        self.output_size = 8
        self.rank = 2
        self.alpha = 32
        self.config = TransformerConfig(
            # mandatory params
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=expert_tensor_parallel_size if expert_tensor_parallel_size > 1 else None,
            pipeline_dtype=torch.float32,
            
            # parallelization params
            expert_model_parallel_size=expert_tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            tensor_model_parallel_size=tensor_model_parallel_size,
        )

        self.base_layer = base_layer(
            input_size=self.input_size,
            output_size=self.output_size,
            config=self.config,
            bias=False,
            skip_bias_add=True,
            init_method=torch.nn.init.zeros_,
            is_expert=is_expert,
        )
        self.lora_adapter = LoraAdapter(
            self.base_layer,
            config=self.config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self) -> None:
        INPUT_INDEX = 1
        OUTPUT_INDEX = 0
        assert self.base_layer.weight.shape[INPUT_INDEX] == self.lora_adapter.lora_a.weight.shape[INPUT_INDEX]
        assert self.base_layer.weight.shape[OUTPUT_INDEX] == self.lora_adapter.lora_b.weight.shape[OUTPUT_INDEX]

    def test_load_state_dict(self) -> None:
        model = torch.nn.Module()
        model.add_module("output_layer", self.lora_adapter)
        state_dict = {
            "output_layer.weight": torch.ones(self.base_layer.weight.shape),
            "output_layer._extra_state": None,
        }
        if type(self.base_layer) is TELayerNormColumnParallelLinear:
            norm_shape = self.base_layer.weight.shape[1]
            state_dict["output_layer.layer_norm_weight"] = torch.ones(norm_shape)
            state_dict["output_layer.layer_norm_bias"] = torch.ones(norm_shape)
        
        missing_keys, unexpected_keys = model.load_state_dict(state_dict)

        assert missing_keys == []
        assert unexpected_keys == []
        assert self.lora_adapter.base_layer.weight.all()
        assert torch.any(self.lora_adapter.lora_a.weight)
        assert self.lora_adapter.lora_b.weight.sum() == 0

    def test_forward(self) -> None:
        batch_size = 1
        sequence_length = self.base_layer.weight.shape[1]
        input_data = torch.rand(batch_size, sequence_length).cuda()

        non_parallel_non_zero_layer = "lora_a"
        parallel_zero_layer = "lora_b"
        if type(self.base_layer) in [RowParallelLinear, TERowParallelLinear]:
            non_parallel_non_zero_layer = "lora_b"
            parallel_zero_layer = "lora_a"

        original_weight = getattr(self.lora_adapter, non_parallel_non_zero_layer).weight.clone().detach()
        optimizer = Adam(self.lora_adapter.parameters())
        
        # To propagate gradients through the zero-initialized weights we need at least two iterations.
        # Let's do 10 to see errors accumulation.
        for _ in range(10):
            optimizer.zero_grad()
            output, _ = self.lora_adapter(input_data)
            (1 - output.mean()).backward()
            optimizer.step()

        assert self.base_layer.weight.sum() == 0, "Base layer frozen weights were updated"
        assert torch.all(getattr(self.lora_adapter, parallel_zero_layer).weight), f"LoRA zero layer ({parallel_zero_layer}) weights weren't updated"
        
        current_local_weight = getattr(self.lora_adapter, non_parallel_non_zero_layer).weight
        assert not torch.allclose(original_weight, current_local_weight), f"Local weight wasn't updated for {non_parallel_non_zero_layer}"
        
        full_weight = _gather_along_first_dim(current_local_weight)
        for idx, weight in enumerate(torch.split(full_weight, current_local_weight.shape[0])):
            assert torch.allclose(weight, current_local_weight), f"Weight on rank {idx} doesn't match for {non_parallel_non_zero_layer}"


class TestLoraAdapterWithUnknownBaseLayer:

    @pytest.fixture(scope='function', autouse=True)
    def setup_and_teardown(self):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)

        self.input_size = 64
        self.output_size = 64
        self.rank = 16
        self.alpha = 32
        self.transformer_config = TransformerConfig(
            num_layers=1,
            add_bias_linear=False,
            hidden_size=12,
            num_attention_heads=4,
        )
        self.base_layer = torch.nn.Linear(self.input_size, self.output_size, bias=False)
        self.lora_adapter = LoraAdapter(
            self.base_layer,
            config=self.transformer_config,
            rank=self.rank,
            alpha=self.alpha,
            dropout=0.01,
        )
        yield
        Utils.destroy_model_parallel()

    def test_constructor(self):
        assert _get_nparams(self.lora_adapter) == self.input_size * self.output_size


def _get_nparams(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])
