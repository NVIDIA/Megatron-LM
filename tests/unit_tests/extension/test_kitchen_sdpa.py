# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import queue
from typing import Literal, Tuple

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.kitchen import KitchenDotProductAttention, KitchenFlashAttention
from megatron.core.extensions.transformer_engine import TEDotProductAttention
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.quantization.quant_config import RecipeConfig
from megatron.core.quantization.utils import get_quant_config_or_none
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.dot_product_attention import DotProductAttention
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import nvidia_kitchen  # type: ignore[import-not-found]

    HAVE_KITCHEN = True
except ImportError:
    from unittest.mock import MagicMock

    HAVE_KITCHEN = False
    nvidia_kitchen = MagicMock()

try:
    import transformer_engine  # type: ignore[import-untyped]
    from transformer_engine.pytorch.attention import (  # type: ignore[import-untyped]
        dot_product_attention,
    )

    HAVE_TE = True
except ImportError:
    from unittest.mock import MagicMock

    HAVE_TE = False
    transformer_engine = MagicMock()
    dot_product_attention = MagicMock()


# Create custom process groups
Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=1)
model_parallel_cuda_manual_seed(123)

# Get TP and CP process groups from device mesh
tp_group = parallel_state.get_tensor_model_parallel_group()
cp_group = parallel_state.get_context_parallel_group()

pg_collection = ProcessGroupCollection(tp=tp_group, cp=cp_group)


def get_attention_implementation(
    impl: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
    config: TransformerConfig,
    layer_number: int,
    attn_mask_type: AttnMaskType,
    attention_type: str,
    attention_dropout: float,
    softmax_scale: float,
    cp_comm_type: str = "a2a",
) -> MegatronModule:
    if impl == "megatron":
        return DotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
    elif impl == "te-fa" or impl == "te-unfused":
        if attention_type == "self_attention":
            attention_type = "self"
        return TEDotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type=cp_comm_type,
            pg_collection=pg_collection,
        )
    elif impl == "kitchen":
        attn = KitchenDotProductAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
        attn.finish_init(
            get_quant_config_or_none("self_attention.core_attention", config.quant_recipe)
        )
        return attn
    elif impl == "kitchen-fa":
        if attention_type == "self_attention":
            attention_type = "self"
        attn = KitchenFlashAttention(
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
            cp_comm_type,
            pg_collection,
        )
        attn.finish_init(
            get_quant_config_or_none("self_attention.core_attention", config.quant_recipe)
        )
        return attn
    else:
        raise ValueError(f"Invalid implementation: {impl}")


class DotProductAttentionModel(torch.nn.Module):
    def __init__(
        self,
        impl: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float,
        softmax_scale: float,
    ):
        super().__init__()
        self.impl = impl
        self.attention_module = get_attention_implementation(
            impl,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor,
        attn_mask_type: AttnMaskType,
    ):
        return self.attention_module(query, key, value, attention_mask, attn_mask_type)

    @property
    def last_attention_probs(self):
        return self.attention_module._last_attention_probs


class CompareImplementations:

    def _prepare_data(
        self, config: TransformerConfig, seed: int, dtype: torch.dtype = torch.bfloat16
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        b = 4
        # np = number of attention heads per partition
        np = config.num_attention_heads // config.tensor_model_parallel_size
        # hn = hidden size per attention head (same as kv_channels)
        hn = config.hidden_size // config.num_attention_heads
        # sk = number of key tokens
        sk = 256
        # sq = number of query tokens
        sq = 256

        # bshd layout

        shape = (sq, b, np, hn)

        q = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
        k = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
        v = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)

        grad = torch.randn((sq, b, np * hn), dtype=dtype, device="cuda")
        return q, k, v, grad

    def run_attention_one_step(
        self,
        layer: DotProductAttentionModel,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gradient: torch.Tensor,
        attn_mask_type: AttnMaskType,
    ):
        layer.zero_grad()
        query.grad = None
        key.grad = None
        value.grad = None

        attention_mask = None

        out = layer(query, key, value, attention_mask, attn_mask_type)

        out.backward(gradient)

        qgrad = query.grad
        kgrad = key.grad
        vgrad = value.grad

        return out, qgrad, kgrad, vgrad  # , layer.last_attention_probs

    def compare_implementations(
        self,
        impl1: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
        impl2: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float,
        softmax_scale: float,
        out_error: float,
        q_grad_error: float,
        k_grad_error: float,
        v_grad_error: float,
        seed: int = 0,
    ) -> None:
        os.environ["NVTE_ALLOW_NONDETERMINISTIC_ALGO"] = "0"
        if impl1 == "te-fa" or impl2 == "te-fa":
            dot_product_attention._attention_backends = {
                "attention_params": None,
                "use_flash_attention": None,
                "flash_attention_backend": None,
                "use_fused_attention": None,
                "fused_attention_backend": None,
                "use_unfused_attention": None,
                "backend_selection_requires_update": False,
            }
            os.environ["NVTE_FLASH_ATTN"] = "1"
        elif impl1 == "te-unfused" or impl2 == "te-unfused":
            dot_product_attention._attention_backends = {
                "attention_params": None,
                "use_flash_attention": None,
                "flash_attention_backend": None,
                "use_fused_attention": None,
                "fused_attention_backend": None,
                "use_unfused_attention": None,
                "backend_selection_requires_update": False,
            }
            os.environ["NVTE_FUSED_ATTN"] = "0"
            os.environ["NVTE_FLASH_ATTN"] = "0"

        # qkv are (sq, b, np, hn)
        # grad is (sq, b, np * hn)
        q, k, v, grad = self._prepare_data(config, seed)
        layer1 = DotProductAttentionModel(
            impl1,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
        )
        layer2 = DotProductAttentionModel(
            impl2,
            config,
            layer_number,
            attn_mask_type,
            attention_type,
            attention_dropout,
            softmax_scale,
        )

        query_layer, key_layer, value_layer = q, k, v

        out1, q_grad1, k_grad1, v_grad1 = self.run_attention_one_step(
            layer1,
            query_layer.clone().detach().requires_grad_(True),
            key_layer.clone().detach().requires_grad_(True),
            value_layer.clone().detach().requires_grad_(True),
            grad.clone().detach().requires_grad_(True),
            attn_mask_type,
        )
        out2, q_grad2, k_grad2, v_grad2 = self.run_attention_one_step(
            layer2,
            query_layer.clone().detach().requires_grad_(True),
            key_layer.clone().detach().requires_grad_(True),
            value_layer.clone().detach().requires_grad_(True),
            grad.clone().detach().requires_grad_(True),
            attn_mask_type,
        )

        torch.testing.assert_close(out1, out2, atol=out_error, rtol=0.0)
        torch.testing.assert_close(q_grad1, q_grad2, atol=q_grad_error, rtol=0.0)
        torch.testing.assert_close(k_grad1, k_grad2, atol=k_grad_error, rtol=0.0)
        torch.testing.assert_close(v_grad1, v_grad2, atol=v_grad_error, rtol=0.0)


@pytest.mark.skipif(
    not HAVE_KITCHEN or not HAVE_TE,
    reason="Kitchen and Transformer Engine required for using kitchen backend.",
)
@pytest.mark.parametrize(
    "impl1, impl2, errors",
    [
        ("megatron", "kitchen", (0.0, 0.0625, 0.125, 0.0625)),
        ("kitchen", "te-fa", (0.0625, 0.1875, 0.125, 0.05)),
        ("kitchen", "te-unfused", (0.05, 0.1875, 0.09375, 0.05)),
    ],
)
def test_attention_implementations(
    impl1: Literal["megatron", "te-fa", "te-unfused", "kitchen"],
    impl2: Literal["megatron", "te-fa", "te-unfused", "kitchen"],
    errors: Tuple[float, float, float, float],
) -> None:
    out_error, q_grad_error, k_grad_error, v_grad_error = errors

    config = TransformerConfig(
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=8,
        use_cpu_initialization=False,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        add_bias_linear=False,
        use_kitchen=True,
        use_kitchen_attention=True,
        tensor_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
        quant_recipe=RecipeConfig.from_config_dict(
            {
                "matchers": {
                    "attention": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*self_attention.core_attention",
                        "config": "bf16_attn",
                    },
                    "attention_fa": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*self_attention.core_attention",
                        "config": "bf16_fa",
                    },
                    "keep_in_hp": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*fc2",
                        "config": "bf16",
                    },
                    "use_fp8_cs": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*",
                        "config": "fp8_cs",
                    },
                },
                "configs": {
                    "bf16": {"kitchen_config_type": "QLinearParams", "recipe_idx": 1},
                    "fp8_cs": {"kitchen_config_type": "QLinearParams", "recipe_idx": 2},
                    "bf16_attn": {"kitchen_config_type": "QAttentionParams", "recipe_idx": 1},
                    "bf16_fa": {
                        "kitchen_config_type": "QFlashAttentionParams",
                        "recipe_name": "triton_fa_bf16_for_all_natural",
                    },
                },
            }
        ),
    )

    CompareImplementations().compare_implementations(
        impl1=impl1,
        impl2=impl2,
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self_attention",
        attention_dropout=0.0,
        softmax_scale=0.23,
        seed=0,
        out_error=out_error,
        q_grad_error=q_grad_error,
        k_grad_error=k_grad_error,
        v_grad_error=v_grad_error,
    )


@pytest.mark.skipif(
    not HAVE_KITCHEN or not HAVE_TE,
    reason="Kitchen and Transformer Engine required for using kitchen backend.",
)
@pytest.mark.parametrize(
    "impl1, impl2, errors",
    [
        ("kitchen-fa", "te-fa", (0.016, 0.07, 0.04, 0.01)),
        ("kitchen-fa", "kitchen", (0.125, 0.25, 0.25, 0.125)),
    ],
)
def test_kitchen_flash_attention_implementations(
    impl1: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
    impl2: Literal["megatron", "te-fa", "te-unfused", "kitchen", "kitchen-fa"],
    errors: Tuple[float, float, float, float],
) -> None:
    """Test KitchenFlashAttention against other implementations."""
    out_error, q_grad_error, k_grad_error, v_grad_error = errors

    config = TransformerConfig(
        num_layers=2,
        hidden_size=1024,
        num_attention_heads=8,
        use_cpu_initialization=False,
        gated_linear_unit=True,
        bias_activation_fusion=True,
        add_bias_linear=False,
        use_kitchen=True,
        use_kitchen_attention=True,
        kitchen_attention_backend="fa",
        tensor_model_parallel_size=1,
        bf16=True,
        params_dtype=torch.bfloat16,
        deterministic_mode=True,
        quant_recipe=RecipeConfig.from_config_dict(
            {
                "matchers": {
                    "attention": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*self_attention.core_attention",
                        "config": "bf16_fa",
                    },
                    "keep_in_hp": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*fc2",
                        "config": "bf16",
                    },
                    "use_fp8_cs": {
                        "type": "glob",
                        "enabled": True,
                        "pattern": "*",
                        "config": "fp8_cs",
                    },
                },
                "configs": {
                    "bf16": {"kitchen_config_type": "QLinearParams", "recipe_idx": 1},
                    "fp8_cs": {"kitchen_config_type": "QLinearParams", "recipe_idx": 2},
                    "bf16_fa": {
                        "kitchen_config_type": "QFlashAttentionParams",
                        "recipe_name": "triton_fa_bf16_for_all_natural",
                    },
                },
            }
        ),
    )

    CompareImplementations().compare_implementations(
        impl1=impl1,
        impl2=impl2,
        config=config,
        layer_number=1,
        attn_mask_type=AttnMaskType.causal,
        attention_type="self_attention",
        attention_dropout=0.0,
        softmax_scale=0.23,
        seed=0,
        out_error=out_error,
        q_grad_error=q_grad_error,
        k_grad_error=k_grad_error,
        v_grad_error=v_grad_error,
    )
