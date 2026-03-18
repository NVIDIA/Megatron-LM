# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import get_align_size_for_quantization
from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.paged_stash import (
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class MoEModelTestContainer:
    def __init__(
        self,
        tp_size,
        ep_size,
        pp_size,
        cp_size=1,
        moe_tp_size=None,
        data_parallel_random_init=False,
        num_moe_experts=8,
        num_layers=1,
        moe_router_topk=2,
        moe_router_load_balancing_type="aux_loss",
        moe_token_dispatcher_type="alltoall",
        moe_expert_capacity_factor=None,
        moe_pad_expert_input_to_capacity=False,
        moe_aux_loss_coeff=0.1,
        test_dtype=torch.float32,
        **kwargs,
    ):
        self.num_local_experts = num_moe_experts // ep_size
        self.num_layers = num_layers
        self.test_dtype = test_dtype
        if moe_tp_size is None:
            moe_tp_size = tp_size
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=data_parallel_random_init)
        self.config = TransformerConfig(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_tensor_parallel_size=moe_tp_size,
            fp8='e4m3',
            fp8_recipe='mxfp8',
            fp8_wgrad=True,
            fp8_amax_compute_algo='most_recent',
            fp8_amax_history_len=1,
            fp8_interval=1,
            fp8_margin=0,
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=num_layers,
            moe_router_dtype="fp32",
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            moe_use_device_initiated_grouped_gemm=kwargs.get(
                "moe_use_device_initiated_grouped_gemm", False
            ),
            moe_use_legacy_grouped_gemm=kwargs.get("moe_use_legacy_grouped_gemm", False),
            moe_paged_stash=kwargs.get("moe_paged_stash", False),
            stash_modules=kwargs.get("stash_modules", None),
            moe_expert_rank_capacity_factor=kwargs.get("moe_expert_rank_capacity_factor", None),
            moe_router_padding_for_fp8=kwargs.get("moe_router_padding_for_fp8", True),
            use_transformer_engine_op_fuser=kwargs.get("use_transformer_engine_op_fuser", False),
            moe_mlp_glu_interleave_size=kwargs.get("moe_mlp_glu_interleave_size", None),
            moe_router_padding_for_quantization=kwargs.get("moe_router_padding_for_quantization", False),
            gated_linear_unit=kwargs.get("gated_linear_unit", False),
            activation_func=kwargs.get("activation_func", F.gelu),
            moe_router_force_biased=kwargs.get("moe_router_force_biased", None),
        )
        self.moe_layer = self._create_moe_layer(layer_number=0)

    def _create_moe_layer(self, layer_number=0):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=True
        )
        quantization_context = get_fp8_context(self.config, layer_number, is_init=True)
        with quantization_context:
            moe_layer = (
                MoELayer(self.config, transformer_layer_spec.submodules.mlp.submodules)
                .cuda()
                .to(dtype=self.test_dtype)
            )
            moe_layer.set_layer_number(layer_number)
            return moe_layer

    def zero_grad(self):
        self.moe_layer.zero_grad()

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    def forward_backward(self, hidden_states):
        """Run one forward and backward pass through the MoE layer.

        Returns:
            output: MoE layer output (detached).
            hidden_states_grad: Gradient w.r.t. hidden_states.
            routing_map: Token-to-expert routing map from the dispatcher (after forward).
            tokens_per_expert: Number of tokens per local expert on this EP rank (after forward).
        """
        hidden_states = hidden_states.cuda().requires_grad_(True)
        quantization_context = get_fp8_context(self.config)
        with quantization_context:
            output, _ = self.moe_layer(hidden_states)
        # Capture routing_map and tokens_per_expert after forward (before backward)
        comm = getattr(self.moe_layer.token_dispatcher, "_comm_manager", None)
        routing_map = getattr(comm, "routing_map", None)
        tokens_per_expert = (
            comm.get_number_of_tokens_per_expert()
            if comm is not None and hasattr(comm, "get_number_of_tokens_per_expert")
            else None
        )
        # Use contiguous gradient to avoid non-contiguous grad in HybridEP combine backward
        # (output.sum().backward() produces a broadcast gradient that is non-contiguous)
        output.backward(torch.ones_like(output))
        return output.detach(), hidden_states.grad, routing_map, tokens_per_expert

    def destroy(self):
        Utils.destroy_model_parallel()


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
    return HAVE_HYBRIDEP


@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestPagedStashing:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_forward_backward(self):
        if not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")

        config.ENABLE_EXPERIMENTAL = True

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=2,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="hybridep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_device_initiated_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_paged_stash=True,
            stash_modules=["expert_fc1", "moe_act", "expert_fc2"],
            moe_expert_rank_capacity_factor=1.5,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
        )
        if not isinstance(container.moe_layer.experts, TEGroupedMLP) or not container.moe_layer.experts._is_fused_impl_supported():
            container.destroy()
            pytest.skip("TEGroupedMLP fused impl not supported")

        # [sequence_length, batch_size, hidden_size] for MoELayer.forward
        seq_length = 1024
        batch_size = 1
        hidden_size = container.config.hidden_size
        hidden_states = torch.randn(
            (seq_length, batch_size, hidden_size), dtype=torch.bfloat16
        )

        # First iteration: capture schedule, capacity, etc.
        paged_stash_reset(True)
        paged_stash_init_chunk_handler(1, 0)
        output_ref, hidden_states_grad_ref, routing_map_ref, tokens_per_expert_ref = (
            container.forward_backward(hidden_states)
        )

        container.zero_grad()

        # Second iteration: run with paged stash.
        paged_stash_reset(True)
        paged_stash_init_chunk_handler(1, 0)
        output, hidden_states_grad, routing_map, tokens_per_expert = container.forward_backward(
            hidden_states
        )

        # Verify output and input gradient match the first iteration.
        torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            hidden_states_grad, hidden_states_grad_ref, atol=1e-4, rtol=1e-4
        )
        # Routing and token counts available after forward (e.g. for debugging or further checks)
        if routing_map is not None and tokens_per_expert is not None:
            num_tokens_per_ep_rank = tokens_per_expert.sum().item()
            assert num_tokens_per_ep_rank > 0
            assert routing_map_ref is not None and tokens_per_expert_ref is not None
            torch.testing.assert_close(tokens_per_expert, tokens_per_expert_ref)


@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestPagedStashingOverBudget:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    def test_overload_factor_and_over_budget(self):
        """Test budget computation (same as token_dispatcher lines 1017-1025) and assert
        over_budget flag is set when tokens_per_ep_rank exceeds budget."""
        if not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")

        config.ENABLE_EXPERIMENTAL = True

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=4,
            pp_size=1,
            num_moe_experts=8,
            num_layers=1,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=True,
            hidden_size=1024,
            moe_flex_dispatcher_backend="hybridep",
            test_dtype=torch.bfloat16,
            moe_grouped_gemm=True,
            moe_use_device_initiated_grouped_gemm=True,
            moe_use_legacy_grouped_gemm=False,
            moe_paged_stash=True,
            stash_modules=["expert_fc1", "moe_act", "expert_fc2"],
            moe_expert_rank_capacity_factor=1.0,
            use_transformer_engine_op_fuser=True,
            moe_mlp_glu_interleave_size=32,
            moe_router_padding_for_quantization=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            moe_router_force_biased=1,
        )
        if not isinstance(container.moe_layer.experts, TEGroupedMLP) or not container.moe_layer.experts._is_fused_impl_supported():
            container.destroy()
            pytest.skip("TEGroupedMLP fused impl not supported")

        seq_length = 4096
        batch_size = 1
        topk = container.config.moe_router_topk
        capacity_factor = container.config.moe_expert_rank_capacity_factor
        hidden_size = container.config.hidden_size
        hidden_states = torch.randn(
            (seq_length, batch_size, hidden_size), dtype=torch.bfloat16
        )

        # Budget computed like token_dispatcher._HybridEPManager.setup_metadata (lines 1017-1025)
        num_tokens = seq_length * batch_size
        pad_multiple = get_align_size_for_quantization(container.config)
        budget = int(num_tokens * topk * capacity_factor)
        budget += -budget % pad_multiple

        paged_stash_reset(True)
        paged_stash_init_chunk_handler(1, 0)
        _, _, _, tokens_per_expert = container.forward_backward(hidden_states)

        assert tokens_per_expert is not None
        tokens_per_ep_rank = tokens_per_expert.sum().item()
        over_budget_tensor = container.moe_layer.token_dispatcher.check_over_budget()
        over_budget = over_budget_tensor.item() if over_budget_tensor is not None else False

        # When tokens_per_ep_rank > budget, over_budget flag must be raised
        if tokens_per_ep_rank >= budget:
            assert over_budget, (
                f"tokens_per_ep_rank ({tokens_per_ep_rank}) > budget ({budget}), "
                "but over_budget flag was not set"
            )
        else:
            assert not over_budget, (
                f"tokens_per_ep_rank ({tokens_per_ep_rank}) <= budget ({budget}), "
                "but over_budget flag was set"
            )
