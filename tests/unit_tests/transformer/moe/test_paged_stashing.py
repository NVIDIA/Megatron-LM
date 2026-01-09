# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import dataclasses

import pytest
import torch

from megatron.core import config, parallel_state
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.paged_stash import (
    paged_stash_init_chunk_handler,
    paged_stash_reset,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def token_permutation(token_dispatcher, hidden_states, probs, indices):
    residual = hidden_states
    hidden_states, probs = token_dispatcher.dispatch_preprocess(hidden_states, indices, probs)
    hidden_states, probs = token_dispatcher.token_dispatch(hidden_states, probs)
    return hidden_states, probs, residual


def token_unpermutation(token_dispatcher, hidden_states):
    hidden_states = token_dispatcher.token_combine(hidden_states)
    hidden_states = token_dispatcher.combine_postprocess(hidden_states)
    return hidden_states, None


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
        local_expert_indices_offset = (
            parallel_state.get_expert_model_parallel_rank() * self.num_local_experts
        )
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]
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
        )
        # init moe layers
        self.moe_layers = [self.new_moe_layer(layer_number=i) for i in range(num_layers)]

    def new_moe_layer(self, **kargs):
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=self.config.num_moe_experts, moe_grouped_gemm=True
        )
        layer_number = kargs.get("layer_number", 0)
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
        for moe_layer in self.moe_layers:
            moe_layer.zero_grad()

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self, inp_hidden_states=None):
        moe_layers = self.moe_layers

        inp_hidden_states = inp_hidden_states.cuda()
        # Permute and then unpermute data are supposed to restore original data
        inp_hidden_states.requires_grad = True
        hidden_states = inp_hidden_states
        for i, moe_layer in enumerate(moe_layers):
            quantization_context = get_fp8_context(self.config)
            with quantization_context:
                probs, indices = moe_layer.router(hidden_states)
                probs = torch.ones_like(probs) / moe_layer.router.topk

                (dispatched_input, probs, residual) = token_permutation(
                    moe_layer.token_dispatcher, hidden_states, probs, indices
                )
                output, _ = moe_layer.routed_experts_compute(dispatched_input, probs, residual)
                output, _ = token_unpermutation(moe_layer.token_dispatcher, output)
                hidden_states = output
        torch.autograd.backward(output, inp_hidden_states)
        return output, inp_hidden_states.grad

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


permute_fusion_params = [False]
if is_te_min_version("2.1.0"):
    permute_fusion_params.append(True)


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP
    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP
    return HAVE_HYBRIDEP


@pytest.mark.skipif(not is_hybrid_ep_available(), reason="Hybrid EP are not available")
class TestFlexDispatcher:
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
        )
        bs = 32
        seql = 8

        inp_hidden_states = torch.randn(
            (bs, seql, container.moe_layers[0].config.hidden_size), dtype=torch.bfloat16
        )
        # First iteration to capture schedule, calculate capacity, etc.
        paged_stash_reset(True)
        paged_stash_init_chunk_handler(1, 0)
        output_ref, inp_hidden_states_grad_ref = container.dispatcher_dropless_test(
            inp_hidden_states
        )

        container.zero_grad()

        # Second iteration to run with paged stash.
        paged_stash_reset(True)
        paged_stash_init_chunk_handler(1, 0)
        output, inp_hidden_states_grad = container.dispatcher_dropless_test(inp_hidden_states)

        # verify output and input gradient are the same as the first iteration.
        torch.testing.assert_close(output, output_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(
            inp_hidden_states_grad, inp_hidden_states_grad_ref, atol=1e-4, rtol=1e-4
        )
