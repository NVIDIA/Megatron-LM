# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from megatron.core import config, parallel_state
from megatron.core.extensions.transformer_engine import get_thd_partitioned_indices
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.fused_a2a import reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.core.transformer.moe.token_dispatcher import MoETokenDispatcher
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


def token_permutation(token_dispatcher, hidden_states, probs, indices):
    hidden_states, probs = token_dispatcher.dispatch_preprocess(hidden_states, indices, probs)
    hidden_states, probs = token_dispatcher.token_dispatch(hidden_states, probs)
    hidden_states, tokens_per_expert, permuted_probs = token_dispatcher.dispatch_postprocess(
        hidden_states, probs
    )
    return hidden_states, tokens_per_expert, permuted_probs


def token_unpermutation(token_dispatcher, hidden_states):
    hidden_states = token_dispatcher.combine_preprocess(hidden_states)
    hidden_states = token_dispatcher.token_combine(hidden_states)
    hidden_states = token_dispatcher.combine_postprocess(hidden_states)
    return hidden_states, None


class _NestedAttrTestDispatcher(MoETokenDispatcher):
    def dispatch_preprocess(self, tokens, routing_map, probs):
        raise NotImplementedError

    def token_dispatch(self, hidden_states, probs):
        raise NotImplementedError

    def dispatch_postprocess(self, hidden_states, probs):
        raise NotImplementedError

    def combine_preprocess(self, hidden_states):
        raise NotImplementedError

    def token_combine(self, hidden_states):
        raise NotImplementedError

    def combine_postprocess(self, hidden_states):
        raise NotImplementedError


def test_get_cudagraph_attr_supports_nested_paths():
    dispatcher = object.__new__(_NestedAttrTestDispatcher)
    token_probs = torch.randn(2, 3)
    dispatcher._comm_manager = SimpleNamespace(
        token_probs=token_probs, nested=SimpleNamespace(routing_map=torch.randn(2, 4))
    )

    assert dispatcher.get_cudagraph_attr("_comm_manager.token_probs") is token_probs
    assert dispatcher.get_cudagraph_attr("_comm_manager.nested.routing_map") is not None
    assert dispatcher.get_cudagraph_attr("_comm_manager.missing_attr") is None


def test_set_cudagraph_attr_supports_nested_paths():
    dispatcher = object.__new__(_NestedAttrTestDispatcher)
    dispatcher._comm_manager = SimpleNamespace(routing_map=None)
    routing_map = torch.randn(4, 5)

    dispatcher.set_cudagraph_attr("_comm_manager.routing_map", routing_map)

    assert dispatcher._comm_manager.routing_map is routing_map


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
            moe_router_topk=moe_router_topk,
            num_moe_experts=num_moe_experts,
            moe_router_load_balancing_type=moe_router_load_balancing_type,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_expert_capacity_factor=moe_expert_capacity_factor,
            moe_pad_expert_input_to_capacity=moe_pad_expert_input_to_capacity,
            moe_aux_loss_coeff=moe_aux_loss_coeff,
            num_layers=1,
            moe_router_dtype="fp32",
            moe_grouped_gemm=kwargs.get("moe_grouped_gemm", False),
            hidden_size=kwargs.get("hidden_size", 16),
            num_attention_heads=kwargs.get("num_attention_heads", 8),
            use_cpu_initialization=kwargs.get("use_cpu_initialization", True),
            sequence_parallel=tp_size > 1,
            add_bias_linear=kwargs.get("add_bias_linear", False),
            moe_permute_fusion=kwargs.get("moe_permute_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
            calculate_per_token_loss=kwargs.get("calculate_per_token_loss", False),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        submodules = get_submodules(
            get_gpt_layer_local_submodules(
                num_experts=self.config.num_moe_experts,
                moe_grouped_gemm=self.config.moe_grouped_gemm,
            ).mlp
        )
        assert isinstance(submodules, MoESubmodules)
        new_config = dataclasses.replace(self.config, **kargs)
        moe_layer = MoELayer(new_config, submodules).cuda().to(dtype=self.test_dtype)
        moe_layer.set_layer_number(0)
        return moe_layer

    def __del__(self):
        torch.distributed.barrier()
        torch.cuda.synchronize()
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    def dispatcher_dropless_test(self):
        moe_layer = self.moe_layer
        bs = 32
        seql = 8
        # TODO: Find why setting manual seed can cause the test to fail
        # Manual seed to differentiate input data for each rank
        # rank = torch.distributed.get_rank()
        # torch.manual_seed(1000 + rank)
        hidden_states = torch.randn((bs, seql, moe_layer.config.hidden_size), dtype=self.test_dtype)
        hidden_states = hidden_states.cuda()
        # Permute and then unpermute data are supposed to restore original data
        ans = hidden_states
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)
        probs = torch.ones_like(probs) / moe_layer.router.topk

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs, indices
        )

        permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)
        permuted_local_hidden_states = permuted_local_hidden_states.to(dtype=self.test_dtype)

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_local_hidden_states
        )

        # reduce across TP rank equals to multiply data by a scale of ETP
        scale = moe_layer.config.expert_tensor_parallel_size
        restored_hidden_states = restored_hidden_states / scale

        torch.testing.assert_close(
            restored_hidden_states, ans
        ), "Restored hidden states do not match original hidden states"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, ans
        ), "Restored hidden states do not match original hidden states"

    @pytest.mark.internal
    def dispatcher_capacity_test(self):
        moe_layer = self.moe_layer
        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        )
        hidden_states = hidden_states.cuda()
        hidden_states.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(hidden_states)

        # Create the answer.
        prob_mask = probs != 0
        probs = torch.ones_like(probs) * prob_mask / moe_layer.router.topk
        local_probss = probs
        restored_hidden_states_answer = hidden_states * local_probss.sum(dim=1).unsqueeze(1)
        restored_hidden_states_answer = restored_hidden_states_answer.to(dtype=self.test_dtype)

        (permuted_local_hidden_states, tokens_per_expert, permuted_probs) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs, indices
        )

        # Check tokens per expert not exceed the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            <= capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert exceed the capacity"

        permuted_local_hidden_states = permuted_local_hidden_states * permuted_probs.unsqueeze(-1)

        permuted_local_hidden_states /= moe_layer.config.tensor_model_parallel_size
        permuted_local_hidden_states = permuted_local_hidden_states.to(dtype=self.test_dtype)

        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_local_hidden_states
        )
        torch.testing.assert_close(
            restored_hidden_states, restored_hidden_states_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, restored_hidden_states_answer
        ), "Gradient of hidden states should be same as hidden states"

    @pytest.mark.internal
    def dispatcher_drop_and_pad_test(self):
        """Test if the tokens are dropped and padded correctly.

        Since the probs of padded tokens are 0, the combined results for
        dispatching with or without padding should be the same.
        """
        moe_layer = self.new_moe_layer(moe_pad_expert_input_to_capacity=False)

        num_tokens = 16
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        (permuted_input_1, tokens_per_expert, permuted_probs_1) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        forward_answer, restored_bias = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(forward_answer, forward_answer)
        backward_answer = hidden_states.grad.clone()
        hidden_states.grad = None
        torch.cuda.synchronize()
        # End

        moe_layer_2 = self.new_moe_layer(moe_pad_expert_input_to_capacity=True)
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        (permuted_input_2, tokens_per_expert, permuted_probs_2) = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states, restored_bias = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # # Check tokens per expert equals to the capacity.
        capacity = get_capacity(
            num_tokens * self.config.moe_router_topk,
            self.config.num_moe_experts,
            self.config.moe_expert_capacity_factor,
        )
        assert torch.all(
            tokens_per_expert
            == capacity
            * self.config.expert_model_parallel_size
            * self.config.tensor_model_parallel_size
        ), "Tokens per expert should be the same as the capacity"
        torch.testing.assert_close(
            restored_hidden_states, forward_answer
        ), "Restored hidden states does not match"

        # check if the grad of the hidden states is same as the hidden states
        torch.autograd.backward(restored_hidden_states, restored_hidden_states)
        torch.testing.assert_close(
            hidden_states.grad, backward_answer
        ), "Gradient of hidden states should be same as hidden states"

    @pytest.mark.internal
    def dispatcher_router_padding_for_fp8_test(self):
        """Test if the routing map is padded correctly for FP8 training.

        The test runs the forward flow twice:
        1. First with moe_router_padding_for_quantization=False
        2. Then with moe_router_padding_for_quantization=True

        We verify that:
        1. The results are the same in both cases
        2. The number of tokens received by each expert is padded to a multiple of 16
        """
        # First run with moe_router_padding_for_quantization = False
        moe_layer = self.new_moe_layer(moe_router_padding_for_quantization=False)

        num_tokens = 32
        hidden_states = torch.randn(
            (num_tokens, moe_layer.config.hidden_size), dtype=self.test_dtype
        ).cuda()
        hidden_states.requires_grad = True

        probs_1, indices_1 = apply_module(moe_layer.router)(hidden_states)
        (permuted_input_1, tokens_per_expert_1, permuted_probs_1) = token_permutation(
            moe_layer.token_dispatcher, hidden_states, probs_1, indices_1
        )
        permuted_input_1 = permuted_input_1 * permuted_probs_1.unsqueeze(-1)
        permuted_input_1 = permuted_input_1.to(dtype=self.test_dtype)
        restored_hidden_states_1, _ = token_unpermutation(
            moe_layer.token_dispatcher, permuted_input_1
        )
        torch.autograd.backward(restored_hidden_states_1, restored_hidden_states_1)
        grad_1 = hidden_states.grad.clone()
        hidden_states.grad = None

        # Run with moe_router_padding_for_quantization = True
        moe_layer_2 = self.new_moe_layer(moe_router_padding_for_quantization=True, fp8="hybrid")
        moe_layer_2.load_state_dict(moe_layer.state_dict())

        probs_2, indices_2 = apply_module(moe_layer_2.router)(hidden_states)
        (permuted_input_2, tokens_per_expert_2, permuted_probs_2) = token_permutation(
            moe_layer_2.token_dispatcher, hidden_states, probs_2, indices_2
        )
        assert (
            sum(tokens_per_expert_2) == permuted_input_2.shape[0]
        ), f"number of tokens is not the same, {sum(tokens_per_expert_2)} != {permuted_input_2.shape[0]}"
        # when there is only one expert, the tokens is not enough for router padding
        if moe_layer_2.num_local_experts > 1:
            assert torch.all(
                tokens_per_expert_2 % 16 == 0
            ), "number of tokens for expert is not a multiple of 16"

        permuted_input_2 = permuted_input_2 * permuted_probs_2.unsqueeze(-1)
        permuted_input_2 = permuted_input_2.to(dtype=self.test_dtype)
        restored_hidden_states_2, _ = token_unpermutation(
            moe_layer_2.token_dispatcher, permuted_input_2
        )

        # Check that the results are the same
        torch.testing.assert_close(
            restored_hidden_states_1, restored_hidden_states_2
        ), "Restored hidden states do not match between padded and non-padded versions"

        # Check gradients
        torch.autograd.backward(restored_hidden_states_2, restored_hidden_states_2)
        torch.testing.assert_close(
            grad_1, hidden_states.grad
        ), "Gradients do not match between padded and non-padded versions"

    def set_params(self):
        # TODO: Set consistent parameters for various parallelisms.
        raise NotImplementedError

    def destroy(self):
        Utils.destroy_model_parallel()


permute_fusion_params = [False]
if is_te_min_version("2.1.0"):
    permute_fusion_params.append(True)


class TestAllgatherDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(8, 1), (1, 8), (2, 4), (1, 1)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    def test_forward_backward(self, tp_size, ep_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_permute_fusion=permute_fusion,
        )

        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "tp_size,ep_size,moe_tp_size", [(1, 1, 8), (1, 2, 4), (1, 4, 2), (2, 2, 4)]
    )
    def test_moe_tp_forward_backward(self, tp_size, ep_size, moe_tp_size, permute_fusion):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            moe_tp_size=moe_tp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            sequence_parallel=True,
            moe_permute_fusion=permute_fusion,
            use_cpu_initialization=False,
        )

        container.dispatcher_dropless_test()


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_deep_ep_v2_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2

    return HAVE_DEEP_EP_V2


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def _round_up(value, divisor):
    return value if divisor <= 1 else (value + divisor - 1) // divisor * divisor


def _get_thd_padded_seqlens(seqlens, cp_size, tp_size):
    # This follows the runtime packed-sequence path used by the Moonlight script:
    # per-sequence lengths must be CP partitionable, and the packed token count
    # must be even for TP/SP slicing.
    cp_divisor = 2 * cp_size if cp_size > 1 else 1
    padded_seqlens = [_round_up(seqlen, cp_divisor) for seqlen in seqlens]
    total_seqlen = sum(padded_seqlens)
    total_alignment = math.lcm(cp_divisor, tp_size)
    padded_seqlens[-1] += _round_up(total_seqlen, total_alignment) - total_seqlen
    return padded_seqlens


def _to_cu_seqlens(seqlens):
    cu_seqlens = torch.empty(len(seqlens) + 1, dtype=torch.int32, device="cuda")
    cu_seqlens[0] = 0
    cu_seqlens[1:] = torch.cumsum(torch.tensor(seqlens, dtype=torch.int32, device="cuda"), dim=0)
    return cu_seqlens


def _make_thd_packed_seq_params(seqlens, cp_size, tp_size):
    padded_seqlens = _get_thd_padded_seqlens(seqlens, cp_size, tp_size)
    cu_seqlens_padded = _to_cu_seqlens(padded_seqlens)
    max_seqlen = max(padded_seqlens)
    # Match get_batch_on_this_rank_for_sequence_packing(): TE consumes padded
    # cumulative lengths as both cu_seqlens and cu_seqlens_padded for THD.
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens_padded,
        cu_seqlens_kv=cu_seqlens_padded,
        cu_seqlens_q_padded=cu_seqlens_padded,
        cu_seqlens_kv_padded=cu_seqlens_padded,
        max_seqlen_q=max_seqlen,
        max_seqlen_kv=max_seqlen,
    )


def _make_sharded_thd_hidden_states(seqlens, hidden_size, cp_size, tp_size, dtype):
    padded_seqlens = _get_thd_padded_seqlens(seqlens, cp_size, tp_size)
    padded_sequences = []
    for seqlen, padded_seqlen in zip(seqlens, padded_seqlens):
        sequence = torch.randn(seqlen, hidden_size, device="cuda", dtype=dtype)
        if padded_seqlen > seqlen:
            sequence = torch.cat(
                [
                    sequence,
                    torch.zeros(padded_seqlen - seqlen, hidden_size, device="cuda", dtype=dtype),
                ],
                dim=0,
            )
        padded_sequences.append(sequence)

    hidden_states = torch.cat(padded_sequences, dim=0)
    if cp_size > 1:
        cu_seqlens_padded = _to_cu_seqlens(padded_seqlens)
        cp_rank = parallel_state.get_context_parallel_rank()
        index = get_thd_partitioned_indices(
            cu_seqlens_padded, hidden_states.shape[0], cp_size, cp_rank
        )
        hidden_states = hidden_states.index_select(0, index)

    tp_rank = parallel_state.get_tensor_model_parallel_rank()
    sequence_parallel_length = hidden_states.shape[0] // tp_size
    hidden_states = hidden_states[
        tp_rank * sequence_parallel_length : (tp_rank + 1) * sequence_parallel_length
    ]
    return hidden_states.unsqueeze(1).contiguous().requires_grad_(True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(
    Utils.world_size % 8 != 0, reason="requires world size divisible by 8 for pp2/cp2/tp2/ep2/etp2"
)
@pytest.mark.internal
@pytest.mark.parametrize("dispatcher", ["alltoall", "deepep", "hybridep"])
def test_sequence_packing_thd_e2e_proxy_model(dispatcher):
    """Run packed THD attention + MoE forward/backward with major parallelisms enabled."""
    if not is_te_min_version("2.9.0"):
        pytest.skip("SFT sequence packing requires Transformer Engine >= 2.9.0")
    if dispatcher == "deepep" and not is_deep_ep_available():
        pytest.skip("Deep EP is not available")
    if dispatcher == "hybridep" and not is_hybrid_ep_available():
        pytest.skip("Hybrid EP is not available")

    tp_size, pp_size, cp_size, ep_size, etp_size = 2, 2, 2, 2, 2
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp_size,
        pipeline_model_parallel_size=pp_size,
        context_parallel_size=cp_size,
        expert_model_parallel_size=ep_size,
        expert_tensor_parallel_size=etp_size,
    )
    _set_random_seed(seed_=123, data_parallel_random_init=False)

    try:
        spec = get_gpt_layer_with_transformer_engine_spec(num_experts=4, moe_grouped_gemm=False)
        transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=1024,
            ffn_hidden_size=2048,
            moe_ffn_hidden_size=2048,
            num_attention_heads=8,
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=pp_size,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
            expert_tensor_parallel_size=etp_size,
            sequence_parallel=True,
            sequence_packing_scheduler="dp_balanced",
            max_seqlen_per_dp_cp_rank=1024,
            cp_comm_type="p2p",
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type=(
                "flex" if dispatcher in ("deepep", "hybridep") else dispatcher
            ),
            moe_flex_dispatcher_backend=(
                dispatcher if dispatcher in ("deepep", "hybridep") else "deepep"
            ),
            moe_grouped_gemm=False,
            moe_router_dtype="fp32",
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            autocast_dtype=torch.bfloat16,
            bf16=True,
            add_bias_linear=False,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            use_cpu_initialization=True,
        )
        transformer_block = TransformerBlock(transformer_config, spec).cuda().to(torch.bfloat16)

        torch.manual_seed(1000 + torch.distributed.get_rank())
        seqlens = [257, 509, 1021]
        hidden_states = _make_sharded_thd_hidden_states(
            seqlens, transformer_config.hidden_size, cp_size, tp_size, torch.bfloat16
        )
        packed_seq_params = _make_thd_packed_seq_params(seqlens, cp_size, tp_size)

        output = transformer_block(
            hidden_states=hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all()

        loss = output.float().square().mean()
        loss.backward()

        assert hidden_states.grad is not None
        assert hidden_states.grad.shape == hidden_states.shape
        assert torch.isfinite(hidden_states.grad).all()
        assert any(
            param.grad is not None and torch.isfinite(param.grad).all()
            for param in transformer_block.parameters()
            if param.requires_grad
        )
    finally:
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()


def skip_if_flex_backend_unavailable(moe_flex_dispatcher_backend):
    if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
        pytest.skip("Deep EP is not available")
    if moe_flex_dispatcher_backend == "deepepv2" and not is_deep_ep_v2_available():
        pytest.skip("Deep EP v2 is not available")
    if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
        pytest.skip("Hybrid EP is not available")


@pytest.mark.skipif(
    not is_deep_ep_available() and not is_deep_ep_v2_available() and not is_hybrid_ep_available(),
    reason="Deep EP, Deep EP v2 and Hybrid EP are not available",
)
class TestFlexDispatcher:
    def setup_method(self, method):
        pass

    def teardown_method(self, method):
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "deepepv2", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        skip_if_flex_backend_unavailable(moe_flex_dispatcher_backend)
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()
        # reset experimental flag to False
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "deepepv2", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_capacity_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        skip_if_flex_backend_unavailable(moe_flex_dispatcher_backend)
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_token_drop_policy="probs",
            moe_expert_capacity_factor=0.5,
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_capacity_test()
        config.ENABLE_EXPERIMENTAL = False

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0"), reason="TE 1.7.0 is required for MoE with FP8."
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", [True])
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "deepepv2", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        skip_if_flex_backend_unavailable(moe_flex_dispatcher_backend)
        if moe_permute_fusion_into_hybridep:
            if permute_fusion or moe_flex_dispatcher_backend != "hybridep":
                pytest.skip(
                    "moe_permute_fusion_into_hybridep skipped because permute_fusion or hybridep is not set"
                )
        if permute_fusion:
            config.ENABLE_EXPERIMENTAL = True
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=32,
            moe_router_topk=4,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_pad_expert_input_to_capacity=False,
            moe_permute_fusion=permute_fusion,
            hidden_size=1024,
            moe_flex_dispatcher_backend=moe_flex_dispatcher_backend,
            moe_permute_fusion_into_hybridep=moe_permute_fusion_into_hybridep,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_router_padding_for_fp8_test()
        config.ENABLE_EXPERIMENTAL = False
