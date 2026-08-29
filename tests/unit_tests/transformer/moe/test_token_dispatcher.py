# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import os

import pytest
import torch
import torch.nn.functional as F

from megatron.core import config, parallel_state
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.transformer.moe.fused_a2a import HYBRIDEP_TOKEN_ALIGNMENT, reset_hybrid_ep_buffer
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_capacity
from megatron.core.transformer.moe.token_dispatcher import _HybridEPManager
from megatron.core.transformer.spec_utils import get_submodules
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
            moe_expert_rank_capacity_factor=kwargs.get("moe_expert_rank_capacity_factor", None),
            moe_ncclep_zero_copy=kwargs.get("moe_ncclep_zero_copy", False),
            moe_dispatch_fwd_dtype=kwargs.get("moe_dispatch_fwd_dtype", 'bf16'),
            moe_combine_bwd_dtype=kwargs.get("moe_combine_bwd_dtype", 'bf16'),
            use_transformer_engine_op_fuser=kwargs.get("use_transformer_engine_op_fuser", False),
            moe_use_transformer_engine_fused_moe=kwargs.get(
                "moe_use_transformer_engine_fused_moe", False
            ),
            moe_single_grouped_weight=kwargs.get("moe_single_grouped_weight", False),
            gated_linear_unit=kwargs.get("gated_linear_unit", False),
            activation_func=kwargs.get("activation_func", F.gelu),
            fp8=kwargs.get("fp8", None),
            fp8_recipe=kwargs.get("fp8_recipe", "delayed"),
            calculate_per_token_loss=kwargs.get("calculate_per_token_loss", False),
        )

        # init moe layer
        self.moe_layer = self.new_moe_layer()

    def new_moe_layer(self, **kargs):
        new_config = dataclasses.replace(self.config, **kargs)
        if new_config.use_transformer_engine_op_fuser:
            # op-fuser needs the TE grouped-MLP experts (they accept output_buffer/grad_input_buffer
            # for the ncclEP zero-copy path); the local spec yields SequentialMLP, which does not.
            mlp_spec = get_gpt_layer_with_transformer_engine_spec(
                num_experts=new_config.num_moe_experts, moe_grouped_gemm=new_config.moe_grouped_gemm
            ).submodules.mlp
        else:
            mlp_spec = get_gpt_layer_local_submodules(
                num_experts=self.config.num_moe_experts,
                moe_grouped_gemm=self.config.moe_grouped_gemm,
            ).mlp
        submodules = get_submodules(mlp_spec)
        assert isinstance(submodules, MoESubmodules)
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

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = token_permutation(
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
    def moe_layer_variant_parity_test(self, variant="zero_copy"):
        """Full MoE-layer fwd+bwd, reference vs IO-variant (identical weights), asserting parity.

        variant="zero_copy": ncclEP zero-copy OFF then ON. Runs the real op-fuser experts so
        fc2-out/fc1-dgrad are written straight into the symm combine/dispatch buffers (verified
        via is_symm_backed) -- the pure permute/unpermute harness cannot exercise this path.
        variant="mxfp8_wire": bf16 wire then MXFP8 dispatch-fwd/combine-bwd wire. The recv
        payload is an opaque carrier the op-fuser grouped GEMM rebuilds, so the tolerance is
        quantization-sized, not exactness-sized."""
        from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize
        from megatron.core.transformer.moe.token_dispatcher import _NCCLEPManager

        torch.manual_seed(42)
        x = torch.randn((32, 8, self.config.hidden_size), dtype=self.test_dtype).cuda()

        def run(layer):
            inp = x.clone().detach().requires_grad_(True)
            out, _ = layer(inp)  # full fwd: dispatch -> op-fuser experts -> combine
            out.sum().backward()  # bwd: dispatch-bwd reads the symm grad buffer
            return out.detach(), inp.grad.detach()

        def reset_ep():
            # zero_copy mode is fixed at ep_bootstrap (process-global); finalize + drop the shared
            # symm classvars so the next layer re-bootstraps in the other mode.
            nccl_ep_finalize()
            _NCCLEPManager._zc_fwd_token_buf = None
            _NCCLEPManager._zc_bwd_token_buf = None
            _NCCLEPManager._zc_recv_topk_weights_buf = None

        if variant == "zero_copy":
            variant_overrides = dict(moe_ncclep_zero_copy=True)
            rtol = atol = 1e-2
        else:
            assert variant == "mxfp8_wire", f"unknown variant {variant!r}"
            variant_overrides = dict(moe_dispatch_fwd_dtype='mxfp8', moe_combine_bwd_dtype='mxfp8')
            rtol = atol = 2e-1

        ref_layer = self.new_moe_layer()
        out_ref, grad_ref = run(ref_layer)

        reset_ep()
        var_layer = self.new_moe_layer(**variant_overrides)
        var_layer.load_state_dict(ref_layer.state_dict())  # identical weights
        out_var, grad_var = run(var_layer)

        if variant == "zero_copy":
            from transformer_engine.pytorch.ep import is_symm_backed

            # the combine forward buffer must be an allocated, registered symm window
            # (zero-copy engaged)
            fwd_buf = _NCCLEPManager._zc_fwd_token_buf
            assert fwd_buf is not None, "zero-copy forward symm buffer was not allocated"
            assert is_symm_backed(fwd_buf), "zero-copy forward buffer is not symm-mem-backed"
        else:
            # the quant recipes must have reached the dispatch manager (config -> manager wiring);
            # manager -> EpBuffer wiring hard-fails inside fused_dispatch when TE lacks support.
            manager = var_layer.token_dispatcher._comm_manager
            assert manager.dispatch_fwd_quant_recipe is not None
            assert manager.combine_bwd_quant_recipe is not None
        reset_ep()

        assert not torch.isnan(out_var).any() and not torch.isnan(grad_var).any()
        torch.testing.assert_close(out_var, out_ref, rtol=rtol, atol=atol)
        torch.testing.assert_close(grad_var, grad_ref, rtol=rtol, atol=atol)

    @pytest.mark.internal
    def fused_moe_sequential_parity_test(self):
        """Compare MXFP8 MegaMoE with Megatron's split BF16 NCCL-EP path."""
        from megatron.core.transformer.moe.fused_a2a import nccl_ep_finalize

        torch.manual_seed(42)
        x = torch.randn((16, 4, self.config.hidden_size), dtype=self.test_dtype).cuda()
        dy = (torch.randn_like(x, dtype=torch.float32) * 0.1).to(self.test_dtype)

        def run(layer):
            layer.zero_grad(set_to_none=True)
            inp = x.clone().detach().requires_grad_(True)
            with get_fp8_context(layer.config):
                out, _ = layer(inp)
            out.backward(dy)
            grads = {
                name: param.grad.detach().clone()
                for name, param in layer.named_parameters()
                if param.grad is not None
            }
            return out.detach(), inp.grad.detach(), grads

        reference = self.new_moe_layer(moe_use_transformer_engine_fused_moe=False)
        try:
            out_ref, dgrad_ref, grads_ref = run(reference)
        finally:
            nccl_ep_finalize()

        fused = self.new_moe_layer(
            moe_use_transformer_engine_fused_moe=True,
            # NCCL-EP transports MXFP8 gradients as E4M3; HYBRID would select E5M2 in backward.
            fp8="e4m3",
            fp8_recipe="mxfp8",
        )
        fused.load_state_dict(reference.state_dict())
        try:
            out_fused, dgrad_fused, grads_fused = run(fused)

            # Match TE's MXFP8 MegaMoE-vs-reference numerical contract.
            tolerances = {"rtol": 0.125, "atol": 0.25}
            torch.testing.assert_close(out_fused, out_ref, **tolerances)
            torch.testing.assert_close(dgrad_fused, dgrad_ref, **tolerances)
            assert grads_fused.keys() == grads_ref.keys()
            for name in grads_ref:
                torch.testing.assert_close(grads_fused[name], grads_ref[name], **tolerances)

            (sequence,) = fused.experts._last_fused_moe_ops
            op_names = [type(op).__name__ for op in sequence]
            assert op_names == [
                "MoeDispatch",
                "GroupedLinear",
                "ScaledSwiGLU",
                "GroupedLinear",
                "MoeCombine",
            ]
            forward_ops = sequence._module_groups[0]._forward_ops
            is_megamoe = any(
                type(op).__name__ == "FusedMoeEp" for group in forward_ops for op in group
            )
            try:
                from transformer_engine.pytorch.ops.fused.moe_ep import _cudnn_megamoe_supported
            except ImportError:
                megamoe_supported = False
            else:
                megamoe_supported = (
                    torch.cuda.get_device_capability() == (10, 7)
                    and _cudnn_megamoe_supported()
                )
            assert is_megamoe == megamoe_supported
        finally:
            nccl_ep_finalize()

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

        permuted_local_hidden_states, tokens_per_expert, permuted_probs = token_permutation(
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
        permuted_input_1, tokens_per_expert, permuted_probs_1 = token_permutation(
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
        permuted_input_2, tokens_per_expert, permuted_probs_2 = token_permutation(
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
        permuted_input_1, tokens_per_expert_1, permuted_probs_1 = token_permutation(
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
        permuted_input_2, tokens_per_expert_2, permuted_probs_2 = token_permutation(
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


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def is_nccl_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_TE_EP

    return HAVE_TE_EP


def is_nccl_ep_zero_copy_available():
    """Zero-copy needs the newer TE symm-mem APIs (symm_mem_alloc/is_symm_backed), which a plain
    NCCL-EP build lacks -- gate zero-copy tests on these separately from is_nccl_ep_available()."""
    if not is_nccl_ep_available():
        return False
    try:
        from transformer_engine.pytorch.ep import is_symm_backed, symm_mem_alloc  # noqa: F401
    except ImportError:
        return False
    return True


def is_op_fuser_available():
    """The static-shape/zero-copy path runs the TE op-fuser grouped GEMM (needs TE>=2.14 ops)."""
    try:
        from transformer_engine.pytorch.ops import GroupedLinear, ScaledSwiGLU  # noqa: F401
    except ImportError:
        return False
    return is_te_min_version("2.14.0")


def is_fused_moe_sequential_available():
    if not is_nccl_ep_available() or not is_op_fuser_available():
        return False
    try:
        from transformer_engine.pytorch.ep import EpConfig  # noqa: F401
        from transformer_engine.pytorch.ops import MoeCombine, MoeDispatch  # noqa: F401
    except ImportError:
        return False
    return True


def is_nccl_ep_fp8_dispatch_available():
    """MXFP8 wire dtypes need a TE build whose EpBuffer takes the quant recipes AND that returns
    the plain-tensor MXFP8 carrier (mxfp8_carrier_to_grouped, TE PR #3355 -- older quant-recipe
    builds return a GroupedTensor payload the op-fuser attrs cannot rebuild), plus MXFP8 hardware
    support (Blackwell) for the quantize kernels and the grouped GEMM."""
    if not is_nccl_ep_available():
        return False
    import inspect

    try:
        import transformer_engine.pytorch.ep as te_ep
        from transformer_engine.pytorch.fp8 import check_mxfp8_support
    except ImportError:
        return False
    if "dispatch_fwd_quant_recipe" not in inspect.signature(te_ep.EpBuffer).parameters:
        return False
    if not hasattr(te_ep, "mxfp8_carrier_to_grouped"):
        return False
    return check_mxfp8_support()[0]


def test_hybridep_pad_uneven_dispatch_inputs_metadata(monkeypatch):
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = object()
    manager.num_local_experts = 2
    manager.num_experts = 4
    manager.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_hybridep_pad_uneven_dispatch_inputs=True,
    )
    manager.moe_expert_rank_capacity_factor = None
    manager.drop_and_pad = False

    local_num_tokens = 17
    max_num_tokens_across_ep = 70
    padded_num_tokens = (
        max_num_tokens_across_ep + -max_num_tokens_across_ep % HYBRIDEP_TOKEN_ALIGNMENT
    )
    routing_map = torch.ones((local_num_tokens, manager.num_experts), dtype=torch.bool)
    probs = torch.ones((local_num_tokens, manager.num_experts), dtype=torch.float32)

    def fake_all_reduce(tensor, op=None, group=None):
        assert op == torch.distributed.ReduceOp.MAX
        assert group is manager.group
        tensor.fill_(max_num_tokens_across_ep)

    monkeypatch.setattr(torch.distributed, "all_reduce", fake_all_reduce)

    manager.setup_metadata(routing_map, probs)

    assert manager._original_num_tokens == local_num_tokens
    assert manager._padded_num_tokens == padded_num_tokens
    assert manager.routing_map.shape == (padded_num_tokens, manager.num_experts)
    assert manager.token_probs.shape == (padded_num_tokens, manager.num_experts)
    torch.testing.assert_close(manager.routing_map[:local_num_tokens], routing_map)
    torch.testing.assert_close(manager.token_probs[:local_num_tokens], probs)
    assert not manager.routing_map[local_num_tokens:].any()
    assert not manager.token_probs[local_num_tokens:].any()


@pytest.mark.skipif(
    not is_deep_ep_available() and not is_hybrid_ep_available() and not is_nccl_ep_available(),
    reason="No flex dispatcher backend is available",
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
    @pytest.mark.parametrize(
        "moe_flex_dispatcher_backend",
        [
            "deepep",
            "hybridep",
            # NCCL EP aborts in dev CI with a pybind11 GIL dec_ref failure.
            pytest.param("ncclep", marks=pytest.mark.flaky_in_dev),
        ],
    )
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if moe_flex_dispatcher_backend == "ncclep" and not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
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
    @pytest.mark.skipif(
        not is_op_fuser_available(), reason="op-fuser (static-shape/zero-copy) needs TE>=2.14"
    )
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    @pytest.mark.parametrize(
        "variant", ["zero_copy", pytest.param("mxfp8_wire", marks=pytest.mark.launch_on_gb200)]
    )
    def test_forward_backward_variant_parity(self, tp_size, ep_size, variant):
        # The op-fuser needs tp=1 and a SwiGLU activation. Parity: the variant IO path must match
        # the plain staged/eager path.
        # zero_copy requires a capacity factor (fixed symm buffers), which requires BOTH op-fuser
        # and grouped_gemm; bf16 so no fp8/Blackwell dependency.
        # mxfp8_wire runs eager (the validated fp8-wire mode) and needs a TE build with EpBuffer
        # quant recipes plus MXFP8 hardware.
        if variant == "zero_copy" and not is_nccl_ep_zero_copy_available():
            pytest.skip("NCCL EP zero-copy TE API is not available")
        if variant == "mxfp8_wire" and not is_nccl_ep_fp8_dispatch_available():
            pytest.skip("NCCL EP MXFP8 wire needs EpBuffer quant-recipe support and MXFP8 hardware")
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            # ncclep sizes a per-rank recv buffer from this and overflow HARD-TRAPS; size generously.
            moe_expert_rank_capacity_factor=8.0 if variant == "zero_copy" else None,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.moe_layer_variant_parity_test(variant)

    @pytest.mark.skipif(
        not is_fused_moe_sequential_available(),
        reason="TE Dispatch/Combine operation-fuser APIs are not available",
    )
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 4)])
    def test_fused_moe_sequential(self, tp_size, ep_size):
        previous_single_param = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        try:
            self._run_fused_moe_sequential(tp_size, ep_size)
        finally:
            if previous_single_param is None:
                os.environ.pop("NVTE_GROUPED_LINEAR_SINGLE_PARAM", None)
            else:
                os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = previous_single_param

    def _run_fused_moe_sequential(self, tp_size, ep_size):
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            moe_grouped_gemm=True,
            use_transformer_engine_op_fuser=True,
            moe_single_grouped_weight=True,
            gated_linear_unit=True,
            activation_func=F.silu,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.fused_moe_sequential_parity_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_capacity_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
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
            use_cpu_initialization=False,
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
    @pytest.mark.parametrize("moe_flex_dispatcher_backend", ["deepep", "hybridep"])
    @pytest.mark.parametrize("moe_permute_fusion_into_hybridep", [True, False])
    def test_router_padding_for_fp8_forward_backward(
        self,
        tp_size,
        ep_size,
        permute_fusion,
        moe_flex_dispatcher_backend,
        moe_permute_fusion_into_hybridep,
    ):
        if moe_flex_dispatcher_backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if moe_flex_dispatcher_backend == "hybridep" and not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
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
