# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses
import weakref
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import megatron.core.transformer.moe.fused_a2a as fused_a2a
import megatron.core.transformer.moe.token_dispatcher as token_dispatcher
from megatron.core import config, parallel_state
from megatron.core.extensions.transformer_engine import (
    fused_topk_with_score_function_supports_topk_indices,
)
from megatron.core.fp8_utils import get_fp8_context
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.moe.fused_a2a import (
    HAVE_HYBRIDEP_DENSE_ROUTING,
    HYBRIDEP_TOKEN_ALIGNMENT,
    reset_hybrid_ep_buffer,
)
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.transformer.moe.moe_utils import get_capacity, pad_routing_map
from megatron.core.transformer.moe.token_dispatcher import (
    MoEAlltoAllTokenDispatcher,
    MoEFlexTokenDispatcher,
    _DeepepManager,
    _HybridEPManager,
    _NCCLEPManager,
)
from megatron.core.transformer.spec_utils import get_submodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import (
    Utils,
    is_nccl_ep_available,
    is_nccl_ep_fp8_dispatch_available,
    is_nccl_ep_zero_copy_available,
    is_op_fuser_available,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAVE_HYBRIDEP_DENSE_ROUTING,
    reason="requires CUDA and HybridEP dense routing",
)
def test_hybridep_explicit_capacity_and_dynamic_retry():
    """Overflow, recover dynamically, then use an explicit budget without losing gradients."""
    Utils.initialize_model_parallel(expert_model_parallel_size=4)
    try:
        group = parallel_state.get_expert_model_parallel_group()
        model_config = TransformerConfig(
            num_layers=1,
            hidden_size=128,
            num_attention_heads=4,
            num_moe_experts=4,
            expert_model_parallel_size=4,
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="hybridep",
            moe_expert_rank_capacity_factor=1.0,
            moe_grouped_gemm=True,
            moe_use_grouped_tensor=True,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        manager = _HybridEPManager(group, 1, 4, model_config)
        tokens = 256  # Explicit budgets must satisfy grouped-tensor alignment.
        ids = torch.zeros((tokens, 1), device="cuda", dtype=torch.int64)
        probs = torch.zeros((tokens, 4), device="cuda", dtype=torch.float32)
        probs[:, 0] = 1
        # All ranks route to expert 0: a factor of 1 cannot hold the aggregate load.
        for factor, capacity, overflow in (
            (1.0, None, True),
            (None, None, False),  # PagedStashRunner's dropless retry.
            (1.0, None, True),  # Restore the ordinary configured budget.
            (None, tokens, True),  # Explicit budgets must also report overflow.
            (None, tokens * group.size(), False),
            (None, None, False),  # An explicit budget must not leak into the next dispatch.
        ):
            manager.moe_expert_rank_capacity_factor = factor
            manager.over_budget.zero_()
            manager.setup_metadata(ids, probs)
            hidden = torch.full(
                (tokens, 128),
                group.rank() + 1,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            dispatched = manager.dispatch(hidden, rank_capacity=capacity)
            restored = manager.combine(dispatched)
            any_overflow = manager.over_budget.to(torch.int32)
            torch.distributed.all_reduce(any_overflow, group=group)
            assert bool(any_overflow.item()) == overflow
            if not overflow:
                torch.testing.assert_close(restored, hidden, rtol=0, atol=0)
                restored.backward(torch.ones_like(restored))
                torch.testing.assert_close(hidden.grad, torch.ones_like(hidden), rtol=0, atol=0)
        torch.cuda.synchronize()
    finally:
        reset_hybrid_ep_buffer()
        Utils.destroy_model_parallel()


def test_pad_routing_map_does_not_modify_input():
    routing_map = torch.tensor(
        [[True, False], [False, True], [False, False], [False, False]], dtype=torch.bool
    )
    original = routing_map.clone()
    original_version = routing_map._version

    padded = pad_routing_map(routing_map, pad_multiple=2)

    assert torch.equal(routing_map, original)
    assert routing_map._version == original_version
    assert torch.equal(padded.sum(dim=0), torch.tensor([2, 2]))


def test_pad_routing_map_pads_each_expert_to_multiple():
    routing_map = torch.zeros(100, 3, dtype=torch.bool)
    routing_map[:90, 0] = True
    routing_map[:5, 1] = True
    # Expert 2 receives no tokens and must stay empty.

    padded = pad_routing_map(routing_map, pad_multiple=16)

    assert torch.equal(padded.sum(dim=0), torch.tensor([96, 16, 0]))
    # Padding only ever flips zeros to ones.
    assert padded[routing_map].all()


def test_pad_routing_map_uses_every_remaining_token_when_needed():
    # Expert 0 needs all 38 remaining tokens to reach 128, which is still feasible.
    routing_map = torch.zeros(128, 2, dtype=torch.bool)
    routing_map[:90, 0] = True

    padded = pad_routing_map(routing_map, pad_multiple=128)

    assert torch.equal(padded.sum(dim=0), torch.tensor([128, 0]))


def test_pad_routing_map_raises_when_not_enough_tokens():
    # Expert 0 has 90 of 100 tokens, so it cannot be padded to 128.
    routing_map = torch.zeros(100, 2, dtype=torch.bool)
    routing_map[:90, 0] = True

    with pytest.raises(RuntimeError, match="not enough tokens"):
        pad_routing_map(routing_map, pad_multiple=128)


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
            moe_router_fusion=kwargs.get("moe_router_fusion", False),
            moe_flex_dispatcher_backend=kwargs.get("moe_flex_dispatcher_backend", None),
            moe_hybridep_routing_map_mode=kwargs.get("moe_hybridep_routing_map_mode", "bool"),
            moe_expert_rank_capacity_factor=kwargs.get("moe_expert_rank_capacity_factor", None),
            moe_ncclep_zero_copy=kwargs.get("moe_ncclep_zero_copy", False),
            moe_dispatch_fwd_dtype=kwargs.get("moe_dispatch_fwd_dtype", 'bf16'),
            moe_combine_bwd_dtype=kwargs.get("moe_combine_bwd_dtype", 'bf16'),
            use_transformer_engine_op_fuser=kwargs.get("use_transformer_engine_op_fuser", False),
            gated_linear_unit=kwargs.get("gated_linear_unit", False),
            activation_func=kwargs.get("activation_func", F.gelu),
            fp8=kwargs.get("fp8", None),
            fp8_recipe=kwargs.get("fp8_recipe", "delayed"),
            calculate_per_token_loss=kwargs.get("calculate_per_token_loss", False),
            max_seq_length_per_sample=kwargs.get("max_seq_length_per_sample", None),
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
    def dispatcher_thd_uneven_tokens_test(self, seq_length):
        """Packed (THD) input: per-rank token counts differ and change between dispatches. The
        dispatcher sizes its buffers once from seq_length and must round-trip every count under
        it, and reject a count above it."""
        moe_layer = self.moe_layer
        dispatcher = moe_layer.token_dispatcher
        rank = torch.distributed.get_rank()
        packed_seq_params = PackedSeqParams(qkv_format="thd")

        def round_trip(num_tokens):
            hidden_states = torch.randn(
                (num_tokens, 1, moe_layer.config.hidden_size), dtype=self.test_dtype, device="cuda"
            )
            hidden_states.requires_grad = True
            probs, indices = apply_module(moe_layer.router)(hidden_states)
            probs = torch.ones_like(probs) / moe_layer.router.topk
            permuted, permuted_probs = dispatcher.dispatch_preprocess(
                hidden_states, indices, probs, packed_seq_params=packed_seq_params
            )
            permuted, permuted_probs = dispatcher.token_dispatch(permuted, permuted_probs)
            permuted, _, permuted_probs = dispatcher.dispatch_postprocess(permuted, permuted_probs)
            permuted = (permuted * permuted_probs.unsqueeze(-1)).to(dtype=self.test_dtype)
            restored, _ = token_unpermutation(dispatcher, permuted)
            torch.testing.assert_close(restored, hidden_states)
            torch.autograd.backward(restored, hidden_states)
            torch.testing.assert_close(hidden_states.grad, hidden_states)

        round_trip(64 + 16 * rank)
        round_trip(seq_length - 8 * rank)
        bound = dispatcher._comm_manager._max_tokens_per_rank
        assert bound == seq_length, f"bound {bound} should come from seq_length {seq_length}"
        with pytest.raises(RuntimeError, match="sized for"):
            round_trip(bound + 64)

    @pytest.mark.internal
    def dispatcher_thd_parity_vs_alltoall_test(self, rtol, atol):
        """Full MoE-layer fwd+bwd on packed (THD) input with a padding mask and a different token
        count per rank: ncclep against the alltoall dispatcher with identical weights. ncclep
        routes the padded rows to -1 and never ships them, so the reference is alltoall run on
        the valid rows alone (the alltoall dispatcher has no padding handling of its own). The
        valid rows' outputs and input grads and every parameter grad must agree within
        tolerance; ncclep's padded rows must come out exactly zero."""
        rank = torch.distributed.get_rank()
        torch.manual_seed(1000 + rank)
        num_tokens = 64 + 16 * rank
        hidden_size = self.config.hidden_size
        x = torch.randn((num_tokens, 1, hidden_size), dtype=self.test_dtype, device="cuda")
        padding_mask = torch.zeros((1, num_tokens), dtype=torch.bool, device="cuda")
        padding_mask[0, num_tokens - 4 - rank :] = True
        padding_rows = padding_mask[0]
        valid_rows = ~padding_rows
        packed_seq_params = PackedSeqParams(qkv_format="thd")

        def run(layer, inputs, **forward_kwargs):
            inp = inputs.clone().detach().requires_grad_(True)
            out, _ = layer(inp, **forward_kwargs)
            out.sum().backward()
            param_grads = {
                name: p.grad.detach().clone()
                for name, p in layer.named_parameters()
                if p.grad is not None
            }
            return out.detach(), inp.grad.detach(), param_grads

        ref_layer = self.new_moe_layer(
            moe_token_dispatcher_type="alltoall", moe_flex_dispatcher_backend=None
        )
        out_ref, grad_ref, param_grads_ref = run(ref_layer, x[valid_rows])
        ncclep_layer = self.moe_layer
        ncclep_layer.load_state_dict(ref_layer.state_dict())
        # NCCL EP reads NCCL_EP_HT_EM_PULL_PUSH when its process-wide group is created, and
        # earlier tests may already have created it: rebuild under this test's environment,
        # and again afterwards so later tests do not inherit the mode.
        from megatron.core.transformer.moe.token_dispatcher import nccl_ep_release_context

        nccl_ep_release_context()
        try:
            out, grad, param_grads = run(
                ncclep_layer, x, padding_mask=padding_mask, packed_seq_params=packed_seq_params
            )
        finally:
            nccl_ep_release_context()

        assert ncclep_layer.token_dispatcher._comm_manager._is_thd
        assert not torch.isnan(out).any() and not torch.isnan(grad).any()
        assert torch.equal(out[padding_rows], torch.zeros_like(out[padding_rows]))
        assert torch.equal(grad[padding_rows], torch.zeros_like(grad[padding_rows]))
        out, grad = out[valid_rows], grad[valid_rows]

        def report(name, actual, expected):
            diff = (actual.float() - expected.float()).abs()
            rel = diff / expected.float().abs().clamp_min(1e-6)
            if rank == 0:
                print(
                    f"[thd parity {self.test_dtype}] {name}: max_abs={diff.max().item():.3e} "
                    f"max_rel={rel.max().item():.3e} ref_absmax={expected.abs().max().item():.3e}"
                )
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, msg=name)

        report("output", out, out_ref)
        report("input_grad", grad, grad_ref)
        assert param_grads.keys() == param_grads_ref.keys()
        for name in sorted(param_grads):
            report(f"grad[{name}]", param_grads[name], param_grads_ref[name])

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


def is_deep_ep_v2_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP_V2

    return HAVE_DEEP_EP_V2


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def test_hybridep_pad_uneven_dispatch_inputs_metadata(monkeypatch):
    manager = _HybridEPManager.__new__(_HybridEPManager)
    manager.group = object()
    manager.num_local_experts = 2
    manager.num_experts = 4
    manager.router_topk = 2
    manager.config = TransformerConfig(
        num_layers=1,
        hidden_size=16,
        num_attention_heads=4,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_hybridep_pad_uneven_dispatch_inputs=True,
    )
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = token_dispatcher.HAVE_HYBRIDEP_DENSE_ROUTING
    manager.use_dense_routing_for_bool_map = (
        manager.dense_routing_supported
        and manager.config.moe_hybridep_routing_map_mode == "indices"
    )

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


def _ncclep_manager_stub(**config_overrides):
    """An _NCCLEPManager with the state setup_metadata and _ensure_bootstrap read, no NCCL EP."""
    manager = _NCCLEPManager.__new__(_NCCLEPManager)
    manager.group = object()
    manager.num_local_experts = 2
    manager.num_experts = 4
    manager.router_topk = 2
    manager.hidden_dim = 16
    manager.alignment = 0
    manager.moe_expert_rank_capacity_factor = None
    manager._recv_capacity_override = None
    manager._bootstrapped = False
    manager._is_thd = False
    cfg = dict(
        moe_ncclep_zero_copy=False,
        fp8=None,
        fp4=None,
        moe_flex_dispatcher_num_sms=None,
        overlap_moe_expert_parallel_comm=False,
        tensor_model_parallel_size=1,
        context_parallel_size=1,
        sequence_parallel=False,
        max_seqlen_per_dp_cp_rank=None,
        max_seq_length_per_sample=None,
    )
    cfg.update(config_overrides)
    manager.config = SimpleNamespace(**cfg)
    return manager


@pytest.mark.parametrize("routing_mode", ["bool", "indices"])
def test_ncclep_padding_mask_marks_routes_invalid(routing_mode):
    manager = _ncclep_manager_stub()
    seq_len, batch = 3, 2  # tokens flattened from sequence-first [s, b]
    num_tokens = seq_len * batch
    # Experts 0 and 1 are the top-2 of every row, in that order (topk sorts by probability).
    probs = torch.tensor([[0.6, 0.3, 0.07, 0.03]] * num_tokens)
    if routing_mode == "bool":
        routing_map = torch.zeros(num_tokens, manager.num_experts, dtype=torch.bool)
        routing_map[:, :2] = True
    else:
        routing_map = torch.tensor([[0, 1]] * num_tokens, dtype=torch.int32)
    padding_mask = torch.zeros(batch, seq_len, dtype=torch.bool)  # batch-first [b, s]
    padding_mask[1, 2] = True  # (s=2, b=1) -> flattened row 2 * batch + 1
    padded_row = 2 * batch + 1

    manager.setup_metadata(
        routing_map, probs, padding_mask=padding_mask, packed_seq_params=PackedSeqParams("thd")
    )

    assert manager.num_local_tokens == num_tokens
    assert manager._is_thd
    assert manager.token_indices.shape == (num_tokens, manager.router_topk)
    assert (manager.token_indices[padded_row] == -1).all()
    assert (manager.token_probs[padded_row] == 0).all()
    valid = torch.arange(num_tokens) != padded_row
    torch.testing.assert_close(
        manager.token_indices[valid], torch.tensor([[0, 1]] * (num_tokens - 1))
    )
    torch.testing.assert_close(manager.token_probs[valid], probs[valid][:, :2])


@pytest.mark.parametrize(
    "is_thd,num_local_tokens,config_overrides,expected",
    [
        # BSHD: the first dispatch's count, rounded up to the HT chunk.
        (False, 100, dict(max_seq_length_per_sample=4096), 128),
        # THD: max_seqlen_per_dp_cp_rank wins over max_seq_length_per_sample; SP slice.
        (
            True,
            100,
            dict(
                max_seqlen_per_dp_cp_rank=1000,
                max_seq_length_per_sample=4096,
                tensor_model_parallel_size=2,
                sequence_parallel=True,
            ),
            512,
        ),
        # THD: max_seq_length_per_sample split over CP and the sequence-parallel TP slice.
        (
            True,
            100,
            dict(
                max_seq_length_per_sample=4096,
                context_parallel_size=2,
                tensor_model_parallel_size=2,
                sequence_parallel=True,
            ),
            1024,
        ),
        # THD: no sequence parallel, TP does not shrink the per-rank count.
        (
            True,
            100,
            dict(
                max_seq_length_per_sample=4096,
                context_parallel_size=2,
                tensor_model_parallel_size=2,
            ),
            2048,
        ),
    ],
)
def test_ncclep_bootstrap_token_bound(
    monkeypatch, is_thd, num_local_tokens, config_overrides, expected
):
    bootstrap_kwargs = {}
    monkeypatch.setattr(token_dispatcher, "is_nccl_ep_bootstrapped", lambda: False)
    monkeypatch.setattr(
        token_dispatcher,
        "ensure_nccl_ep_bootstrapped",
        lambda *args, **kwargs: bootstrap_kwargs.update(kwargs),
    )
    manager = _ncclep_manager_stub(**config_overrides)
    manager.num_local_tokens = num_local_tokens
    manager._is_thd = is_thd

    manager._ensure_bootstrap()

    assert manager._max_tokens_per_rank == expected
    assert bootstrap_kwargs["max_tokens_per_rank"] == expected


def test_ncclep_thd_requires_a_sequence_length_bound(monkeypatch):
    monkeypatch.setattr(token_dispatcher, "is_nccl_ep_bootstrapped", lambda: False)
    manager = _ncclep_manager_stub()
    manager.num_local_tokens = 100
    manager._is_thd = True
    with pytest.raises(ValueError, match="max_seqlen_per_dp_cp_rank"):
        manager._ensure_bootstrap()


def test_ncclep_thd_rejects_first_microbatch_above_the_bound(monkeypatch):
    """The bound must be identical on every rank for the NCCL EP bootstrap, so a rank above it
    fails before the collective instead of silently enlarging its own buffers."""
    bootstrap_calls = []
    monkeypatch.setattr(token_dispatcher, "is_nccl_ep_bootstrapped", lambda: False)
    monkeypatch.setattr(
        token_dispatcher, "ensure_nccl_ep_bootstrapped", lambda *a, **kw: bootstrap_calls.append(kw)
    )
    manager = _ncclep_manager_stub(max_seq_length_per_sample=2048)
    manager.num_local_tokens = 3000
    manager._is_thd = True
    with pytest.raises(RuntimeError, match="per-rank bound from max_seq_length_per_sample"):
        manager._ensure_bootstrap()
    assert not bootstrap_calls


def test_ncclep_fixed_length_packed_input_uses_the_measured_bound():
    """tokens_per_sample marks fixed-length samples (inter-document masking, any micro batch
    size): their per-rank count is fixed like BSHD, so the config bound is not required."""
    manager = _ncclep_manager_stub()
    routing_map = torch.tensor([[0, 1]] * 4, dtype=torch.int32)
    probs = torch.full((4, manager.num_experts), 0.25)
    manager.setup_metadata(
        routing_map, probs, packed_seq_params=PackedSeqParams(qkv_format="thd", tokens_per_sample=2)
    )
    assert not manager._is_thd
    manager.setup_metadata(routing_map, probs, packed_seq_params=PackedSeqParams(qkv_format="thd"))
    assert manager._is_thd


@pytest.mark.parametrize("is_thd", [False, True])
def test_ncclep_dispatch_guard_names_the_right_remedy(monkeypatch, is_thd):
    monkeypatch.setattr(token_dispatcher, "is_nccl_ep_bootstrapped", lambda: True)
    manager = _ncclep_manager_stub()
    manager._bootstrapped = True
    manager._max_tokens_per_rank = 128
    manager._is_thd = is_thd
    with pytest.raises(RuntimeError, match="sized for") as excinfo:
        manager.dispatch(torch.zeros(192, manager.hidden_dim))
    message = str(excinfo.value)
    assert ("max_seq_length_per_sample" in message) is is_thd
    assert ("first microbatch" in message) is not is_thd


def test_hybridep_sparse_fallback_marks_empty_routes_invalid(monkeypatch):
    monkeypatch.setattr(token_dispatcher, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False, moe_hybridep_routing_map_mode="indices"
    )
    manager.group = object()
    manager.num_experts = 2
    manager.router_topk = 1
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = token_dispatcher.HAVE_HYBRIDEP_DENSE_ROUTING
    manager.use_dense_routing_for_bool_map = (
        manager.dense_routing_supported
        and manager.config.moe_hybridep_routing_map_mode == "indices"
    )

    routing_map = torch.tensor([[True, False], [False, False]])
    probs = torch.tensor([[1.0, 0.0], [0.0, 0.0]])

    manager.setup_metadata(routing_map, probs)

    assert torch.equal(manager.topk_idx, torch.tensor([[0], [-1]], dtype=torch.int16))


def test_hybridep_indices_mode_honors_bool_map_with_full_width_probs(monkeypatch):
    """Dense ids must follow the bool map even when probs are nonzero outside the selected
    routes (direct callers and the dispatcher test helper pass full-width weights); a topk over
    probs would pick expert 1 here and drop the token's only real route."""
    monkeypatch.setattr(token_dispatcher, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False, moe_hybridep_routing_map_mode="indices"
    )
    manager.group = object()
    manager.num_experts = 4
    manager.router_topk = 2
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = token_dispatcher.HAVE_HYBRIDEP_DENSE_ROUTING
    manager.use_dense_routing_for_bool_map = (
        manager.dense_routing_supported
        and manager.config.moe_hybridep_routing_map_mode == "indices"
    )

    routing_map = torch.tensor(
        [[True, False, False, True], [False, True, False, False], [False, False, False, False]]
    )
    probs = torch.tensor([[0.1, 0.5, 0.3, 0.1], [0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]])

    manager.setup_metadata(routing_map, probs)

    assert manager.topk_idx.dtype == torch.int16
    routes = [sorted(row[row >= 0].tolist()) for row in manager.topk_idx]
    assert routes == [[0, 3], [1], []]
    assert (manager.topk_idx[1] == -1).sum() == 1 and (manager.topk_idx[2] == -1).all()
    # Full-width probs are passed through untouched for HybridEP to gather by index.
    assert torch.equal(manager.token_probs, probs)


@pytest.mark.parametrize("flex", [False, True])
def test_moe_layer_preprocess_forwards_padding_mask_only_to_flex_dispatcher(flex):
    """MoELayer.preprocess hands the batch-first mask and the packed-sequence metadata to the
    flex dispatcher (whose backend decides what to do with them); the other dispatchers keep
    their 3-arg call."""
    calls = []

    class _FlexSpy(MoEFlexTokenDispatcher):
        def __init__(self):
            pass

        def dispatch_preprocess(
            self, hidden_states, routing_map, probs, padding_mask=None, packed_seq_params=None
        ):
            calls.append((padding_mask, packed_seq_params))
            return hidden_states, probs

    class _OtherSpy:
        def dispatch_preprocess(self, hidden_states, routing_map, probs):
            calls.append("no-mask")
            return hidden_states, probs

    layer = object.__new__(MoELayer)
    torch.nn.Module.__init__(layer)
    layer.config = SimpleNamespace(cuda_graph_impl="local", moe_latent_size=None)
    layer.shared_expert_overlap = False
    layer.token_dispatcher = _FlexSpy() if flex else _OtherSpy()
    hidden_states = torch.zeros(4, 8)
    probs = torch.zeros(4, 2)
    routing_map = torch.zeros(4, 2, dtype=torch.bool)
    padding_mask = torch.tensor([[False, True], [False, False]])
    packed_seq_params = PackedSeqParams(qkv_format="thd")

    layer.preprocess(
        hidden_states, probs, routing_map, padding_mask, packed_seq_params=packed_seq_params
    )

    if flex:
        assert len(calls) == 1
        assert calls[0][0] is padding_mask and calls[0][1] is packed_seq_params
    else:
        assert calls == ["no-mask"]


def test_alltoall_combine_postprocess_releases_router_probs():
    """combine_postprocess drops the dispatcher's reference to the router probs, so their
    autograd graph is not kept alive until the next forward (under full recompute, through
    the rest of backward). The combined output is unchanged.
    """
    dispatcher = object.__new__(MoEAlltoAllTokenDispatcher)
    dispatcher.config = SimpleNamespace(moe_permute_fusion=False)
    dispatcher.drop_and_pad = False
    dispatcher.batch_invariant_inverse_permutation_mapping = None
    dispatcher.shared_experts = None
    num_tokens, hidden, topk, num_experts = 3, 4, 2, 2
    dispatcher.hidden_shape = torch.Size([num_tokens, 1, hidden])
    dispatcher.hidden_shape_before_permute = torch.Size([num_tokens, hidden])
    dispatcher.routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool)
    # Permuted rows are grouped by expert: expert 0 gets tokens [0, 1, 2], then expert 1.
    dispatcher.reversed_local_input_permutation_mapping = torch.tensor([0, 1, 2, 0, 1, 2])

    logits = torch.randn(num_tokens, num_experts, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)  # non-leaf, carries the router's grad_fn
    probs_ref = weakref.ref(probs)
    dispatcher.probs = probs
    del probs

    permuted = torch.arange(num_tokens * topk * hidden, dtype=torch.float32).view(-1, hidden)
    output = dispatcher.combine_postprocess(permuted)

    expected = (permuted[:num_tokens] + permuted[num_tokens:]).view(num_tokens, 1, hidden)
    torch.testing.assert_close(output, expected)
    assert dispatcher.probs is None
    assert probs_ref() is None, "the router probs (and their graph) are still referenced"


def test_hybridep_dropless_excludes_padding_rows(monkeypatch):
    """Dropless HybridEP drops padded tokens in the dispatcher (the router keeps them routed):
    padding rows lose their routes and weights, valid rows are untouched. The mask arrives
    batch-first [b, s] and is transposed to the sequence-first token order."""
    monkeypatch.setattr(token_dispatcher, "HAVE_HYBRIDEP_DENSE_ROUTING", False)
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False, moe_hybridep_routing_map_mode="bool"
    )
    manager.group = object()
    manager.num_experts = 2
    manager.router_topk = 1
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = False
    manager.use_dense_routing_for_bool_map = False

    # s=2, b=2 -> 4 tokens in sequence-first order (s0b0, s0b1, s1b0, s1b1).
    routing_map = torch.tensor([[True, False], [False, True], [True, False], [False, True]])
    probs = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.4, 0.6]])
    # Batch-first mask: sample 1 has its second position padded -> token s1b1.
    padding_mask = torch.tensor([[False, False], [False, True]])

    manager.setup_metadata(routing_map, probs, padding_mask=padding_mask)

    expected_map = routing_map.clone()
    expected_map[3] = False
    assert torch.equal(manager.routing_map, expected_map)
    # Dropless setup_metadata leaves tokens_per_expert to dispatch(); count from the map.
    assert torch.equal(manager.routing_map.sum(dim=0), torch.tensor([2, 1]))
    assert torch.count_nonzero(manager.token_probs[3]) == 0
    torch.testing.assert_close(manager.token_probs[:3], probs[:3])


def test_hybridep_indices_mode_keeps_bool_map_with_pad_to_capacity(monkeypatch):
    """With pad-to-capacity the routing map is the capacity mask (a token can carry more than
    topk assignments); the indices-mode topk reconstruction would drop the padded assignments
    while tokens_per_expert still declares the full capacity, so the bool map must be kept."""
    monkeypatch.setattr(token_dispatcher, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    num_tokens, num_experts, topk, capacity_factor = 64, 2, 1, 2.0
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False,
        moe_hybridep_routing_map_mode="indices",
        moe_router_topk=topk,
    )
    manager.group = SimpleNamespace(size=lambda: 1)
    manager.num_experts = num_experts
    manager.num_local_experts = num_experts
    manager.router_topk = topk
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = capacity_factor
    manager.drop_and_pad = True
    manager.dense_routing_supported = True
    # Pad-to-capacity keeps the bool map even in "indices" mode.
    manager.use_dense_routing_for_bool_map = False

    # Every token picked expert 0; pad-to-capacity fills expert 1 with the same tokens.
    capacity = get_capacity(num_tokens * topk, num_experts, capacity_factor)
    assert capacity == num_tokens
    routing_map = torch.ones(num_tokens, num_experts, dtype=torch.bool)
    probs = torch.zeros(num_tokens, num_experts)
    probs[:, 0] = 1.0

    manager.setup_metadata(routing_map, probs)

    assert manager.topk_idx is None
    assert torch.equal(manager.routing_map.sum(dim=0), torch.tensor([capacity, capacity]))
    assert torch.equal(manager.tokens_per_expert, torch.tensor([capacity, capacity]))


def test_hybridep_dense_input_requires_backend_support(monkeypatch):
    monkeypatch.setattr(token_dispatcher, "HAVE_HYBRIDEP_DENSE_ROUTING", False)
    manager = object.__new__(_HybridEPManager)
    manager.config = SimpleNamespace(
        moe_hybridep_pad_uneven_dispatch_inputs=False, moe_hybridep_routing_map_mode="indices"
    )
    manager.group = object()
    manager.num_experts = 4
    manager.router_topk = 2
    manager.moe_expert_rank_capacity_factor = None
    manager.capacity_factor = None
    manager.drop_and_pad = False
    manager.dense_routing_supported = token_dispatcher.HAVE_HYBRIDEP_DENSE_ROUTING
    manager.use_dense_routing_for_bool_map = (
        manager.dense_routing_supported
        and manager.config.moe_hybridep_routing_map_mode == "indices"
    )

    with pytest.raises(RuntimeError, match="does not support dense topk_idx metadata"):
        manager.setup_metadata(torch.tensor([[0, 2]], dtype=torch.int16), torch.ones(1, 4))


def test_flex_dense_metadata_preserves_invalid_routes():
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher.tp_size = 2
    dispatcher.ep_size = 2
    dispatcher.num_local_experts = 2
    # HybridEP dense indices come with full-width [num_tokens, num_experts] probs.
    dispatcher.config = SimpleNamespace(moe_flex_dispatcher_backend="hybridep")
    routing_map = torch.tensor([[0, -1], [3, 1]], dtype=torch.int16)
    probs = torch.ones((2, 4))

    expanded_routes, expanded_probs = dispatcher._initialize_metadata(routing_map, probs)

    expected = torch.tensor([[0, 2, -1, -1], [5, 7, 1, 3]], dtype=torch.int16)
    assert torch.equal(expanded_routes, expected)
    # Full-width probs keep the [num_tokens, world_size, num_local_experts] expansion.
    assert expanded_probs.shape == (2, 4, 2)


@pytest.mark.parametrize("backend", ["deepep", "ncclep"])
def test_flex_dense_metadata_keeps_full_width_probs(backend):
    """deepep/ncclep: dense indices arrive with the router's full-width probs, which are expanded
    like a bool map; the backend manager selects the per-route weights afterwards."""
    dispatcher = object.__new__(MoEFlexTokenDispatcher)
    dispatcher.tp_size = 2
    dispatcher.ep_size = 2
    dispatcher.num_local_experts = 2
    dispatcher.config = SimpleNamespace(moe_flex_dispatcher_backend=backend)
    routing_map = torch.tensor([[0, 3], [3, 1]], dtype=torch.int64)
    probs = torch.tensor([[0.6, 0.0, 0.0, 0.4], [0.0, 0.3, 0.0, 0.7]])

    expanded_routes, expanded_probs = dispatcher._initialize_metadata(routing_map, probs)

    assert torch.equal(expanded_routes, torch.tensor([[0, 2, 5, 7], [5, 7, 1, 3]]))
    # [num_tokens, world_size, num_local_experts], one copy per expert-TP rank.
    assert expanded_probs.shape == (2, 4, 2)
    torch.testing.assert_close(expanded_probs[:, 0, :], torch.tensor([[0.6, 0.0], [0.0, 0.3]]))
    torch.testing.assert_close(expanded_probs[:, 2, :], torch.tensor([[0.0, 0.4], [0.0, 0.7]]))


@pytest.mark.parametrize("manager_cls", [_DeepepManager, _NCCLEPManager])
def test_dense_required_manager_selects_weights_at_dense_indices(manager_cls):
    """The manager, not the router, gathers the per-route weights from the full-width probs;
    -1 routes get zero weight."""
    manager = object.__new__(manager_cls)
    manager.num_experts = 4
    manager.router_topk = 2
    manager.capacity_factor = None
    manager.config = SimpleNamespace(moe_expert_capacity_factor=None)
    routing_map = torch.tensor([[0, 3], [2, -1]], dtype=torch.int64)
    probs = torch.tensor([[0.6, 0.1, 0.1, 0.4], [0.2, 0.2, 0.7, 0.9]])

    manager.setup_metadata(routing_map, probs)

    assert manager.token_indices.dtype == torch.int64
    assert torch.equal(manager.token_indices, routing_map)
    torch.testing.assert_close(manager.token_probs, torch.tensor([[0.6, 0.4], [0.7, 0.0]]))


def test_hybridep_dispatch_passes_dense_routing_metadata(monkeypatch):
    class FakeHybridEPBuffer:
        def __init__(self):
            self.kwargs = None

        def dispatch_with_permute(self, **kwargs):
            self.kwargs = kwargs
            return (
                kwargs["hidden"],
                kwargs["probs"],
                None,
                torch.ones(2, dtype=torch.int32),
                ("handle",),
            )

    fake_buffer = FakeHybridEPBuffer()
    monkeypatch.setattr(fused_a2a, "_hybrid_ep_buffer", fake_buffer)
    monkeypatch.setattr(fused_a2a, "HAVE_HYBRIDEP_DENSE_ROUTING", True)
    hidden = torch.randn(2, 4)
    probs = torch.randn(2, 4)
    topk_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int16)

    fused_a2a.HybridEPDispatch.forward(
        SimpleNamespace(), hidden, None, probs, object(), 2, topk_idx=topk_idx, num_of_experts=4
    )

    assert fake_buffer.kwargs["topk_idx"] is topk_idx
    assert fake_buffer.kwargs["num_of_experts"] == 4
    assert "routing_map" not in fake_buffer.kwargs
    assert "dense_routing" not in fake_buffer.kwargs


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
    @pytest.mark.parametrize("routing_map_mode", ["bool", "indices"])
    @pytest.mark.parametrize("moe_router_fusion", [True, False])
    def test_hybridep_routing_map_modes_forward_backward(self, routing_map_mode, moe_router_fusion):
        """With router fusion the router hands HybridEP dense ids directly; without it the router
        emits a bool map and indices mode must rebuild the ids from that map (the test helper's
        uniform full-width probs would otherwise select unrouted experts)."""
        if not is_hybrid_ep_available():
            pytest.skip("Hybrid EP is not available")
        if routing_map_mode == "indices" and not HAVE_HYBRIDEP_DENSE_ROUTING:
            pytest.skip("Dense HybridEP routing API is not available")
        if (
            routing_map_mode == "indices"
            and moe_router_fusion
            and not fused_topk_with_score_function_supports_topk_indices
        ):
            pytest.skip("Dense TE fused top-k routing API is not available")

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=8,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_router_fusion=moe_router_fusion,
            moe_flex_dispatcher_backend="hybridep",
            moe_hybridep_routing_map_mode=routing_map_mode,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("backend", ["deepep", "ncclep"])
    def test_dense_required_backend_forward_backward(self, backend):
        if backend == "deepep" and not is_deep_ep_available():
            pytest.skip("Deep EP is not available")
        if backend == "ncclep" and not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        if not fused_topk_with_score_function_supports_topk_indices:
            pytest.skip("Dense TE routing output is not available")

        container = MoEModelTestContainer(
            tp_size=1,
            ep_size=8,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_router_fusion=True,
            moe_flex_dispatcher_backend=backend,
            hidden_size=1024,
            test_dtype=torch.bfloat16,
        )
        container.dispatcher_dropless_test()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8), (8, 1), (4, 2)])
    @pytest.mark.parametrize("permute_fusion", permute_fusion_params)
    @pytest.mark.parametrize(
        "moe_flex_dispatcher_backend", ["deepep", "deepepv2", "hybridep", "ncclep"]
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
        if moe_flex_dispatcher_backend == "deepepv2" and not is_deep_ep_v2_available():
            pytest.skip("Deep EP v2 is not available")
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

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    def test_ncclep_thd_uneven_tokens(self, tp_size, ep_size):
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        seq_length = 256
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            hidden_size=1024,
            test_dtype=torch.bfloat16,
            max_seq_length_per_sample=seq_length,
        )
        container.dispatcher_thd_uneven_tokens_test(seq_length)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.internal
    @pytest.mark.timeout(120)
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 8)])
    def test_ncclep_thd_parity_vs_alltoall(self, monkeypatch, tp_size, ep_size):
        # NCCL EP carries bf16 (or MXFP8) payloads only, so the comparison runs in bf16. In
        # eager mode the pull-push EM mode builds its expert-major slot map with the
        # order-preserving routing-map scan, so every received row has a fixed slot and the
        # comparison is bit-exact. The default local-permute mode builds the map in-kernel
        # with atomic per-expert cursors, so the row order inside an expert zone depends on
        # scheduling and the token-reduced weight grads can drift by one bf16 ulp.
        if not is_nccl_ep_available():
            pytest.skip("NCCL EP is not available")
        monkeypatch.setenv("NCCL_EP_HT_EM_PULL_PUSH", "1")
        rtol = atol = 0.0
        test_dtype = torch.bfloat16
        container = MoEModelTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            # The reference sees only the valid rows; an aux loss would make the router grads
            # depend on the token count rather than on the dispatcher.
            moe_aux_loss_coeff=0.0,
            moe_token_dispatcher_type="flex",
            moe_flex_dispatcher_backend="ncclep",
            hidden_size=1024,
            test_dtype=test_dtype,
            max_seq_length_per_sample=256,
        )
        container.dispatcher_thd_parity_vs_alltoall_test(rtol=rtol, atol=atol)
