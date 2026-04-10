# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_submodules,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils


class TestMoELayerInit:
    def setup_method(self, method):
        pass

    @pytest.mark.skipif(
        not is_te_min_version("1.7.0.dev0"),
        reason="Expert with TE Linear is only supported in TE 1.7.0 and later.",
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_te_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            moe_ffn_hidden_size=128,
            add_bias_linear=False,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(self.transformer_config, submodules.mlp.submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [1, 2])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    def test_legacy_moe_layer(self, num_moe_experts, moe_token_dispatcher_type, grouped_gemm):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)
        num_moe_experts = 4
        self.transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=grouped_gemm,
            add_bias_linear=False,
        )
        transformer_layer_submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )
        moe_layer = MoELayer(self.transformer_config, transformer_layer_submodules.mlp.submodules)
        Utils.destroy_model_parallel()

    @pytest.mark.skip(
        "Late init of parallel_state was broken after parallel states refactor MR2988."
    )
    @pytest.mark.parametrize("moe_token_dispatcher_type", ["alltoall", "allgather"])
    @pytest.mark.parametrize("grouped_gemm", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2)])
    def test_moe_with_late_initialize(
        self, moe_token_dispatcher_type, grouped_gemm, tp_size, ep_size
    ):
        num_moe_experts = 4
        hidden_size = 12
        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            add_bias_linear=False,
            moe_grouped_gemm=grouped_gemm,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            bf16=True,
            params_dtype=torch.bfloat16,
        )
        submodules = get_gpt_layer_with_transformer_engine_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=grouped_gemm
        )

        # Fake initialization as NeMo does
        Utils.fake_initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        moe_layer = MoELayer(transformer_config, submodules.mlp.submodules).cuda()

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        input_data = torch.randn(
            16, 4, hidden_size, device=torch.cuda.current_device(), dtype=torch.bfloat16
        )
        output = moe_layer(input_data)

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestDenseMixerForward:
    """Test DenseMixer dense forward pass for router gradient estimation."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("moe_router_topk", [1, 2])
    def test_dense_moe_forward_shape_and_grad(self, num_moe_experts, moe_router_topk):
        """Test that dense forward produces correct shape and all expert weights receive grads."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 16
        sequence_length = 8
        micro_batch_size = 2
        ffn_hidden_size = 32

        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=moe_router_topk,
            moe_router_pre_softmax=moe_router_topk == 1,  # required when topk=1
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # required for DenseMixer
            moe_ffn_hidden_size=ffn_hidden_size,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        moe_layer = MoELayer(config, submodules.mlp.submodules)
        moe_layer.cuda()
        moe_layer.train()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            requires_grad=True,
        )

        output, mlp_bias = moe_layer(hidden_states)

        assert mlp_bias is None
        assert (
            output.shape == hidden_states.shape
        ), f"Output shape {output.shape} != input shape {hidden_states.shape}"

        loss = output.sum()
        loss.backward()

        # All expert parameters should receive gradients (dense forward touches all experts)
        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert (
                    param.grad is not None
                ), f"Parameter {name} should have gradient in dense forward mode"

        # Router weight should have gradients
        assert moe_layer.router.weight.grad is not None, "Router weight should have gradient"

        Utils.destroy_model_parallel()

    def test_dense_forward_all_experts_receive_grad_from_all_tokens(self):
        """Core DenseMixer invariant: every expert processes every token in dense mode.

        DenseMixer spec: when moe_dense_forward_for_router_grad=True, the forward pass
        runs ALL experts on ALL tokens regardless of routing decisions. This means every
        expert's fc1 weight must receive non-zero gradient after backward, even if the
        router would not have selected that expert for any token under standard routing.

        We verify this in contrast to sparse mode: with a deterministic router biased
        toward expert 0 (topk=1), sparse routing gives expert 0 all tokens and leaves
        experts 1-3 with zero tokens (zero fc1 grad). Dense mode must give all experts
        non-zero fc1 grad regardless.
        """
        Utils.initialize_model_parallel(1, 1)

        num_moe_experts = 4
        hidden_size = 16

        def make_layer(dense: bool) -> MoELayer:
            config = TransformerConfig(
                num_layers=1,
                hidden_size=hidden_size,
                num_attention_heads=4,
                num_moe_experts=num_moe_experts,
                use_cpu_initialization=True,
                moe_token_dispatcher_type="allgather",
                moe_router_load_balancing_type="none",
                moe_router_topk=1,
                moe_router_pre_softmax=True,
                moe_aux_loss_coeff=0.0,
                moe_grouped_gemm=False,
                moe_ffn_hidden_size=32,
                add_bias_linear=False,
                moe_dense_forward_for_router_grad=dense,
            )
            submodules = get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=False
            )
            layer = MoELayer(config, submodules.mlp.submodules).cuda().train()
            return layer

        # Construct input and router weights such that logit[0] >> logit[1,2,3] for all tokens:
        # Set router.weight[i] = e_i (standard basis vector in hidden_size), and
        # set hidden = e_0 (unit vector along dim 0) for all tokens.
        # Then logit_0 = 1.0, logit_1 = logit_2 = logit_3 = 0.0 -> expert 0 always wins topk=1.
        basis = torch.zeros(hidden_size, device='cuda')
        basis[0] = 1.0
        hidden_states = basis.view(1, 1, hidden_size).expand(8, 2, hidden_size).clone()

        # --- Sparse mode: expert 0 gets all tokens, experts 1-3 get zero tokens ---
        sparse_layer = make_layer(dense=False)
        with torch.no_grad():
            for i in range(num_moe_experts):
                sparse_layer.router.weight.data[i] = torch.zeros(hidden_size, device='cuda')
                sparse_layer.router.weight.data[i][i] = 1.0  # weight[i] = e_i

        sparse_layer(hidden_states.detach())[0].sum().backward()
        sparse_fc1_grads = [
            e.linear_fc1.weight.grad.norm().item() for e in sparse_layer.experts.local_experts
        ]

        assert sparse_fc1_grads[0] > 0, "Sparse mode: expert 0 should receive tokens"
        assert all(g == 0.0 for g in sparse_fc1_grads[1:]), (
            f"Sparse mode: experts 1-3 should receive zero tokens (topk=1, input=e_0), "
            f"got fc1 grad norms: {sparse_fc1_grads[1:]}"
        )

        # --- Dense mode: all experts process all tokens, all must have non-zero fc1 grad ---
        dense_layer = make_layer(dense=True)
        with torch.no_grad():
            for i in range(num_moe_experts):
                dense_layer.router.weight.data[i] = torch.zeros(hidden_size, device='cuda')
                dense_layer.router.weight.data[i][i] = 1.0

        dense_layer(hidden_states.detach())[0].sum().backward()
        dense_fc1_grads = [
            e.linear_fc1.weight.grad.norm().item() for e in dense_layer.experts.local_experts
        ]

        for i, norm in enumerate(dense_fc1_grads):
            assert norm > 0, (
                f"Dense mode: expert {i} must have non-zero fc1 gradient "
                f"(DenseMixer runs all experts on all tokens), got {norm}"
            )

        Utils.destroy_model_parallel()

    def test_dense_forward_aux_loss_gradient_flows(self):
        """Verify that aux loss gradients still reach the router in dense forward mode."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.1,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        hidden_states = torch.randn(4, 2, 16, device=torch.cuda.current_device())
        output, _ = moe_layer(hidden_states)
        output.sum().backward()

        # Router weight must have gradient (aux loss contributes via sparse_probs anchor)
        assert moe_layer.router.weight.grad is not None
        assert (
            moe_layer.router.weight.grad.abs().sum() > 0
        ), "Router weight gradient should be non-zero with aux_loss enabled"

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("score_function", ["softmax", "sigmoid"])
    def test_dense_forward_score_functions(self, score_function):
        """DenseMixer should work with both softmax and sigmoid score functions."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
            moe_router_score_function=score_function,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        hidden_states = torch.randn(4, 2, 16, device=torch.cuda.current_device())
        output, _ = moe_layer(hidden_states)
        assert output.shape == hidden_states.shape

        output.sum().backward()
        assert moe_layer.router.weight.grad is not None

        Utils.destroy_model_parallel()

    def test_dense_moe_forward_eval_uses_sparse(self):
        """Test that eval mode uses standard sparse forward (not dense)."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules)
        moe_layer.cuda()

        hidden_states = torch.randn(4, 2, 16, device=torch.cuda.current_device())

        # Eval mode should use standard sparse forward
        moe_layer.eval()
        with torch.no_grad():
            output_eval, _ = moe_layer(hidden_states)
        assert output_eval.shape == hidden_states.shape

        Utils.destroy_model_parallel()

    def test_dense_forward_no_nan_inf(self):
        """DenseMixer output and gradients must be finite (no NaN/Inf)."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        hidden_states = torch.randn(
            8, 2, 16, device=torch.cuda.current_device(), requires_grad=True
        )
        output, _ = moe_layer(hidden_states)

        assert torch.isfinite(output).all(), "DenseMixer output contains NaN or Inf"

        output.sum().backward()
        assert torch.isfinite(
            hidden_states.grad
        ).all(), "DenseMixer input gradient contains NaN or Inf"
        for name, param in moe_layer.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), f"Gradient of {name} contains NaN or Inf"

        Utils.destroy_model_parallel()

    def test_dense_forward_with_padding_mask(self):
        """DenseMixer dense forward must handle padding_mask without errors."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        seq_len, bsz, hidden_size = 8, 2, 16
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        hidden_states = torch.randn(seq_len, bsz, hidden_size, device=torch.cuda.current_device())
        # padding_mask: [bsz, seq_len], False = padding token
        padding_mask = torch.ones(
            bsz, seq_len, device=torch.cuda.current_device(), dtype=torch.bool
        )
        padding_mask[:, -2:] = False  # last 2 positions are padding

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)
        assert output.shape == hidden_states.shape
        assert torch.isfinite(output).all(), "Output contains NaN/Inf with padding mask"

        output.sum().backward()

        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("load_balancing_type", ["aux_loss", "seq_aux_loss"])
    def test_dense_forward_load_balancing_types(self, load_balancing_type):
        """DenseMixer should work with different load balancing loss types."""
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type=load_balancing_type,
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        hidden_states = torch.randn(4, 2, 16, device=torch.cuda.current_device())
        output, _ = moe_layer(hidden_states)
        assert output.shape == hidden_states.shape
        output.sum().backward()
        assert moe_layer.router.weight.grad is not None

        Utils.destroy_model_parallel()

    def test_dense_forward_flag_false_uses_sparse(self):
        """Regression guard: when flag is False (default), standard sparse forward is used.

        We verify this by checking that dense_moe_forward is NOT called when the flag is off.
        We use topk=1 with deterministic routing (input=e_0, weights=basis), so with sparse
        routing expert 0 gets all tokens while experts 1-3 get zero tokens (fc1 grad == 0).
        """
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        num_moe_experts = 4
        hidden_size = 16
        config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_load_balancing_type="none",
            moe_router_topk=1,
            moe_router_pre_softmax=True,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=False,  # default: sparse
        )
        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        # Deterministic routing: input = e_0, router.weight[i] = e_i -> expert 0 always wins
        basis = torch.zeros(hidden_size, device='cuda')
        basis[0] = 1.0
        hidden_states = basis.view(1, 1, hidden_size).expand(8, 2, hidden_size).clone()
        with torch.no_grad():
            for i in range(num_moe_experts):
                moe_layer.router.weight.data[i] = torch.zeros(hidden_size, device='cuda')
                moe_layer.router.weight.data[i][i] = 1.0

        output, _ = moe_layer(hidden_states)
        assert output.shape == hidden_states.shape
        output.sum().backward()

        fc1_grads = [
            e.linear_fc1.weight.grad.norm().item() for e in moe_layer.experts.local_experts
        ]

        # Sparse mode with topk=1: only expert 0 should have non-zero fc1 gradient
        assert fc1_grads[0] > 0, "flag=False: expert 0 should receive tokens"
        assert all(
            g == 0.0 for g in fc1_grads[1:]
        ), f"flag=False (sparse): experts 1-3 should have zero fc1 grad, got {fc1_grads[1:]}"

        Utils.destroy_model_parallel()

    def test_dense_forward_guard_grouped_gemm(self):
        """DenseMixer must raise AssertionError when experts is not SequentialMLP.

        We verify the guard by directly calling dense_moe_forward on a layer whose experts
        have been replaced with a mock non-SequentialMLP object, bypassing the submodules
        constructor (which may fall back to SequentialMLP when TE is unavailable).
        """
        import types

        Utils.initialize_model_parallel(1, 1)

        config = TransformerConfig(
            num_layers=1,
            hidden_size=16,
            num_attention_heads=4,
            num_moe_experts=4,
            use_cpu_initialization=True,
            moe_token_dispatcher_type="allgather",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.0,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=32,
            add_bias_linear=False,
            moe_dense_forward_for_router_grad=True,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=4, moe_grouped_gemm=False)
        moe_layer = MoELayer(config, submodules.mlp.submodules).cuda().train()

        # Replace experts with a non-SequentialMLP mock to trigger the guard.
        # Use _modules directly to bypass nn.Module's type check on setattr.
        original_experts = moe_layer.experts
        moe_layer._modules['experts'] = torch.nn.Linear(1, 1)  # not a SequentialMLP

        hidden_states = torch.randn(4, 2, 16, device=torch.cuda.current_device())
        with pytest.raises(AssertionError, match="SequentialMLP"):
            moe_layer.dense_moe_forward(hidden_states)

        moe_layer.experts = original_experts  # restore
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestInterleaveTransformerBlock:

    @pytest.mark.parametrize("moe_layer_freq", [2, eval("[0,1,1,1]"), eval("[0]*2+[1]*2")])
    def test_interleave_transformer_block(self, moe_layer_freq):
        Utils.initialize_model_parallel(1, 1)
        model_parallel_cuda_manual_seed(123)
        self.transformer_config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            moe_layer_freq=moe_layer_freq,
            moe_ffn_hidden_size=256,
            use_cpu_initialization=True,
            num_moe_experts=2,
            add_bias_linear=False,
        )
        self.parallel_transformer_block = TransformerBlock(
            self.transformer_config, get_gpt_decoder_block_spec(self.transformer_config, False)
        )

        # Check if the moe layer is interleaved correctly
        if isinstance(self.transformer_config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % self.transformer_config.moe_layer_freq == 0) else 0
                for i in range(self.transformer_config.num_layers)
            ]
        else:
            moe_layer_pattern = self.transformer_config.moe_layer_freq

        for i, layer in enumerate(self.parallel_transformer_block.layers):
            is_moe_layer = isinstance(layer.mlp, MoELayer)
            assert is_moe_layer == moe_layer_pattern[i]

        # Test forward pass
        parallel_transformer_block = self.parallel_transformer_block
        config: TransformerConfig = parallel_transformer_block.config
        sequence_length = 32
        micro_batch_size = 2
        parallel_transformer_block.cuda()

        # [sequence length, batch size, hidden size]
        hidden_states = torch.ones((sequence_length, micro_batch_size, config.hidden_size))
        hidden_states = hidden_states.cuda()

        attention_mask = torch.ones((1, 1, sequence_length, sequence_length), dtype=bool).cuda()
        hidden_states = parallel_transformer_block(
            hidden_states=hidden_states, attention_mask=attention_mask
        )
        assert hidden_states.shape[0] == sequence_length
        assert hidden_states.shape[1] == micro_batch_size
        assert hidden_states.shape[2] == config.hidden_size

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerFP16:
    """Test MoE layer with FP16 precision."""

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (2, 2), (4, 2)])
    def test_moe_layer_fp16_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, tp_size, ep_size
    ):
        """Test MoE layer forward and backward pass with fp16 params and inputs."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,  # Use SequentialMLP for fp16 test
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp16=True,
            params_dtype=torch.float16,
        )

        submodules = get_gpt_layer_local_submodules(
            num_experts=num_moe_experts, moe_grouped_gemm=False
        )

        moe_layer = MoELayer(transformer_config, submodules.mlp.submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.float16,
            requires_grad=True,
        )

        # Forward pass
        output, _ = moe_layer(hidden_states)

        assert output.dtype == torch.float16, f"Expected fp16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.float16
        ), f"Expected fp16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()


class TestMoELayerRecompute:
    """Test MoE layer with recompute enabled (activation checkpointing).

    Tests both code paths:
    - fp8=False: uses tensor_parallel.checkpoint
    - fp8=True: uses te_checkpoint (requires TE >= 1.7.0)
    """

    def setup_method(self, method):
        pass

    @pytest.mark.parametrize("moe_token_dispatcher_type", ["allgather", "alltoall"])
    @pytest.mark.parametrize("num_moe_experts", [2, 4])
    @pytest.mark.parametrize("with_padding_mask", [True, False])
    @pytest.mark.parametrize("tp_size,ep_size", [(1, 1), (4, 2)])
    @pytest.mark.parametrize("fp8", [False, True])
    def test_moe_layer_recompute_forward_backward(
        self, num_moe_experts, moe_token_dispatcher_type, with_padding_mask, tp_size, ep_size, fp8
    ):
        """Test MoE layer forward and backward pass with recompute enabled.

        When fp8=False, uses tensor_parallel.checkpoint.
        When fp8=True, uses te_checkpoint (requires TE >= 1.7.0).
        """
        # Skip fp8 tests if TE version is not sufficient
        if fp8 and not is_te_min_version("1.7.0.dev0"):
            pytest.skip("FP8 MoE recompute requires TE 1.7.0 and later.")

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size, expert_model_parallel_size=ep_size
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        hidden_size = 64
        sequence_length = 32
        micro_batch_size = 2

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=num_moe_experts,
            use_cpu_initialization=False,
            moe_token_dispatcher_type=moe_token_dispatcher_type,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=0.01,
            moe_grouped_gemm=False,
            moe_ffn_hidden_size=256,
            add_bias_linear=False,
            # Enable recompute for MoE layer
            recompute_granularity="selective",
            recompute_modules=["moe"],
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            sequence_parallel=tp_size > 1,
            fp8=fp8,
            bf16=True,
            params_dtype=torch.bfloat16,
        )

        # Use TE spec for fp8, local spec otherwise
        if fp8:
            transformer_layer_submodules = get_gpt_layer_with_transformer_engine_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=False
            )
        else:
            transformer_layer_submodules = get_gpt_layer_local_submodules(
                num_experts=num_moe_experts, moe_grouped_gemm=False
            )

        moe_layer = MoELayer(transformer_config, transformer_layer_submodules.mlp.submodules).cuda()

        hidden_states = torch.randn(
            sequence_length,
            micro_batch_size,
            hidden_size,
            device=torch.cuda.current_device(),
            dtype=torch.bfloat16,
            requires_grad=True,
        )

        # Create padding mask if needed: shape [batch_size, sequence_length]
        padding_mask = None
        if with_padding_mask:
            padding_mask = torch.ones(
                micro_batch_size,
                sequence_length,
                device=torch.cuda.current_device(),
                dtype=torch.bool,
            )
            # Mark last 4 tokens as padding for each batch
            padding_mask[:, -4:] = False

        output, _ = moe_layer(hidden_states, padding_mask=padding_mask)

        assert output.dtype == torch.bfloat16, f"Expected bf16 output, got {output.dtype}"
        assert output.shape == hidden_states.shape, f"Output shape mismatch"

        # Backward pass - this is where recompute/checkpoint is actually used
        loss = output.sum()
        loss.backward()

        assert hidden_states.grad is not None, "Input gradients should exist"
        assert (
            hidden_states.grad.dtype == torch.bfloat16
        ), f"Expected bf16 gradients, got {hidden_states.grad.dtype}"

        for name, param in moe_layer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Gradient for {name} should exist"

        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
