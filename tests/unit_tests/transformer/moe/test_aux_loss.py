# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import dataclasses

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.tensor_parallel.mappings import reduce_from_tensor_model_parallel_region
from megatron.core.tensor_parallel.random import (
    get_cuda_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.moe.moe_utils import (
    clear_aux_losses_tracker,
    get_default_pg_collection,
    get_moe_layer_wise_logging_tracker,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.typed_torch import apply_module
from megatron.training.initialize import _set_random_seed
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.moe.test_token_dispatcher import MoEModelTestContainer

try:
    # Check availability of TE fused router aux ops
    from megatron.core.extensions.transformer_engine import (
        fused_compute_score_for_moe_aux_loss as _fused_compute_score_for_moe_aux_loss,
    )
    from megatron.core.extensions.transformer_engine import (
        fused_moe_aux_loss as _fused_moe_aux_loss,
    )

    HAVE_ROUTER_FUSION = (
        _fused_compute_score_for_moe_aux_loss is not None and _fused_moe_aux_loss is not None
    )
except Exception:  # pragma: no cover - defensive
    HAVE_ROUTER_FUSION = False


class AuxlossTestContainer(MoEModelTestContainer):
    def partition_input(self, input):
        partitioned_input = input.chunk(
            parallel_state.get_tensor_and_context_parallel_world_size(), dim=0
        )[parallel_state.get_tensor_and_context_parallel_rank()]
        output = partitioned_input.clone().detach()
        output.requires_grad = True
        return output

    @pytest.mark.internal
    def aux_loss_test(self, input, baseline_grad, loss_name):
        partitioned_input = self.partition_input(input)
        moe_layer = self.moe_layer
        probs, indices = apply_module(moe_layer.router)(partitioned_input)
        probs.sum().mul_(0).backward()
        aux_loss_grad = partitioned_input.grad
        torch.distributed.barrier()
        ans = self.partition_input(baseline_grad)
        assert torch.allclose(aux_loss_grad, ans), f"Diff: {(aux_loss_grad/ans).mean()}"
        loss = get_moe_layer_wise_logging_tracker()[loss_name]['values']
        assert loss > 0, "Loss should be greater than 0"
        clear_aux_losses_tracker()

        with torch.no_grad():
            probs, indices = apply_module(moe_layer.router)(partitioned_input)
            loss = get_moe_layer_wise_logging_tracker()[loss_name]['values']
            assert loss == 0, "Loss should be 0"
            clear_aux_losses_tracker()


class TestAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_allgather_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="allgather",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")


class TestSeqAuxLoss:
    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()  # zero out the main gradients
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_a2a_dispatcher(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "seq_load_balancing_loss")


class TestPerTokenAuxLoss:
    """Regression test for the aux_loss TP/CP scaling fix under
    --calculate-per-token-loss. Computes a baseline aux-loss input
    gradient at (tp=1, cp=1) and asserts that each parametrized
    (tp, ep, cp) config produces a matching gradient on each rank's
    local input slice. Without the fix, the per-rank scale on aux_loss
    would shrink with tp_cp_size and the assertion would fail at any
    config with tp_size > 1 or cp_size > 1.
    """

    def setup_method(self, method):
        baseline_container = AuxlossTestContainer(
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
            calculate_per_token_loss=True,
        )
        moe_layer = baseline_container.moe_layer
        self.input = torch.randn((32, 8, moe_layer.config.hidden_size)).cuda()
        self.input.requires_grad = True
        probs, indices = apply_module(moe_layer.router)(self.input)
        probs.sum().mul_(0).backward()
        self.baseline_grad = self.input.grad
        self.input.grad = None
        clear_aux_losses_tracker()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_per_token_aux_loss_invariant_to_tp_cp(self, tp_size, ep_size, cp_size):
        container = AuxlossTestContainer(
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=cp_size,
            num_moe_experts=8,
            moe_router_topk=2,
            moe_router_load_balancing_type="aux_loss",
            moe_token_dispatcher_type="alltoall",
            moe_aux_loss_coeff=0.1,
            calculate_per_token_loss=True,
        )
        container.aux_loss_test(self.input, self.baseline_grad, "load_balancing_loss")


class TestRouterAuxLoss:
    def setup_method(self, method):
        Utils.initialize_model_parallel(1, 1)
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        # Default configuration
        self.default_transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=8,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
        )

    def new_router(self, **kwargs):
        """Create a new router with updated configuration.

        Args:
            **kwargs: Configuration parameters to update in the default config.

        Returns:
            Router: A new router instance with the specified configuration.
        """
        pg_collection = get_default_pg_collection()
        # Create a new config with updated parameters
        new_transformer_config = dataclasses.replace(self.default_transformer_config, **kwargs)

        # Create the router with the updated config
        router = TopKRouter(config=new_transformer_config, pg_collection=pg_collection)
        router.set_layer_number(0)
        return router

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_seq_aux_loss(self, tp_size, ep_size, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(42)

        # Test that with batch_size=1, aux_loss and seq_aux_loss should be the same
        aux_loss_router = self.new_router(
            moe_router_load_balancing_type="aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()
        seq_aux_loss_router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        # Set identical weights for fair comparison
        with torch.no_grad():
            seq_aux_loss_router.weight.copy_(aux_loss_router.weight)

        ### MBS=1 case: results should be identical ###
        clear_aux_losses_tracker()
        seq_len = 32
        batch_size = 1
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, batch_size, aux_loss_router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        # Forward pass for aux_loss router
        aux_loss_router.weight.grad = None
        scores1, routing_map1 = aux_loss_router(hidden_states)
        loss1 = scores1.sum()
        loss1.backward()
        grad1 = aux_loss_router.weight.grad.clone()

        # Forward pass for seq_aux_loss router
        seq_aux_loss_router.weight.grad = None
        scores2, routing_map2 = seq_aux_loss_router(hidden_states)
        loss2 = scores2.sum()
        loss2.backward()
        grad2 = seq_aux_loss_router.weight.grad.clone()

        # For batch_size=1, they should produce the same results
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss = tracker["load_balancing_loss"]["values"][0]
        seq_aux_loss = tracker["seq_load_balancing_loss"]["values"][0]

        reduce_from_tensor_model_parallel_region(aux_loss, aux_loss_router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, aux_loss_router.tp_cp_group)

        assert torch.equal(routing_map1, routing_map2)
        assert torch.equal(grad1, grad2)
        assert torch.equal(scores1, scores2)
        assert aux_loss == seq_aux_loss, f"aux_loss: {aux_loss}, seq_aux_loss: {seq_aux_loss}"

        ### MBS=2 case ###
        clear_aux_losses_tracker()
        batch_size = 2
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, batch_size, aux_loss_router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

        # Forward pass for aux_loss router
        aux_loss_router.weight.grad = None
        scores_first_batch, _ = aux_loss_router(hidden_states[:, 0:1, :])
        scores_second_batch, _ = aux_loss_router(hidden_states[:, 1:, :])

        # setting grad to 0 to only backward aux loss
        (scores_first_batch + scores_second_batch).backward(torch.zeros_like(scores_first_batch))

        grad1 = aux_loss_router.weight.grad.clone()

        # Forward pass for seq_aux_loss router
        seq_aux_loss_router.weight.grad = None
        scores2, routing_map2 = seq_aux_loss_router(hidden_states)
        # setting grad to 0 to only backward aux loss
        scores2.backward(torch.zeros_like(scores2))
        grad2 = seq_aux_loss_router.weight.grad.clone() * 2

        aux_loss = tracker["load_balancing_loss"]["values"][0] / 2
        seq_aux_loss = tracker["seq_load_balancing_loss"]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss, aux_loss_router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, aux_loss_router.tp_cp_group)

        torch.testing.assert_close(aux_loss, seq_aux_loss)
        torch.testing.assert_close(grad1, grad2)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("with_padding", [False, True])
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_seq_aux_loss_mbs_invariant_per_token_loss(
        self, tp_size, ep_size, cp_size, with_padding
    ):
        """seq_aux_loss gradient must be invariant to MBS under --calculate-per-token-loss.

        The same global batch is processed as N micro-batches of size 1 (MBS=1) and as one
        micro-batch of size N (MBS=N). Both cover the same tokens, so the finalize-time
        1/total_tokens normalization is an identical constant and the accumulated
        router-weight aux gradients must match. Before the fix (valid_token_count dropped the
        bsz factor), the MBS=N gradient is scaled by 1/N and the assertion fails. The padding
        case additionally checks the correction uses valid (non-padded) token counts.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        model_parallel_cuda_manual_seed(42)
        clear_aux_losses_tracker()

        router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            calculate_per_token_loss=True,
            # fp32 weights so the MBS=1 gradient (accumulated over N backward passes)
            # is not degraded by bf16 rounding relative to the single MBS=N backward.
            params_dtype=torch.float32,
            bf16=False,
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        seq_len = 32
        num_seqs = 4
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn(
                (seq_len, num_seqs, router.config.hidden_size),
                device=torch.device("cuda"),
                dtype=torch.float32,
            )
        padding_mask = None
        if with_padding:
            # True marks padding tokens (second half of each sequence).
            padding_mask = torch.zeros((seq_len, num_seqs), dtype=torch.bool, device="cuda")
            padding_mask[seq_len // 2 :, :] = True

        def run(indices):
            pmask = None if padding_mask is None else padding_mask[:, indices]
            scores, _ = router(hidden_states[:, indices, :].contiguous(), padding_mask=pmask)
            scores.backward(torch.zeros_like(scores))  # isolate the aux-loss gradient
            clear_aux_losses_tracker()

        # MBS=1: N micro-batches of size 1, accumulating the aux-loss gradient.
        router.weight.grad = None
        for b in range(num_seqs):
            run(slice(b, b + 1))
        grad_mbs1 = router.weight.grad.clone()

        # MBS=N: a single micro-batch of size N.
        router.weight.grad = None
        run(slice(0, num_seqs))
        grad_mbsN = router.weight.grad.clone()

        torch.testing.assert_close(grad_mbs1, grad_mbsN)

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="CUDA or TE fused router ops not available",
    )
    @pytest.mark.parametrize("aux_type", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
    def test_aux_loss_fusion_equivalence(self, aux_type):
        # Compare fused vs unfused aux loss path to ensure numerical equivalence
        router_ref = self.new_router(
            moe_router_load_balancing_type=aux_type,
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            moe_router_fusion=False,
        ).cuda()
        router_fused = self.new_router(
            moe_router_load_balancing_type=aux_type,
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            moe_router_fusion=True,
        ).cuda()

        with torch.no_grad():
            router_fused.weight.copy_(router_ref.weight)

        hidden_states = torch.randn((32, 2, router_ref.config.hidden_size)).cuda().bfloat16()

        # Map aux type to its tracker key
        loss_name_map = {
            "aux_loss": "load_balancing_loss",
            "seq_aux_loss": "seq_load_balancing_loss",
            "global_aux_loss": "global_load_balancing_loss",
        }
        loss_name = loss_name_map[aux_type]

        # Unfused
        clear_aux_losses_tracker()
        router_ref.weight.grad = None
        scores_ref, routing_ref = router_ref(hidden_states)
        # Backward zeros to isolate aux-loss-only gradient contribution
        scores_ref.backward(torch.zeros_like(scores_ref))
        grad_ref = router_ref.weight.grad.clone()
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss_ref = tracker[loss_name]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss_ref, router_ref.tp_cp_group)

        # Fused
        clear_aux_losses_tracker()
        router_fused.weight.grad = None
        scores_fused, routing_fused = router_fused(hidden_states)
        scores_fused.backward(torch.zeros_like(scores_fused))
        grad_fused = router_fused.weight.grad.clone()
        tracker = get_moe_layer_wise_logging_tracker()
        aux_loss_fused = tracker[loss_name]["values"][0]
        reduce_from_tensor_model_parallel_region(aux_loss_fused, router_fused.tp_cp_group)

        # Checks
        assert torch.equal(routing_ref, routing_fused)
        torch.testing.assert_close(scores_ref, scores_fused, rtol=2.0e-2, atol=1.0e-3)
        torch.testing.assert_close(aux_loss_ref, aux_loss_fused)
        torch.testing.assert_close(grad_ref, grad_fused)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_global_aux_loss(self, tp_size, ep_size, cp_size):
        clear_aux_losses_tracker()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )

        router = self.new_router(
            moe_router_load_balancing_type="global_aux_loss",
            moe_aux_loss_coeff=1.0,
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        seq_len = 32
        # Verify global tokens tracker initialized
        assert router.global_tokens_per_expert is not None
        assert router.ga_steps == 0

        # First microbatch
        with get_cuda_rng_tracker().fork():
            hidden_states = torch.randn((seq_len, 2, router.config.hidden_size)).cuda().bfloat16()
        num_local_tokens = seq_len * 2
        scores, routing_map = router(hidden_states)
        # Check that global tokens were counted
        assert torch.all(router.global_tokens_per_expert >= 0)
        assert (
            router.global_tokens_per_expert.sum()
            == num_local_tokens * router.tp_dp_cp_group.size() * router.ga_steps * router.topk
        )
        global_aux_loss_1 = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]
        reduce_from_tensor_model_parallel_region(global_aux_loss_1, router.tp_dp_cp_group)
        assert global_aux_loss_1 >= 1

        # When DP size is 1, the global aux loss should match the aux loss
        # for the first microbatch
        if get_default_pg_collection().tp_dp_cp.size() == tp_size:
            ref_router = self.new_router(
                moe_router_load_balancing_type="aux_loss", moe_aux_loss_coeff=1.0
            ).cuda()
            with torch.no_grad():
                ref_router.weight.copy_(router.weight)
            ref_scores, ref_routing_map = ref_router(hidden_states)
            aux_loss = get_moe_layer_wise_logging_tracker()["load_balancing_loss"]["values"][0]
            reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)

            assert torch.equal(
                aux_loss, global_aux_loss_1
            ), f"aux_loss: {aux_loss}, global_aux_loss_1: {global_aux_loss_1}"

        clear_aux_losses_tracker()

        # Get current tokens count to verify accumulation
        current_per_expert = router.global_tokens_per_expert.clone()

        # Second microbatch - should accumulate
        hidden_states = torch.randn((seq_len, 2, router.config.hidden_size)).cuda().bfloat16()
        scores, routing_map = router(hidden_states)
        global_aux_loss_2 = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]
        reduce_from_tensor_model_parallel_region(global_aux_loss_2, router.tp_dp_cp_group)
        assert torch.all(global_aux_loss_2 >= 1), f"global_aux_loss_2: {global_aux_loss_2}"

        # Verify tokens were accumulated
        assert router.ga_steps == 2
        assert torch.any(router.global_tokens_per_expert > current_per_expert)
        clear_aux_losses_tracker()

        # Reset global tracker
        router.reset_global_aux_loss_tracker()
        assert router.ga_steps == 0
        assert torch.all(router.global_tokens_per_expert == 0)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_combined_aux_loss(self, tp_size, ep_size, cp_size):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        clear_aux_losses_tracker()

        # Test combined aux loss types
        router = self.new_router(
            moe_router_load_balancing_type=["aux_loss", "seq_aux_loss", "global_aux_loss"],
            moe_aux_loss_coeff=[0.5, 1.0, 2.0],
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        ).cuda()

        # Verify all aux loss trackers initialized
        assert router.global_tokens_per_expert is not None
        assert router.ga_steps == 0

        # Execute forward pass
        hidden_states = torch.randn((32, 2, router.config.hidden_size)).cuda().bfloat16()
        router.weight.grad = None
        scores, routing_map = router(hidden_states)
        loss = scores.sum()
        loss.backward()

        aux_loss = get_moe_layer_wise_logging_tracker()["load_balancing_loss"]["values"][0]
        seq_aux_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
        global_aux_loss = get_moe_layer_wise_logging_tracker()["global_load_balancing_loss"][
            "values"
        ][0]

        reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(seq_aux_loss, router.tp_cp_group)
        reduce_from_tensor_model_parallel_region(global_aux_loss, router.tp_dp_cp_group)

        assert aux_loss >= 1
        assert seq_aux_loss >= 1
        assert global_aux_loss >= 1

        # Verify gradient is non-zero (aux losses are being applied)
        assert router.weight.grad.abs().sum() > 0

        # Verify method to get aux loss coeffs works properly
        assert router.get_aux_loss_coeff("aux_loss") == 0.5
        assert router.get_aux_loss_coeff("seq_aux_loss") == 1.0
        assert router.get_aux_loss_coeff("global_aux_loss") == 2.0
        assert router.get_aux_loss_coeff("non_existent_type") == 0.0

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_force_balanced_aux_loss(self, tp_size, ep_size, cp_size):
        """Test if aux loss is 1.0 when using uniform routing"""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_tensor_parallel_size=ep_size,
            context_parallel_size=cp_size,
        )
        clear_aux_losses_tracker()
        seq_len = 32
        batch_size = 2

        # Create router with each aux loss type
        for aux_loss_type in ["aux_loss", "seq_aux_loss", "global_aux_loss"]:
            router = self.new_router(
                moe_router_load_balancing_type=aux_loss_type,
                moe_aux_loss_coeff=1.0,
                moe_router_dtype="fp32",
                tensor_model_parallel_size=tp_size,
                expert_tensor_parallel_size=ep_size,
                context_parallel_size=cp_size,
            ).cuda()
            # create uniform weights
            with torch.no_grad():
                router.weight.copy_(torch.ones_like(router.weight) / router.weight.numel())

            # Create uniform logits (all experts equally likely)
            hidden_size = router.config.hidden_size
            num_experts = router.config.num_moe_experts

            loss_name = {
                "aux_loss": "load_balancing_loss",
                "seq_aux_loss": "seq_load_balancing_loss",
                "global_aux_loss": "global_load_balancing_loss",
            }[aux_loss_type]

            hidden_states = torch.randn(
                (seq_len, batch_size, hidden_size),
                device=torch.device("cuda"),
                dtype=torch.bfloat16,
            )

            # Get routing scores and map
            scores, routing_map = router(hidden_states)
            aux_loss = get_moe_layer_wise_logging_tracker()[loss_name]["values"][0]
            if aux_loss_type == "global_aux_loss":
                reduce_from_tensor_model_parallel_region(aux_loss, router.tp_dp_cp_group)
            else:
                reduce_from_tensor_model_parallel_region(aux_loss, router.tp_cp_group)
            assert aux_loss.item() == 1, f"{aux_loss_type}: {aux_loss.item()}"
            clear_aux_losses_tracker()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_seq_aux_loss_flattened_packed_sequences(self):
        """Test that TransformerLayer reshapes flattened packed sequences for MoE.

        When inter-document masking flattens MBS > 1 into [mbs*S, 1, H],
        TransformerLayer._maybe_reshape_for_moe should restore [S, mbs, H] so
        the router computes seq_aux_loss per sample. This test runs a forward
        pass through a real TransformerLayer with an MoE MLP and verifies that passing
        packed_seq_params with the flattened input produces the same
        seq_load_balancing_loss as the un-flattened input.
        """
        from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_submodules
        from megatron.core.packed_seq_params import PackedSeqParams
        from megatron.core.transformer.transformer_layer import TransformerLayer

        seq_len = 128
        batch_size = 4
        hidden_size = 12

        transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=hidden_size,
            num_attention_heads=4,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_router_topk=2,
            moe_aux_loss_coeff=1.0,
            moe_ffn_hidden_size=64,
            add_bias_linear=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            hidden_dropout=0.0,
        )
        submodules = get_gpt_layer_local_submodules(num_experts=32, moe_grouped_gemm=False)
        layer = TransformerLayer(transformer_config, submodules).cuda().bfloat16()
        assert layer.is_moe_layer

        hidden_states = torch.randn(
            (seq_len, batch_size, hidden_size), device=torch.device("cuda"), dtype=torch.bfloat16
        )

        def _get_seq_aux_loss(hidden_states, packed_seq_params=None):
            clear_aux_losses_tracker()
            output = layer._forward_mlp(hidden_states, packed_seq_params=packed_seq_params)
            loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
            return output, loss

        # Baseline: forward with the original [seq_len, mbs, H] shape.
        _, loss_baseline = _get_seq_aux_loss(hidden_states)

        # Flatten to [mbs*seq_len, 1, H] the same way the dataloader does.
        flattened = hidden_states.transpose(0, 1).reshape(batch_size * seq_len, 1, -1)

        # With packed_seq_params, _maybe_reshape_for_moe restores [S, mbs, H]
        # before the router, recovering the correct per-sample loss.
        packed_seq_params = PackedSeqParams(tokens_per_sample=seq_len)
        flattened_output, loss_with_implicit_reshape = _get_seq_aux_loss(
            flattened, packed_seq_params=packed_seq_params
        )

        assert flattened_output.shape == flattened.shape
        torch.testing.assert_close(loss_with_implicit_reshape, loss_baseline)

        # Variable-length packs cannot be unflattened into a dense [S, mbs, H] tensor. Verify
        # the segmented router path through a real MoE layer while retaining the THD layout.
        logical_lengths = (3, 5)
        physical_lengths = (4, 8)
        sequences = [hidden_states[:length, idx] for idx, length in enumerate(logical_lengths)]
        padded_hidden = torch.zeros(
            (max(logical_lengths), len(logical_lengths), hidden_size),
            device="cuda",
            dtype=torch.bfloat16,
        )
        padded_mask = torch.ones(
            (len(logical_lengths), max(logical_lengths)), device="cuda", dtype=torch.bool
        )
        for sequence_idx, sequence in enumerate(sequences):
            padded_hidden[: sequence.shape[0], sequence_idx] = sequence
            padded_mask[sequence_idx, : sequence.shape[0]] = False

        clear_aux_losses_tracker()
        layer._forward_mlp(padded_hidden, padding_mask=padded_mask)
        variable_loss_baseline = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"][
            "values"
        ][0]

        total_physical_tokens = sum(physical_lengths)
        variable_packed = torch.zeros(
            (total_physical_tokens, 1, hidden_size), device="cuda", dtype=torch.bfloat16
        )
        variable_mask = torch.ones((1, total_physical_tokens), device="cuda", dtype=torch.bool)
        cursor = 0
        for sequence, physical_length in zip(sequences, physical_lengths):
            variable_packed[cursor : cursor + sequence.shape[0], 0] = sequence
            variable_mask[0, cursor : cursor + sequence.shape[0]] = False
            cursor += physical_length

        variable_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, 4, 12], device="cuda", dtype=torch.int32),
        )
        clear_aux_losses_tracker()
        variable_output = layer._forward_mlp(
            variable_packed, padding_mask=variable_mask, packed_seq_params=variable_params
        )
        variable_packed_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"][
            "values"
        ][0]

        assert variable_output.shape == variable_packed.shape
        torch.testing.assert_close(variable_packed_loss, variable_loss_baseline)

        # Chunking the variable THD stream would split logical sequences and apply
        # seq_aux_loss once per chunk. The layer should keep this case unchunked.
        layer.config.mlp_chunks_for_training = 2
        clear_aux_losses_tracker()
        chunk_config_output = layer._forward_mlp(
            variable_packed, padding_mask=variable_mask, packed_seq_params=variable_params
        )
        chunk_config_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"][
            "values"
        ][0]

        torch.testing.assert_close(chunk_config_output, variable_output)
        torch.testing.assert_close(chunk_config_loss, variable_packed_loss)

        with pytest.raises(ValueError, match="not supported with Transformer Engine CUDA graphs"):
            layer._te_cuda_graph_capture(variable_packed, packed_seq_params=variable_params)

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("with_alignment_padding", [False, True])
    @pytest.mark.parametrize("calculate_per_token_loss", [False, True])
    @pytest.mark.parametrize("with_unused_boundary_slots", [False, True])
    def test_seq_aux_loss_variable_length_packed_sequences(
        self, with_alignment_padding, calculate_per_token_loss, with_unused_boundary_slots
    ):
        """Variable-length THD packs must match an equivalent padded batch.

        The packed representation keeps the logical sequences flattened and may
        include physical alignment gaps. Sequence-level routing statistics must
        still be computed independently for each logical sequence.
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            calculate_per_token_loss=calculate_per_token_loss,
        ).cuda()

        logical_lengths = (3, 5)
        physical_lengths = (4, 8) if with_alignment_padding else logical_lengths
        max_logical_length = max(logical_lengths)
        hidden_size = router.config.hidden_size

        with get_cuda_rng_tracker().fork():
            sequences = [
                torch.randn(
                    (length, hidden_size), device=torch.device("cuda"), dtype=torch.bfloat16
                )
                for length in logical_lengths
            ]

        padded_hidden = torch.zeros(
            (max_logical_length, len(sequences), hidden_size),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        padded_mask = torch.ones(
            (max_logical_length, len(sequences)), device=torch.device("cuda"), dtype=torch.bool
        )
        for sequence_idx, sequence in enumerate(sequences):
            padded_hidden[: sequence.shape[0], sequence_idx] = sequence
            padded_mask[: sequence.shape[0], sequence_idx] = False

        total_physical_tokens = sum(physical_lengths)
        packed_hidden = torch.zeros(
            (total_physical_tokens, 1, hidden_size),
            device=torch.device("cuda"),
            dtype=torch.bfloat16,
        )
        packed_mask = torch.ones(
            (total_physical_tokens, 1), device=torch.device("cuda"), dtype=torch.bool
        )
        physical_cursor = 0
        for sequence, physical_length in zip(sequences, physical_lengths):
            logical_length = sequence.shape[0]
            packed_hidden[physical_cursor : physical_cursor + logical_length, 0] = sequence
            packed_mask[physical_cursor : physical_cursor + logical_length, 0] = False
            physical_cursor += physical_length

        logical_cu_seqlens = torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32)
        physical_cu_seqlens = torch.tensor(
            [0, physical_lengths[0], total_physical_tokens], device="cuda", dtype=torch.int32
        )
        if with_unused_boundary_slots:
            logical_cu_seqlens = torch.nn.functional.pad(logical_cu_seqlens, (0, 2), value=8)
            physical_cu_seqlens = torch.nn.functional.pad(
                physical_cu_seqlens, (0, 2), value=total_physical_tokens
            )
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=logical_cu_seqlens,
            cu_seqlens_q_padded=(physical_cu_seqlens if with_alignment_padding else None),
        )

        def run(hidden_states, padding_mask, packed_params=None):
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states.requires_grad_(True)
            probs, _ = router(
                hidden_states, padding_mask=padding_mask, packed_seq_params=packed_params
            )
            probs.backward(torch.zeros_like(probs))
            loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
            return (
                loss.detach().clone(),
                router.weight.grad.detach().clone(),
                hidden_states.grad.detach().clone(),
            )

        padded_loss, padded_weight_grad, padded_input_grad = run(padded_hidden, padded_mask)
        packed_loss, packed_weight_grad, packed_input_grad = run(
            packed_hidden, packed_mask, packed_seq_params
        )

        torch.testing.assert_close(packed_loss, padded_loss)
        torch.testing.assert_close(packed_weight_grad, padded_weight_grad)

        physical_cursor = 0
        for sequence_idx, (logical_length, physical_length) in enumerate(
            zip(logical_lengths, physical_lengths)
        ):
            torch.testing.assert_close(
                packed_input_grad[physical_cursor : physical_cursor + logical_length, 0],
                padded_input_grad[:logical_length, sequence_idx],
            )
            alignment_grad = packed_input_grad[
                physical_cursor + logical_length : physical_cursor + physical_length
            ]
            assert torch.count_nonzero(alignment_grad) == 0
            physical_cursor += physical_length

    @pytest.mark.internal
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="CUDA or TE fused router ops not available",
    )
    def test_seq_aux_loss_variable_length_packed_sequences_fusion(self):
        """Fused and unfused aux-loss kernels must agree for variable THD packs."""
        from megatron.core.packed_seq_params import PackedSeqParams

        router_ref = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            moe_router_aux_loss_fusion=False,
        ).cuda()
        router_fused = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp32",
            moe_router_aux_loss_fusion=True,
        ).cuda()
        with torch.no_grad():
            router_fused.weight.copy_(router_ref.weight)

        hidden_states = torch.randn(
            (12, 1, router_ref.config.hidden_size), device="cuda", dtype=torch.bfloat16
        )
        padding_mask = torch.tensor(
            [[False, False, False, True, False, False, False, False, False, True, True, True]],
            device="cuda",
        ).T
        packed_seq_params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=torch.tensor([0, 3, 8], device="cuda", dtype=torch.int32),
            cu_seqlens_q_padded=torch.tensor([0, 4, 12], device="cuda", dtype=torch.int32),
        )

        def run(router):
            clear_aux_losses_tracker()
            router.weight.grad = None
            local_hidden = hidden_states.clone().requires_grad_(True)
            scores, routing_map = router(
                local_hidden, padding_mask=padding_mask, packed_seq_params=packed_seq_params
            )
            scores.backward(torch.zeros_like(scores))
            loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
            return scores, routing_map, loss, router.weight.grad, local_hidden.grad

        ref = run(router_ref)
        fused = run(router_fused)
        assert torch.equal(ref[1], fused[1])
        torch.testing.assert_close(ref[0], fused[0], rtol=2.0e-2, atol=1.0e-3)
        torch.testing.assert_close(ref[2], fused[2])
        torch.testing.assert_close(ref[3], fused[3])
        torch.testing.assert_close(ref[4], fused[4])

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("tp_size,cp_size", [(1, 1), (1, 2), (2, 1), (1, 8), (4, 2)])
    def test_seq_aux_loss_variable_length_packed_sequences_parallel(self, tp_size, cp_size):
        """Packed seq_aux_loss must preserve sequence ownership across CP and SP."""
        from megatron.core.context_parallel import get_batches_on_this_cp_rank

        world_size = torch.distributed.get_world_size()
        if world_size % (tp_size * cp_size) != 0:
            pytest.skip(f"requires a world size divisible by TP={tp_size} * CP={cp_size}")

        baseline_router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            params_dtype=torch.float32,
            bf16=False,
        ).cuda()
        hidden_size = baseline_router.config.hidden_size
        logical_lengths = (3, 5)
        with get_cuda_rng_tracker().fork():
            flat_hidden = torch.randn(
                (sum(logical_lengths), hidden_size), device="cuda", dtype=torch.float32
            )

        padded_hidden = torch.zeros(
            (max(logical_lengths), len(logical_lengths), hidden_size),
            device="cuda",
            dtype=torch.float32,
        )
        padded_mask = torch.ones(
            (max(logical_lengths), len(logical_lengths)), device="cuda", dtype=torch.bool
        )
        cursor = 0
        for sequence_idx, length in enumerate(logical_lengths):
            padded_hidden[:length, sequence_idx] = flat_hidden[cursor : cursor + length]
            padded_mask[:length, sequence_idx] = False
            cursor += length

        clear_aux_losses_tracker()
        padded_hidden.requires_grad_(True)
        baseline_probs, _ = baseline_router(padded_hidden, padding_mask=padded_mask)
        baseline_probs.backward(torch.zeros_like(baseline_probs))
        baseline_loss = get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][
            0
        ].detach()
        baseline_weight_grad = baseline_router.weight.grad.detach()
        baseline_input_grad = torch.cat(
            [
                padded_hidden.grad[: logical_lengths[0], 0],
                padded_hidden.grad[: logical_lengths[1], 1],
            ]
        )

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
        )
        router = self.new_router(
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=1.0,
            moe_router_dtype="fp64",
            params_dtype=torch.float32,
            bf16=False,
            tensor_model_parallel_size=tp_size,
            context_parallel_size=cp_size,
            sequence_parallel=tp_size > 1,
        ).cuda()
        with torch.no_grad():
            router.weight.copy_(baseline_router.weight)

        pg_collection = get_default_pg_collection()
        token_ids = torch.arange(1, sum(logical_lengths) + 1, device="cuda").view(1, -1)
        cp_batch = get_batches_on_this_cp_rank(
            {
                "tokens": token_ids,
                "labels": None,
                "loss_mask": torch.ones_like(token_ids),
                "position_ids": token_ids - 1,
                "attention_mask": None,
                "cu_seqlens": torch.tensor([[0, 3, 8]], device="cuda", dtype=torch.int32),
                "cu_seqlens_padded": None,
                "max_seqlen": torch.tensor([5], device="cuda", dtype=torch.int32),
                "local_cp_size": None,
                "hybrid_cp_group": None,
            },
            boundary_layout="zigzag",
            is_hybrid_cp=False,
            cp_group=pg_collection.cp,
            use_per_sequence_balancing=True,
            sequence_parallel=tp_size > 1,
            tp_group=pg_collection.tp,
            tp_cp_group=pg_collection.tp_cp,
            tokens_per_sample=None,
        )
        local_token_ids = cp_batch.get_batch()["tokens"].reshape(-1)
        if tp_size > 1:
            local_token_ids = local_token_ids.chunk(tp_size)[pg_collection.tp.rank()]

        valid_tokens = local_token_ids > 0
        local_hidden = torch.zeros(
            (local_token_ids.numel(), 1, hidden_size), device="cuda", dtype=torch.float32
        )
        local_hidden[valid_tokens, 0] = flat_hidden[local_token_ids[valid_tokens] - 1]
        local_hidden.requires_grad_(True)
        local_padding_mask = (~valid_tokens).view(-1, 1)

        clear_aux_losses_tracker()
        packed_probs, _ = router(
            local_hidden,
            padding_mask=local_padding_mask,
            packed_seq_params=cp_batch.get_packed_seq_params(),
        )
        assert packed_probs.shape == (local_hidden.shape[0], router.config.num_moe_experts)
        packed_probs.backward(torch.zeros_like(packed_probs))

        expected_input_grad = torch.zeros_like(local_hidden.grad)
        expected_input_grad[valid_tokens, 0] = baseline_input_grad[
            local_token_ids[valid_tokens] - 1
        ]
        torch.testing.assert_close(local_hidden.grad, expected_input_grad)

        packed_loss = (
            get_moe_layer_wise_logging_tracker()["seq_load_balancing_loss"]["values"][0]
            .detach()
            .clone()
        )
        reduce_from_tensor_model_parallel_region(packed_loss, router.tp_cp_group)
        torch.testing.assert_close(packed_loss, baseline_loss)

        packed_weight_grad = router.weight.grad.detach().clone()
        torch.distributed.all_reduce(packed_weight_grad, group=router.tp_cp_group)
        torch.testing.assert_close(packed_weight_grad, baseline_weight_grad)


class TestPaddingMaskAuxLoss:
    """Test padding mask support in various aux loss types."""

    def setup_model_parallel(self, tp_size=1, ep_size=1, cp_size=1, sequence_parallel=False):
        """Initialize model parallel with given configuration.

        Args:
            tp_size: Tensor parallel size.
            ep_size: Expert parallel size.
            cp_size: Context parallel size.
        """
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            pipeline_model_parallel_size=1,
            context_parallel_size=cp_size,
            expert_model_parallel_size=ep_size,
        )
        _set_random_seed(seed_=123, data_parallel_random_init=False)

        # Store parallel configuration
        self.tp_size = tp_size
        self.ep_size = ep_size
        self.cp_size = cp_size

        # Default configuration
        self.default_transformer_config = TransformerConfig(
            num_layers=1,
            hidden_size=12,
            num_attention_heads=8,
            num_moe_experts=32,
            use_cpu_initialization=True,
            moe_router_load_balancing_type="aux_loss",
            moe_router_topk=8,
            moe_aux_loss_coeff=1.0,
            bf16=True,
            params_dtype=torch.bfloat16,
            add_bias_linear=False,
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=ep_size,
            context_parallel_size=cp_size,
            sequence_parallel=sequence_parallel and tp_size > 1,
        )

    def new_router(self, **kwargs):
        """Create a new router with updated configuration."""
        pg_collection = get_default_pg_collection()
        new_transformer_config = dataclasses.replace(self.default_transformer_config, **kwargs)
        router = TopKRouter(config=new_transformer_config, pg_collection=pg_collection)
        router.set_layer_number(0)
        return router

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("aux_loss_type", ["aux_loss", "seq_aux_loss", "global_aux_loss"])
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_padding_mask_removes_padding_tokens(self, aux_loss_type, tp_size, ep_size, cp_size):
        """Test that padding tokens are correctly excluded from aux loss calculation."""
        # Initialize model parallel with given configuration
        self.setup_model_parallel(tp_size=tp_size, ep_size=ep_size, cp_size=cp_size)

        try:
            clear_aux_losses_tracker()

            router = self.new_router(
                moe_router_load_balancing_type=aux_loss_type,
                moe_aux_loss_coeff=1.0,
                moe_router_dtype="fp64",
            ).cuda()

            seq_len = 32
            batch_size = 2
            hidden_size = router.config.hidden_size

            # Create input with padding
            hidden_states_full = torch.randn(
                (seq_len, batch_size, hidden_size), dtype=torch.bfloat16, device='cuda'
            )

            # Create padding mask: first half valid, second half padding
            padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
            padding_mask[seq_len // 2 :, :] = True

            # Test with padding mask
            router.weight.grad = None
            scores_with_mask, routing_map_with_mask = router(
                hidden_states_full, padding_mask=padding_mask
            )
            scores_with_mask.backward(torch.zeros_like(scores_with_mask))

            loss_name = {
                "aux_loss": "load_balancing_loss",
                "seq_aux_loss": "seq_load_balancing_loss",
                "global_aux_loss": "global_load_balancing_loss",
            }[aux_loss_type]

            tracker = get_moe_layer_wise_logging_tracker()
            aux_loss_with_mask = tracker[loss_name]["values"][0].clone()
            grad_with_mask = router.weight.grad.clone()

            # Test without padding (with only half of the tokens)
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states_valid = hidden_states_full[: seq_len // 2, :, :]
            scores_without_mask, routing_map_without_mask = router(hidden_states_valid)
            scores_without_mask.backward(torch.zeros_like(scores_without_mask))

            aux_loss_without_mask = tracker[loss_name]["values"][0].clone()
            grad_without_mask = router.weight.grad.clone()

            # The aux loss with mask should be equal to the aux loss without mask
            assert torch.equal(aux_loss_with_mask, aux_loss_without_mask)
            assert torch.equal(grad_with_mask, grad_without_mask)

            clear_aux_losses_tracker()
        finally:
            # Always cleanup model parallel
            Utils.destroy_model_parallel()

    @pytest.mark.internal
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize(
        "tp_size,ep_size,cp_size", [(8, 1, 1), (4, 2, 1), (1, 1, 8), (2, 1, 4), (2, 2, 2)]
    )
    def test_padding_mask_with_z_loss(self, tp_size, ep_size, cp_size):
        """Test that padding mask works correctly with z_loss."""
        # Initialize model parallel with given configuration
        self.setup_model_parallel(tp_size=tp_size, ep_size=ep_size, cp_size=cp_size)

        try:
            clear_aux_losses_tracker()

            router = self.new_router(
                moe_router_load_balancing_type="aux_loss",
                moe_aux_loss_coeff=0.0,
                moe_z_loss_coeff=1.0,
                moe_router_dtype="fp32",
            ).cuda()

            seq_len = 32
            batch_size = 2
            hidden_size = router.config.hidden_size

            # Create input
            hidden_states_full = torch.randn(
                (seq_len, batch_size, hidden_size), dtype=torch.bfloat16, device='cuda'
            )

            # Create padding mask: first half valid, second half padding
            padding_mask = torch.zeros((seq_len, batch_size), dtype=torch.bool, device='cuda')
            padding_mask[seq_len // 2 :, :] = True

            # Test with padding mask
            router.weight.grad = None
            scores_with_mask, _ = router(hidden_states_full, padding_mask=padding_mask)
            scores_with_mask.sum().backward()

            tracker = get_moe_layer_wise_logging_tracker()
            z_loss_with_mask = tracker["z_loss"]["values"][0].clone()
            grad_with_mask = router.weight.grad.clone()

            # Test without padding (with only half of the tokens)
            clear_aux_losses_tracker()
            router.weight.grad = None
            hidden_states_valid = hidden_states_full[: seq_len // 2, :, :]
            scores_without_mask, _ = router(hidden_states_valid)
            scores_without_mask.sum().backward()

            z_loss_without_mask = tracker["z_loss"]["values"][0].clone()
            grad_without_mask = router.weight.grad.clone()

            # The z_loss with mask should be close to the z_loss without mask
            assert torch.equal(z_loss_with_mask, z_loss_without_mask)
            assert torch.equal(grad_with_mask, grad_without_mask)

            clear_aux_losses_tracker()
        finally:
            # Always cleanup model parallel
            Utils.destroy_model_parallel()
