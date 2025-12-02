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
        probs, indices = moe_layer.router(partitioned_input)
        probs.sum().mul_(0).backward()
        aux_loss_grad = partitioned_input.grad
        torch.distributed.barrier()
        ans = self.partition_input(baseline_grad)
        assert torch.allclose(aux_loss_grad, ans), f"Diff: {(aux_loss_grad/ans).mean()}"
        loss = get_moe_layer_wise_logging_tracker()[loss_name]['values']
        assert loss > 0, "Loss should be greater than 0"
        clear_aux_losses_tracker()

        with torch.no_grad():
            probs, indices = moe_layer.router(partitioned_input)
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
        probs, indices = moe_layer.router(self.input)
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
        probs, indices = moe_layer.router(self.input)
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
    @pytest.mark.skipif(
        not torch.cuda.is_available() or not HAVE_ROUTER_FUSION,
        reason="CUDA or TE fused router ops not available",
    )
    @pytest.mark.parametrize("aux_type", ["aux_loss", "seq_aux_loss"])
    def test_aux_loss_fusion_equivalence(self, aux_type):
        # Compare fused vs unfused aux loss path to ensure numerical equivalence
        router_ref = self.new_router(
            moe_router_load_balancing_type=aux_type, moe_aux_loss_coeff=1.0, moe_router_dtype="fp32"
        ).cuda()
        router_fused = self.new_router(
            moe_router_load_balancing_type=aux_type, moe_aux_loss_coeff=1.0, moe_router_dtype="fp32"
        ).cuda()

        with torch.no_grad():
            router_fused.weight.copy_(router_ref.weight)

        hidden_states = torch.randn((32, 2, router_ref.config.hidden_size)).cuda().bfloat16()

        # Map aux type to its tracker key
        loss_name_map = {
            "aux_loss": "load_balancing_loss",
            "seq_aux_loss": "seq_load_balancing_loss",
        }
        loss_name = loss_name_map[aux_type]

        # Unfused
        router_ref.config.moe_router_fusion = False
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
        router_fused.config.moe_router_fusion = True
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
