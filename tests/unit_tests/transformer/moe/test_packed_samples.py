# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Packed sample statistics must not expand MoE router or dispatcher inputs."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.context_parallel import get_batches_on_this_cp_rank
from megatron.core.extensions.transformer_engine import fused_moe_aux_loss
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import (
    MoEAuxLossAutoScaler,
    clear_aux_losses_tracker,
    get_moe_layer_wise_logging_tracker,
)
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import TransformerLayer
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize(
    "local_tokens, sample_lengths",
    [(18, (6, 6, 6)), (9, (6, 6, 6)), (4, (4, 4, 4, 4))],
    ids=["cp-local", "tp2-sp", "tp4-sp"],
)
def test_packed_moe_keeps_flat_activations(local_tokens, sample_lengths):
    layer = SimpleNamespace(is_moe_layer=True)
    packed = PackedSeqParams(tokens_per_sample=8, moe_sample_lengths=sample_lengths)
    hidden = torch.randn(local_tokens, 1, 64)
    mask = torch.zeros(1, local_tokens, dtype=torch.bool)
    ids = torch.arange(local_tokens).unsqueeze(0)

    routed, routed_mask, routed_ids, reshape_state = TransformerLayer._maybe_unflatten_for_moe(
        layer, hidden, mask, ids, packed
    )

    assert routed is hidden
    assert routed_mask is mask
    assert routed_ids is ids
    assert reshape_state is None
    restored = TransformerLayer._maybe_reflatten_from_moe(layer, routed, packed, reshape_state)
    assert restored is hidden


def _router_loss_and_gradients(router, hidden, **kwargs):
    clear_aux_losses_tracker()
    router.zero_grad(set_to_none=True)
    hidden = hidden.detach().requires_grad_()
    probs, _ = router(hidden, **kwargs)
    # Backpropagate only the attached auxiliary loss, without a main-loss gradient.
    probs.backward(torch.zeros_like(probs))
    tracker = get_moe_layer_wise_logging_tracker()
    loss = tracker["seq_load_balancing_loss"]["values"][0].clone()
    return loss, hidden.grad.clone(), router.weight.grad.clone()


@pytest.mark.parametrize(
    "tp_size, per_token_loss, fused",
    [(1, False, False), (2, False, False), (2, True, False), (4, True, False), (2, True, True)],
    ids=["cp2", "tp2-cp2", "tp2-cp2-per-token", "tp4-cp2-per-token", "tp2-cp2-fused"],
)
def test_packed_seq_aux_loss_matches_rectangular_input(tp_size, per_token_loss, fused):
    Utils.initialize_distributed()
    if torch.distributed.get_world_size() % (tp_size * 2):
        pytest.skip(f"Requires a world size divisible by TP{tp_size} x CP2")
    if fused and fused_moe_aux_loss is None:
        pytest.skip("Transformer Engine fused auxiliary loss is unavailable")

    def create_router(tensor_parallel_size, context_parallel_size, sequence_parallel, fused):
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_router_load_balancing_type="seq_aux_loss",
            moe_aux_loss_coeff=0.1,
            moe_router_dtype="fp32",
            moe_router_fusion=False,
            moe_router_aux_loss_fusion=fused,
            calculate_per_token_loss=per_token_loss,
            add_bias_linear=False,
            tensor_model_parallel_size=tensor_parallel_size,
            context_parallel_size=context_parallel_size,
            sequence_parallel=sequence_parallel,
        )
        router = TopKRouter(config, ProcessGroupCollection.use_mpu_process_groups()).cuda()
        router.set_layer_number(1)
        return router

    # Three original length-8 samples. Their document lengths are [3,5],
    # [4,4], [3,5], so CP padding produces unequal physical sample lengths.
    generator = torch.Generator().manual_seed(456)
    Utils.initialize_model_parallel()
    MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
    try:
        hidden = torch.randn(8, 3, 8, generator=generator).cuda()
        torch.manual_seed(123)
        reference = create_router(
            tensor_parallel_size=1, context_parallel_size=1, sequence_parallel=False, fused=False
        )
        expected_loss, expected_grad, expected_weight_grad = _router_loss_and_gradients(
            reference, hidden
        )
    finally:
        clear_aux_losses_tracker()
        Utils.destroy_model_parallel()

    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size, context_parallel_size=2)
    try:
        router = create_router(
            tensor_parallel_size=tp_size,
            context_parallel_size=2,
            sequence_parallel=tp_size > 1,
            fused=fused,
        )
        router.load_state_dict(reference.state_dict())
        cp_batch = get_batches_on_this_cp_rank(
            {
                "tokens": torch.arange(24, device="cuda").view(1, 24),
                "cu_seqlens": torch.tensor(
                    [[0, 3, 8, 12, 16, 19, 24]], dtype=torch.int32, device="cuda"
                ),
                "max_seqlen": torch.tensor([5], dtype=torch.int32, device="cuda"),
            },
            boundary_layout="zigzag",
            dynamic_cp=False,
            cp_group=parallel_state.get_context_parallel_group(),
            use_per_sequence_balancing=True,
            sequence_parallel=tp_size > 1,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            tokens_per_sample=8,
        )
        batch = cp_batch.get_batch()
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        token_ids = batch["tokens"].flatten().chunk(tp_size)[tp_rank]
        padding = batch["padding_mask"].flatten().chunk(tp_size)[tp_rank]

        # Match the sample-major token order used by CP batching.
        flat_hidden = hidden.transpose(0, 1).flatten(0, 1)
        flat_expected_grad = expected_grad.transpose(0, 1).flatten(0, 1)
        local_hidden = flat_hidden[token_ids]
        expected_local_grad = flat_expected_grad[token_ids].masked_fill(padding[:, None], 0)

        loss, gradient, weight_grad = _router_loss_and_gradients(
            router,
            local_hidden.unsqueeze(1),
            padding_mask=padding,
            packed_seq_params=cp_batch.get_packed_seq_params(),
        )
        torch.distributed.all_reduce(loss, group=router.tp_cp_group)
        torch.distributed.all_reduce(weight_grad, group=router.tp_cp_group)

        torch.testing.assert_close(loss, expected_loss, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(gradient[:, 0], expected_local_grad, rtol=1e-4, atol=1e-6)
        torch.testing.assert_close(weight_grad, expected_weight_grad, rtol=1e-4, atol=1e-6)
    finally:
        clear_aux_losses_tracker()
        Utils.destroy_model_parallel()
