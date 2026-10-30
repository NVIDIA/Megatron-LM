# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Target masks change auxiliary statistics without changing token dispatch."""

import pytest
import torch

from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, clear_aux_losses_tracker
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils


def make_router(kind, per_token, masking, groups):
    """Use double precision to distinguish weighting errors from GEMM precision."""
    config = TransformerConfig(
        num_layers=1,
        hidden_size=8,
        num_attention_heads=2,
        num_moe_experts=4,
        add_bias_linear=False,
        moe_router_topk=2,
        moe_router_dtype="fp64",
        moe_router_load_balancing_type=kind if kind != "z_loss" else "none",
        moe_aux_loss_coeff=0.1 if kind != "z_loss" else 0.0,
        moe_z_loss_coeff=0.1 if kind == "z_loss" else None,
        calculate_per_token_loss=per_token,
        moe_aux_loss_masking=masking,
    )
    router = TopKRouter(config, pg_collection=groups).cuda().double().train()
    router.set_layer_number(0)
    return router


@pytest.mark.parametrize("kind", ["aux_loss", "seq_aux_loss", "global_aux_loss", "z_loss"])
@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("empty", [False, True])
def test_masked_router_matches_selected_token_reference(kind, per_token, empty):
    """Compare actual gradients with a router fed only the scored token subset."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        hidden = torch.randn(16, 1, 8, device="cuda", dtype=torch.float64)
        padding = torch.zeros(16, 1, dtype=torch.bool, device="cuda")
        padding[-2:] = True
        selected = (torch.arange(16, device="cuda") % 3 == 0).view(16, 1)
        if empty:
            selected.zero_()
        keep = (selected & ~padding).squeeze(1)
        n = int(keep.sum())
        candidate = make_router(kind, per_token, True, groups)
        reference = make_router(kind, False, False, groups)
        unmasked = make_router(kind, per_token, False, groups)
        reference.load_state_dict(candidate.state_dict())
        unmasked.load_state_dict(candidate.state_dict())
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        candidate_probs, candidate_routes = candidate(
            hidden, padding_mask=padding, moe_aux_loss_mask=selected
        )
        unmasked_probs, unmasked_routes = unmasked(hidden, padding_mask=padding)
        assert torch.equal(candidate_probs, unmasked_probs)
        assert torch.equal(candidate_routes, unmasked_routes)
        (candidate_probs.sum() * 0).backward()
        actual = candidate.weight.grad
        if per_token:
            actual = actual / max(n, 1)
        if n:
            expected_probs, _ = reference(hidden[keep])
            (expected_probs.sum() * 0).backward()
            expected = reference.weight.grad
        else:
            expected = torch.zeros_like(actual)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("first_empty", [False, True])
def test_sequence_aux_weights_and_empty_sequences(per_token, first_empty):
    """Use token means with per-token loss, otherwise means over nonempty sequences."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(333)
        hidden = torch.randn(8, 2, 8, device="cuda", dtype=torch.float64)
        mask = torch.zeros(8, 2, dtype=torch.bool, device="cuda")
        mask[: 0 if first_empty else 2, 0] = True
        mask[:6, 1] = True
        router = make_router("seq_aux_loss", per_token, True, groups)
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        scores, _ = router(hidden, moe_aux_loss_mask=mask)
        (scores.sum() * 0).backward()
        actual = router.weight.grad / (int(mask.sum()) if per_token else 1)
        expected = torch.zeros_like(actual)
        divisor = int(mask.sum()) if per_token else int(mask.any(dim=0).sum())
        for column in range(2):
            n = int(mask[:, column].sum())
            if not n:
                continue
            reference = make_router("seq_aux_loss", False, False, groups)
            reference.load_state_dict(router.state_dict())
            scores, _ = reference(hidden[mask[:, column], column : column + 1])
            (scores.sum() * 0).backward()
            expected.add_(reference.weight.grad * (n if per_token else 1) / divisor)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("per_token", [False, True])
def test_global_auxiliary_gradient_matches_global_target_mean(per_token):
    """A global statistic must not acquire an extra DP mean or local-count weight."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        rank = groups.dp.rank()
        hidden = torch.arange(128, device="cuda", dtype=torch.float64).reshape(16, 1, 8) / 64 + rank
        mask = (torch.arange(16, device="cuda") < 3 + rank).reshape(16, 1)
        all_hidden = [torch.empty_like(hidden) for _ in range(groups.dp.size())]
        all_masks = [torch.empty_like(mask) for _ in range(groups.dp.size())]
        torch.distributed.all_gather(all_hidden, hidden, group=groups.dp)
        torch.distributed.all_gather(all_masks, mask, group=groups.dp)
        global_hidden = torch.cat(all_hidden, dim=0)
        global_mask = torch.cat(all_masks, dim=0).squeeze(1)
        candidate = make_router("global_aux_loss", per_token, True, groups)
        reference = make_router("global_aux_loss", False, False, groups)
        reference.weight.data.copy_(candidate.weight.data)
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        scores, _ = candidate(hidden, moe_aux_loss_mask=mask)
        (scores.sum() * 0).backward()
        actual = candidate.weight.grad.clone()
        torch.distributed.all_reduce(actual, group=groups.dp)
        actual /= int(global_mask.sum()) if per_token else groups.dp.size()
        scores, _ = reference(global_hidden[global_mask])
        (scores.sum() * 0).backward()
        torch.testing.assert_close(actual, reference.weight.grad, atol=2e-6, rtol=2e-5)
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("per_token", [False, True])
def test_global_statistics_ignore_prefix_partition_and_empty_batches(per_token):
    """A cumulative expert distribution depends on tokens, not how they were batched."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        split = make_router("global_aux_loss", per_token, True, groups)
        merged = make_router("global_aux_loss", per_token, True, groups)
        merged.weight.data.copy_(split.weight.data)
        first = torch.randn(12, 1, 8, device="cuda", dtype=torch.float64)
        second = torch.randn(12, 1, 8, device="cuda", dtype=torch.float64) + 2
        mask1 = (torch.arange(12, device="cuda") < 2).reshape(12, 1)
        mask2 = (torch.arange(12, device="cuda") < 9).reshape(12, 1)
        split(first, moe_aux_loss_mask=mask1)
        split(second, moe_aux_loss_mask=torch.zeros_like(mask2))
        split(second, moe_aux_loss_mask=mask2)
        merged(torch.cat([first, second]), moe_aux_loss_mask=torch.cat([mask1, mask2]))
        assert split.global_num_tokens == merged.global_num_tokens
        probe = torch.randn(8, 1, 8, device="cuda", dtype=torch.float64)
        probe_mask = torch.ones(8, 1, device="cuda", dtype=torch.bool)
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        scores, _ = split(probe, moe_aux_loss_mask=probe_mask)
        (scores.sum() * 0).backward()
        scores, _ = merged(probe, moe_aux_loss_mask=probe_mask)
        (scores.sum() * 0).backward()
        torch.testing.assert_close(split.weight.grad, merged.weight.grad, atol=2e-6, rtol=2e-5)
        split.reset_global_aux_loss_tracker()
        assert split.global_num_tokens == 0
        assert split.global_tokens_per_expert.count_nonzero() == 0
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("per_token", [False, True])
@pytest.mark.parametrize("empty_shard", [False, True])
@pytest.mark.parametrize("kind", ["aux_loss", "seq_aux_loss", "z_loss"])
def test_aux_loss_is_token_weighted_across_context_shards(per_token, empty_shard, kind):
    """Compare CP aggregation with direct auxiliary formulas on all selected tokens."""
    Utils.initialize_model_parallel(context_parallel_size=2)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        router = make_router(kind, per_token, True, groups)
        hidden = torch.randn(8, 1, 8, device="cuda", dtype=torch.float64) + groups.cp.rank()
        count = 5 if groups.cp.rank() == 0 else (0 if empty_shard else 1)
        mask = (torch.arange(8, device="cuda") < count).reshape(8, 1)
        hidden_parts = [torch.empty_like(hidden) for _ in range(2)]
        mask_parts = [torch.empty_like(mask) for _ in range(2)]
        torch.distributed.all_gather(hidden_parts, hidden, group=groups.cp)
        torch.distributed.all_gather(mask_parts, mask, group=groups.cp)
        full_mask = torch.cat(mask_parts).squeeze(1)
        full_hidden = torch.cat(hidden_parts)[full_mask]
        reference_weight = router.weight.detach().clone().requires_grad_(True)
        logits = full_hidden @ reference_weight.T
        if kind == "z_loss":
            loss = 0.1 * torch.logsumexp(logits, dim=-1).square().mean()
        else:
            probabilities = logits.softmax(dim=-1)
            counts = torch.bincount(logits.topk(2, dim=-1).indices.flatten(), minlength=4)
            loss = 0.1 * 4 * (probabilities.sum(dim=0) * counts).sum() / (2 * logits.shape[0] ** 2)
        expected = torch.autograd.grad(loss, reference_weight)[0]
        MoEAuxLossAutoScaler.set_loss_scale(torch.ones((), device="cuda"))
        scores, _ = router(hidden, moe_aux_loss_mask=mask)
        (scores.sum() * 0).backward()
        actual = router.weight.grad.clone()
        torch.distributed.all_reduce(actual, group=groups.cp)
        if per_token:
            actual /= int(full_mask.sum())
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


def test_global_count_buffers_survive_low_precision_conversion():
    """BF16 module conversion must not round cumulative expert/token counts."""
    Utils.initialize_model_parallel()
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        router = make_router("global_aux_loss", True, True, groups)
        counts = torch.arange(4, device="cuda", dtype=torch.int64) + 2**25
        router.global_tokens_per_expert.copy_(counts)
        router.global_num_tokens.fill_(2**25 + 1)
        router.bfloat16()
        assert router.global_tokens_per_expert.dtype == torch.int64
        assert router.global_num_tokens.dtype == torch.int64
        assert torch.equal(router.global_tokens_per_expert, counts)
        assert router.global_num_tokens == 2**25 + 1
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize(
    "feature, value",
    [
        ("sequence_packing_scheduler", "dp_balanced"),
        ("mtp_num_layers", 1),
        ("cuda_graph_impl", "local"),
        ("enable_cuda_graph", True),
        ("external_cuda_graph", True),
        ("dynamic_context_parallel", True),
        ("overlap_moe_expert_parallel_comm", True),
        ("enable_mhc_connections", True),
        ("mlp_chunks_for_training", 2),
    ],
)
def test_unsupported_mask_execution_paths_fail_explicitly(feature, value):
    """Reject paths that cannot carry the mask, rather than silently using all tokens."""
    with pytest.raises(ValueError, match="MoE auxiliary target masking"):
        TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            num_moe_experts=4,
            moe_aux_loss_masking=True,
            max_seqlen_per_dp_cp_rank=16,
            **{feature: value},
        )
