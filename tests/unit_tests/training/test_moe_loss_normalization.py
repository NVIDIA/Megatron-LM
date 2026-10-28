# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Check independent main/auxiliary normalization with real router gradients."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from megatron.core.pipeline_parallel.schedules import _get_moe_loss_scale
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, clear_aux_losses_tracker
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.moe_loss_normalization import (
    input_token_moe_loss_scaling,
    prepare_moe_loss_normalization,
)
from tests.unit_tests.test_utilities import Utils


@pytest.mark.parametrize("custom", [False, True])
def test_scaling_composes_amp_and_restores_callback(custom):
    """Preserve existing scaling and restore it even when a schedule raises."""
    original = (lambda: 8.0) if custom else None
    config = SimpleNamespace(moe_grad_scale_func=original, grad_scale_func=lambda x: x * 8)
    with pytest.raises(RuntimeError, match="schedule failed"):
        with input_token_moe_loss_scaling(config, torch.tensor(0.25)):
            torch.testing.assert_close(config.moe_grad_scale_func(), torch.tensor(2.0))
            raise RuntimeError("schedule failed")
    assert config.moe_grad_scale_func is original


@pytest.mark.parametrize("cp_size", [1, 2])
@pytest.mark.parametrize("num_microbatches", [1, 3])
@pytest.mark.parametrize("kind", ["aux_loss", "z_loss"])
@pytest.mark.parametrize("empty_targets", [False, True])
def test_router_gradients_use_input_count_not_target_count(
    cp_size, num_microbatches, kind, empty_targets
):
    """Mask changes affect the main gradient, never the input-token regularizer.

    The reference averages the router's ordinary mean auxiliary loss using real
    input counts. The tested path uses per-token router backward, the actual
    schedule scaling helper, and the finalizer's summed-gradient/token division.
    Tokens, target fractions, and padding differ across DP ranks/microbatches.
    """
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        dp_rank = torch.distributed.get_rank(groups.dp)
        cp_rank = torch.distributed.get_rank(groups.cp)
        torch.manual_seed(123)
        torch.cuda.manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=8,
            num_attention_heads=2,
            num_moe_experts=4,
            add_bias_linear=False,
            moe_router_topk=2,
            moe_router_dtype="fp64",
            moe_router_load_balancing_type="aux_loss" if kind == "aux_loss" else "none",
            moe_aux_loss_coeff=0.1 if kind == "aux_loss" else 0.0,
            moe_z_loss_coeff=0.1 if kind == "z_loss" else None,
            calculate_per_token_loss=True,
            context_parallel_size=cp_size,
        )
        router = TopKRouter(config, pg_collection=groups).cuda().double().train()
        router.set_layer_number(0)
        batches = []
        activations = []
        for mb in range(num_microbatches):
            n = 10 + (dp_rank + mb) % 5
            t = 0 if empty_targets else 1 + (3 * dp_rank + mb) % 7
            tokens = torch.zeros(1, 16, dtype=torch.int64)
            tokens[:, :n] = 1
            mask = torch.zeros(1, 16)
            mask[:, :t] = 1
            batches.append({"tokens": tokens, "loss_mask": mask})
            activations.append(torch.randn(16, 1, 8, device="cuda", dtype=torch.float64) + dp_rank)

        iterator, ratio = prepare_moe_loss_normalization(
            iter(batches), num_microbatches, pad_token_id=0, tp_group=groups.tp, dp_group=groups.dp
        )
        replay = list(iterator)
        assert all(a is b for a, b in zip(replay, batches))
        counts = torch.tensor(
            [
                sum(int(b["loss_mask"].sum()) for b in batches),
                sum(int(b["tokens"].count_nonzero()) for b in batches),
            ],
            dtype=torch.float32,
            device="cuda",
        )
        torch.distributed.all_reduce(counts, group=groups.dp)
        torch.testing.assert_close(ratio, counts[0].clamp_min(1) / counts[1])

        def gradients(per_token, corrected):
            router.weight.grad = None
            config.calculate_per_token_loss = per_token
            router.calculate_per_token_loss = per_token
            main_weight = torch.tensor(1.0, device="cuda", requires_grad=True)
            main_numerator = torch.zeros((), device="cuda")
            context = input_token_moe_loss_scaling(config, ratio) if corrected else nullcontext()
            with context:
                for batch, hidden in zip(batches, activations):
                    n = int(batch["tokens"].count_nonzero())
                    hidden = hidden.chunk(cp_size, dim=0)[cp_rank].contiguous()
                    padding = batch["tokens"].eq(0).T.cuda().chunk(cp_size, dim=0)[cp_rank]
                    scale = _get_moe_loss_scale(config, hidden.device)
                    if not per_token:
                        # A DP replica's CP-reduced router mean has denominator n.
                        weight = n if kind == "aux_loss" else int((~padding).sum()) * cp_size
                        scale = scale * weight / counts[1]
                    MoEAuxLossAutoScaler.set_loss_scale(scale)
                    scores, _ = router(hidden, padding_mask=padding)
                    features = hidden[..., 0].squeeze(1)
                    mask = batch["loss_mask"].squeeze(0).cuda().chunk(cp_size)[cp_rank]
                    local_main = (features * mask).sum()
                    main_numerator.add_(local_main.detach())
                    loss = scores.sum() * 0 + main_weight * local_main
                    loss.backward()
            aux_grad = router.weight.grad.clone()
            # Same reductions/denominator as finalize_model_grads. These are sums,
            # not DDP's legacy averaging, when calculate_per_token_loss is enabled.
            torch.distributed.all_reduce(aux_grad, group=groups.dp_cp)
            torch.distributed.all_reduce(main_weight.grad, group=groups.dp_cp)
            torch.distributed.all_reduce(main_numerator, group=groups.dp_cp)
            if per_token:
                aux_grad.div_(counts[0].clamp_min(1))
            main_weight.grad.div_(counts[0].clamp_min(1))
            torch.testing.assert_close(main_weight.grad, main_numerator / counts[0].clamp_min(1))
            clear_aux_losses_tracker()
            return aux_grad

        expected = gradients(False, False)
        actual = gradients(True, True)
        biased = gradients(True, False)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(
            biased, expected * counts[1] / counts[0].clamp_min(1), atol=2e-6, rtol=2e-5
        )
        assert expected.norm() > 0
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("tp_size", [1, 2])
def test_prefetch_consumes_one_step_only_on_tp_source(tp_size):
    """TP peers must use broadcast counts and must not advance their iterators."""
    Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
    try:
        groups = ProcessGroupCollection.use_mpu_process_groups()
        dp_rank = torch.distributed.get_rank(groups.dp)
        dp_size = torch.distributed.get_world_size(groups.dp)
        tp_rank = torch.distributed.get_rank(groups.tp)
        batches = [
            {
                "tokens": torch.ones(1, 4 + dp_rank + mb, dtype=torch.int64),
                "loss_mask": torch.ones(1, 1 + mb),
            }
            for mb in range(4)
        ]
        original = iter(batches) if tp_rank == 0 else None
        for start in [0, 2]:
            prepared, ratio = prepare_moe_loss_normalization(
                original, 2, pad_token_id=0, tp_group=groups.tp, dp_group=groups.dp
            )
            target_count = dp_size * (3 + 2 * start)
            input_count = sum(9 + 2 * rank + 2 * start for rank in range(dp_size))
            torch.testing.assert_close(ratio, ratio.new_tensor(target_count / input_count))
            if tp_rank == 0:
                assert list(prepared) == batches[start : start + 2]
            else:
                assert prepared is None
        if tp_rank == 0:
            with pytest.raises(StopIteration):
                next(original)
    finally:
        Utils.destroy_model_parallel()
