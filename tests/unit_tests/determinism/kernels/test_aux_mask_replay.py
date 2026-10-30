# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of target-masked routing through model/recompute boundaries."""

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe.moe_layer import MoELayer
from megatron.core.transformer.moe.moe_utils import MoEAuxLossAutoScaler, clear_aux_losses_tracker
from megatron.core.transformer.moe.router import TopKRouter
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_batch_on_this_tp_rank
from tests.unit_tests.determinism.kernels.harness import bytes_equal, deterministic_algorithms
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.mark.parametrize("architecture", ["gpt", "hybrid"])
@pytest.mark.parametrize("recompute", [False, True])
def test_masked_model_replays_with_retained_microbatches(architecture, recompute):
    """Two retained graphs must keep their own masks, including during recomputation."""
    Utils.initialize_model_parallel()
    try:
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            num_layers=1,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            num_moe_experts=4,
            moe_router_topk=2,
            moe_grouped_gemm=True,
            moe_token_dispatcher_type="alltoall",
            add_bias_linear=False,
            params_dtype=torch.bfloat16,
            bf16=True,
            hidden_dropout=0,
            attention_dropout=0,
            moe_aux_loss_coeff=0.1,
            moe_z_loss_coeff=0.01,
            calculate_per_token_loss=True,
            moe_aux_loss_masking=True,
            recompute_granularity="full" if recompute else None,
            recompute_method="uniform" if recompute else None,
            recompute_num_layers=1 if recompute else None,
        )
        common = dict(config=config, vocab_size=128, max_sequence_length=16)
        if architecture == "gpt":
            model = GPTModel(
                **common,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                    num_experts=4, moe_grouped_gemm=True
                ),
            )
        else:
            model = HybridModel(
                **common, hybrid_stack_spec=hybrid_stack_spec, hybrid_layer_pattern="E"
            )
        model = model.cuda().to(dtype=torch.bfloat16).train()
        moe_layers = [module for module in model.modules() if isinstance(module, MoELayer)]
        assert moe_layers, "the replay must exercise actual MoE dispatch"
        assert all(isinstance(layer.router, TopKRouter) for layer in moe_layers)
        tokens = torch.randint(1, 128, (2, 16), device="cuda")
        positions = torch.arange(16, device="cuda").expand_as(tokens)
        masks = [torch.zeros_like(tokens, dtype=torch.bool) for _ in range(2)]
        masks[0][0, 3:9] = True
        masks[1][0, 8:] = True
        masks[1][1, 2:13] = True
        MoEAuxLossAutoScaler.set_loss_scale(
            torch.tensor(1.0 / sum(int(mask.sum()) for mask in masks), device="cuda")
        )

        def run(retain):
            model.zero_grad(set_to_none=True)
            clear_aux_losses_tracker()
            outputs = []
            for mask in masks:
                output = model(tokens, positions, None, loss_mask=mask)
                outputs.append(output)
                if not retain:
                    (output.sum() * 0).backward()
            if retain:
                sum(output.sum() * 0 for output in outputs).backward()
            return (
                [output.detach().clone() for output in outputs],
                {
                    name: param.grad.detach().clone()
                    for name, param in model.named_parameters()
                    if param.grad is not None
                },
            )

        with deterministic_algorithms(True):
            reference_outputs, reference_grads = run(retain=True)
            assert any("router" in name for name in reference_grads)
            assert bytes_equal(*reference_outputs), "the auxiliary mask must not change logits"
            for _ in range(2):
                outputs, grads = run(retain=True)
                assert grads.keys() == reference_grads.keys()
                for reference, current in zip(reference_outputs, outputs):
                    assert bytes_equal(reference, current)
                for name in grads:
                    assert bytes_equal(reference_grads[name], grads[name]), name
            # Different backward order may round differently; this independent
            # execution checks that retained/recomputed graphs keep their own masks.
            _, sequential_grads = run(retain=False)
            for name in reference_grads:
                torch.testing.assert_close(
                    reference_grads[name], sequential_grads[name], atol=2e-4, rtol=0.025
                )
    finally:
        clear_aux_losses_tracker()
        MoEAuxLossAutoScaler.main_loss_backward_scale = None
        Utils.destroy_model_parallel()


def test_pipeline_moe_metadata_broadcast_replays():
    """Every PP stage receives exactly the current loss mask on its TP peers."""
    if Utils.world_size % 4:
        pytest.skip("requires a world size divisible by four")
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=2)
    try:
        group = parallel_state.get_tensor_model_parallel_group()
        for offset in (0, 5, 0):
            tokens = torch.arange(16, device="cuda").reshape(1, 16)
            mask = (tokens >= offset).float()
            sample = dict(
                tokens=tokens,
                labels=tokens + 1,
                loss_mask=mask,
                position_ids=tokens,
                attention_mask=None,
            )
            result = get_batch_on_this_tp_rank(
                sample if group.rank() == 0 else {},
                has_cu_seqlens=False,
                dynamic_cp=False,
                create_attention_mask_in_dataloader=False,
                broadcast_src_rank=torch.distributed.get_global_rank(group, 0),
                broadcast_group=group,
                cp_size=1,
                tp_rank=group.rank(),
                micro_batch_size=1,
                seq_length=16,
                mtp_on_this_rank=False,
                pipeline_model_parallel_size=2,
                is_pipeline_first_stage=parallel_state.is_pipeline_first_stage(),
                is_pipeline_last_stage=parallel_state.is_pipeline_last_stage(),
                broadcast_moe_metadata=True,
            )
            for key in ("tokens", "labels", "loss_mask", "position_ids"):
                assert bytes_equal(result[key], sample[key].cuda()), key
    finally:
        Utils.destroy_model_parallel()
