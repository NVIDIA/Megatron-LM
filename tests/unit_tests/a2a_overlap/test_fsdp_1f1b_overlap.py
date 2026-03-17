# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import contextlib
import gc

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.pipeline_parallel.combined_1f1b import combined_1f1b_schedule_for_no_pipelining
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    deterministic_mode,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_fp8_flags,
    get_valid_token_dispatcher_types,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

SEQ_LEN = 32
MAX_SEQ_LEN = 300
VOCAB_SIZE = 128
NUM_STEPS = 3
LR = 0.01


def _build_data():
    """Build fixed input data for testing."""
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.int64).cuda(),
        "labels": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.int64).cuda(),
        "position_ids": torch.arange(SEQ_LEN, dtype=torch.int64).unsqueeze(0).cuda(),
        "attention_mask": torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=bool).cuda(),
    }


def _build_gpt_model(config):
    """Build a GPTModel on CUDA from the given config."""
    layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=VOCAB_SIZE,
        pre_process=True,
        post_process=True,
        max_sequence_length=MAX_SEQ_LEN,
    )
    model.cuda()
    return model


def _loss_func(output_tensor):
    """Sum per-token losses to produce a scalar suitable for backward."""
    loss = output_tensor.float().sum()
    return loss, {'lm loss': loss}


def _forward_step_func(data_iterator, model, return_schedule_plan=False):
    """Forward step function compatible with combined_1f1b_schedule_for_no_pipelining.

    When return_schedule_plan=True (used by the overlap scheduler), builds and
    returns a TransformerModelChunkSchedulePlan instead of eagerly executing
    the forward pass.
    """
    data = next(data_iterator)
    if return_schedule_plan:
        schedule_plan = model.build_schedule_plan(**data)
        return schedule_plan, _loss_func
    output = model(**data)
    return output, _loss_func


def _train_step(model, optimizer, data):
    """One standard forward-backward-optimizer step through FSDP. Return scalar loss."""
    optimizer.zero_grad()
    loss = model(**data)
    loss = float16_to_fp32(loss).sum()
    loss.backward()
    optimizer.step()
    return loss.detach().clone()


def _overlap_train_step(model, optimizer, config, data):
    """One overlap forward-backward-optimizer step. Return scalar loss."""
    optimizer.zero_grad()
    forward_data_store = []
    combined_1f1b_schedule_for_no_pipelining(
        forward_step_func=_forward_step_func,
        data_iterator=iter([data]),
        model=model,
        num_microbatches=1,
        input_tensor=None,
        output_tensor_grad=None,
        forward_data_store=forward_data_store,
        config=config,
        collect_non_loss_data=False,
        first_val_step=None,
        forward_only=False,
        no_sync_func=contextlib.nullcontext,
        total_num_tokens=torch.zeros([], dtype=torch.int, device="cuda"),
        check_first_val_step=lambda cond: cond,
    )
    torch.cuda.synchronize()
    loss = forward_data_store[0]['lm loss'].detach().clone()
    optimizer.step()
    return loss


class TestFSDP1F1BOverlap:
    """Verify FSDP integration with the 1F1B overlap schedule.

    Uses combined_1f1b_schedule_for_no_pipelining (the production code path)
    to drive the overlap schedule, then compares per-step loss and final
    weights against standard FSDP forward/backward.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )
        set_streams()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    @pytest.mark.parametrize(
        "sharding_strategy", ["optim_grads_params", "optim_grads"]
    )
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    @pytest.mark.parametrize(
        "recompute_modules",
        [[], ["core_attn", "mla_up_proj", "layernorm", "moe_act", "mlp", "shared_experts"]],
    )
    def test_fsdp_1f1b_training_step(
        self,
        dispatcher_type,
        fp8_flag,
        sharding_strategy,
        shared_expert_intermediate_size,
        recompute_modules,
    ):
        """Verify multi-step FSDP training with overlap produces identical
        per-step loss and final weights as standard FSDP training.

        Covers FP8 recipes, activation recomputation, shared experts, and
        different token dispatcher types.  Reference uses standard
        forward/backward; test uses combined_1f1b_schedule_for_no_pipelining.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            fully_shard_model,
            fully_shard_optimizer,
        )

        num_layers = 2
        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size
        if recompute_modules:
            extra_kwargs["recompute_granularity"] = "selective"
            extra_kwargs["recompute_modules"] = recompute_modules

        with deterministic_mode():
            data = _build_data()

            # --- Reference: FSDP model with standard training loop ---
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            ref_model = _build_gpt_model(ref_config)
            init_params = reset_model(ref_model)

            ref_fsdp = fully_shard_model(
                module=ref_model,
                fsdp_unit_modules=[TransformerLayer],
                zero_dp_strategy=sharding_strategy,
            )
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            # --- Test: FSDP model with overlap training loop ---
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            test_model = _build_gpt_model(test_config)
            reset_model(test_model, init_params)

            test_fsdp = fully_shard_model(
                module=test_model,
                fsdp_unit_modules=[TransformerLayer],
                enable_fine_grained_param_gather=True,
                zero_dp_strategy=sharding_strategy,
            )
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                ref_loss = _train_step(ref_fsdp, ref_opt, data)
                test_loss = _overlap_train_step(test_fsdp, test_opt, test_config, data)

                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            for (name_r, param_r), (_, param_t) in zip(
                ref_fsdp.named_parameters(), test_fsdp.named_parameters()
            ):
                assert torch.equal(
                    param_r.data, param_t.data
                ), f"[rank {rank}] Parameter mismatch after training: {name_r}"

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()
