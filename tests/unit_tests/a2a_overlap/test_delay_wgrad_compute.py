# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import gc

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from megatron.core.models.gpt.gpt_model import GPTModel
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

NUM_STEPS = 3
SEQ_LEN = 128
VOCAB_SIZE = 512
LR = 0.01


def _build_gpt_model(config):
    """Build and return a GPTModel on CUDA from the given config."""
    layer_spec = get_gpt_decoder_block_spec(config=config, use_transformer_engine=True)
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        vocab_size=VOCAB_SIZE,
        pre_process=True,
        post_process=True,
        max_sequence_length=300,
    )
    model.cuda()
    return model


def _build_input_data():
    """Build fixed input data for the model."""
    return {
        "input_ids": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.int64).cuda(),
        "labels": torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN), dtype=torch.int64).cuda(),
        "position_ids": torch.arange(SEQ_LEN, dtype=torch.int64).unsqueeze(0).cuda(),
        "attention_mask": torch.ones((1, 1, SEQ_LEN, SEQ_LEN), dtype=bool).cuda(),
    }


def _train_step(model, optimizer, data):
    """Run one forward-backward-optimizer step. Return the detached loss."""
    optimizer.zero_grad()
    loss = model.forward(**data)
    loss = float16_to_fp32(loss)
    loss.backward(torch.ones_like(loss))
    optimizer.step()
    return loss.detach().clone()


def _assert_models_equal(ref_model, test_model):
    """Assert that all parameters of two models are bit-identical."""
    rank = torch.distributed.get_rank()
    for (name_r, param_r), (_, param_t) in zip(
        ref_model.named_parameters(), test_model.named_parameters()
    ):
        assert torch.equal(
            param_r.data, param_t.data
        ), f"[rank {rank}] Parameter mismatch after training: {name_r}"


class TestDelayWgradCompute:
    """Verify that delay_wgrad_compute_for_te_grouped_gemm produces identical
    training behaviour (per-step loss and final weights) as the non-delayed baseline
    across multiple forward-backward-optimizer steps on the full GPTModel.
    """

    def setup_method(self, method):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("fp8_flag", get_valid_fp8_flags())
    def test_delay_wgrad_compute_for_te_grouped_gemm(
        self, shared_expert_intermediate_size, dispatcher_type, fp8_flag
    ):
        """Verify that delay_wgrad_compute_for_te_grouped_gemm produces identical
        per-step loss and final weights as the non-delayed baseline across multiple
        forward-backward-optimizer steps on the full GPTModel.

        Covers single/multi-layer, with/without shared experts, dispatcher types,
        and FP8 modes.
        """
        num_layers = 4
        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        if fp8_flag is not None:
            extra_kwargs["fp8"] = fp8_flag[0]
            extra_kwargs["fp8_recipe"] = fp8_flag[1]
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size

        with deterministic_mode():
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            ref_model = _build_gpt_model(ref_config)
            init_params = reset_model(ref_model)

            delay_kwargs = {**extra_kwargs, "delay_wgrad_compute_for_te_grouped_gemm": True}
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=delay_kwargs)
            test_model = _build_gpt_model(test_config)
            reset_model(test_model, init_params)

            data = _build_input_data()
            ref_opt = torch.optim.SGD(ref_model.parameters(), lr=LR)
            test_opt = torch.optim.SGD(test_model.parameters(), lr=LR)

            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                ref_loss = _train_step(ref_model, ref_opt, data)
                test_loss = _train_step(test_model, test_opt, data)
                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            _assert_models_equal(ref_model, test_model)

            del ref_model, test_model
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    def test_delay_wgrad_compute_for_te_grouped_gemm_with_fsdp(
        self, shared_expert_intermediate_size, dispatcher_type
    ):
        """Verify delayed wgrad with MegatronFSDP wrapping.

        The delayed wgrad path defers the FSDP reduce-scatter for expert
        parameters until the wgrad computation completes on a separate stream.
        This test checks that the deferred reduce-scatter produces identical
        per-step loss and final weights as the non-delayed FSDP baseline.
        """
        from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import (
            fully_shard_model,
            fully_shard_optimizer,
        )

        num_layers = 4
        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size

        with deterministic_mode():
            # Build reference model (no delay) and wrap with FSDP
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            ref_model = _build_gpt_model(ref_config)
            init_params = reset_model(ref_model)

            ref_fsdp = fully_shard_model(module=ref_model, fsdp_unit_modules=[TransformerLayer])
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            # Build test model (with delay) and wrap with FSDP
            delay_kwargs = {**extra_kwargs, "delay_wgrad_compute_for_te_grouped_gemm": True}
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=delay_kwargs)
            test_model = _build_gpt_model(test_config)
            reset_model(test_model, init_params)

            test_fsdp = fully_shard_model(module=test_model, fsdp_unit_modules=[TransformerLayer])
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            data = _build_input_data()
            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                ref_loss = _train_step(ref_fsdp, ref_opt, data)
                test_loss = _train_step(test_fsdp, test_opt, data)
                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            _assert_models_equal(ref_fsdp, test_fsdp)

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()
