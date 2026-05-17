# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import gc

import pytest
import torch

from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
from megatron.core.distributed.fsdp.src.megatron_fsdp.fully_shard import fully_shard_optimizer
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.transformer import TransformerLayer
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version
from tests.unit_tests.a2a_overlap.utils import (
    assert_models_equal,
    build_gpt_model,
    build_input_data,
    deterministic_mode,
    fsdp_train_step,
    get_test_config,
    get_valid_flex_dispatcher_backend,
    get_valid_fp8_flags,
    get_valid_token_dispatcher_types,
    overlap_train_step,
    reset_model,
)
from tests.unit_tests.test_utilities import Utils

NUM_STEPS = 3
SEQ_LEN = 128
VOCAB_SIZE = 512
LR = 0.01


def _train_step(model, optimizer, data):
    """Run one forward-backward-optimizer step. Return the detached loss."""
    optimizer.zero_grad()
    loss = model.forward(**data)
    loss = float16_to_fp32(loss)
    loss.backward(torch.ones_like(loss))
    optimizer.step()
    return loss.detach().clone()


class TestDelayWgradCompute:
    """Verify that overlap_dispatch_backward_with_experts_wgrad produces identical
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
    def test_overlap_dispatch_backward_with_experts_wgrad(
        self, shared_expert_intermediate_size, dispatcher_type, fp8_flag
    ):
        """Verify that overlap_dispatch_backward_with_experts_wgrad produces identical
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
            ref_model = build_gpt_model(ref_config, vocab_size=VOCAB_SIZE)
            init_params = reset_model(ref_model)

            delay_kwargs = {**extra_kwargs, "overlap_dispatch_backward_with_experts_wgrad": True}
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=delay_kwargs)
            test_model = build_gpt_model(test_config, vocab_size=VOCAB_SIZE)
            reset_model(test_model, init_params)

            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
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

            assert_models_equal(ref_model, test_model)

            del ref_model, test_model
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("shared_expert_intermediate_size", [None, 512])
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    def test_overlap_dispatch_backward_with_experts_wgrad_with_fsdp(
        self, shared_expert_intermediate_size, dispatcher_type
    ):
        """Verify delayed wgrad with MegatronFSDP wrapping.

        The delayed wgrad path defers the FSDP reduce-scatter for expert
        parameters until the wgrad computation completes on a separate stream.
        This test checks that the deferred reduce-scatter produces identical
        per-step loss and final weights as the non-delayed FSDP baseline.
        """

        def _make_ddp_config():
            return DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy="optim_grads_params",
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                megatron_fsdp_main_params_dtype=None,
            )

        num_layers = 4
        extra_kwargs = {"moe_token_dispatcher_type": dispatcher_type}
        if dispatcher_type == "flex":
            extra_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()
        if shared_expert_intermediate_size is not None:
            extra_kwargs["moe_shared_expert_intermediate_size"] = shared_expert_intermediate_size

        with deterministic_mode():
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=extra_kwargs)
            ref_model = build_gpt_model(ref_config, vocab_size=VOCAB_SIZE)
            init_params = reset_model(ref_model)

            ref_fsdp = FullyShardedDataParallel(
                config=ref_config,
                ddp_config=_make_ddp_config(),
                module=ref_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            delay_kwargs = {**extra_kwargs, "overlap_dispatch_backward_with_experts_wgrad": True}
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=delay_kwargs)
            test_model = build_gpt_model(test_config, vocab_size=VOCAB_SIZE)
            reset_model(test_model, init_params)

            test_fsdp = FullyShardedDataParallel(
                config=test_config,
                ddp_config=_make_ddp_config(),
                module=test_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                ref_loss = _train_step(ref_fsdp, ref_opt, data)
                test_loss = _train_step(test_fsdp, test_opt, data)
                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            assert_models_equal(ref_fsdp, test_fsdp)

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()

    @pytest.mark.skipif(not is_te_min_version("2.3.0"), reason="Requires TE >= 2.3.0")
    @pytest.mark.parametrize("dispatcher_type", get_valid_token_dispatcher_types())
    @pytest.mark.parametrize("sharding_strategy", ["optim_grads_params", "optim_grads"])
    def test_fsdp_1f1b_delay_wgrad(self, dispatcher_type, sharding_strategy):
        """Verify FSDP + 1F1B overlap + delay_wgrad_compute.

        Compares per-step loss and final weights between:
          - Reference: FSDP + 1F1B overlap (without delay_wgrad_compute)
          - Test:      FSDP + 1F1B overlap (with delay_wgrad_compute)

        Both use combined_1f1b_schedule_for_no_pipelining with
        overlap_moe_expert_parallel_comm.  The delay_wgrad_compute flag only
        reorders wgrad computation, so results must be identical.
        """
        set_streams()

        def _make_ddp_config():
            return DistributedDataParallelConfig(
                use_megatron_fsdp=True,
                data_parallel_sharding_strategy=sharding_strategy,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
                megatron_fsdp_main_params_dtype=None,
            )

        num_layers = 2
        base_kwargs = {
            "moe_token_dispatcher_type": dispatcher_type,
            "moe_shared_expert_intermediate_size": 512,
        }
        if dispatcher_type == "flex":
            base_kwargs["moe_flex_dispatcher_backend"] = get_valid_flex_dispatcher_backend()

        with deterministic_mode():
            data = build_input_data(seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)

            # --- Reference: FSDP + 1F1B overlap, no delay_wgrad ---
            ref_kwargs = {**base_kwargs, "delay_wgrad_compute": False}
            ref_config = get_test_config(num_layers=num_layers, extra_kwargs=ref_kwargs)
            ref_model = build_gpt_model(ref_config, vocab_size=VOCAB_SIZE)
            init_params = reset_model(ref_model)

            ref_fsdp = FullyShardedDataParallel(
                config=ref_config,
                ddp_config=_make_ddp_config(),
                module=ref_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            ref_opt = torch.optim.SGD(ref_fsdp.parameters(), lr=LR)
            ref_opt = fully_shard_optimizer(optimizer=ref_opt)

            # --- Test: FSDP + 1F1B overlap, with delay_wgrad ---
            test_kwargs = {
                **base_kwargs,
                "delay_wgrad_compute": True,
                "overlap_moe_expert_parallel_comm": True,
            }
            test_config = get_test_config(num_layers=num_layers, extra_kwargs=test_kwargs)
            test_model = build_gpt_model(test_config, vocab_size=VOCAB_SIZE)
            reset_model(test_model, init_params)

            test_fsdp = FullyShardedDataParallel(
                config=test_config,
                ddp_config=_make_ddp_config(),
                module=test_model,
                fsdp_unit_modules=[TransformerLayer],
            )
            test_opt = torch.optim.SGD(test_fsdp.parameters(), lr=LR)
            test_opt = fully_shard_optimizer(optimizer=test_opt)

            rank = torch.distributed.get_rank()
            for step in range(NUM_STEPS):
                if hasattr(ref_fsdp, 'set_is_first_microbatch'):
                    ref_fsdp.set_is_first_microbatch()
                ref_loss = fsdp_train_step(ref_fsdp, ref_opt, data)
                test_loss = overlap_train_step(test_fsdp, test_opt, test_config, data)

                assert torch.equal(ref_loss, test_loss), (
                    f"[rank {rank}] Loss mismatch at step {step}: "
                    f"ref={ref_loss.item()}, test={test_loss.item()}"
                )

            assert_models_equal(ref_fsdp, test_fsdp)

            del ref_fsdp, test_fsdp, ref_opt, test_opt
            gc.collect()
            torch.cuda.empty_cache()
