# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Deterministic real-MoE resume through the production checkpoint APIs."""

import gc
import random
import sys
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from megatron.core import dist_checkpointing, parallel_state, tensor_parallel
from megatron.core.distributed import DistributedDataParallel, DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.optimizer import ChainedOptimizer, get_megatron_optimizer
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.training import global_vars
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import get_rng_state, load_checkpoint, save_checkpoint
from megatron.training.determinism import apply_determinism_to_args
from megatron.training.training import get_megatron_optimizer_config
from tests.unit_tests.test_utilities import Utils


def _snapshot(model, optimizer, scheduler):
    children = optimizer.chained_optimizers
    states = []
    for child in children:
        groups = []
        for group in child.optimizer.param_groups:
            metadata = {key: deepcopy(value) for key, value in group.items() if key != "params"}
            # TE never updates empty groups; resume may normalize their missing step.
            if not group["params"]:
                metadata.pop("step", None)
            values = []
            for parameter in group["params"]:
                values.append(
                    {
                        key: (
                            value.detach().cpu().clone()
                            if torch.is_tensor(value)
                            else deepcopy(value)
                        )
                        for key, value in child.optimizer.state[parameter].items()
                    }
                )
            groups.append((metadata, values))
        states.append(groups)
    return {
        "model": {name: value.detach().cpu().clone() for name, value in model.named_parameters()},
        "buffers": {name: value.detach().cpu().clone() for name, value in model.named_buffers()},
        "optimizer": states,
        "scheduler": deepcopy(scheduler.state_dict()),
    }


def _assert_equal(actual, expected):
    if torch.is_tensor(expected):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected):
            _assert_equal(left, right)
    else:
        assert actual == expected


def _rng_state():
    return deepcopy(
        get_rng_state(
            "torch",
            parallel_state.get_tensor_model_parallel_group(),
            parallel_state.get_pipeline_model_parallel_group(),
        )
    )


@pytest.mark.parametrize(
    ("sharding", "remainders"),
    [("dp_reshardable", False), ("dp_reshardable", True), ("fully_reshardable", False)],
)
def test_moe_production_checkpoint_resume(tmp_path_dist_ckpt, monkeypatch, sharding, remainders):
    """Restore dense/expert state, scheduler and RNG, then reproduce the next update."""
    if not torch.cuda.is_available() or Utils.world_size != 8:
        pytest.skip("requires eight CUDA ranks")
    te = pytest.importorskip("transformer_engine.pytorch.optimizers")
    monkeypatch.setenv("CUDA_DEVICE_MAX_CONNECTIONS", "1")
    monkeypatch.setenv("NCCL_ALGO", "Ring")
    monkeypatch.setenv("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    apply_determinism_to_args(
        SimpleNamespace(cross_entropy_loss_fusion=False, tp_comm_overlap=False)
    )
    Utils.initialize_model_parallel(
        2, 2, expert_model_parallel_size=2, expert_tensor_parallel_size=1
    )
    global_vars.unset_global_variables()
    with patch.object(sys, "argv", [sys.argv[0]]):
        args = parse_args()
    for name, value in dict(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        data_parallel_size=2,
        gtp_weight_remat_size=1,
        micro_batch_size=1,
        global_batch_size=4,
        seq_length=64,
        consumed_train_samples=0,
        skipped_train_samples=0,
        consumed_valid_samples=0,
        ckpt_format="torch_dist",
        use_dist_ckpt=True,
        use_distributed_optimizer=True,
        use_precision_aware_optimizer=True,
        store_param_remainders=remainders,
        optimizer="adam",
        lr=1e-3,
        min_lr=1e-5,
        params_dtype=torch.bfloat16,
        # validate_args normally resolves these CLI dtype strings before construction.
        main_params_dtype=torch.float32,
        main_grads_dtype=torch.float32,
        exp_avg_dtype=torch.float32,
        exp_avg_sq_dtype=torch.float32,
        clip_grad=1.0,
        dist_ckpt_optim_fully_reshardable=sharding == "fully_reshardable",
        distrib_optim_fully_reshardable_mem_efficient=False,
        save_tokenizer_assets=False,
        enable_one_logger=False,
        bf16=True,
        fp16=False,
        num_experts=4,
        swiglu=False,
        add_position_embedding=False,
        dist_ckpt_strictness="raise_all",
    ).items():
        setattr(args, name, value)
    global_vars.set_global_variables(args, build_tokenizer=False)

    config = TransformerConfig(
        num_layers=4,
        hidden_size=128,
        num_attention_heads=4,
        ffn_hidden_size=256,
        tensor_model_parallel_size=2,
        pipeline_model_parallel_size=2,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        num_moe_experts=4,
        moe_router_topk=2,
        moe_ffn_hidden_size=128,
        moe_token_dispatcher_type="allgather",
        moe_permute_fusion=False,
        moe_grouped_gemm=False,
        moe_router_load_balancing_type="none",
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=True,
        hidden_dropout=0.1,
        attention_dropout=0.0,
        gradient_accumulation_fusion=False,
        cross_entropy_loss_fusion=False,
        attention_backend=AttnBackend.unfused,
        deterministic_mode=True,
        use_cpu_initialization=False,
        add_bias_linear=False,
    )

    def build():
        model = (
            GPTModel(
                config=config,
                transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(num_experts=4),
                vocab_size=256,
                max_sequence_length=64,
                pre_process=parallel_state.is_pipeline_first_stage(),
                post_process=parallel_state.is_pipeline_last_stage(),
                position_embedding_type="rope",
                share_embeddings_and_output_weights=False,
            )
            .cuda()
            .bfloat16()
        )
        ddp = DistributedDataParallel(
            config,
            DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=False,
                overlap_param_gather=False,
            ),
            model,
        )
        optimizer_config, _ = get_megatron_optimizer_config(args)
        assert args.fp8 is None and optimizer_config.fp8_recipe == "delayed"
        optimizer = get_megatron_optimizer(optimizer_config, [ddp])
        assert isinstance(optimizer, ChainedOptimizer)
        assert len(optimizer.chained_optimizers) == 2
        assert all(type(child.optimizer) is te.FusedAdam for child in optimizer.chained_optimizers)
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=1e-3,
            max_lr=1e-3,
            min_lr=1e-5,
            lr_warmup_steps=0,
            lr_decay_steps=40,
            lr_decay_style="linear",
            start_wd=0.01,
            end_wd=0.01,
            wd_incr_steps=40,
            wd_incr_style="constant",
        )
        config.finalize_model_grads_func = finalize_model_grads
        config.grad_scale_func = optimizer.scale_loss
        config.no_sync_func = ddp.no_sync
        return ddp, optimizer, scheduler

    def step(ddp, optimizer, scheduler, step_number):
        ddp.zero_grad_buffer()
        optimizer.zero_grad()
        generator = torch.Generator().manual_seed(
            1000 + step_number + 10 * parallel_state.get_data_parallel_rank()
        )
        batches = [torch.randint(0, 256, (1, 65), generator=generator) for _ in range(2)]

        def forward(iterator, model):
            batch = next(iterator).cuda()
            output = model(
                batch[:, :-1].contiguous(),
                torch.arange(64, device="cuda").unsqueeze(0),
                None,
                labels=batch[:, 1:].contiguous(),
            )

            def loss(value):
                result = value.float().mean()
                return result, {"loss": result.detach().clone()}

            return output, loss

        get_forward_backward_func()(
            forward_step_func=forward,
            data_iterator=iter(batches),
            model=[ddp],
            num_microbatches=2,
            seq_length=64,
            micro_batch_size=1,
            forward_only=False,
        )
        assert optimizer.step()[0]
        scheduler.step(4)
        args.consumed_train_samples += 4

    try:
        random.seed(123)
        np.random.seed(123)
        torch.manual_seed(123)
        tensor_parallel.model_parallel_cuda_manual_seed(123)
        # The shared parent fixture owns cleanup on rank zero. Avoid per-rank
        # TemporaryDirectory finalizers racing on the same distributed directory.
        directory = tmp_path_dist_ckpt / f"production-moe-{sharding}-{remainders}"
        if Utils.rank == 0:
            directory.mkdir()
        torch.distributed.barrier()
        args.save = args.load = str(directory)
        model, optimizer, scheduler = build()
        step(model, optimizer, scheduler, 1)
        saved = _snapshot(model, optimizer, scheduler)
        saved_rng = _rng_state()
        save_checkpoint(1, [model], optimizer, scheduler, 123)
        metadata = dist_checkpointing.load_content_metadata(str(directory / "iter_0000001"))
        assert metadata["distrib_optim_sharding_type"] == sharding
        step(model, optimizer, scheduler, 2)
        expected = _snapshot(model, optimizer, scheduler)
        del model, optimizer, scheduler
        config.grad_scale_func = config.no_sync_func = None
        gc.collect()

        torch.manual_seed(456)
        tensor_parallel.model_parallel_cuda_manual_seed(456)
        model, optimizer, scheduler = build()
        args.consumed_train_samples = 0
        observed = {}
        original_load = DistributedOptimizer.load_state_dict

        def observe(child, state_dict):
            result = original_load(child, state_dict)
            pointers = {
                parameter: {
                    key: value.data_ptr() for key, value in state.items() if torch.is_tensor(value)
                }
                for parameter, state in child.optimizer.state.items()
            }
            assert pointers and all(pointers.values())
            observed.setdefault(id(child), []).append(pointers)
            return result

        with patch.object(DistributedOptimizer, "load_state_dict", observe):
            assert load_checkpoint([model], optimizer, scheduler, strict=True) == (1, 123)
        assert len(observed) == 2
        for calls in observed.values():
            assert len(calls) == 2
            assert calls[0] == calls[1]
        assert args.consumed_train_samples == 4
        _assert_equal(_rng_state(), saved_rng)
        _assert_equal(_snapshot(model, optimizer, scheduler), saved)
        step(model, optimizer, scheduler, 2)
        _assert_equal(_snapshot(model, optimizer, scheduler), expected)
    finally:
        global_vars.unset_global_variables()
        Utils.destroy_model_parallel()
        torch.use_deterministic_algorithms(previous_determinism)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
