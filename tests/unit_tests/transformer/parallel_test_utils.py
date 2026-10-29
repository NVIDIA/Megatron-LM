# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared attention parity and checkpoint-aware parameter-gradient comparisons."""

import os
from contextlib import contextmanager
from dataclasses import replace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.dist_checkpointing.mapping import ShardedTensor, ShardedTensorFactory
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import unwrap_model
from megatron.training.arguments import parse_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.training import get_model
from tests.unit_tests.dist_checkpointing import (
    TempNamedDir,
    init_basic_mock_args,
    init_checkpointing_mock_args,
)
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.packed_seq_test_utils import make_test_packed_seq_params


def parameter_gradient_shards(module, cp_group=None, tp_group=None):
    """Return canonical gradient sections, reducing CP and replicated SP gradients.

    Checkpoint factories describe interleaved Q/K/V projection partitions too,
    so TP parity does not incorrectly concatenate a whole fused projection.
    Collect all gradients before checking values to avoid rank-asymmetric exits
    while peers are still waiting in a parameter collective.
    """
    gradients = {}
    parameters = list(module.named_parameters())
    present = torch.tensor(
        [parameter.grad is not None for _, parameter in parameters],
        device=parameters[0][1].device,
        dtype=torch.int32,
    )
    minimum, maximum = present.clone(), present.clone()
    torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX)
    assert torch.equal(minimum, maximum), "parameter gradient participation differs across ranks"
    for name, param in module.named_parameters():
        if param.grad is None:
            continue
        gradient = param.grad.detach().float().clone()
        if cp_group is not None and cp_group.size() > 1:
            torch.distributed.all_reduce(gradient, group=cp_group)
        if (
            tp_group is not None
            and tp_group.size() > 1
            and getattr(param, "sequence_parallel", False)
        ):
            torch.distributed.all_reduce(gradient, group=tp_group)
        gradients[name] = gradient

    result = {}

    def collect(value):
        if isinstance(value, ShardedTensor):
            result[value.key] = (value.data.detach().clone(), value.global_offset)
        elif isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)

    for name, value in module.sharded_state_dict().items():
        if name not in gradients:
            continue
        if isinstance(value, ShardedTensorFactory):
            collect(replace(value, data=gradients[name]).build())
        elif isinstance(value, ShardedTensor):
            collect(replace(value, data=gradients[name], dtype=torch.float32))
    return result


def assert_parameter_gradient_shards_close(reference, actual, *, atol, rtol):
    """Compare each TP partition against the corresponding serial section."""
    assert reference.keys() == actual.keys()
    for name, (gradient, offset) in actual.items():
        baseline, baseline_offset = reference[name]
        assert not any(baseline_offset), "reference must be unsharded"
        selection = tuple(slice(start, start + size) for start, size in zip(offset, gradient.shape))
        torch.testing.assert_close(
            gradient,
            baseline[selection],
            atol=atol,
            rtol=rtol,
            msg=lambda msg, param_name=name: f"Mismatch in parameter gradient {param_name}: {msg}",
        )


@contextmanager
def synchronized_assertions():
    """Propagate a numerical assertion to every rank before collective teardown."""
    error = None
    try:
        yield
    except AssertionError as caught:
        error = caught
    failed = torch.tensor(int(error is not None), device="cuda", dtype=torch.int32)
    torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
    if failed.item():
        if error is not None:
            raise error
        raise AssertionError("parallel parity comparison failed on another distributed rank")


def _test_parallel_attention_correctness(
    transformer_config,
    transformer_layer_spec,
    tmp_path_dist_ckpt,
    atol,
    rtol,
    tp=1,
    sp=False,
    cp=1,
    seed=123,
    sequence_length=256,
    micro_batch_size=4,
    sequence_packing=False,
    cp_partition_mode="zigzag",
    input_grad_atol=None,
    input_grad_rtol=None,
    dynamic_cp=False,
    check_parameter_gradients=False,
    parameter_atol=None,
    parameter_rtol=None,
):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size < tp * cp or world_size % (tp * cp):
        pytest.skip(f"TP={tp}, CP={cp} parity requires world size divisible by {tp * cp}")
    if dynamic_cp:
        assert cp > 1, "Runtime Dynamic CP parity requires cp > 1"
        assert sequence_packing, "Runtime Dynamic CP parity requires packed THD input"
        if int(os.environ.get("WORLD_SIZE", "1")) < tp * cp:
            pytest.skip(f"Runtime CP={cp} parity requires at least {cp} distributed ranks")

    # Model initialization function
    def initialize_gpt_model(
        config, pre_process=True, post_process=True, vp_stage=None, pg_collection=None
    ):
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
            pg_collection=pg_collection,
        )
        return gpt_model

    # Initialize baseline parallel state
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=1
    )

    # Initialize input hidden states
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)
    input_hidden_states = (
        torch.rand((sequence_length, micro_batch_size, transformer_config.hidden_size))
        .cuda()
        .bfloat16()
        .requires_grad_(True)
    )

    with TempNamedDir(tmp_path_dist_ckpt / 'test_parallel_attn', sync=True) as ckpt_dir:
        # Set argument
        mock_args = parse_args(ignore_unknown_args=True)
        mock_args.save_tokenizer_assets = False
        set_args(mock_args)

        # These tests compare raw per-module gradients and reduce CP/SP
        # explicitly. DDP hooks instead move .grad into .main_grad and clear
        # it, which makes direct gradient inspection race-dependent.
        # Initialize baseline model
        init_basic_mock_args(mock_args, 1, 1, bf16=True)
        mock_args.context_parallel_size = 1
        mock_args.sequence_parallel = 1
        gpt_model = unwrap_model(
            get_model(initialize_gpt_model, config=transformer_config, wrap_with_ddp=False)
        )

        # Initialize args and save checkpoint
        init_checkpointing_mock_args(mock_args, ckpt_dir, False)
        mock_args.no_save_optim = True
        mock_args.no_save_rng = True
        mock_args.no_load_optim = True
        mock_args.no_load_rng = True
        save_checkpoint(10, gpt_model, None, None, 0)

        # Calculate baseline output
        attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_baseline, bias_hidden_states_baseline = attention(
            input_hidden_states, attention_mask=None
        )
        output_hidden_states_baseline.sum().backward()
        parameter_grads_baseline = (
            parameter_gradient_shards(attention) if check_parameter_gradients else None
        )

        # Save baseline output
        input_grad_baseline = input_hidden_states.grad.detach()
        output_hidden_states_baseline = output_hidden_states_baseline.detach()
        bias_hidden_states_baseline = bias_hidden_states_baseline
        if bias_hidden_states_baseline is not None:
            bias_hidden_states_baseline = bias_hidden_states_baseline.detach()
            has_bias = True
        else:
            has_bias = False

        # Initialize parallel model
        Utils.destroy_model_parallel()
        build_time_cp = 1 if dynamic_cp else cp
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=1,
            context_parallel_size=build_time_cp,
            dynamic_context_parallel=dynamic_cp,
            min_dynamic_context_parallel_size=1,
        )
        torch.manual_seed(seed)
        model_parallel_cuda_manual_seed(seed)
        transformer_config.context_parallel_size = build_time_cp
        transformer_config.tensor_model_parallel_size = tp
        transformer_config.sequence_parallel = sp
        init_basic_mock_args(mock_args, tp, 1, bf16=True)
        mock_args.context_parallel_size = build_time_cp
        mock_args.sequence_parallel = sp
        gpt_model = unwrap_model(
            get_model(initialize_gpt_model, config=transformer_config, wrap_with_ddp=False)
        )
        with mock.patch('megatron.training.checkpointing.check_checkpoint_args'):
            load_checkpoint(gpt_model, None, None)

        # Function to get tensor on this tp and cp rank
        cp_group = (
            parallel_state.get_dynamic_data_context_parallel_groups(group_size=cp)
            if dynamic_cp
            else parallel_state.get_context_parallel_group()
        )
        tp_rank = parallel_state.get_tensor_model_parallel_rank()

        def get_tensor_on_this_rank(tensor):
            if cp > 1 and cp_partition_mode == "zigzag":
                tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            if sequence_packing:
                tensor = tensor.transpose(0, 1).contiguous().view(-1, 1, *tensor.shape[2:])
            if cp > 1 and cp_partition_mode == "contiguous":
                # Packed contiguous CP partitions the flattened token stream, not each sequence.
                tensor = tensor.chunk(cp, dim=0)[cp_group.rank()].contiguous()
            if tp > 1 and sp:
                sp_seg = tensor.shape[0] // tp
                tensor = tensor[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
            return tensor

        # Calculate parallel model output
        if sequence_packing:
            cu_seqlens = [i * sequence_length for i in range(micro_batch_size + 1)]
            packed_seq_params = make_test_packed_seq_params(cu_seqlens=cu_seqlens)
            packed_seq_params.cp_partition_mode = cp_partition_mode
            if dynamic_cp:
                packed_seq_params.local_cp_size = cp
                packed_seq_params.cp_group = cp_group
        else:
            packed_seq_params = None
        input_hidden_states = get_tensor_on_this_rank(input_hidden_states)
        input_hidden_states = input_hidden_states.detach().requires_grad_(True)
        parallel_attention = gpt_model[0].decoder.layers[0].self_attention
        output_hidden_states_parallel, bias_hidden_states_parallel = parallel_attention(
            input_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        output_hidden_states_parallel.sum().backward()
        input_grad_parallel = input_hidden_states.grad.detach()

        parameter_grads_parallel = None
        if check_parameter_gradients:
            parameter_grads_parallel = parameter_gradient_shards(
                parallel_attention,
                cp_group=cp_group,
                tp_group=parallel_state.get_tensor_model_parallel_group(),
            )

        try:
            with synchronized_assertions():
                # Check if the output is close
                output_hidden_states_baseline = get_tensor_on_this_rank(
                    output_hidden_states_baseline
                )
                input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)

                assert torch.all(
                    ~torch.isnan(output_hidden_states_baseline)
                ), "output_hidden_states_baseline contains nan"
                assert torch.all(
                    ~torch.isinf(output_hidden_states_baseline)
                ), "output_hidden_states_baseline contains inf"
                assert torch.all(
                    ~torch.isnan(input_grad_baseline)
                ), "input_grad_baseline contains nan"
                assert torch.all(
                    ~torch.isinf(input_grad_baseline)
                ), "input_grad_baseline contains inf"
                assert torch.all(
                    ~torch.isnan(output_hidden_states_parallel)
                ), "output_hidden_states_parallel contains nan"
                assert torch.all(
                    ~torch.isinf(output_hidden_states_parallel)
                ), "output_hidden_states_parallel contains inf"
                assert torch.all(
                    ~torch.isnan(input_grad_parallel)
                ), "input_grad_parallel contains nan"
                assert torch.all(
                    ~torch.isinf(input_grad_parallel)
                ), "input_grad_parallel contains inf"
                if has_bias:
                    assert torch.all(
                        ~torch.isnan(bias_hidden_states_baseline)
                    ), "bias_hidden_states_baseline contains nan"
                    assert torch.all(
                        ~torch.isinf(bias_hidden_states_baseline)
                    ), "bias_hidden_states_baseline contains inf"
                    assert torch.all(
                        ~torch.isnan(bias_hidden_states_parallel)
                    ), "bias_hidden_states_parallel contains nan"
                    assert torch.all(
                        ~torch.isinf(bias_hidden_states_parallel)
                    ), "bias_hidden_states_parallel contains inf"

                torch.testing.assert_close(
                    output_hidden_states_baseline,
                    output_hidden_states_parallel,
                    atol=atol,
                    rtol=rtol,
                    msg=lambda msg: f"Mismatch in output_hidden_states: {msg}",
                )
                torch.testing.assert_close(
                    input_grad_baseline,
                    input_grad_parallel,
                    atol=atol if input_grad_atol is None else input_grad_atol,
                    rtol=rtol if input_grad_rtol is None else input_grad_rtol,
                    msg=lambda msg: f"Mismatch in input_grad: {msg}",
                )
                if has_bias:
                    torch.testing.assert_close(
                        bias_hidden_states_baseline,
                        bias_hidden_states_parallel,
                        atol=atol,
                        rtol=rtol,
                        msg=lambda msg: f"Mismatch in bias_hidden_states: {msg}",
                    )

                if check_parameter_gradients:
                    assert_parameter_gradient_shards_close(
                        parameter_grads_baseline,
                        parameter_grads_parallel,
                        atol=atol if parameter_atol is None else parameter_atol,
                        rtol=rtol if parameter_rtol is None else parameter_rtol,
                    )

        finally:
            Utils.destroy_model_parallel()
