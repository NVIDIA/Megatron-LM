# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Shared MLA parallel parity harness, without collected tests or bucket marks."""

import os
from copy import copy
from dataclasses import replace
from unittest import mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.common.embeddings.rope_utils import (
    get_pos_emb_on_this_cp_rank as get_tensor_on_this_cp_rank,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_layer_with_transformer_engine_submodules,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.multi_latent_attention import MLASelfAttention
from megatron.core.transformer.transformer_config import MLATransformerConfig
from megatron.core.utils import is_te_min_version, is_torch_min_version, unwrap_model
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


def check_mla_parallel_correctness(
    tmp_path_dist_ckpt,
    rope_type,
    apply_rope_fusion,
    tp,
    sp,
    cp,
    *,
    dynamic_cp=False,
    check_parameter_gradients=False,
):
    """Compare serial, static and optional runtime MLA topology with shared weights."""
    if dynamic_cp and int(os.environ.get("WORLD_SIZE", "1")) < cp:
        pytest.skip(f"Runtime CP={cp} parity requires at least {cp} distributed ranks")
    if cp > 1 and not is_te_min_version("2.5.0", check_equality=True):
        pytest.skip("MLA CP requires TransformerEngine >= 2.5.0")
    if rope_type == "yarn" and apply_rope_fusion and not is_torch_min_version("2.5.0"):
        pytest.skip("MLA yarn rope fusion requires PyTorch >= 2.5.0")
    if (
        cp > 1
        and rope_type == "yarn"
        and apply_rope_fusion
        and not is_te_min_version("2.6.0", check_equality=True)
    ):
        pytest.skip("MLA CP + yarn rope fusion requires PyTorch >= 2.6.0")

    # Non-deterministic mode has bug to be fixed with MLA
    _environ = os.environ.copy()
    os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = "1"
    os.environ['NVTE_FUSED_ATTN'] = "1"
    os.environ['NVTE_FLASH_ATTN'] = "0"

    packed_control = check_parameter_gradients and tp == 1 and cp > 1

    # Constants
    seed = 123
    sequence_length = 256
    micro_batch_size = 4
    hidden_size = 128

    # Model initialization function
    def initialize_gpt_model(
        pre_process=True, post_process=True, vp_stage=None, pg_collection=None, config=None
    ):
        layer_spec = get_gpt_layer_with_transformer_engine_spec(multi_latent_attention=True)
        gpt_model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=sequence_length,
            pre_process=pre_process,
            post_process=post_process,
            vp_stage=vp_stage,
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
        torch.rand((sequence_length, micro_batch_size, hidden_size))
        .cuda()
        .bfloat16()
        .requires_grad_(True)
    )

    # Initialize transformer config
    transformer_config = MLATransformerConfig(
        num_layers=1,
        hidden_size=hidden_size,
        num_attention_heads=4,
        q_lora_rank=32,
        kv_lora_rank=32,
        qk_head_dim=128,
        v_head_dim=128,
        qk_pos_emb_head_dim=64,
        rotary_base=10000,
        original_max_position_embeddings=64,
        context_parallel_size=1,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        bf16=True,
        rope_type=rope_type,
        apply_rope_fusion=apply_rope_fusion,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )

    with TempNamedDir(tmp_path_dist_ckpt / 'test_parallel_mla', sync=True) as ckpt_dir:
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
        baseline_packed = (
            make_test_packed_seq_params(
                cu_seqlens=[i * sequence_length for i in range(micro_batch_size + 1)]
            )
            if packed_control
            else None
        )
        baseline_input = input_hidden_states
        if packed_control:
            baseline_input = baseline_input.transpose(0, 1).contiguous().view(-1, 1, hidden_size)
        output_hidden_states_baseline, bias_hidden_states_baseline = attention(
            baseline_input, attention_mask=None, packed_seq_params=baseline_packed
        )
        output_hidden_states_baseline.sum().backward()
        parameter_grads_baseline = (
            {
                name: param.grad.detach().clone()
                for name, param in attention.named_parameters()
                if param.grad is not None
            }
            if check_parameter_gradients
            else {}
        )

        # Save baseline output
        input_grad_baseline = input_hidden_states.grad.detach()
        output_hidden_states_baseline = output_hidden_states_baseline.detach()
        if packed_control:
            output_hidden_states_baseline = (
                output_hidden_states_baseline.view(micro_batch_size, sequence_length, hidden_size)
                .transpose(0, 1)
                .contiguous()
            )

        bias_hidden_states_baseline = bias_hidden_states_baseline.detach()

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
            if cp > 1:
                tensor = get_tensor_on_this_cp_rank(tensor, 0, cp_group)
            if packed_control:
                tensor = tensor.transpose(0, 1).contiguous().view(-1, 1, *tensor.shape[2:])
            if tp > 1 and sp:
                sp_seg = sequence_length // tp // cp
                tensor = tensor[tp_rank * sp_seg : (tp_rank + 1) * sp_seg]
            return tensor

        # Calculate parallel model output
        input_hidden_states = get_tensor_on_this_rank(input_hidden_states)
        input_hidden_states = input_hidden_states.detach().requires_grad_(True)
        parallel_attention = gpt_model[0].decoder.layers[0].self_attention
        packed_seq_params = None
        if packed_control:
            packed_seq_params = make_test_packed_seq_params(
                cu_seqlens=[i * sequence_length for i in range(micro_batch_size + 1)]
            )
            if dynamic_cp:
                packed_seq_params.local_cp_size = cp
                packed_seq_params.cp_group = cp_group
        output_hidden_states_parallel, bias_hidden_states_parallel = parallel_attention(
            input_hidden_states, attention_mask=None, packed_seq_params=packed_seq_params
        )
        output_hidden_states_parallel.sum().backward()
        input_grad_parallel = input_hidden_states.grad.detach()

        parameter_gradient_pairs = []
        if check_parameter_gradients and tp == 1 and cp > 1:
            # Accumulate CP partial gradients in FP32: each local BF16 GEMM has
            # already rounded its result, so a serial BF16 GEMM is not bit-exact.
            # Reduce every parameter before asserting, so a mismatch on one rank
            # cannot leave its peers waiting in the next parameter collective.
            for name, param in parallel_attention.named_parameters():
                if name not in parameter_grads_baseline:
                    continue
                assert param.grad is not None, f"Missing parallel gradient for {name}"
                parallel_grad = param.grad.detach().float().clone()
                torch.distributed.all_reduce(parallel_grad, group=cp_group)
                parameter_gradient_pairs.append((name, parallel_grad))

        static_gradient_pairs = []
        if dynamic_cp:
            # Compare the runtime topology to a genuine CP2-built module using
            # identical THD inputs. This isolates runtime dispatch from the BF16
            # serial-vs-partitioned GEMM/attention accumulation difference.
            static_groups = copy(parallel_attention.pg_collection)
            static_groups.cp = cp_group
            static_attention = (
                MLASelfAttention(
                    replace(transformer_config, context_parallel_size=cp),
                    get_gpt_layer_with_transformer_engine_submodules(
                        multi_latent_attention=True
                    ).self_attention.submodules,
                    layer_number=1,
                    attn_mask_type=AttnMaskType.causal,
                    pg_collection=static_groups,
                )
                .cuda()
                .bfloat16()
            )
            static_attention.load_state_dict(parallel_attention.state_dict())
            static_input = input_hidden_states.detach().clone().requires_grad_()
            static_output, _ = static_attention(
                static_input,
                attention_mask=None,
                packed_seq_params=replace(packed_seq_params, cp_group=None, local_cp_size=None),
            )
            static_output.sum().backward()
            for name, parameter in static_attention.named_parameters():
                if name not in parameter_grads_baseline:
                    continue
                gradient = parameter.grad.detach().float().clone()
                torch.distributed.all_reduce(gradient, group=cp_group)
                static_gradient_pairs.append((name, gradient))
            torch.testing.assert_close(
                static_output, output_hidden_states_parallel, atol=5e-3, rtol=5e-3
            )
            torch.testing.assert_close(static_input.grad, input_grad_parallel, atol=5e-3, rtol=5e-3)

        # Check if the output is the same
        if cp:
            atol, rtol = 5e-3, 5e-3
        else:
            atol, rtol = 5e-4, 5e-4
        output_hidden_states_baseline = get_tensor_on_this_rank(output_hidden_states_baseline)
        input_grad_baseline = get_tensor_on_this_rank(input_grad_baseline)

        assert torch.all(
            ~torch.isnan(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_baseline)
        ), "output_hidden_states_baseline contains inf"
        assert torch.all(
            ~torch.isnan(bias_hidden_states_baseline)
        ), "bias_hidden_states_baseline contains nan"
        assert torch.all(
            ~torch.isinf(bias_hidden_states_baseline)
        ), "bias_hidden_states_baseline contains inf"
        assert torch.all(~torch.isnan(input_grad_baseline)), "input_grad_baseline contains nan"
        assert torch.all(~torch.isinf(input_grad_baseline)), "input_grad_baseline contains inf"
        assert torch.all(
            ~torch.isnan(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(output_hidden_states_parallel)
        ), "output_hidden_states_parallel contains inf"
        assert torch.all(
            ~torch.isnan(bias_hidden_states_parallel)
        ), "bias_hidden_states_parallel contains nan"
        assert torch.all(
            ~torch.isinf(bias_hidden_states_parallel)
        ), "bias_hidden_states_parallel contains inf"
        assert torch.all(~torch.isnan(input_grad_parallel)), "input_grad_parallel contains nan"
        assert torch.all(~torch.isinf(input_grad_parallel)), "input_grad_parallel contains inf"

        torch.testing.assert_close(
            output_hidden_states_baseline,
            output_hidden_states_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in output_hidden_states: {msg}",
        )
        torch.testing.assert_close(
            bias_hidden_states_baseline,
            bias_hidden_states_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in bias_hidden_states: {msg}",
        )
        torch.testing.assert_close(
            input_grad_baseline,
            input_grad_parallel,
            atol=atol,
            rtol=rtol,
            msg=lambda msg: f"Mismatch in input_grad: {msg}",
        )

        for name, parallel_grad in parameter_gradient_pairs:
            torch.testing.assert_close(
                parameter_grads_baseline[name].float(),
                parallel_grad,
                # BF16 attention and GEMM backward round before the CP sum.
                # The same bound applies to the static CP2 control; the dynamic
                # vs static topology comparison below retains the tight bound.
                atol=2e-2,
                rtol=2e-2,
                msg=lambda msg, param_name=name: (
                    f"Mismatch in MLA parameter gradient {param_name}: {msg}"
                ),
            )
        # Reuse the static module already built above as the CP1-to-static-CP2
        # gradient control, instead of collecting duplicate standalone controls.
        for name, static_gradient in static_gradient_pairs:
            torch.testing.assert_close(
                parameter_grads_baseline[name].float(),
                static_gradient,
                atol=2e-2,
                rtol=2e-2,
                msg=lambda msg, param_name=name: (
                    f"Mismatch in static MLA parameter gradient {param_name}: {msg}"
                ),
            )
        for (name, gradient), (static_name, static_gradient) in zip(
            parameter_gradient_pairs, static_gradient_pairs
        ):
            assert name == static_name
            torch.testing.assert_close(gradient, static_gradient, atol=5e-3, rtol=5e-3, msg=name)

        Utils.destroy_model_parallel()

    os.environ.clear()
    os.environ.update(_environ)
