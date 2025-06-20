# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import os
import sys

import pytest
import torch
import torch.distributed

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import upcycling_utils
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.utils import get_te_version, is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    unwrap_model,
)
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

_SEED = 42


def _find_submodule(model, submodule_name):
    """
    Find sub-module in model
    """
    for name, submodule in model.named_modules():
        if name.endswith("." + submodule_name) or name == submodule_name:
            return submodule
    return None


def model_provider(
    pre_process=True,
    post_process=True,
    layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
    **config_kwargs,
):
    model_parallel_cuda_manual_seed(_SEED)
    args = get_args()

    config = core_transformer_config_from_args(args)
    use_te = args.transformer_impl == "transformer_engine"
    if use_te:
        layer_spec_fn = get_gpt_layer_with_transformer_engine_spec
    else:
        layer_spec_fn = get_gpt_layer_local_spec

    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec_fn(
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        ),
        vocab_size=args.vocal_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=True,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )
    return model


def create_test_args(tp, grouped_gemm, swiglu, squared_relu, use_te):
    destroy_global_vars()
    destroy_num_microbatches_calculator()

    sys.argv = ['test_upcycling.py']
    args = parse_args()
    args.num_layers = 2
    args.vocal_size = 256
    args.hidden_size = 128
    args.num_attention_heads = 8
    args.max_position_embeddings = 256
    args.micro_batch_size = 1
    args.create_attention_mask_in_dataloader = True
    args.seq_length = 256
    args.tensor_model_parallel_size = tp
    if tp > 1:
        # During training, performance may degrade if MoE and tensor
        # parallelismare enabled without also enabling sequence parallelism.
        args.sequence_parallel = True
    args.context_parallel_size = 1
    args.num_experts = None
    args.train_iters = 1
    args.ckpt_format = 'torch_dist'
    args.moe_router_topk = 2
    args.moe_router_pre_softmax = False
    args.lr = 3e-5
    args.attention_dropout = 0.0
    args.hidden_dropout = 0.0
    args.async_tensor_model_parallel_allreduce = False
    args.no_save_optim = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.moe_grouped_gemm = grouped_gemm
    args.transformer_impl = "transformer_engine" if use_te else "local"
    args.bf16 = True
    args.add_bias_linear = False
    args.moe_token_dispatcher_type = "alltoall"

    args.swiglu = swiglu
    args.squared_relu = squared_relu
    if args.squared_relu == True:
        assert args.swiglu == False, 'must set swiglu=False while squared_relu==True'
        args.bias_gelu_fusion = False
        args.bias_swiglu_fusion = False

    validate_args(args)
    set_global_variables(args, False)
    return args


def set_upcycling_args(ep, granularity, num_experts=8):
    args = get_args()
    args.moe_use_upcycling = True
    args.num_experts = num_experts
    args.expert_model_parallel_size = ep
    args.moe_upcycling_granularity = granularity
    dense_ffn_hidden_size = args.ffn_hidden_size
    args.ffn_hidden_size = dense_ffn_hidden_size // args.moe_upcycling_granularity
    args.moe_ffn_hidden_size = dense_ffn_hidden_size // args.moe_upcycling_granularity
    set_args(args)


def set_bias_value(dense_model):
    # change the bias value, make sure they are not zero
    state_dict = dense_model[0].state_dict()
    for name in state_dict:
        if name.endswith("bias"):
            value = state_dict[name]
            value = torch.randn(value.shape)
            state_dict[name] = value
    dense_model[0].load_state_dict(state_dict, strict=True)


def get_batch(data_iterator):
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


class TestGPTModel:
    def setup_method(self, method):
        Utils.destroy_model_parallel()
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    @pytest.mark.parametrize(
        ('tp_ep', 'granularity', 'grouped_gemm', 'swiglu', 'squared_relu'),
        [pytest.param((1, 1), 1, False, False, False)],
    )
    def test_upcycling_Local(self, tp_ep, granularity, grouped_gemm, swiglu, squared_relu):
        tp = tp_ep[0]
        ep = tp_ep[1]
        args = create_test_args(tp, grouped_gemm, swiglu, squared_relu, use_te=False)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        )

        dense_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder
        )
        data = list(range(args.seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        )
        attention_mask = torch.ones(
            (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=bool
        ).cuda()
        dense_model = unwrap_model(dense_model)
        set_bias_value(dense_model)
        dense_logits = dense_model[0].forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, expert_model_parallel_size=ep
        )
        set_upcycling_args(ep, granularity, num_experts=2)
        # model_parallel_cuda_manual_seed(_SEED+1)
        moe_model = get_model(model_provider, ModelType.encoder_or_decoder)

        # Upcycle the dense model to the MoE model
        moe_model = unwrap_model(moe_model)

        state_dict = upcycling_utils.upcycle_state_dict(moe_model, dense_model)
        if len(moe_model) == 1:
            moe_model[0].load_state_dict(state_dict['model'], strict=True)
        else:
            for i in range(len(moe_model)):
                moe_model[i].load_state_dict(state_dict['model%d' % i], strict=True)

        moe_logits = moe_model[0].forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        # Compare the outputs of the MoE model and the dense model.
        assert torch.allclose(
            moe_logits, dense_logits, rtol=1e-01, atol=1e-01
        ), "The output of moe model do not match the output of dense model."

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(
        ('tp_ep', 'granularity', 'grouped_gemm', 'swiglu', 'squared_relu'),
        [
            pytest.param((1, 2), 1, False, False, False),
            pytest.param((1, 2), 2, False, False, False),
            pytest.param((1, 2), 1, True, False, False),
            pytest.param((2, 1), 1, True, False, False),
            pytest.param((1, 2), 2, True, False, False),
            pytest.param((1, 2), 2, True, False, True),
            pytest.param((1, 2), 2, True, True, False),
        ],
    )
    def test_upcycling_TE(self, tp_ep, granularity, grouped_gemm, swiglu, squared_relu):
        tp = tp_ep[0]
        ep = tp_ep[1]
        args = create_test_args(tp, grouped_gemm, swiglu, squared_relu, use_te=True)
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        )

        dense_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder
        )
        data = list(range(args.seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        )
        attention_mask = torch.ones(
            (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=bool
        ).cuda()
        dense_model = unwrap_model(dense_model)
        set_bias_value(dense_model)
        dense_logits = dense_model[0].forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp, expert_model_parallel_size=ep
        )
        set_upcycling_args(ep, granularity)
        # model_parallel_cuda_manual_seed(_SEED+1)
        moe_model = get_model(model_provider, ModelType.encoder_or_decoder)

        # Upcycle the dense model to the MoE model
        moe_model = unwrap_model(moe_model)

        state_dict = upcycling_utils.upcycle_state_dict(moe_model, dense_model)
        if len(moe_model) == 1:
            moe_model[0].load_state_dict(state_dict['model'], strict=True)
        else:
            for i in range(len(moe_model)):
                mpu.set_virtual_pipeline_model_parallel_rank(i)
                moe_model[i].load_state_dict(state_dict['model%d' % i], strict=True)

        moe_logits = moe_model[0].forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )
        # Compare the outputs of the MoE model and the dense model.
        assert torch.allclose(
            moe_logits, dense_logits, rtol=1e-01, atol=1e-01
        ), "The output of moe model do not match the output of dense model."
