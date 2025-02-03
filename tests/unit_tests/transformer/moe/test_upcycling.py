# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
import sys

import pytest
import torch
import torch.distributed

from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.moe import upcycling_utils
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

_SEED = 42


def model_provider(
    pre_process=True, post_process=True, layer_spec_fn=get_gpt_layer_local_spec, **config_kwargs
):
    model_parallel_cuda_manual_seed(_SEED)
    args = get_args()

    config = core_transformer_config_from_args(args)

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


def create_test_args(
    tensor_model_parallel_size, pipeline_model_parallel_size, enable_vp, enable_grouped_gemm
):
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
    args.pipeline_model_parallel_size = pipeline_model_parallel_size
    args.tensor_model_parallel_size = tensor_model_parallel_size
    args.context_parallel_size = 1
    args.num_experts = None
    args.train_iters = 1
    if enable_vp:
        args.num_layers_per_virtual_pipeline_stage = 1
    args.ckpt_format = 'torch_dist'
    args.moe_router_topk = 2
    args.moe_router_pre_softmax = False
    args.moe_token_dispatcher_type = "alltoall"
    args.lr = 3e-5
    args.attention_dropout = 0.0
    args.hidden_dropout = 0.0
    args.async_tensor_model_parallel_allreduce = False
    args.no_save_optim = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.moe_grouped_gemm = enable_grouped_gemm
    args.add_bias_linear = False

    validate_args(args)
    set_global_variables(args, False)
    return args


def set_upcycling_args(enable_grouped_gemm, ep):
    args = get_args()
    args.moe_use_upcycling = True
    args.num_experts = 2
    args.moe_grouped_gemm = enable_grouped_gemm
    args.expert_model_parallel_size = ep
    set_args(args)


def get_batch(data_iterator):
    if (not mpu.is_pipeline_first_stage()) and (not mpu.is_pipeline_last_stage()):
        return None, None, None, None, None

    batch = get_batch_on_this_tp_rank(data_iterator)
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


class TestGPTModel:
    def setup_method(self, method):
        Utils.destroy_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    @pytest.mark.internal
    @pytest.mark.parametrize(
        ('tp_pp_ep', 'enable_vp', 'enable_grouped_gemm'), [((1, 1, 2), (False), (False))]
    )
    def test_upcycling(self, tp_pp_ep, enable_vp, enable_grouped_gemm):
        tp = tp_pp_ep[0]
        pp = tp_pp_ep[1]
        ep = tp_pp_ep[2]
        args = create_test_args(tp, pp, enable_vp, enable_grouped_gemm)
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        )

        dense_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_provider, ModelType.encoder_or_decoder
        )

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp,
            pipeline_model_parallel_size=pp,
            expert_model_parallel_size=ep,
            virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        )
        set_upcycling_args(enable_grouped_gemm, ep)
        # model_parallel_cuda_manual_seed(_SEED+1)
        moe_model = get_model(model_provider, ModelType.encoder_or_decoder)

        # Upcycle the dense model to the MoE model
        moe_model = unwrap_model(moe_model)
        dense_model = unwrap_model(dense_model)

        data = list(range(args.seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((args.micro_batch_size, 1)).cuda()
        )
        attention_mask = torch.ones(
            (args.micro_batch_size, 1, args.seq_length, args.seq_length), dtype=bool
        ).cuda()

        dense_logits = dense_model[0].forward(
            input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask
        )

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

        torch.allclose(dense_logits, moe_logits, rtol=1e-03, atol=1e-03)
