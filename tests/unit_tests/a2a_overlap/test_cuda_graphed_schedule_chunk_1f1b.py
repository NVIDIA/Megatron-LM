# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
import gc
import os
import sys

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import HAVE_TE, model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.transformer.module import float16_to_fp32
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def is_deep_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_DEEP_EP

    return HAVE_DEEP_EP


def is_hybrid_ep_available():
    from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

    return HAVE_HYBRIDEP


def save(fn, message):
    with open(fn, 'w') as f:
        f.write(message)


class TestPartialCudaGraphedA2AOverlap:
    """Test that CUDA graph outputs match ep-overlapped CUDA graph outputs for various scopes."""

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        # Store original environment variable values
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
        }
        self.cuda_graph_helper = None
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'

    def teardown_method(self, method):
        # Restore original environment variable values
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_decoder_block_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(123)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            config,
            use_transformer_engine=True,
            normalization=args.normalization,
            qk_l2_norm=args.qk_l2_norm,
        )
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
        else:
            mtp_block_spec = None
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            mtp_block_spec=mtp_block_spec,
        )

    def create_test_args(
        self, cuda_graph_impl, cuda_graph_scope, cuda_graph_warmup_steps, ep_size, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_cuda_graphs.py']
        args = parse_args()
        args.num_layers = 1
        args.mtp_num_layers = None
        args.vocab_size = 1024
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.global_batch_size = self.micro_batch_size * 8
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = 2
        args.sequence_parallel = True
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.expert_model_parallel_size = ep_size
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = True
        args.position_embedding_type = "rope"
        args.rotary_percent = 1.0
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0
        args.untie_embeddings_and_output_weights = True

        # MoE settings
        args.num_experts = 16
        args.expert_model_parallel_size = ep_size
        args.moe_shared_expert_intermediate_size = 1024
        args.moe_layer_freq = kwargs.get("moe_layer_freq", "[0,0,1,1]")
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2

        # CUDA graph settings
        args.cuda_graph_impl = cuda_graph_impl
        args.cuda_graph_scope = cuda_graph_scope
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        args.use_te_rng_tracker = cuda_graph_impl != "none"

        for key, value in kwargs.items():
            assert hasattr(args, key)
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, seq_length, micro_batch_size):
        data = list(range(seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, seq_length, seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_1f1b_helper(self, gpt_model, optimizer, data, num_iters, cuda_graph_warmup_steps):
        from megatron.core.models.common.model_chunk_schedule_plan import (
            TransformerModelChunkSchedulePlan,
        )
        from megatron.core.pipeline_parallel.schedules import set_current_microbatch

        schedule_plans = []
        losses = []
        set_current_microbatch(gpt_model[0], 1)

        gpt_model[0].zero_grad_buffer()
        optimizer.zero_grad()
        assert cuda_graph_warmup_steps > 0, "cuda_graph_warmup_steps must be greater than 0"
        for fwd_mb_idx in range(num_iters + 1):
            # Capture CUDA graphs after warmup if helper is provided
            if self.cuda_graph_helper is not None and fwd_mb_idx == cuda_graph_warmup_steps:
                self.cuda_graph_helper.create_cudagraphs()

            if fwd_mb_idx < cuda_graph_warmup_steps:
                gpt_model[0].zero_grad_buffer()
                optimizer.zero_grad()
                output = gpt_model[0].forward(**data)
                schedule_plans.append(None)
            else:
                if fwd_mb_idx == cuda_graph_warmup_steps:
                    extra_schedule_plan = unwrap_model(gpt_model[0]).build_schedule_plan(**data)
                    TransformerModelChunkSchedulePlan.run(extra_schedule_plan, None)
                    schedule_plans[-1] = extra_schedule_plan
                f_schedule_plan = unwrap_model(gpt_model[0]).build_schedule_plan(**data)
                b_schedule_plan = schedule_plans[-1]
                schedule_plans.append(f_schedule_plan)
                if b_schedule_plan is not None:
                    gpt_model[0].zero_grad_buffer()
                    optimizer.zero_grad()
                output = TransformerModelChunkSchedulePlan.run(
                    f_schedule_plan,
                    b_schedule_plan,
                    b_grad=torch.ones_like(output) if fwd_mb_idx > 0 else None,
                )
            # Check output shapes
            if fwd_mb_idx < num_iters:
                assert output is not None
                assert output.shape[0] == self.micro_batch_size
                assert output.shape[1] == self.seq_length
                losses.append(output)

            if fwd_mb_idx < cuda_graph_warmup_steps:
                output.backward(torch.ones_like(output))

            for param in gpt_model[0].parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

        return losses

    def _run_test_helper(
        self,
        ep_size,
        cuda_graph_impl,
        cuda_graph_scope,
        cuda_graph_warmup_steps,
        ep_overlap=False,
        **kwargs,
    ):
        """Test fp8_param with gpt_model."""
        args = self.create_test_args(
            cuda_graph_impl,
            cuda_graph_scope,
            cuda_graph_warmup_steps,
            ep_size,
            overlap_moe_expert_parallel_comm=ep_overlap,
            **kwargs,
        )
        if ep_overlap:
            set_streams()
        set_args(args)
        torch.manual_seed(123)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=2, expert_model_parallel_size=ep_size
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        loss_list = []

        if cuda_graph_impl == "transformer_engine":
            from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        num_iters = cuda_graph_warmup_steps + 2
        data = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }
        if not ep_overlap:
            for i in range(num_iters):
                gpt_model[0].zero_grad_buffer()
                optimizer.zero_grad()

                # Capture CUDA graphs after warmup if helper is provided
                if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                    self.cuda_graph_helper.create_cudagraphs()

                output = unwrap_model(gpt_model[0]).forward(**data)
                output = float16_to_fp32(output)

                # Check output shapes
                assert output.shape[0] == self.micro_batch_size
                assert output.shape[1] == self.seq_length

                # Verify gradients
                output.backward(torch.ones_like(output))
                for param in gpt_model[0].parameters():
                    assert param.main_grad is not None

                update_successful, _, _ = optimizer.step()
                assert update_successful

                loss_list.append(output)
        else:
            loss_list = self._run_1f1b_helper(
                gpt_model, optimizer, data, num_iters, cuda_graph_warmup_steps
            )

        return loss_list

    @pytest.mark.skipif(
        not (HAVE_TE and is_te_min_version("2.10.0")),
        reason="Partial CUDA graph support requires TransformerEngine version >= 2.10.0",
    )
    @pytest.mark.parametrize("moe_dispatcher_type", ["alltoall", "deepep"])
    def test_moe_partial_cudagraph_with_ep_overlap(self, moe_dispatcher_type):
        extra_kwargs = {"moe_layer_freq": 1}
        if moe_dispatcher_type == "deepep":
            if not is_deep_ep_available():
                pytest.skip("Deep EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "deepep"
            extra_kwargs["moe_router_dtype"] = "fp32"
        elif moe_dispatcher_type == "hybridep":
            if not is_hybrid_ep_available():
                pytest.skip("Hybrid EP is not available")
            extra_kwargs["moe_token_dispatcher_type"] = "flex"
            extra_kwargs["moe_flex_dispatcher_backend"] = "hybridep"
        else:
            extra_kwargs["moe_token_dispatcher_type"] = moe_dispatcher_type

        loss_list_ref = self._run_test_helper(4, "none", None, 3, **extra_kwargs)
        for cuda_graph_scope in [
            [CudaGraphScope.attn],
            [CudaGraphScope.attn, CudaGraphScope.moe_router],
            [CudaGraphScope.attn, CudaGraphScope.moe_router, CudaGraphScope.moe_preprocess],
        ]:
            cuda_graph_warmup_steps = 3
            loss_list = self._run_test_helper(
                4,
                "transformer_engine",
                cuda_graph_scope,
                cuda_graph_warmup_steps,
                ep_overlap=True,
                **extra_kwargs,
            )
            assert len(loss_list) == len(loss_list_ref)
            for i in range(len(loss_list)):
                assert torch.equal(
                    loss_list[i].mean(), loss_list_ref[i].mean()
                ), f"scope={cuda_graph_scope}, i={i},loss_list={loss_list[i]}, loss_list_ref={loss_list_ref[i]}"
            print(f"[DEBUG] Pass {cuda_graph_scope}")
