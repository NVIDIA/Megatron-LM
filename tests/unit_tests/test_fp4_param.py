# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import os
import sys
from datetime import timedelta

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.enums import ModelType
from megatron.core.fp4_utils import is_nvfp4tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
import megatron.core.parallel_state as ps
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version


class Utils:
    """Minimal test utilities for distributed setup."""

    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    inited = False

    @staticmethod
    def initialize_distributed():
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        if not torch.distributed.is_initialized() and Utils.rank >= 0:
            print(
                f'Initializing torch.distributed with rank: {Utils.rank}, '
                f'world_size: {Utils.world_size}'
            )
            torch.cuda.set_device(Utils.rank % torch.cuda.device_count())
            torch.distributed.init_process_group(
                backend='nccl',
                world_size=Utils.world_size,
                rank=Utils.rank,
                timeout=timedelta(minutes=1),
            )
            torch.distributed.barrier()
        Utils.inited = True

    @staticmethod
    def destroy_model_parallel():
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)
        if not Utils.inited:
            return
        torch.distributed.barrier()
        ps.destroy_model_parallel()
        Utils.inited = False

    @staticmethod
    def initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        virtual_pipeline_model_parallel_size=None,
        **kwargs,
    ):
        os.environ.pop('NVTE_FLASH_ATTN', None)
        os.environ.pop('NVTE_FUSED_ATTN', None)
        os.environ.pop('NVTE_UNFUSED_ATTN', None)

        ps.destroy_model_parallel()
        Utils.initialize_distributed()
        ps.initialize_model_parallel(
            tensor_model_parallel_size,
            pipeline_model_parallel_size,
            virtual_pipeline_model_parallel_size,
            **kwargs,
        )
        Utils.inited = True

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()


class TestFP4Param:

    def setup_method(self, method):
        # FP4 GEMM requires dimensions to be multiples of 64
        self.seq_length = 512
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn()
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
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

    def create_test_args(
        self, tp, sequence_length, micro_batch_size, inference, fp4_param_gather, **kwargs
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_fp4_param.py']
        args = parse_args()
        args.num_layers = 4
        args.vocal_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 512
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.use_distributed_optimizer = not inference
        # FP4 settings
        args.fp4 = "e2m1"
        args.fp4_recipe = "nvfp4"
        args.fp4_param = fp4_param_gather
        args.ddp_bucket_size = 1024  # Create more buckets to test the rs/ag overlap.

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

    def _run_test_helper(
        self, tp_size, inference: bool = False, fp4_param_gather: bool = True, **kwargs
    ):
        """Test fp4_param with gpt_model."""
        args = self.create_test_args(
            tp_size,
            self.seq_length,
            self.micro_batch_size,
            inference,
            fp4_param_gather,
            **kwargs,
        )

        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )
        if inference:
            gpt_model = get_model(
                self.model_provider, ModelType.encoder_or_decoder, wrap_with_ddp=False
            )
            gpt_model[0].eval()
            optimizer = None
        else:
            gpt_model, optimizer, _ = setup_model_and_optimizer(
                self.model_provider, ModelType.encoder_or_decoder
            )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.

        num_fp4_params = 0
        for _, param in gpt_model[0].named_parameters():
            if not inference:
                assert param.requires_grad
                assert param.main_grad is not None
            if is_nvfp4tensor(param):
                num_fp4_params += 1

        # Verify the number of fp4 params.
        fp4_layers = args.num_layers
        if kwargs.get("first_last_layers_bf16", False):
            fp4_layers -= kwargs["num_layers_at_start_in_bf16"]
            fp4_layers -= kwargs["num_layers_at_end_in_bf16"]
        # Each layer has 4 GEMM weights: qkv, proj, fc1, fc2.
        if fp4_param_gather:
            assert num_fp4_params == 4 * fp4_layers

        loss_list = []

        for i in range(100):
            if not inference:
                gpt_model[0].zero_grad_buffer()
                optimizer.zero_grad()

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length

            if inference:
                continue

            # Verify gradients
            loss = output.mean()
            loss.backward()

            if args.overlap_grad_reduce:
                gpt_model[0].finish_grad_sync()

            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            loss_list.append(loss.item())

        return torch.tensor(loss_list)

    def run_test(self, tp_size, inference: bool = False, **kwargs):
        """Test fp4_param with gpt_model."""
        if inference:
            with torch.inference_mode():
                self._run_test_helper(tp_size, inference=True, **kwargs)
        else:
            loss_list = self._run_test_helper(tp_size, fp4_param_gather=True, **kwargs)

            # Compare with fp4_param_gather=False to verify numerical consistency
            loss_list_ref = self._run_test_helper(tp_size, fp4_param_gather=False, **kwargs)
            # FP4 has lower precision than FP8, so we use a larger tolerance
            torch.testing.assert_close(loss_list, loss_list_ref, atol=1e-2, rtol=1e-2)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="NVFP4 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.7.0.dev0"), reason="TE 2.7.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (True, True)])
    def test_nvfp4(self, tp_size, dp_overlap):
        """
        Test NVFP4 primary weights with distributed optimizer.
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="NVFP4 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.7.0.dev0"), reason="TE 2.7.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_nvfp4_inference(self, tp_size):
        """Test NVFP4 primary weights in inference mode."""
        self.run_test(tp_size=tp_size, inference=True)

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="NVFP4 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.7.0.dev0"), reason="TE 2.7.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_nvfp4_with_first_last_layers_bf16(self, tp_size):
        """Test NVFP4 primary weights with first/last layers in BF16."""
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, **kwargs)


if __name__ == "__main__":
    # Run tests directly without pytest
    test = TestFP4Param()
    test.setup_method(None)
    try:
        print("Running test_nvfp4 with dp_overlap=(False, False)...")
        test.run_test(tp_size=2, overlap_param_gather=False, overlap_grad_reduce=False)
        test.teardown_method(None)
        print("PASSED: test_nvfp4 (no overlap)")
    except Exception as e:
        print(f"FAILED: {e}")
        raise

