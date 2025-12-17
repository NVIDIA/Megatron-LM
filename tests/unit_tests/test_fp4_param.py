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

        # Force deterministic behavior to eliminate run-to-run variation
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
        args.fp4_param_gather = fp4_param_gather
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
        self,
        tp_size,
        inference: bool = False,
        fp4_param_gather: bool = True,
        grad_ref: dict | None = None,
        collect_grad_ref: bool = False,
        **kwargs,
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
        grad_ref_out: dict | None = {} if collect_grad_ref else None
        first_master_mismatch_iter = None  # Track first master weight divergence

        # Helper to get DistributedOptimizer from ChainedOptimizer
        def _get_dop(opt):
            if hasattr(opt, "chained_optimizers") and opt.chained_optimizers:
                return opt.chained_optimizers[0]
            return opt

        # Helper to collect master weights by name
        def _collect_master_weights(dop, model):
            master_weights = {}
            if hasattr(dop, "model_float16_groups") and hasattr(dop, "shard_fp32_from_float16_groups"):
                for mg, sg in zip(dop.model_float16_groups, dop.shard_fp32_from_float16_groups):
                    for model_param, shard_main in zip(mg, sg):
                        if shard_main is not None:
                            for n, p in model.named_parameters():
                                if id(p) == id(model_param):
                                    master_weights[n] = shard_main.detach().cpu().clone()
                                    break
            return master_weights

        for i in range(200):
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

            # Check master weights every iteration to find first divergence
            if not inference:
                dop = _get_dop(optimizer)
                master_weights = _collect_master_weights(dop, gpt_model[0])
                
                if collect_grad_ref:
                    # Store master weights snapshot for comparison
                    grad_ref_out[f"__master_iter_{i}__"] = master_weights
                elif grad_ref is not None and first_master_mismatch_iter is None:
                    # Compare with reference (only until first mismatch found)
                    ref_key = f"__master_iter_{i}__"
                    if ref_key in grad_ref:
                        ref_masters = grad_ref[ref_key]
                        for name in sorted(master_weights.keys()):
                            if name not in ref_masters:
                                continue
                            cur = master_weights[name]
                            ref = ref_masters[name]
                            if not torch.equal(cur, ref):
                                first_master_mismatch_iter = i
                                flat_c = cur.view(-1)
                                flat_r = ref.view(-1)
                                diff = flat_c != flat_r
                                diff_count = int(diff.sum().item())
                                idx0 = int(diff.nonzero()[0].item())
                                print(f"\n{'='*60}")
                                print(f"[MASTER DEBUG] FIRST MISMATCH at iter {i}")
                                print(f"[MASTER DEBUG]   param: {name}")
                                print(f"[MASTER DEBUG]   total_diff={diff_count}/{flat_c.numel()} ({100*diff_count/flat_c.numel():.4f}%)")
                                print(f"[MASTER DEBUG]   idx={idx0} got={flat_c[idx0].item():.10e} ref={flat_r[idx0].item():.10e}")
                                print(f"{'='*60}\n")
                                break

            loss_list.append(loss.item())

        loss_t = torch.tensor(loss_list)
        if collect_grad_ref:
            assert grad_ref_out is not None
            return loss_t, grad_ref_out
        return loss_t

    def run_test(self, tp_size, inference: bool = False, **kwargs):
        """Test fp4_param with gpt_model."""
        if inference:
            with torch.inference_mode():
                self._run_test_helper(tp_size, inference=True, **kwargs)
        else:
            # Memory profiling for NVFP4 run
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print("\n=== Running with fp4_param_gather=True (NVFP4) ===")
            loss_list, grad_ref = self._run_test_helper(
                tp_size,
                fp4_param_gather=True,
                collect_grad_ref=True,
                **kwargs,
            )
            nvfp4_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Memory profiling for BF16 run
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            print("\n=== Running with fp4_param_gather=False (BF16) ===")
            loss_list_ref = self._run_test_helper(
                tp_size,
                fp4_param_gather=False,
                grad_ref=grad_ref,
                **kwargs,
            )
            bf16_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            # Print memory comparison
            print("\n=== Memory Usage ===")
            print(f"NVFP4 peak memory: {nvfp4_peak_mb:.2f} MB")
            print(f"BF16 peak memory:  {bf16_peak_mb:.2f} MB")
            print(f"Memory savings:    {bf16_peak_mb - nvfp4_peak_mb:.2f} MB ({100*(bf16_peak_mb - nvfp4_peak_mb)/bf16_peak_mb:.1f}%)")

            # Debug: Compare first few iterations
            print("\n=== Loss Comparison ===")
            print("loss_list length: ", len(loss_list))
            for i in range(0, min(100, len(loss_list)), 10):
                fp4_loss = loss_list[i].item()
                bf16_loss = loss_list_ref[i].item()
                match = torch.isclose(loss_list[i], loss_list_ref[i], atol=0, rtol=0)
                print(f"Iter {i}: FP4={fp4_loss:.6f}, BF16={bf16_loss:.6f}, Match={match}")

            # Check if first iteration matches
            first_match = torch.isclose(loss_list[0], loss_list_ref[0], atol=0, rtol=0)
            print(f"\nFirst iteration match: {first_match}")
            if not first_match:
                print(">>> Issue is in FORWARD PASS or MODEL INIT <<<")
            else:
                # Find first mismatch
                for i in range(len(loss_list)):
                    if not torch.isclose(loss_list[i], loss_list_ref[i], atol=0, rtol=0):
                        print(f">>> First mismatch at iteration {i} <<<")
                        print(">>> Issue is in OPTIMIZER STEP or WEIGHT UPDATE <<<")
                        break

            torch.testing.assert_close(loss_list, loss_list_ref, atol=0, rtol=0)

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
        test.run_test(tp_size=1, overlap_param_gather=False, overlap_grad_reduce=False)
        test.teardown_method(None)
        print("PASSED: test_nvfp4 (no overlap)")
    except Exception as e:
        print(f"FAILED: {e}")
        raise

