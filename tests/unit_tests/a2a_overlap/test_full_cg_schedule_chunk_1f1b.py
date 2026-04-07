# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""
Test that validates full-iteration CUDA graph capture + replay correctness for
the combined_1f1b A2A overlap schedule with device-initiated grouped GEMM.

Runs a real model training loop using FullCudaGraphWrapper (the production CG
path) and compares loss values against an eager baseline to verify that the
conditional record_stream() logic and device-initiated grouped GEMM produce
correct results under CUDA graph replay.

Requirements:
    - Blackwell GPU (SM >= 100) for MXFP8 device-initiated CUTLASS grouped GEMM
    - HybridEP dispatcher (flex backend)
    - TE >= 1.9.0.dev0
    - 4 GPUs (EP=4)

Usage:
    torchrun --nproc_per_node 4 -m pytest -xvs \
        "tests/unit_tests/a2a_overlap/test_full_cg_schedule_chunk_1f1b.py"
"""

import gc
import os
import sys
from functools import partial

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.pipeline_parallel.utils import set_streams
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils


def _is_blackwell():
    """Check if the current GPU is Blackwell (SM >= 100)."""
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability()[0] >= 10


def _is_hybrid_ep_available():
    """Check if HybridEP dispatcher backend is available."""
    try:
        from megatron.core.transformer.moe.fused_a2a import HAVE_HYBRIDEP

        return HAVE_HYBRIDEP
    except ImportError:
        return False


def _loss_func(loss_mask, output_tensor, model=None):
    """Loss function matching the production pattern in pretrain_gpt.py.

    Applies loss_mask to per-token cross-entropy loss to produce a scalar loss
    and token count. Returns 3-tuple expected by forward_step_calc_loss.

    Args:
        loss_mask: Tensor of shape [batch, seq_len] masking padding tokens.
        output_tensor: Per-token cross-entropy loss of shape [batch, seq_len].
        model: The GPT model (unused here, kept for API compatibility).

    Returns:
        Tuple of (loss, num_tokens, report_dict).
    """
    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}
    return loss, num_tokens, report


def _reset_full_cuda_graph_wrapper_state():
    """Reset FullCudaGraphWrapper class-level state between runs."""
    FullCudaGraphWrapper.curr_iteration = {'training': 0, 'validation': 0}
    FullCudaGraphWrapper.cuda_graph = {'training': None, 'validation': None}
    FullCudaGraphWrapper.result = {'training': None, 'validation': None}
    StaticBufferLoader.static_buffers = {'training': [], 'validation': []}


class TestFullIterCGScheduleChunk1F1B:
    """
    Test class for validating full-iteration CUDA graph capture + replay of the
    combined_1f1b A2A overlap schedule with device-initiated grouped GEMM.

    Uses FullCudaGraphWrapper to capture the entire forward_backward_func
    (including loss backward, optimizer-excluded) into a single CUDA graph,
    exactly matching the production code path.

    Requires Blackwell (SM >= 100) for MXFP8 + device-initiated CUTLASS grouped
    GEMM, which is the only grouped GEMM path safe for CUDA graph capture.

    Compares loss values from eager training steps against CUDA-graph-replayed
    training steps.
    """

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.original_env = {
            'CUDA_DEVICE_MAX_CONNECTIONS': os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS'),
            'CUBLAS_WORKSPACE_CONFIG': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'NVTE_ALLOW_NONDETERMINISTIC_ALGO': os.environ.get('NVTE_ALLOW_NONDETERMINISTIC_ALGO'),
            'NCCL_NVLS_ENABLE': os.environ.get('NCCL_NVLS_ENABLE'),
            'NCCL_ALGO': os.environ.get('NCCL_ALGO'),
        }
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        os.environ['NVTE_ALLOW_NONDETERMINISTIC_ALGO'] = '0'
        os.environ['NCCL_NVLS_ENABLE'] = '0'
        os.environ['NCCL_ALGO'] = '^NVLS'

    def teardown_method(self, method):
        for key, value in self.original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _reset_full_cuda_graph_wrapper_state()
        # Reset the comm stream singleton so Run 2 creates a fresh one
        # on the correct device after initialize_model_parallel.
        from megatron.core.pipeline_parallel import utils as pp_utils

        pp_utils._COMM_STREAM = None
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(self, pre_process=True, post_process=True, **config_kwargs):
        """Model provider for setup_model_and_optimizer."""
        args = get_args()
        model_parallel_cuda_manual_seed(
            123, te_rng_tracker=args.use_te_rng_tracker, force_reset_rng=True
        )
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = get_gpt_decoder_block_spec(config, use_transformer_engine=True)
        mtp_block_spec = None
        if args.mtp_num_layers:
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec, use_transformer_engine=True
            )
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

    def create_test_args(self, cuda_graph_warmup_steps=3, **kwargs):
        """Create test arguments matching production Qwen3-style MoE config.

        Always builds with the full-iteration CG config (matching production).
        Whether FullCudaGraphWrapper is actually used is controlled by the caller.
        """
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_full_cg.py']
        args = parse_args()

        # Model architecture
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
        args.tensor_model_parallel_size = 1
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.expert_model_parallel_size = 4
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

        # MoE settings -- production-like
        args.num_experts = 8
        args.moe_layer_freq = 1
        args.moe_grouped_gemm = True
        args.moe_permute_fusion = True
        args.moe_router_fusion = True
        args.moe_router_topk = 2
        args.overlap_moe_expert_parallel_comm = True

        # HybridEP flex dispatcher + device-initiated grouped GEMM (CG-safe)
        args.moe_token_dispatcher_type = "flex"
        args.moe_flex_dispatcher_backend = "hybridep"
        args.moe_router_dtype = "fp32"
        args.moe_use_device_initiated_grouped_gemm = True
        args.moe_received_token_capacity = 64

        # MXFP8 -- required by device-initiated CUTLASS path
        args.fp8 = "e4m3"
        args.fp8_recipe = "mxfp8"

        # Always build with full-iteration CG config (matching production).
        # FullCudaGraphWrapper handles warmup vs capture at runtime.
        args.cuda_graph_impl = "local"
        args.cuda_graph_scope = [CudaGraphScope.full_iteration]
        args.check_for_nan_in_loss_and_grad = False
        args.moe_pad_experts_for_cuda_graph_inference = True
        args.cuda_graph_warmup_steps = cuda_graph_warmup_steps
        args.use_te_rng_tracker = True

        # Apply any overrides
        for key, value in kwargs.items():
            assert hasattr(args, key), f"Unknown arg: {key}"
            setattr(args, key, value)

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self, microbatch_idx=0):
        """Create a static input batch.

        Each microbatch_idx produces distinct data (offset by idx) so that the
        combined_1f1b overlap schedule is tested with different data flowing
        through forward and backward of different microbatches simultaneously.
        """
        offset = microbatch_idx * 7  # prime offset to avoid aliasing
        data = list(range(self.seq_length))
        input_ids = (
            (offset + torch.tensor(data, dtype=torch.int64))
            .remainder(1024)  # keep within vocab_size
            .repeat((self.micro_batch_size, 1))
            .cuda()
        )
        labels = (
            (offset + 1 + torch.tensor(data, dtype=torch.int64))
            .remainder(1024)
            .repeat((self.micro_batch_size, 1))
            .cuda()
        )
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((self.micro_batch_size, 1)).cuda()
        )
        attention_mask = torch.ones(
            (self.micro_batch_size, 1, self.seq_length, self.seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(self.seq_length).repeat((self.micro_batch_size, 1)).cuda()
        return {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
        }

    def _run_training_steps(self, use_full_iter_cg, num_steps, cuda_graph_warmup_steps=3):
        """Run training steps and return list of loss values.

        Both eager and CG runs build the model with the same full-iteration CG
        config (matching production). The only difference is whether
        FullCudaGraphWrapper wraps forward_backward_func.

        Args:
            use_full_iter_cg: Whether to wrap with FullCudaGraphWrapper.
            num_steps: Total number of training steps to run.
            cuda_graph_warmup_steps: Number of eager warmup steps before CG capture.

        Returns:
            List of loss tensors from each training step.
        """
        _reset_full_cuda_graph_wrapper_state()

        args = self.create_test_args(cuda_graph_warmup_steps=cuda_graph_warmup_steps)
        set_args(args)
        torch.manual_seed(123)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=4)
        # set_streams must be called after initialize_model_parallel so that
        # torch.cuda.set_device has been called and the comm stream is created
        # on the correct per-rank device.
        set_streams()

        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1

        def forward_step_func(data_iterator, model, return_schedule_plan=False):
            """Forward step matching production forward_step_func signature.

            Mirrors pretrain_gpt.py:forward_step — returns a (output, loss_func)
            tuple where loss_func is a partial that captures loss_mask so it can
            be called as loss_func(output_tensor) by forward_step_calc_loss.
            """
            batch = next(data_iterator)
            loss_mask = batch["loss_mask"]
            if return_schedule_plan:
                schedule_plan = model.build_schedule_plan(**batch)
                return schedule_plan, partial(_loss_func, loss_mask, model=model)
            output = model.forward(**batch)
            return output, partial(_loss_func, loss_mask, model=model)

        forward_backward_func = get_forward_backward_func()

        # Wrap with FullCudaGraphWrapper if requested -- exactly as production does
        if use_full_iter_cg:
            forward_backward_func = FullCudaGraphWrapper(
                forward_backward_func, cuda_graph_warmup_steps=cuda_graph_warmup_steps
            )

        # Use >=2 microbatches to exercise inter-microbatch overlap in
        # combined_1f1b: forward of MB[i+1] runs concurrently with backward
        # of MB[i] on separate CUDA streams.  With num_microbatches=1 the
        # overlap loop in combined_1f1b_schedule_for_no_pipelining is
        # range(0) and never executes, so the multistream schedule is never
        # actually tested.
        num_microbatches = 4
        # Distinct data per microbatch so forward/backward of different MBs
        # process different tokens through the multistream overlap.
        microbatches = [self.get_batch(mb_idx) for mb_idx in range(num_microbatches)]

        losses = []
        for step in range(num_steps):
            gpt_model[0].zero_grad_buffer()
            optimizer.zero_grad()

            fwd_bwd_result = forward_backward_func(
                forward_step_func=forward_step_func,
                data_iterator=iter(microbatches),
                model=gpt_model,
                num_microbatches=num_microbatches,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                forward_only=False,
            )

            # Extract loss from forward_data_store.
            # forward_step_calc_loss appends loss_reduced dicts to forward_data_store.
            # Our _loss_func returns {'lm loss': cat([loss, num_tokens])}, so the
            # first element of the 'lm loss' tensor is the scalar loss value.
            if fwd_bwd_result:
                for result in fwd_bwd_result:
                    if 'lm loss' in result:
                        losses.append(result['lm loss'][0].detach().clone())

            update_successful, _, _ = optimizer.step()
            assert update_successful, f"Optimizer step failed at step {step}"

        return losses

    @pytest.mark.skipif(not _is_blackwell(), reason="Requires Blackwell GPU (SM >= 100) for MXFP8")
    @pytest.mark.skipif(not _is_hybrid_ep_available(), reason="HybridEP dispatcher not available")
    @pytest.mark.skipif(not is_te_min_version("1.9.0.dev0"), reason="Requires TE >= 1.9.0.dev0")
    def test_full_iter_cg_combined_1f1b(self):
        """
        Verifies that full-iteration CUDA graph capture + replay of the
        combined_1f1b A2A overlap schedule produces matching loss values
        compared to eager execution.

        Uses FullCudaGraphWrapper with the production code path:
        - Device-initiated CUTLASS grouped GEMM (MXFP8)
        - HybridEP flex dispatcher
        - conditional record_stream() (skipped during capture)

        The test:
        1. Runs N training steps in eager mode (same CG config, no wrapper).
        2. Runs N training steps with FullCudaGraphWrapper (warmup + capture + replay).
        3. Compares loss values.
        """
        cuda_graph_warmup_steps = 3
        # Total steps = warmup + a few CG-replayed steps
        num_steps = cuda_graph_warmup_steps + 2

        # --- Run 1: Eager baseline (same model config, no FullCudaGraphWrapper) ---
        eager_losses = self._run_training_steps(use_full_iter_cg=False, num_steps=num_steps)

        # --- Run 2: Full-iteration CUDA graph ---
        cg_losses = self._run_training_steps(
            use_full_iter_cg=True,
            num_steps=num_steps,
            cuda_graph_warmup_steps=cuda_graph_warmup_steps,
        )

        # --- Compare ---
        assert len(eager_losses) == len(
            cg_losses
        ), f"Loss count mismatch: eager={len(eager_losses)}, cg={len(cg_losses)}"
        for i, (eager_loss, cg_loss) in enumerate(zip(eager_losses, cg_losses)):
            assert torch.equal(eager_loss, cg_loss), (
                f"[rank {torch.distributed.get_rank()}] "
                f"step {i}: loss mismatch: eager={eager_loss.item()}, cg={cg_loss.item()}"
            )
