# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bitwise ON-vs-OFF check for muon + mxfp8 + ``--fp8-param-gather``.

Compares two trajectories of the same small GPT model over a handful of
training steps:

* **OFF**: ``--fp8-recipe=mxfp8`` only (bf16 primary weights, MXFP8 computed
  fresh in each forward).
* **ON**: ``--fp8-recipe=mxfp8 --fp8-param-gather
  --reuse-grad-buf-for-mxfp8-param-ag`` (persistent MXFP8 primary weights,
  bf16 staging buffer routed through the LayerWise param all-gather).

With deterministic CUBLAS / NCCL / TE kernels enabled (see
:func:`deterministic_mode`), both paths must produce bitwise-identical
per-step loss, forward output, per-parameter ``main_grad``, and per-parameter
fp32 master. Any divergence means the LayerWise bf16⇒MXFP8 round-trip is
perturbing numerics relative to OFF — which is the bug class addressed by
``_restore_high_precision_init_val`` in ``layer_wise_optimizer.py``.
"""

import gc
import os
import sys
from contextlib import contextmanager

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core import config
from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import get_device_arch_version, is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.initialize import _set_random_seed
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()


@contextmanager
def deterministic_mode():
    """Enable deterministic CUDA/CUBLAS/NCCL/TE kernels for bitwise comparison.

    Mirrors ``tests/unit_tests/a2a_overlap/utils.py::deterministic_mode`` —
    same env-var sweep + ``_set_random_seed`` invocation. Restores prior env
    on exit.
    """
    config.ENABLE_EXPERIMENTAL = True
    envs = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_NCCL_AVOID_RECORD_STREAMS": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "NCCL_NVLS_ENABLE": "0",
        "NVTE_FUSED_ATTN": "0",
        "NCCL_ALGO": "^NVLS",
        "NVTE_FWD_LAYERNORM_SM_MARGIN": "8",
        "NVTE_BWD_LAYERNORM_SM_MARGIN": "8",
    }
    origin_envs = {}
    for k, v in envs.items():
        origin_envs[k] = os.environ.get(k)
        os.environ[k] = v
    _set_random_seed(seed_=_SEED, data_parallel_random_init=False)
    try:
        yield
    finally:
        for k in envs:
            if origin_envs[k] is not None:
                os.environ[k] = origin_envs[k]
            elif k in os.environ:
                del os.environ[k]


def _snapshot_masters(optimizer):
    """Collect fp32 masters from every Float16-wrapped inner optimizer in the
    chain, keyed by stable (chain-position, group, index) tuple so the same
    parameter lands under the same key across ON / OFF runs."""
    snapshot = {}
    for outer_idx, child in enumerate(optimizer.chained_optimizers):
        if isinstance(child, LayerWiseDistributedOptimizer):
            for inner_idx, inner in enumerate(child.chained_optimizers):
                main_groups = getattr(inner, 'fp32_from_float16_groups', None)
                if main_groups is None:
                    continue
                for grp_idx, group in enumerate(main_groups):
                    for p_idx, p in enumerate(group):
                        key = f"lw[{outer_idx}][{inner_idx}][{grp_idx}][{p_idx}]"
                        snapshot[key] = p.detach().clone()
            continue
        main_groups = getattr(child, 'shard_fp32_from_float16_groups', None)
        if main_groups is None:
            main_groups = getattr(child, 'fp32_from_float16_groups', None)
        if main_groups is None:
            continue
        for grp_idx, group in enumerate(main_groups):
            for p_idx, p in enumerate(group):
                key = f"do[{outer_idx}][{grp_idx}][{p_idx}]"
                snapshot[key] = p.detach().clone()
    return snapshot


class TestMuonMXFP8FP8ParamGather:
    """Bitwise ON-vs-OFF check for muon LayerWise + mxfp8 + fp8_param_gather."""

    def setup_method(self, method):
        self.seq_length = 128
        self.micro_batch_size = 1
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(self, pre_process=True, post_process=True):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        transformer_config = core_transformer_config_from_args(args)
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        return GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
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

    def _create_args(self, fp8_param_gather):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        sys.argv = ['test_muon_mxfp8_fp8_param_gather.py']
        args = parse_args()
        args.num_layers = 2
        args.vocal_size = 128
        args.hidden_size = 64
        args.num_attention_heads = 4
        args.max_position_embeddings = self.seq_length
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = self.seq_length
        args.tensor_model_parallel_size = 1
        args.sequence_parallel = False
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.train_iters = 10
        args.lr = 3e-5
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        # ``--optimizer muon --use-distributed-optimizer`` auto-flips to the
        # LayerWiseDistributedOptimizer path in ``arguments.py``.
        args.optimizer = 'muon'
        args.muon_momentum = 0.9
        args.muon_scale_mode = 'spectral'
        args.muon_num_ns_steps = 5
        args.muon_coefficient_type = 'quintic'
        args.muon_tp_mode = 'duplicated'
        args.use_precision_aware_optimizer = False
        args.exp_avg_dtype = 'fp32'
        args.exp_avg_sq_dtype = 'fp32'
        args.use_distributed_optimizer = True
        # Disable AG / RS overlap so the timing is deterministic and the test
        # exercises the synchronous post-AG quantize path (see
        # ``_post_param_sync`` in ``param_and_grad_buffer.py``).
        args.overlap_param_gather = False
        args.overlap_grad_reduce = False
        # MXFP8 + fp8_param_gather config.
        args.fp8 = "e4m3"
        args.fp8_recipe = "mxfp8"
        args.fp8_param_gather = fp8_param_gather
        if fp8_param_gather:
            args.reuse_grad_buf_for_mxfp8_param_ag = True
        args.ddp_bucket_size = 1024
        validate_args(args)
        set_global_variables(args, False)
        return args

    def _build_batch(self):
        data = list(range(self.seq_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((self.micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((self.micro_batch_size, 1)).cuda()
        position_ids = (
            torch.tensor(data, dtype=torch.int64).repeat((self.micro_batch_size, 1)).cuda()
        )
        attention_mask = torch.ones(
            (self.micro_batch_size, 1, self.seq_length, self.seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(self.seq_length).repeat((self.micro_batch_size, 1)).cuda()
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def _run_steps(self, fp8_param_gather, num_steps):
        """Build model + LayerWise optimizer, run ``num_steps`` deterministic
        training steps, and return per-step snapshots of loss, forward output,
        per-parameter ``main_grad`` (before step), and per-parameter fp32
        master (after step)."""
        args = self._create_args(fp8_param_gather=fp8_param_gather)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)

        input_ids, labels, position_ids, attention_mask, loss_mask = self._build_batch()
        gpt_model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(gpt_model) == 1
        # Muon + use_distributed_optimizer must auto-promote to LayerWise.
        assert isinstance(optimizer.chained_optimizers[0], LayerWiseDistributedOptimizer), (
            "muon + use_distributed_optimizer should route to "
            "LayerWiseDistributedOptimizer; got "
            f"{type(optimizer.chained_optimizers[0]).__name__}"
        )

        losses, outputs, grads_per_step, masters_per_step = [], [], [], []

        for _ in range(num_steps):
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
            loss = output.mean()
            loss.backward()
            if args.overlap_grad_reduce:
                gpt_model[0].finish_grad_sync()

            # Snapshot main_grad before the optimizer step zeros / overwrites it.
            grad_snapshot = {
                name: p.main_grad.detach().clone()
                for name, p in gpt_model[0].named_parameters()
                if p.main_grad is not None
            }

            update_successful, _, _ = optimizer.step()
            assert update_successful

            masters_per_step.append(_snapshot_masters(optimizer))
            grads_per_step.append(grad_snapshot)
            losses.append(loss.detach().clone())
            outputs.append(output.detach().clone())

        return losses, outputs, grads_per_step, masters_per_step

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 requires Blackwell architecture or newer"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required for MXFP8"
    )
    def test_on_vs_off_bitwise_identical(self):
        """fp8_param_gather=ON must produce bitwise-identical loss, forward
        output, per-parameter gradient, and per-parameter fp32 master vs
        fp8_param_gather=OFF for muon + mxfp8 over multiple training steps."""
        num_steps = 5

        with deterministic_mode():
            losses_off, outputs_off, grads_off, masters_off = self._run_steps(
                fp8_param_gather=False, num_steps=num_steps
            )
            losses_on, outputs_on, grads_on, masters_on = self._run_steps(
                fp8_param_gather=True, num_steps=num_steps
            )

        assert len(losses_on) == len(losses_off) == num_steps

        for step in range(num_steps):
            torch.testing.assert_close(
                losses_on[step],
                losses_off[step],
                atol=0,
                rtol=0,
                msg=lambda m, s=step: f"loss mismatch at step {s}: {m}",
            )
            torch.testing.assert_close(
                outputs_on[step],
                outputs_off[step],
                atol=0,
                rtol=0,
                msg=lambda m, s=step: f"output mismatch at step {s}: {m}",
            )

            assert set(grads_on[step].keys()) == set(grads_off[step].keys()), (
                f"grad parameter set mismatch at step {step}: "
                f"on={sorted(grads_on[step].keys())} "
                f"off={sorted(grads_off[step].keys())}"
            )
            for name in grads_on[step]:
                torch.testing.assert_close(
                    grads_on[step][name],
                    grads_off[step][name],
                    atol=0,
                    rtol=0,
                    msg=lambda m, s=step, n=name: f"grad mismatch at step {s} for {n}: {m}",
                )

            assert set(masters_on[step].keys()) == set(masters_off[step].keys()), (
                f"master parameter set mismatch at step {step}: "
                f"on={sorted(masters_on[step].keys())} "
                f"off={sorted(masters_off[step].keys())}"
            )
            for name in masters_on[step]:
                torch.testing.assert_close(
                    masters_on[step][name],
                    masters_off[step][name],
                    atol=0,
                    rtol=0,
                    msg=lambda m, s=step, n=name: f"master mismatch at step {s} for {n}: {m}",
                )
