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

import copy
import gc
import os
import sys

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support, check_mxfp8_support

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.a2a_overlap.utils import deterministic_mode
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()
mxfp8_available, reason_for_no_mxfp8 = check_mxfp8_support()


def _clone_optimizer_state_value(value):
    if torch.is_tensor(value):
        return value.detach().clone()
    if isinstance(value, dict):
        return {k: _clone_optimizer_state_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clone_optimizer_state_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_clone_optimizer_state_value(v) for v in value)
    return copy.deepcopy(value)


def _iter_leaf_optimizers(optimizer):
    for child in getattr(optimizer, 'chained_optimizers', []):
        yield from _iter_leaf_optimizers(child)
    if not hasattr(optimizer, 'chained_optimizers'):
        yield optimizer


def _snapshot_masters(model):
    """Collect fp32 masters keyed by model parameter name.

    Optimizer-internal group layouts can legitimately differ between
    fp8_param_gather ON / OFF, especially for MXFP8 tensors that may be split
    into internal quantized members. The model parameter name is the stable
    identity we want to compare.
    """
    snapshot = {}
    for name, param in model.named_parameters():
        main_param = getattr(param, 'main_param', None)
        if main_param is None:
            continue
        snapshot[name] = main_param.detach().clone()
    return snapshot


def _snapshot_model_params(model):
    return {
        name: param.detach().clone()
        for name, param in model.named_parameters()
        if not _is_quantized_param(param)
    }


def _is_quantized_param(param):
    return hasattr(param, 'dequantize') or hasattr(param.data, 'dequantize')


def _assert_tensor_equal(actual, expected, message):
    if torch.equal(actual, expected):
        return
    diff = (actual.float() - expected.float()).abs()
    max_index = int(diff.argmax().item()) if diff.numel() > 0 else 0
    actual_flat = actual.detach().reshape(-1)
    expected_flat = expected.detach().reshape(-1)
    raise AssertionError(
        f"{message}: max_diff={diff.max().item()}, "
        f"max_index={max_index}, "
        f"actual_at_max={actual_flat[max_index].item() if actual_flat.numel() else None}, "
        f"expected_at_max={expected_flat[max_index].item() if expected_flat.numel() else None}, "
        f"actual_dtype={actual.dtype}, expected_dtype={expected.dtype}, "
        f"actual={actual}, expected={expected}"
    )


def _snapshot_optimizer_states(model, optimizer):
    param_to_name = {param: name for name, param in model.named_parameters()}
    states = {}
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        torch_optimizer = getattr(leaf_optimizer, 'optimizer', None)
        if torch_optimizer is None:
            continue
        for name, param in model.named_parameters():
            main_param = getattr(param, 'main_param', None)
            if main_param is None or main_param not in torch_optimizer.state:
                continue
            states[name] = _clone_optimizer_state_value(torch_optimizer.state[main_param])

        for model_group, main_group in zip(
            getattr(leaf_optimizer, 'model_float16_groups', []),
            getattr(leaf_optimizer, 'shard_fp32_from_float16_groups', []),
        ):
            for model_param, main_param in zip(model_group, main_group):
                name = param_to_name.get(model_param)
                if name is None or main_param is None or main_param not in torch_optimizer.state:
                    continue
                states[name] = _clone_optimizer_state_value(torch_optimizer.state[main_param])
    return states


def _snapshot_initial_state(model, optimizer):
    return {
        'model_params': {name: param.detach().clone() for name, param in model.named_parameters()},
        'masters': _snapshot_masters(model),
        'optimizer_states': _snapshot_optimizer_states(model, optimizer),
    }


def _restore_optimizer_states(model, optimizer, optimizer_states):
    param_to_name = {param: name for name, param in model.named_parameters()}
    for leaf_optimizer in _iter_leaf_optimizers(optimizer):
        torch_optimizer = getattr(leaf_optimizer, 'optimizer', None)
        if torch_optimizer is None:
            continue
        for name, param in model.named_parameters():
            main_param = getattr(param, 'main_param', None)
            if main_param is None or name not in optimizer_states:
                continue
            torch_optimizer.state[main_param] = _clone_optimizer_state_value(optimizer_states[name])

        for model_group, main_group in zip(
            getattr(leaf_optimizer, 'model_float16_groups', []),
            getattr(leaf_optimizer, 'shard_fp32_from_float16_groups', []),
        ):
            for model_param, main_param in zip(model_group, main_group):
                name = param_to_name.get(model_param)
                if name is None or main_param is None or name not in optimizer_states:
                    continue
                torch_optimizer.state[main_param] = _clone_optimizer_state_value(
                    optimizer_states[name]
                )


@torch.no_grad()
def _restore_initial_state(model, optimizer, initial_state):
    source_model_params = initial_state['model_params']
    source_masters = initial_state['masters']

    for name, param in model.named_parameters():
        if name not in source_model_params:
            continue
        param.data.copy_(source_model_params[name].to(device=param.device))

    optimizer.reload_model_params()

    for name, param in model.named_parameters():
        main_param = getattr(param, 'main_param', None)
        if main_param is not None and name in source_masters:
            main_param.data.copy_(source_masters[name].to(device=main_param.device))

    _restore_optimizer_states(model, optimizer, initial_state['optimizer_states'])


class TestMuonMXFP8FP8ParamGather:
    """Bitwise ON-vs-OFF check for muon LayerWise + mxfp8 + fp8_param_gather."""

    def setup_method(self, method):
        self.seq_length = 128
        self.micro_batch_size = 1
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(self, pre_process=True, post_process=True, **config_kwargs):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        transformer_config = core_transformer_config_from_args(args)
        layer_spec = get_gpt_layer_with_transformer_engine_spec()
        return GPTModel(
            config=transformer_config,
            transformer_layer_spec=layer_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            pg_collection=config_kwargs.get("pg_collection"),
            vp_stage=config_kwargs.get("vp_stage"),
        )

    def _create_args(self, fp8_param_gather, fp8_recipe="mxfp8"):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        sys.argv = ['test_muon_mxfp8_fp8_param_gather.py']
        args = parse_args()
        args.num_layers = 2
        args.vocal_size = 128
        args.hidden_size = 128
        args.ffn_hidden_size = 256
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
        args.clip_grad = 0.0
        args.bf16 = True
        args.add_bias_linear = False
        args.swiglu = True
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0
        args.attention_backend = "unfused"
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
        # FP8 + fp8_param_gather config. Only ``mxfp8`` is wired through the
        # LayerWise bf16-staging + post-AG quantize round-trip; other recipes
        # are blocked by ``arguments.py`` and the parametrized test skips
        # them explicitly.
        args.fp8 = "e4m3"
        args.fp8_recipe = fp8_recipe
        args.fp8_param_gather = fp8_param_gather
        if fp8_param_gather and fp8_recipe == "mxfp8":
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

    def _build_model_and_optimizer(self, fp8_param_gather, fp8_recipe="mxfp8"):
        args = self._create_args(fp8_param_gather=fp8_param_gather, fp8_recipe=fp8_recipe)
        set_args(args)
        torch.manual_seed(_SEED)

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
        return args, gpt_model, optimizer

    def _run_steps(self, args, gpt_model, optimizer, num_steps):
        """Run ``num_steps`` deterministic training steps and return per-step
        snapshots of loss, forward output, per-parameter ``main_grad`` (before
        step), and per-parameter fp32 master (after step)."""
        input_ids, labels, position_ids, attention_mask, loss_mask = self._build_batch()

        losses, outputs, grads_per_step, masters_per_step, params_per_step = [], [], [], [], []

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

            params_per_step.append(_snapshot_model_params(gpt_model[0]))
            masters_per_step.append(_snapshot_masters(gpt_model[0]))
            grads_per_step.append(grad_snapshot)
            losses.append(loss.detach().clone())
            outputs.append(output.detach().clone())

        return losses, outputs, grads_per_step, masters_per_step, params_per_step

    @pytest.mark.parametrize("fp8_recipe", ["mxfp8", "blockwise"])
    @pytest.mark.skipif(not mxfp8_available, reason=reason_for_no_mxfp8)
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 requires Blackwell architecture or newer"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required for MXFP8"
    )
    def test_on_vs_off_bitwise_identical(self, fp8_recipe):
        """fp8_param_gather=ON must produce bitwise-identical loss, forward
        output, per-parameter gradient, and per-parameter fp32 master vs
        fp8_param_gather=OFF for muon + mxfp8 over multiple training steps.

        Only mxfp8 is wired up for the LayerWise FP8 param-gather path;
        other recipes (e.g. blockwise) are blocked by ``arguments.py`` and
        skipped here so the test surface reflects what is supported.
        """
        if fp8_recipe != "mxfp8":
            pytest.skip(
                f"--fp8-recipe={fp8_recipe} is not supported with the LayerWise "
                "FP8 param-gather path; only mxfp8 is wired up."
            )
        num_steps = 5

        with deterministic_mode():
            off_args, off_model, off_optimizer = self._build_model_and_optimizer(
                fp8_param_gather=False, fp8_recipe=fp8_recipe
            )
            initial_state = _snapshot_initial_state(off_model[0], off_optimizer)
            on_args, on_model, on_optimizer = self._build_model_and_optimizer(
                fp8_param_gather=True, fp8_recipe=fp8_recipe
            )
            _restore_initial_state(on_model[0], on_optimizer, initial_state)

            losses_off, outputs_off, grads_off, masters_off, params_off = [], [], [], [], []
            losses_on, outputs_on, grads_on, masters_on, params_on = [], [], [], [], []

            for _ in range(num_steps):
                off_step = self._run_steps(off_args, off_model, off_optimizer, 1)
                on_step = self._run_steps(on_args, on_model, on_optimizer, 1)

                for dst, src in zip(
                    (losses_off, outputs_off, grads_off, masters_off, params_off), off_step
                ):
                    dst.extend(src)
                for dst, src in zip(
                    (losses_on, outputs_on, grads_on, masters_on, params_on), on_step
                ):
                    dst.extend(src)

            del off_model, on_model, off_optimizer, on_optimizer
            gc.collect()
            torch.cuda.empty_cache()

        assert len(losses_on) == len(losses_off) == num_steps

        for step in range(num_steps):
            _assert_tensor_equal(losses_on[step], losses_off[step], f"loss mismatch at step {step}")
            _assert_tensor_equal(
                outputs_on[step], outputs_off[step], f"output mismatch at step {step}"
            )

            assert set(grads_on[step].keys()) == set(grads_off[step].keys()), (
                f"grad parameter set mismatch at step {step}: "
                f"on={sorted(grads_on[step].keys())} "
                f"off={sorted(grads_off[step].keys())}"
            )
            for name in grads_on[step]:
                _assert_tensor_equal(
                    grads_on[step][name],
                    grads_off[step][name],
                    f"grad mismatch at step {step} for {name}",
                )

            assert set(params_on[step].keys()) == set(params_off[step].keys()), (
                f"model parameter set mismatch at step {step}: "
                f"on={sorted(params_on[step].keys())} "
                f"off={sorted(params_off[step].keys())}"
            )
            for name in params_on[step]:
                _assert_tensor_equal(
                    params_on[step][name],
                    params_off[step][name],
                    f"model parameter mismatch after step {step} for {name}",
                )

            common_master_names = set(masters_on[step].keys()) & set(masters_off[step].keys())
            assert common_master_names, (
                f"no common local fp32 masters to compare at step {step}: "
                f"on={sorted(masters_on[step].keys())} "
                f"off={sorted(masters_off[step].keys())}"
            )
            for name in common_master_names:
                _assert_tensor_equal(
                    masters_on[step][name],
                    masters_off[step][name],
                    f"master mismatch at step {step} for {name}",
                )
