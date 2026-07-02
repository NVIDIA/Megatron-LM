# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bitwise ON-vs-OFF check for muon + ``--fp8-param-gather`` on the DECOUPLED
compact LayerWise layout (``--no-use-layer-wise-param-layout``).

Construction mirrors tests/unit_tests/test_muon_mxfp8_fp8_param_gather.py
(PR #4987), but on the decouple path — the route where FP8 param-gather is
*supported* (it is rejected on the padded layout in ``validate_args``): blockwise
(Hopper) needs no ``reuse_grad_buf``; mxfp8 (Blackwell) does. Under deterministic
kernels, fp8_param_gather ON must match OFF bitwise on per-step loss, forward
output, per-param ``main_grad``, fp32 master, and (bf16) model param. Covers
``overlap_grad_reduce`` + ``overlap_param_gather`` both ON and OFF.
"""

import gc
import os
import sys

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

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


def _is_quantized(p):
    return hasattr(p, 'dequantize') or hasattr(p.data, 'dequantize')


def _assert_equal(actual, expected, msg):
    if torch.equal(actual, expected):
        return
    diff = (actual.float() - expected.float()).abs()
    raise AssertionError(
        f"{msg}: max_diff={diff.max().item()} dtype={actual.dtype}/{expected.dtype}"
    )


def _snapshot_masters(model):
    # fp32 masters keyed by param name (stable identity across ON/OFF group layouts).
    return {
        n: p.main_param.detach().clone()
        for n, p in model.named_parameters()
        if getattr(p, 'main_param', None) is not None
    }


def _snapshot_params(model):
    # Only non-quantized (bf16) params; fp8 weights are compared via master + grad.
    return {n: p.detach().clone() for n, p in model.named_parameters() if not _is_quantized(p)}


@torch.no_grad()
def _restore_initial_state(model, optimizer, params0, masters0):
    # Start the ON run from the OFF run's exact init (params + fp32 masters); muon
    # momentum state is empty pre-step, so it needs no restore.
    for n, p in model.named_parameters():
        if n in params0:
            p.data.copy_(params0[n].to(p.device))
    optimizer.reload_model_params()
    for n, p in model.named_parameters():
        mp = getattr(p, 'main_param', None)
        if mp is not None and n in masters0:
            mp.data.copy_(masters0[n].to(mp.device))


class TestMuonDecoupleFP8ParamGather:

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

    def model_provider(self, pre_process=True, post_process=True, **kw):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        return GPTModel(
            config=core_transformer_config_from_args(args),
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
        )

    def _create_args(self, fp8_param_gather, fp8_recipe, overlap):
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        sys.argv = ['test_muon_decouple_fp8_param_gather.py']
        args = parse_args()
        args.num_layers = 2
        args.vocab_size = 128
        args.hidden_size = 128
        args.ffn_hidden_size = 256
        args.num_attention_heads = 4
        args.max_position_embeddings = self.seq_length
        args.seq_length = self.seq_length
        args.micro_batch_size = self.micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.tensor_model_parallel_size = 1
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
        # muon + use_distributed_optimizer auto-routes to LayerWiseDistributedOptimizer.
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
        # DECOUPLED compact LayerWise layout (the supported FP8 param-gather route).
        args.use_layer_wise_param_layout = False
        # --overlap-param-gather requires --overlap-grad-reduce (arguments.py); co-enable.
        args.overlap_param_gather = overlap
        args.overlap_grad_reduce = overlap
        args.fp8 = "e4m3"
        args.fp8_recipe = fp8_recipe
        args.fp8_param_gather = fp8_param_gather
        if fp8_param_gather and fp8_recipe == "mxfp8":
            args.reuse_grad_buf_for_mxfp8_param_ag = (
                True  # mxfp8 columnwise needs the bf16 round-trip
            )
        args.ddp_bucket_size = 1024  # more buckets -> exercise rs/ag overlap
        validate_args(args)
        set_global_variables(args, False)
        return args

    def _batch(self):
        d = list(range(self.seq_length))
        ids = torch.tensor(d, dtype=torch.int64).repeat((self.micro_batch_size, 1)).cuda()
        labels = 1 + ids
        pos = ids.clone()
        mask = torch.ones(
            (self.micro_batch_size, 1, self.seq_length, self.seq_length), dtype=bool
        ).cuda()
        loss_mask = torch.ones(self.seq_length).repeat((self.micro_batch_size, 1)).cuda()
        return ids, labels, pos, mask, loss_mask

    def _build(self, fp8_param_gather, fp8_recipe, overlap):
        args = self._create_args(fp8_param_gather, fp8_recipe, overlap)
        set_args(args)
        torch.manual_seed(_SEED)
        model, optimizer, _ = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        assert len(model) == 1
        assert isinstance(optimizer.chained_optimizers[0], LayerWiseDistributedOptimizer), (
            "muon + use_distributed_optimizer should route to LayerWiseDistributedOptimizer; got "
            f"{type(optimizer.chained_optimizers[0]).__name__}"
        )
        return args, model, optimizer

    def _run_steps(self, args, model, optimizer, n):
        """Run ``n`` deterministic steps; return per-step loss, forward output,
        per-param ``main_grad`` (pre-step), fp32 master and (bf16) param (post-step)."""
        ids, labels, pos, mask, loss_mask = self._batch()
        losses, outs, grads, masters, params = [], [], [], [], []
        for _ in range(n):
            model[0].zero_grad_buffer()
            optimizer.zero_grad()
            # reuse_grad_buf aliases the bf16 staging buffer onto the just-zeroed grad buffer, so
            # re-stage masters before the deferred (overlap) param all-gather. No-op otherwise.
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                for opt in optimizer.chained_optimizers:
                    if isinstance(opt, LayerWiseDistributedOptimizer):
                        opt._copy_main_params_to_param_buffer()
            model[0].set_is_first_microbatch()
            out = model[0].forward(
                input_ids=ids,
                position_ids=pos,
                attention_mask=mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            loss = out.mean()
            loss.backward()
            if args.overlap_grad_reduce:
                model[0].finish_grad_sync()
            grad = {
                name: p.main_grad.detach().clone()
                for name, p in model[0].named_parameters()
                if p.main_grad is not None
            }
            ok, _, _ = optimizer.step()
            assert ok
            params.append(_snapshot_params(model[0]))
            masters.append(_snapshot_masters(model[0]))
            grads.append(grad)
            losses.append(loss.detach().clone())
            outs.append(out.detach().clone())
        return losses, outs, grads, masters, params

    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("fp8_recipe", ["blockwise", "mxfp8"])
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_on_vs_off_bitwise_identical(self, fp8_recipe, overlap):
        """fp8_param_gather ON must match OFF bitwise on the decoupled layout, for
        overlap grad-reduce + param-gather both ON and OFF."""
        arch = get_device_arch_version()
        if fp8_recipe == "blockwise" and arch != 9:
            pytest.skip("blockwise FP8 is Hopper-only")
        if fp8_recipe == "mxfp8" and arch < 10:
            pytest.skip("mxfp8 requires Blackwell architecture or newer")
        n = 5

        with deterministic_mode():
            off_args, off_model, off_opt = self._build(False, fp8_recipe, overlap)
            params0, masters0 = _snapshot_params(off_model[0]), _snapshot_masters(off_model[0])
            on_args, on_model, on_opt = self._build(True, fp8_recipe, overlap)
            _restore_initial_state(on_model[0], on_opt, params0, masters0)

            off = [[], [], [], [], []]  # loss, out, grad, master, param
            on = [[], [], [], [], []]
            for _ in range(n):
                for dst, src in zip(off, self._run_steps(off_args, off_model, off_opt, 1)):
                    dst.extend(src)
                for dst, src in zip(on, self._run_steps(on_args, on_model, on_opt, 1)):
                    dst.extend(src)
            del off_model, on_model, off_opt, on_opt
            gc.collect()
            torch.cuda.empty_cache()

        lo, oo, go, mo, po = off
        ln, on_, gn, mn, pn = on
        for s in range(n):
            _assert_equal(ln[s], lo[s], f"loss step {s}")
            _assert_equal(on_[s], oo[s], f"output step {s}")
            assert gn[s].keys() == go[s].keys(), f"grad param set mismatch step {s}"
            for k in gn[s]:
                _assert_equal(gn[s][k], go[s][k], f"grad step {s} {k}")
            assert pn[s].keys() == po[s].keys(), f"param set mismatch step {s}"
            for k in pn[s]:
                _assert_equal(pn[s][k], po[s][k], f"param step {s} {k}")
            common = mn[s].keys() & mo[s].keys()
            assert common, f"no common masters step {s}"
            for k in common:
                _assert_equal(mn[s][k], mo[s][k], f"master step {s} {k}")
