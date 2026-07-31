# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bitwise ON-vs-OFF check for muon + ``--fp8-param-gather`` on the DECOUPLED
compact LayerWise layout (the default; ``--use-layer-wise-param-layout`` opts into
the padded layout).

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
from megatron.core.fp8_utils import is_float8tensor
from megatron.core.inference.utils import InferenceMode
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
    # NB: do not probe for a ``dequantize`` attribute -- ``torch.Tensor.dequantize`` is defined
    # for every tensor (it returns ``self.to(float32)`` for dense ones), so ``hasattr`` is always
    # True and would classify plain bf16 params as quantized.
    return is_float8tensor(p) or is_float8tensor(p.data)


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


def _snapshot_params(model, include_quantized=False):
    """Model params for ON-vs-OFF comparison.

    fp8 params are skipped by default and covered via the fp32 master + reduced grad instead:
    with fp8_param_gather OFF ``param.data`` is plain bf16, with it ON the same weight is
    Float8/MXFP8 holding ``Q(bf16(master))``, so the two storages differ by the quantization
    step by construction and cannot be compared directly. What remains -- layernorm, biases,
    embeddings, any non-quantized weight -- is directly comparable and IS compared.

    ``include_quantized`` dequantizes instead, for the single-run comparisons (force_sync)
    where both sides hold the same fp8 storage and the gathered bytes are the thing under test.
    """
    out = {}
    for n, p in model.named_parameters():
        if _is_quantized(p):
            if include_quantized:
                out[n] = p.data.dequantize().detach().clone().float()
        else:
            out[n] = p.detach().clone()
    assert out, "snapshot is empty -- the param comparison would assert nothing"
    return out


def _snapshot_layerwise_grad_data(ddp):
    # grad_data of the non-DistOpt LayerWise buffers (reused as the fp8 all-gather's bf16
    # receive buffer, which the param-sync finalize must re-zero).
    return [
        buf.grad_data.detach().clone()
        for buf in (ddp.buffers + ddp.expert_parallel_buffers)
        if not buf.ddp_config.use_distributed_optimizer
    ]


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
        # InferenceMode is a process-global (class-level) flag. Another test file
        # in the same pytest shard can leave it active (e.g. an inference engine
        # test that aborts before unset). These are training tests, so the GPT
        # postprocess "Inference must always gather TP logits" assertion would
        # then fire spuriously. Force training mode before each test.
        InferenceMode.unset_active()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=1,
        )

    def teardown_method(self, method):
        InferenceMode.unset_active()
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

    def _create_args(
        self, fp8_param_gather, fp8_recipe, overlap, num_experts=0, expert_model_parallel_size=1
    ):
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
        args.expert_model_parallel_size = expert_model_parallel_size
        args.train_iters = 10
        # Larger lr than a real run: amplifies any fp8-path discrepancy so an ON-vs-OFF
        # mismatch (if one exists) shows up within a handful of steps rather than being
        # lost in the low bits (per PR #5470 review). The model is tiny so this stays stable.
        args.lr = 1e-3
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
        if num_experts > 0:
            # MoE variant: expert weights are 2D matrices -> Muon-managed, so they ride the
            # LayerWise param path. At expt_dp == 1 (expert_model_parallel_size == world size)
            # the experts are NOT all-gathered and only get the fp8 master->model copy-back; at
            # expt_dp > 1 they are gathered. This covers both branches (PR #5470 review).
            args.num_experts = num_experts
            args.moe_router_topk = 2
            args.moe_ffn_hidden_size = args.ffn_hidden_size
            args.moe_token_dispatcher_type = 'alltoall'
            args.moe_grouped_gemm = False
            # Deterministic routing comes from the fixed seed + deterministic_mode; drop the
            # aux-loss gradient term so ON and OFF compare cleanly without router-bias drift.
            args.moe_router_load_balancing_type = 'none'
            args.moe_aux_loss_coeff = 0.0
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

    def _build(
        self, fp8_param_gather, fp8_recipe, overlap, num_experts=0, expert_model_parallel_size=1
    ):
        args = self._create_args(
            fp8_param_gather,
            fp8_recipe,
            overlap,
            num_experts=num_experts,
            expert_model_parallel_size=expert_model_parallel_size,
        )
        set_args(args)
        torch.manual_seed(_SEED)
        model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder, self.model_provider
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
            # re-stage masters before the deferred (overlap) param all-gather. This is what
            # ``force_param_sync`` / ``disable_forward_pre_hook`` do in the real training loop;
            # it recurses into the sibling DistributedOptimizer (embeddings / biases / layernorm)
            # that owns a byte-shard param buffer. No-op otherwise.
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                optimizer.prepare_model_params_for_param_sync()
            if args.overlap_param_gather:
                # The deferred gather must still be pending here. A forced sync that forgets to
                # re-arm leaves param_gather_dispatched=True and turns the forward pre-hook into
                # a silent no-op -- the test would keep passing while losing the overlap
                # coverage it exists for.
                # ``all``, not ``any``: with ``any`` a single armed dense group would mask a
                # refactor that drops expert_parallel_bucket_groups from the reset, leaving
                # every expert group permanently dispatched. ``_groups and`` because all([])
                # is vacuously True.
                _groups = model[0].bucket_groups + model[0].expert_parallel_bucket_groups
                assert _groups and all(not g.param_gather_dispatched for g in _groups), (
                    "forward pre-hook is not armed for every bucket group: the deferred param "
                    "all-gather would be skipped, so this iteration does not exercise the "
                    "overlap path"
                )
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
            if args.overlap_param_gather:
                # Under overlap the param all-gather is deferred to the next forward pre-hook,
                # so right after ``step()`` the param buffer still holds the values gathered
                # at THIS iteration's forward, i.e. pre-update -- there is no well-defined
                # instant at which ON and OFF can be compared. Finalize it here.
                #
                # Order matters. The LayerWise (Muon) buckets are self-sufficient: their
                # gather re-stages its source from ``param.main_param`` every time
                # (``_stage_param_to_bf16``), so they never read stale buffer content --
                # which is why no Muon param ever diverged here. The sibling plain
                # DistributedOptimizer (embeddings / biases / layernorm) instead gathers
                # from a param_data shard written back at step time, and backward has since
                # refilled the aliased grad buffer. Restage those masters first, otherwise
                # the forced gather broadcasts gradients as parameters. This is the same
                # sequence the real training loop performs around eval/checkpoint;
                # ``prepare_model_params_for_param_sync`` self-guards on
                # reuse_grad_buf_for_mxfp8_param_ag + overlap_param_gather.
                optimizer.prepare_model_params_for_param_sync()
                model[0].disable_forward_pre_hook(param_sync=True)
                model[0].enable_forward_pre_hook()
                # The forced sync above takes the synchronous branch and leaves
                # ``param_gather_dispatched=True``, which is only cleared by finish_grad_sync
                # (already past) or here. Without this reset the next forward pre-hook is a
                # strict no-op, so steps 2..n would silently stop exercising the deferred
                # (async dispatch -> wait -> finalize -> bucket chaining) path that the
                # ``overlap`` parametrization exists to cover. Re-gathering already-correct
                # values is idempotent, so the ON-vs-OFF comparison is unaffected.
                model[0].reset_param_sync_dispatch_state()
            # NB: include_quantized must stay False here. This snapshot feeds the ON-vs-OFF
            # comparison, where OFF holds plain bf16 and ON holds Q(bf16(master)); a
            # dequantized fp32 view of the latter differs from the former by the
            # quantization step by construction (max_diff == 2**-9 for MXFP8).
            params.append(_snapshot_params(model[0]))
            masters.append(_snapshot_masters(model[0]))
            grads.append(grad)
            losses.append(loss.detach().clone())
            outs.append(out.detach().clone())
        return losses, outs, grads, masters, params

    def _check_on_vs_off(self, fp8_recipe, overlap, n, num_experts=0, expert_model_parallel_size=1):
        """fp8_param_gather ON must match OFF bitwise for ``n`` deterministic steps on
        per-step loss / forward output / per-param main_grad / fp32 master / bf16 param."""
        with deterministic_mode():
            off_args, off_model, off_opt = self._build(
                False, fp8_recipe, overlap, num_experts, expert_model_parallel_size
            )
            params0, masters0 = _snapshot_params(off_model[0]), _snapshot_masters(off_model[0])
            on_args, on_model, on_opt = self._build(
                True, fp8_recipe, overlap, num_experts, expert_model_parallel_size
            )
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
            # ON stores the fp8 weights as Float8/MXFP8 and skips them; OFF keeps the same
            # weights as plain bf16, so ON's key set is a subset. What survives -- layernorms,
            # biases, embeddings, the output layer -- are real bf16 param tensors compared
            # bitwise. The fp8 weights are covered by the master + reduced-grad checks above.
            assert pn[s].keys() <= po[s].keys(), f"param set mismatch step {s}"
            assert pn[s], f"no comparable params captured step {s}"
            # Pin the skipped (fp8-on-ON-only) set: a bare subset assert would silently accept
            # a param dropping out of the comparison, e.g. if a copy-back regression replaced
            # a bf16 ``param.data`` with quantized storage on the ON side only.
            _dropped = po[s].keys() - pn[s].keys()
            if s == 0:
                _dropped_0 = _dropped
                assert _dropped, "no fp8 params skipped -- the ON run is not using fp8 storage"
            assert _dropped == _dropped_0, f"fp8-skipped param set changed at step {s}"
            for k in pn[s]:
                _assert_equal(pn[s][k], po[s][k], f"param step {s} {k}")
            assert mn[s].keys() == mo[s].keys(), f"master param set mismatch step {s}"
            assert mn[s], f"no masters captured step {s}"
            common = mn[s].keys()
            for k in common:
                _assert_equal(mn[s][k], mo[s][k], f"master step {s} {k}")

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
        # 30 steps: fp8-quantization ON-vs-OFF mismatches often only surface after many
        # iterations (PR #5470 review), so a handful of steps can miss a real divergence.
        self._check_on_vs_off(fp8_recipe, overlap, n=30)

    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("expt_dp_gt_1", [False, True])
    @pytest.mark.parametrize("fp8_recipe", ["blockwise", "mxfp8"])
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_moe_on_vs_off_bitwise_identical(self, fp8_recipe, overlap, expt_dp_gt_1):
        """MoE variant (PR #5470 review): the 2D expert weights are Muon-managed, so they
        ride the LayerWise param path. Covers both expert-data-parallel regimes:

        - ``expt_dp == 1`` (expert_model_parallel_size == world size): experts are NOT
          all-gathered — they only get the fp8 master->model copy-back
          (``_copy_main_params_to_model_params``'s non-gathered branch).
        - ``expt_dp > 1`` (expert_model_parallel_size == 1): experts ARE gathered.

        Needs world size >= 2 to realize both regimes (at dp==1 only expt_dp==1 exists);
        run with ``torchrun --nproc_per_node>=2``.
        """
        world = torch.distributed.get_world_size()
        if world < 2:
            pytest.skip(
                "MoE expt_dp coverage needs data-parallel size >= 2 (dp==1 only realizes "
                "expt_dp==1); run with --nproc_per_node>=2"
            )
        arch = get_device_arch_version()
        if fp8_recipe == "blockwise" and arch != 9:
            pytest.skip("blockwise FP8 is Hopper-only")
        if fp8_recipe == "mxfp8" and arch < 10:
            pytest.skip("mxfp8 requires Blackwell architecture or newer")

        # expt_dp = world_size / expert_model_parallel_size.
        ep = 1 if expt_dp_gt_1 else world
        # setup_method initialized model parallel with ep=1; re-init with the target EP.
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=ep,
        )
        self._check_on_vs_off(
            fp8_recipe, overlap, n=30, num_experts=8, expert_model_parallel_size=ep
        )

    @pytest.mark.parametrize("fp8_recipe", ["blockwise", "mxfp8"])
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_force_sync_finalizes_pending_layerwise_gather(self, fp8_recipe):
        """force_sync finalize (eval/ckpt ``disable_forward_pre_hook`` path) must match the
        ``finish_param_sync`` forward-pre-hook path: gathered params land in every rank's
        ``param.data`` and the reused grad buffer is re-zeroed. Both asserted equal against
        the reference path. Needs dp>=2 (dp==1 early-returns the all-gather); run with
        ``torchrun --nproc_per_node>=2``.
        """
        if torch.distributed.get_world_size() < 2:
            pytest.skip(
                "force_sync LayerWise gather copy-back is only exercised at data-parallel "
                "size >= 2 (dp==1 skips the all-gather); run with --nproc_per_node>=2"
            )
        arch = get_device_arch_version()
        if fp8_recipe == "blockwise" and arch != 9:
            pytest.skip("blockwise FP8 is Hopper-only")
        if fp8_recipe == "mxfp8" and arch < 10:
            pytest.skip("mxfp8 requires Blackwell architecture or newer")

        def _step_and_dispatch():
            args, model, opt = self._build(True, fp8_recipe, True)
            self._run_steps(args, model, opt, 1)
            ddp = model[0]
            ddp.start_param_sync()
            groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
            assert any(
                g.param_gather_handle is not None for g in groups
            ), "test precondition: expected a pending async param-gather handle"
            return model, ddp, groups

        with deterministic_mode():
            # Reference: finish the pending gather through the forward-pre-hook path.
            ref_model, ref_ddp, ref_groups = _step_and_dispatch()
            for g in ref_groups:
                if g.param_gather_handle is not None:
                    g.finish_param_sync(skip_next_bucket_dispatch=True)
            ref_params = _snapshot_params(ref_model[0], include_quantized=True)
            ref_grads = _snapshot_layerwise_grad_data(ref_ddp)
            del ref_model, ref_ddp, ref_groups
            gc.collect()
            torch.cuda.empty_cache()

            # Under test: force-sync with the handle still pending.
            model, ddp, groups = _step_and_dispatch()
            ddp.disable_forward_pre_hook(param_sync=True)
            got_params = _snapshot_params(model[0], include_quantized=True)
            got_grads = _snapshot_layerwise_grad_data(ddp)

            for g in groups:
                for bucket in g.buckets:
                    assert (
                        getattr(bucket, 'layerwise_gather_list', None) is None
                    ), "force_sync left an unconsumed layerwise_gather_list"

        assert ref_params.keys() == got_params.keys()
        for k in ref_params:
            _assert_equal(got_params[k], ref_params[k], f"param after force_sync {k}")
        assert len(ref_grads) == len(got_grads) and ref_grads, "expected LayerWise grad buffers"
        for i, (gr, gg) in enumerate(zip(ref_grads, got_grads)):
            _assert_equal(gg, gr, f"grad_data buffer {i} after force_sync")
            # Both routes end in _finalize_layerwise_param_sync -> grad_data.zero_(), so equality
            # alone is zeros-vs-zeros on the happy path. State the intended post-condition
            # directly: force_sync must not leave the gather payload behind.
            assert (
                torch.count_nonzero(gg) == 0
            ), f"grad_data buffer {i} still holds the gather payload after force_sync"
