# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Muon FP8 gather parity across compact/padded layouts and overlap modes.

Compare native initialization, losses, gradients, masters and persistent BF16
weights with FP8 primary weights ON/OFF. Quantized bytes are additionally checked
by the checkpoint and force-sync tests.
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
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    initialize_runtime_services,
    set_args,
    set_run_config,
)
from megatron.training.training import setup_model_and_optimizer
from tests.unit_tests.a2a_overlap.utils import deterministic_mode
from tests.unit_tests.test_fp8_param import TestFP8Param as _FP8ParamHarness
from tests.unit_tests.test_fp8_param_gather_policy import (
    assert_param_storage_policy,
    require_fp8_recipe,
)
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.launch_on_gb200

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
    # FP8 storage grouping can change padded whole-matrix ownership. Compare the
    # same global Muon master by name, not whichever rank happens to own it.
    param_buffers = {
        p: buffer for buffer in model.buffers + model.expert_parallel_buffers for p in buffer.params
    }
    result = {}
    for name, param in model.named_parameters():
        main = getattr(param, "main_param", None)
        if getattr(param, "is_managed_by_layer_wise_optimizer", False):
            value = torch.zeros(param.shape, dtype=torch.float32, device=param.device)
            owner_count = torch.tensor(int(main is not None), device=param.device)
            if main is not None:
                value.copy_(main.detach())
            group = param_buffers[param].data_parallel_group
            torch.distributed.all_reduce(value, group=group)
            torch.distributed.all_reduce(owner_count, group=group)
            assert owner_count.item() == 1, f"Expected one Muon owner for {name}"
            result[name] = value
        elif main is not None:
            result[name] = main.detach().clone()
    return result


def _snapshot_grads(model):
    param_buffers = {
        p: buffer for buffer in model.buffers + model.expert_parallel_buffers for p in buffer.params
    }
    result = {}
    for name, param in model.named_parameters():
        if param.main_grad is None:
            continue
        value = param.main_grad.detach().clone()
        buffer = param_buffers[param]
        if (
            getattr(param, "is_managed_by_layer_wise_optimizer", False)
            and buffer.ddp_config.use_distributed_optimizer
        ):
            # Reduce-scatter publishes a complete Muon matrix only on its owner.
            if getattr(param, "main_param", None) is None:
                value.zero_()
            torch.distributed.all_reduce(value, group=buffer.data_parallel_group)
        result[name] = value
    return result


def _snapshot_params(model, include_quantized=False):
    """Model params for ON-vs-OFF comparison.

    fp8 params are skipped by default and covered via the fp32 master + reduced grad instead:
    with fp8_param_gather OFF ``param.data`` is plain bf16, with it ON the same weight is
    Float8/MXFP8 holding ``Q(bf16(master))``, so the two storages differ by the quantization
    step by construction and cannot be compared directly. What remains -- layernorm, biases,
    embeddings, any non-quantized weight -- is directly comparable and IS compared.

    ``include_quantized`` dequantizes instead, for the single-run comparisons (force_sync)
    where both sides hold the same fp8 storage. Raw codes are checked separately.
    """
    out = {}
    for n, p in model.named_parameters():
        if _is_quantized(p):
            if include_quantized:
                out[n] = p.detach().float().clone()
        else:
            out[n] = p.detach().clone()
    assert out, "snapshot is empty -- the param comparison would assert nothing"
    return out


def _snapshot_layerwise_grad_data(ddp):
    # Compact buffers can contain high-precision-only buckets alongside FP8 buckets.
    # Finalization clears only buckets whose gradients supplied FP8 gather storage.
    return [
        bucket.grad_data.detach().clone()
        for buf in (ddp.buffers + ddp.expert_parallel_buffers)
        if not buf.ddp_config.use_distributed_optimizer
        for bucket in buf.buckets
        if bucket.reuse_grad_buffer_for_param_ag
    ]


class TestMuonFP8ParamGather:

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
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
            ),
            vocab_size=args.padded_vocab_size,
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
        sys.argv = ['test_muon_fp8_param_gather.py']
        args = parse_args()
        args.num_layers = 2
        args.padded_vocab_size = 256
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
        args.use_layer_wise_param_layout = self.use_param_layout
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
            # Expert matrices use LayerWise/Muon. Expert-DP1 stages and publishes
            # local FP8 parameters through DDP without a collective; larger groups
            # additionally gather the staged parameters from their owners.
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
        set_args(args)
        set_run_config(pretrain_cfg_container_from_args(args))
        initialize_runtime_services(args, build_tokenizer=False)
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
        # The config builder bypasses model_provider, so reset its CUDA RNG tracker
        # here as well. torch.manual_seed does not reset Megatron's tracked streams.
        model_parallel_cuda_manual_seed(
            _SEED,
            te_rng_tracker=args.te_rng_tracker,
            use_cudagraphable_rng=args.cuda_graph_impl != "none",
            force_reset_rng=True,
        )
        model, optimizer, _ = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=Utils.pretrain_config_from_global_args(args, "gpt"),
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert len(model) == 1
        assert isinstance(optimizer.chained_optimizers[0], LayerWiseDistributedOptimizer), (
            "muon + use_distributed_optimizer should route to LayerWiseDistributedOptimizer; got "
            f"{type(optimizer.chained_optimizers[0]).__name__}"
        )
        counts = assert_param_storage_policy(model[0], args)
        if fp8_param_gather:
            assert counts["muon_fp8"], "No native Muon FP8 parameters were exercised"
        return args, model, optimizer

    def _run_steps(self, args, model, optimizer, n):
        """Run ``n`` deterministic steps; return per-step loss, forward output,
        per-param ``main_grad`` (pre-step), fp32 master (post-step), and forward-time params."""
        # Each trajectory runs under the globals initialized by its own _build.
        assert get_args() is args
        ids, labels, pos, mask, loss_mask = self._batch()
        losses, outs, grads, masters, params = [], [], [], [], []
        for _ in range(n):
            model[0].zero_grad_buffer()
            optimizer.zero_grad()
            # Restage reused FP8 transport after gradient reset, as production does
            # before a deferred gather. High-precision parameters keep separate storage.
            if args.overlap_param_gather:
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
            # Observe the weights consumed by this forward. Do not publish updated
            # weights between steps: the next forward pre-hook must complete the gather.
            # Native FP8 storage is compared through outputs, gradients and masters;
            # dequantized FP8 values need not equal the gather-OFF BF16 parameters.
            params.append(_snapshot_params(model[0]))
            loss = out.mean()
            loss.backward()
            model[0].finish_grad_sync()
            grad = _snapshot_grads(model[0])
            ok, _, _ = optimizer.step()
            assert ok
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
            masters0 = _snapshot_masters(off_model[0])
            off = self._run_steps(off_args, off_model, off_opt, n)
            del off_model, off_opt
            gc.collect()
            torch.cuda.empty_cache()

            # Run each trajectory with its own initialized runtime services instead
            # of rebinding the process-global run config between optimizer steps.
            on_args, on_model, on_opt = self._build(
                True, fp8_recipe, overlap, num_experts, expert_model_parallel_size
            )
            # Compare independently initialized native masters. Copying the OFF state into
            # ON here would conceal a regression that seeds masters from lossy FP8 values.
            on_masters0 = _snapshot_masters(on_model[0])
            assert masters0.keys() == on_masters0.keys()
            assert masters0, "No FP32 master parameters captured"
            for name in masters0:
                _assert_equal(on_masters0[name], masters0[name], f"native initialization {name}")

            on = self._run_steps(on_args, on_model, on_opt, n)
            del on_model, on_opt
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
    @pytest.mark.parametrize("use_param_layout", [False, True])
    def test_on_vs_off_bitwise_identical(self, fp8_recipe, overlap, use_param_layout):
        """fp8_param_gather ON must match OFF bitwise for each layout, for
        overlap grad-reduce + param-gather both ON and OFF."""
        self.use_param_layout = use_param_layout
        require_fp8_recipe(fp8_recipe)
        # 30 steps: fp8-quantization ON-vs-OFF mismatches often only surface after many
        # iterations (PR #5470 review), so a handful of steps can miss a real divergence.
        self._check_on_vs_off(fp8_recipe, overlap, n=30)

    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize("expt_dp_gt_1", [False, True])
    @pytest.mark.parametrize("fp8_recipe", ["blockwise", "mxfp8"])
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("use_param_layout", [False, True])
    def test_moe_on_vs_off_bitwise_identical(
        self, fp8_recipe, overlap, expt_dp_gt_1, use_param_layout
    ):
        """MoE variant (PR #5470 review): the 2D expert weights are Muon-managed, so they
        ride the LayerWise param path. Covers both expert-data-parallel regimes:

        - ``expt_dp == 1`` (expert_model_parallel_size == world size): experts need no
          collective; DDP must still publish their locally staged FP8 parameters.
        - ``expt_dp > 1`` (expert_model_parallel_size == 1): experts ARE gathered.

        Needs world size >= 2 to realize both regimes (at dp==1 only expt_dp==1 exists);
        run with ``torchrun --nproc_per_node>=2``.
        """
        self.use_param_layout = use_param_layout
        world = torch.distributed.get_world_size()
        if world < 2:
            pytest.skip(
                "MoE expt_dp coverage needs data-parallel size >= 2 (dp==1 only realizes "
                "expt_dp==1); run with --nproc_per_node>=2"
            )
        require_fp8_recipe(fp8_recipe)

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
        the reference path. Requires DP>=2 to exercise an actual asynchronous collective;
        DP1 still stages and publishes locally. Run with ``torchrun --nproc_per_node>=2``.
        """
        self.use_param_layout = False
        if torch.distributed.get_world_size() < 2:
            pytest.skip(
                "This test needs data-parallel size >= 2 for an asynchronous collective; "
                "DP1 stages and publishes locally. Run with --nproc_per_node>=2"
            )
        require_fp8_recipe(fp8_recipe)

        def _step_and_dispatch():
            args, model, opt = self._build(True, fp8_recipe, True)
            self._run_steps(args, model, opt, 1)
            ddp = model[0]
            unpublished_codes = _FP8ParamHarness.quantized_param_state(ddp)
            opt.prepare_model_params_for_param_sync()
            ddp.start_param_sync()
            groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
            assert any(
                g.param_gather_handle is not None for g in groups
            ), "test precondition: expected a pending async param-gather handle"
            return model, ddp, groups, unpublished_codes

        with deterministic_mode():
            # Reference: finish the pending gather through the forward-pre-hook path.
            ref_model, ref_ddp, ref_groups, unpublished_codes = _step_and_dispatch()
            for g in ref_groups:
                if g.param_gather_handle is not None:
                    g.finish_param_sync(skip_next_bucket_dispatch=True)
            ref_params = _snapshot_params(ref_model[0], include_quantized=True)
            ref_codes = _FP8ParamHarness.quantized_param_state(ref_model[0])
            assert any(
                not torch.equal(value, unpublished_codes[name][kind])
                for name, values in ref_codes.items()
                for kind, value in values.items()
            ), "Pending gather must publish updated weights, not gather already-published values"
            ref_grads = _snapshot_layerwise_grad_data(ref_ddp)
            del ref_model, ref_ddp, ref_groups
            gc.collect()
            torch.cuda.empty_cache()

            # Under test: force-sync with the handle still pending.
            model, ddp, groups, _ = _step_and_dispatch()
            ddp.disable_forward_pre_hook(param_sync=True)
            got_params = _snapshot_params(model[0], include_quantized=True)
            got_codes = _FP8ParamHarness.quantized_param_state(model[0])
            got_grads = _snapshot_layerwise_grad_data(ddp)

            for g in groups:
                for bucket in g.buckets:
                    assert (
                        getattr(bucket, 'layerwise_gather_list', None) is None
                    ), "force_sync left an unconsumed layerwise_gather_list"

        assert ref_codes, "Expected raw FP8 codes in force-sync test"
        _FP8ParamHarness.assert_state_equal(got_codes, ref_codes, "force_sync.quantized_params")
        assert ref_params.keys() == got_params.keys()
        for k in ref_params:
            _assert_equal(got_params[k], ref_params[k], f"param after force_sync {k}")
        assert len(ref_grads) == len(got_grads) and ref_grads, "expected reused LayerWise buckets"
        for i, (gr, gg) in enumerate(zip(ref_grads, got_grads)):
            _assert_equal(gg, gr, f"grad_data bucket {i} after force_sync")
            # Both completion routes must release every reused bucket's staging payload.
            # High-precision-only buckets do not borrow gradients for parameter gather.
            assert (
                torch.count_nonzero(gg) == 0
            ), f"grad_data bucket {i} still holds the gather payload after force_sync"
