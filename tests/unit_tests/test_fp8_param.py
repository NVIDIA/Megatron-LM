# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import gc
import os
import sys
import traceback

import pytest
import torch
from packaging.version import Version
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer import HAVE_EMERGING_OPTIMIZERS
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.optimizer.emerging_optimizers import _is_muon_excluded
from megatron.core.optimizer.layer_wise_optimizer import (
    LayerWiseDistributedOptimizer,
    is_managed_by_layer_wise_optimizer,
)
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import (
    HAVE_EINOPS,
    HAVE_FLA,
    HAVE_MAMBA_SSM,
    GatedDeltaProductMixer,
    causal_conv1d_fn,
    check_fla_sequence_packing_support,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version, unwrap_model
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    get_run_config,
    initialize_runtime_services,
    set_args,
    set_run_config,
)
from megatron.training.training import (
    force_param_sync,
    get_model,
    setup_model_and_optimizer,
    should_disable_forward_pre_hook,
)
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils, sync_checkpoint_config

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()

cuda_graph_supported = False
reason_for_no_cuda_graph = ""
try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    if is_te_min_version("2.10.0"):
        cuda_graph_supported = True
    else:
        reason_for_no_cuda_graph = "Need newer TransformerEngine"
except ImportError:
    reason_for_no_cuda_graph = "Need newer TransformerEngine"


def enable_forward_pre_hook(model_chunks):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.enable_forward_pre_hook()


def disable_forward_pre_hook(model_chunks, param_sync=True):
    for model_chunk in model_chunks:
        assert isinstance(model_chunk, DDP)
        model_chunk.disable_forward_pre_hook(param_sync=param_sync)


def _gtp_grad_fence():
    """GTP's pre-DP-sync fence; no-op when GTP is unavailable (gtp_api guards its exports)."""
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if HAVE_GTP:
        from megatron.core.tensor_parallel.gtp_api import (
            wait_for_gtp_grad_reduction_on_current_stream,
        )

        wait_for_gtp_grad_reduction_on_current_stream()


class TestFP8Param:

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.cuda_graph_helper = None
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
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
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        cfg_container = get_run_config()
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=cfg_container.tokenizer.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
        )

    def _on_model_built(self, model_chunks, optimizer, args):
        """Optional test hook after distributed model and optimizer construction."""

    def _on_forward_complete(self, model_chunks, optimizer, args, step, num_steps):
        """Optional test hook after a forward has consumed synchronized parameters."""

    def create_test_args(
        self,
        tp,
        recipe,
        sequence_length,
        micro_batch_size,
        inference,
        fp8_param_gather,
        use_cuda_graph,
        **kwargs,
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_fp8_param.py']
        args = parse_args()
        args.num_layers = 4
        args.padded_vocab_size = 128800
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
        args.fp8 = "e4m3"
        args.fp8_recipe = recipe
        args.fp8_param_gather = fp8_param_gather
        args.ddp_bucket_size = 1024  # Create more buckets to test the rs/ag overlap.

        # MXFP8 test settings
        if recipe == "mxfp8" and fp8_param_gather:
            args.reuse_grad_buf_for_mxfp8_param_ag = True

        if use_cuda_graph:
            args.cuda_graph_impl = "transformer_engine"
            args.cuda_graph_warmup_steps = 0

        for key, value in kwargs.items():
            assert hasattr(args, key)
            setattr(args, key, value)

        validate_args(args)
        set_args(args)
        # Temporary args/config duplication during the training-loop refactor:
        # migrated settings use config; remaining settings still use legacy args.
        model_class = "hybrid" if args.hybrid_layer_pattern is not None else "gpt"
        cfg = Utils.pretrain_config_from_global_args(args, model_class)
        cfg.finalize()
        set_run_config(cfg)
        initialize_runtime_services(args, build_tokenizer=False, training=True)
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

    def copy_main_params_to_param_buffer(self, model_chunks, optimizer):
        # Mirrors MBridge's pre-eval fix: disable_forward_pre_hook(param_sync=True)
        # force-syncs params before eval callbacks run, so MXFP8 must repopulate
        # the shared param/grad buffer before disabling forward hooks.
        for model_chunk in model_chunks:
            model_chunk.zero_grad_buffer()
        for optim_instance in optimizer.chained_optimizers:
            if isinstance(optim_instance, DistributedOptimizer):
                optim_instance._copy_main_params_to_param_buffer()

    def run_eval_transition(self, args, model_chunks, optimizer, batch):
        input_ids, labels, position_ids, attention_mask, loss_mask = batch

        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            self.copy_main_params_to_param_buffer(model_chunks, optimizer)

        if should_disable_forward_pre_hook(args, optimizer):
            disable_forward_pre_hook(model_chunks, param_sync=True)

        model_chunks[0].eval()
        model_chunks[0].set_is_first_microbatch()
        with torch.no_grad():
            eval_output = model_chunks[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
        eval_loss = eval_output.mean()
        model_chunks[0].train()

        if should_disable_forward_pre_hook(args, optimizer):
            enable_forward_pre_hook(model_chunks)

        return eval_loss.item()

    def _run_test_helper(
        self,
        tp_size,
        recipe,
        inference: bool = False,
        fp8_param_gather: bool = True,
        use_cuda_graph: bool = False,
        eval_transition: bool = False,
        **kwargs,
    ):
        """Test fp8_param with a small GPT model."""
        # Test-only knob: not a model arg, so pop before create_test_args (which asserts every
        # kwarg is a real arg attribute).
        save_at_steps_kw = kwargs.pop("save_at_steps", ())
        num_steps = kwargs.pop("num_steps", 100)
        args = self.create_test_args(
            tp_size,
            recipe,
            self.seq_length,
            self.micro_batch_size,
            inference,
            fp8_param_gather,
            use_cuda_graph,
            **kwargs,
        )

        if recipe == "blockwise" and args.sequence_parallel:
            assert (
                tp_size * 128 <= self.seq_length
            ), "Blockwise recipe and sequence parallelism requires tp_size * 128 <= seq_length"

        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
            # Enable GTP weight-remat when the test requested it (default 1 => no GTP, so
            # non-GTP fp8 tests are unaffected).
            gtp_remat_size=getattr(args, "gtp_weight_remat_size", 1),
            expert_gtp_remat_size=getattr(args, "expert_gtp_weight_remat_size", 1),
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )
        # Mirror production RNG initialization. CUDA-graph validation enables the TE RNG
        # tracker, whose graph-safe states must be torch.Generator objects rather than the
        # Tensor states produced by the default MCore tracker. Each helper invocation may
        # follow a test using a different tracker, so replace the process-global tracker.
        model_parallel_cuda_manual_seed(
            _SEED,
            te_rng_tracker=args.te_rng_tracker,
            use_cudagraphable_rng=args.cuda_graph_impl != "none",
            force_reset_rng=True,
        )
        cfg_container = get_run_config()
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()
        if inference:
            model_cfg = cfg_container.model
            builder_cls = model_cfg.get_builder_cls()
            builder = builder_cls(model_cfg)
            gpt_model = builder.build_distributed_models(
                pg_collection=pg_collection, wrap_with_ddp=False
            )
            gpt_model[0].eval()
            optimizer = None
        else:
            gpt_model, optimizer, _ = setup_model_and_optimizer(
                ModelType.encoder_or_decoder,
                cfg_container=cfg_container,
                pg_collection=pg_collection,
            )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.
        if cfg_container.optimizer.use_layer_wise_distributed_optimizer:
            has_param_layout = getattr(gpt_model[0], "full_param_layout", None) is not None
            assert has_param_layout == args.use_layer_wise_param_layout, (
                "LayerWise test did not enter the requested parameter-layout path: "
                f"requested={args.use_layer_wise_param_layout}, actual={has_param_layout}"
            )
        self._on_model_built(gpt_model, optimizer, args)

        # Hard coded to use cuda_graph_impl="transformer_engine"
        cuda_graph_impl = "transformer_engine"
        if use_cuda_graph and cuda_graph_impl == "transformer_engine":
            from megatron.core.transformer.cuda_graphs import TECudaGraphHelper

            self.cuda_graph_helper = TECudaGraphHelper(
                model=gpt_model,
                config=gpt_model[0].config,
                seq_length=self.seq_length,
                micro_batch_size=self.micro_batch_size,
                optimizers=[optimizer],
            )

        num_fp8_params = 0
        for _, param in gpt_model[0].named_parameters():
            if not inference:
                assert param.requires_grad
                assert param.main_grad is not None
            if is_float8tensor(param):
                num_fp8_params += 1

        fp8_layers = args.num_layers
        if kwargs.get("first_last_layers_bf16", False):
            fp8_layers -= kwargs["num_layers_at_start_in_bf16"]
            fp8_layers -= kwargs["num_layers_at_end_in_bf16"]
        if fp8_param_gather and fp8_layers > 0:
            if args.num_experts is None:
                # Each dense layer has 4 GEMM weights: qkv, proj, fc1, fc2.
                assert num_fp8_params == 4 * fp8_layers
            else:
                assert num_fp8_params > 0
                assert any(
                    not getattr(param, 'allreduce', True) for param in gpt_model[0].parameters()
                )
                if not inference:
                    assert len(optimizer.chained_optimizers) >= 2

        # Verify that bf16 params (embedding, LN, etc.) in the MXFP8 model are mapped
        # to the param buffer (shared with grad buffer) rather than allocated separately.
        if args.reuse_grad_buf_for_mxfp8_param_ag:
            for buffer in gpt_model[0].buffers:
                if buffer.param_data is None:
                    continue
                buf_start = buffer.param_data.data_ptr()
                buf_end = buf_start + buffer.param_data.numel() * buffer.param_data.element_size()
                for param in buffer.param_to_bucket:
                    if is_mxfp8tensor(param):
                        # MXFP8 params keep their own quantized storage.
                        assert not (
                            buf_start <= param.data.data_ptr() < buf_end
                        ), "MXFP8 param should not be mapped to the param buffer"
                    else:
                        # BF16 params should be views into the param buffer
                        # (no double allocation).
                        assert buf_start <= param.data.data_ptr() < buf_end, (
                            "BF16 param should be a view into the param buffer "
                            "(no separate allocation)"
                        )

        loss_list = []
        eval_loss_list = []

        # Production starts overlapped training with parameter-gather pre-hooks disabled: model
        # initialization/checkpoint loading has already populated the forward weights, and the
        # reused grad buffer must stay empty for the first backward. Enable the hooks only after
        # the first successful optimizer step. CUDA-graph tests retain their existing capture-time
        # hook lifecycle, which handles this transition separately.
        first_iteration_pre_hook_disabled = (
            not inference
            and not use_cuda_graph
            and should_disable_forward_pre_hook(args, optimizer)
        )
        if first_iteration_pre_hook_disabled:
            disable_forward_pre_hook(gpt_model, param_sync=False)

        # Optional: generate the sharded_state_dict (the checkpoint-save metadata path) at these
        # steps to catch save side-effects on the live weights — a correct save must not perturb
        # the subsequent training step (regression guard for GTP native-FP8 save corruption).
        save_at_steps = set(save_at_steps_kw or ())

        for i in range(num_steps):
            if not inference:
                gpt_model[0].zero_grad_buffer()
                optimizer.zero_grad()

            if i in save_at_steps:
                # Mirror production save_checkpoint_and_time: when the forward pre-hook is disabled
                # for the save, a forced param-sync runs first. Passing the optimizer makes it copy
                # the FP32 masters into the param buffer before the copy-back re-quantizes, so
                # native-FP8 GTP shards are refreshed from masters (not stale grad scratch).
                # Exercise it so the save-perturbation test is a real regression test for the
                # post-save loss spike.
                if should_disable_forward_pre_hook(args, optimizer):
                    force_param_sync(gpt_model, optimizer=optimizer)
                _ = gpt_model[0].sharded_state_dict()

            # Capture CUDA graphs after warmup if helper is provided.
            # Hard coded cuda_graph_warmup_steps = 0.
            cuda_graph_warmup_steps = 0
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                if should_disable_forward_pre_hook(args, optimizer):
                    disable_forward_pre_hook(gpt_model, param_sync=False)
                self.cuda_graph_helper.create_cudagraphs()
                if should_disable_forward_pre_hook(args, optimizer):
                    enable_forward_pre_hook(gpt_model)
                    self.cuda_graph_helper.cuda_graph_set_manual_hooks()

            # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
            # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
            # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
            forward_pre_hook_enabled = bool(
                getattr(gpt_model[0], 'remove_forward_pre_hook_handles', {})
            )
            if (
                args.reuse_grad_buf_for_mxfp8_param_ag
                and args.overlap_param_gather
                and forward_pre_hook_enabled
            ):
                self.copy_main_params_to_param_buffer(gpt_model, optimizer)

            gpt_model[0].set_is_first_microbatch()
            output = gpt_model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            self._on_forward_complete(gpt_model, optimizer, args, i, num_steps)

            # Check output shapes
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length

            if inference:
                continue

            # Verify gradients
            loss = output.mean()
            loss.backward()

            if args.overlap_grad_reduce:
                # Production order (finalize_model_grads): the GTP fence runs BEFORE the DP
                # grad sync and is what flushes an accumulated wgrad.
                _gtp_grad_fence()
                gpt_model[0].finish_grad_sync()

            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

            update_successful, _, _ = optimizer.step()
            assert update_successful

            if first_iteration_pre_hook_disabled:
                enable_forward_pre_hook(gpt_model)
                first_iteration_pre_hook_disabled = False

            loss_list.append(loss.item())

            if eval_transition:
                eval_loss_list.append(
                    self.run_eval_transition(
                        args,
                        gpt_model,
                        optimizer,
                        (input_ids, labels, position_ids, attention_mask, loss_mask),
                    )
                )

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        if eval_transition:
            return torch.tensor(loss_list), torch.tensor(eval_loss_list)
        return torch.tensor(loss_list)

    def run_test(self, tp_size, recipe, inference: bool = False, **kwargs):
        """Test fp8_param with a small GPT model."""
        if inference:
            with torch.inference_mode():
                self._run_test_helper(tp_size, recipe, inference=True, **kwargs)
        else:
            loss_list = self._run_test_helper(tp_size, recipe, fp8_param_gather=True, **kwargs)

            # Before TE 2.2.0, we cannot guarantee that the main params are the same with/without
            # fp8-param-gather, so skip the checking of tensor values.
            if is_te_min_version("2.2.0"):
                loss_list_ref = self._run_test_helper(
                    tp_size, recipe, fp8_param_gather=False, **kwargs
                )
                torch.testing.assert_close(loss_list, loss_list_ref, atol=1e-4, rtol=1e-4)

    def run_test_with_cuda_graph(self, tp_size, recipe, **kwargs):
        loss = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=True, **kwargs
        )
        loss_ref = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, use_cuda_graph=False, **kwargs
        )
        torch.testing.assert_close(loss, loss_ref, atol=0, rtol=0)

    def run_test_with_eval_transition(self, tp_size, recipe, **kwargs):
        loss, eval_loss = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=True, eval_transition=True, **kwargs
        )
        loss_ref, eval_loss_ref = self._run_test_helper(
            tp_size, recipe, fp8_param_gather=False, eval_transition=True, **kwargs
        )
        torch.testing.assert_close(loss, loss_ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(eval_loss, eval_loss_ref, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_delayed_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_delayed_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "delayed", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_tensorwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_inference(self, tp_size):
        self.run_test(tp_size=tp_size, recipe="tensorwise", inference=True)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_tensorwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "tensorwise", **kwargs)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.2.0"), reason="TE 2.2.0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_tensorwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="tensorwise", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    def test_blockwise_scaling(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="blockwise")

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_blockwise_scaling_with_cuda_graph(self, tp_size, dp_overlap):
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size, "blockwise", **kwargs)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_mxfp8(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [1])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    def test_mxfp8_moe(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {
            "overlap_param_gather": dp_overlap[0],
            "overlap_grad_reduce": dp_overlap[1],
            "num_layers": 4,
            "padded_vocab_size": 128800,
            "hidden_size": 128,
            "num_attention_heads": 8,
            "expert_model_parallel_size": 2,
            "num_experts": 2,
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_topk": 1,
            "moe_router_pre_softmax": True,
            "moe_router_load_balancing_type": "none",
            "moe_aux_loss_coeff": 0.0,
            "moe_ffn_hidden_size": 128,
        }
        self.run_test(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_mxfp8_eval_transition(self, tp_size):
        kwargs = {"overlap_param_gather": True, "overlap_grad_reduce": True}
        self.run_test_with_eval_transition(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (False, True), (True, True)])
    @pytest.mark.skipif(not cuda_graph_supported, reason=reason_for_no_cuda_graph)
    def test_mxfp8_with_cuda_graph(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = {"overlap_param_gather": dp_overlap[0], "overlap_grad_reduce": dp_overlap[1]}
        self.run_test_with_cuda_graph(tp_size=tp_size, recipe="mxfp8", **kwargs)

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_blockwise_scaling_with_first_last_layers_bf16(self, tp_size):
        kwargs = {
            "first_last_layers_bf16": True,
            "num_layers_at_start_in_bf16": 1,
            "num_layers_at_end_in_bf16": 1,
        }
        self.run_test(tp_size=tp_size, recipe="blockwise", **kwargs)

    # ------------------------------------------------------------------
    # Checkpoint round trip
    # ------------------------------------------------------------------

    @staticmethod
    def quantized_param_state(model_chunk):
        """Raw element codes and dequantized values of every quantized param, by name.

        Comparing the dequantized values alone would be weak: a block scale can change
        without moving any value, and that still means the resumed tensor is not the one
        that was saved. Comparing the raw codes alone would be weak the other way round,
        since codes only mean something paired with a scale. So compare both.

        The scale arrays are deliberately not compared directly. TE keeps MXFP8 block scales
        in a padded, swizzled layout, and the padding entries are never read and not
        reproducible across allocations, so they raise false mismatches. A scale that is
        actually used shows up in the dequantized values.
        """
        data_attrs = ("_data", "_rowwise_data", "_columnwise_data")
        state = {}
        for name, param in model_chunk.named_parameters():
            if not is_float8tensor(param):
                continue
            # A quantized tensor may be the Parameter itself or its .data payload.
            for holder in (param, param.data):
                tensors = {
                    attr: getattr(holder, attr).detach().clone()
                    for attr in data_attrs
                    if torch.is_tensor(getattr(holder, attr, None))
                }
                if tensors:
                    break
            assert tensors, f"no quantized storage found on {name} ({type(param).__name__})"
            # .float() and not .dequantize(): on a quantized Parameter the latter recurses
            # through __torch_dispatch__ until the stack overflows.
            tensors["dequantized"] = param.detach().float().clone()
            state[name] = tensors
        return state

    def setup_checkpoint_case(self, tp_size, recipe, ckpt_dir, **kwargs):
        args = self.create_test_args(
            tp_size,
            recipe,
            self.seq_length,
            self.micro_batch_size,
            inference=False,
            fp8_param_gather=True,
            use_cuda_graph=False,
            save=ckpt_dir,
            load=ckpt_dir,
            save_interval=1,
            ckpt_format="torch_dist",
            async_save=False,
            save_tokenizer_assets=False,
            **kwargs,
        )
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        model_parallel_cuda_manual_seed(_SEED)
        cfg_container = get_run_config()
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert len(model) == 1
        return args, model, optimizer, opt_param_scheduler

    def run_train_steps(self, args, model, optimizer, num_steps):
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        input_ids, labels, position_ids, attention_mask, loss_mask = batch
        for _ in range(num_steps):
            model[0].zero_grad_buffer()
            optimizer.zero_grad()
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                self.copy_main_params_to_param_buffer(model, optimizer)
            model[0].set_is_first_microbatch()
            output = model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            output.mean().backward()
            if args.overlap_grad_reduce:
                # Production order (finalize_model_grads): the GTP fence runs BEFORE the DP
                # grad sync and is what flushes an accumulated wgrad.
                _gtp_grad_fence()
                model[0].finish_grad_sync()
            update_successful, _, _ = optimizer.step()
            assert update_successful

    def cleanup_between_runs(self):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()
        torch.cuda.empty_cache()

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    # MXFP8 params cannot be re-homed into the DDP param buffer ("replace_raw_data for
    # MXFP8Tensor is not supported yet"), so mxfp8 + --fp8-param-gather always runs with
    # reuse_grad_buf_for_mxfp8_param_ag. That covers the shared param/grad buffer branch of
    # the re-derivation; tensorwise and delayed cover the direct quantize_param_shard branch.
    # LayerWiseDistributedOptimizer is a ChainedOptimizer nested inside the outer one, so
    # it re-derives through ChainedOptimizer._stage_model_params_from_main_params. Covered
    # on mxfp8 only: tensorwise + muon does not round-trip on some ranks for a reason that
    # predates this change -- the layer-wise path quantizes from the FP32 main param while
    # DistributedOptimizer quantizes from bf16(main), so the two derive different tensorwise
    # scales from a checkpoint that stores BF16.
    @pytest.mark.parametrize(
        ("recipe", "reuse_grad_buf", "optimizer_name"),
        [
            ("mxfp8", True, "adam"),
            ("tensorwise", False, "adam"),
            ("delayed", False, "adam"),
            ("mxfp8", True, "muon"),
        ],
    )
    def test_fp8_param_checkpoint_resume_is_bitwise_exact(
        self, tmp_path_dist_ckpt, monkeypatch, tp_size, recipe, reuse_grad_buf, optimizer_name
    ):
        """A resumed run must hold exactly the quantized weights the saving run held.

        Quantized params are dequantized to BF16 for the checkpoint and MXFP8 block scales
        are not stored, so loading re-quantizes a value that has already been through one
        quantization round trip, whereas a training step quantizes the FP32 main param. Same
        quantizer, same blocks, different input: roughly 5% of 32-element blocks come back
        one E8M0 exponent finer, and every element in those blocks re-encodes. The load path
        re-derives the params from the restored FP32 main params, which reproduces them exactly.

        Tensorwise and delayed scaling are expected to round-trip on their own, and are
        covered here so the re-derivation cannot silently perturb them.
        """
        if recipe == "mxfp8" and get_device_arch_version() < 10:
            pytest.skip("MXFP8 is supported since Blackwell architecture")
        if optimizer_name == "muon" and Version(
            os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")
        ) <= Version("25.05"):
            pytest.skip("Layer-wise optimizer is not supported on LTS")
        # Delayed scaling stores its FP8 metadata as a pickled TE extra state, which TE
        # refuses to load by default. This checkpoint is created by the test and is
        # therefore trusted.
        monkeypatch.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
        kwargs = {
            "overlap_param_gather": True,
            "overlap_grad_reduce": True,
            "reuse_grad_buf_for_mxfp8_param_ag": reuse_grad_buf,
            "optimizer": optimizer_name,
        }
        if optimizer_name == "muon":
            # validate_args turns --optimizer muon + --use-distributed-optimizer into the
            # layer-wise optimizer.
            kwargs["muon_tp_mode"] = "duplicated"
        # TempNamedDir(sync=True) barriers, so the process group has to exist first.
        Utils.initialize_distributed()
        with TempNamedDir(tmp_path_dist_ckpt / "test_fp8_ckpt_resume", sync=True) as ckpt_dir:
            args, model, optimizer, opt_param_scheduler = self.setup_checkpoint_case(
                tp_size, recipe, str(ckpt_dir), **kwargs
            )
            self.run_train_steps(args, model, optimizer, num_steps=3)
            # Mirror save_checkpoint_and_time: the params are staged from the FP32 main params
            # and gathered before the state dict is taken.
            force_param_sync(model, optimizer=optimizer)
            saved_state = self.quantized_param_state(model[0])
            sync_checkpoint_config(get_args())
            save_checkpoint(3, model, optimizer, opt_param_scheduler, 0)
            torch.distributed.barrier()

            self.cleanup_between_runs()

            args, model, optimizer, opt_param_scheduler = self.setup_checkpoint_case(
                tp_size, recipe, str(ckpt_dir), **kwargs
            )
            sync_checkpoint_config(get_args())
            iteration, _ = load_checkpoint(
                model, optimizer, opt_param_scheduler, strict=True, restore_training_state=True
            )
            assert iteration == 3
            loaded_state = self.quantized_param_state(model[0])

        # Each layer contributes 4 GEMM weights: qkv, proj, fc1, fc2. Assert the count rather
        # than just non-emptiness, so a config change that quietly drops --fp8-param-gather
        # cannot turn this into a vacuous pass.
        assert len(saved_state) == 4 * args.num_layers, sorted(saved_state)
        assert saved_state.keys() == loaded_state.keys()
        mismatches = []
        for name, saved_tensors in saved_state.items():
            for attr, saved_tensor in saved_tensors.items():
                loaded_tensor = loaded_state[name][attr]
                num_differing = int((saved_tensor != loaded_tensor).sum())
                if num_differing:
                    mismatches.append(f"{name}{attr}: {num_differing}/{saved_tensor.numel()}")
        assert (
            not mismatches
        ), "quantized params changed across a checkpoint round trip:\n" + "\n".join(mismatches)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("recipe", ["mxfp8", "tensorwise", "blockwise"])
    def test_fp8_param_weights_only_load_stream_ckpt_dequant_is_bitwise_exact(
        self, tmp_path_dist_ckpt, monkeypatch, tp_size, recipe
    ):
        """A weights-only load (optimizer=None, as inference does) must produce the same
        quantized params with --stream-ckpt-dequant as with the upfront dequantize path.

        The streaming path quantizes each param in place during the distributed checkpoint
        load instead of dequantizing the whole state dict first, and hands `load_state_dict`
        the quantized param itself. Compared bitwise on the element codes and on the
        dequantized values, which must also match the values the saving run held.
        """
        if recipe == "mxfp8" and get_device_arch_version() < 10:
            pytest.skip("MXFP8 is supported since Blackwell architecture")
        if recipe == "blockwise" and get_device_arch_version() != 9:
            pytest.skip("blockwise is only supported on Hopper architecture")
        # TE refuses to load a pickled extra state by default. This checkpoint is created by
        # the test and is therefore trusted.
        monkeypatch.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
        kwargs = {
            "overlap_param_gather": True,
            "overlap_grad_reduce": True,
            "reuse_grad_buf_for_mxfp8_param_ag": recipe == "mxfp8",
        }
        Utils.initialize_distributed()
        with TempNamedDir(
            tmp_path_dist_ckpt / "test_fp8_ckpt_stream_dequant", sync=True
        ) as ckpt_dir:
            args, model, optimizer, opt_param_scheduler = self.setup_checkpoint_case(
                tp_size, recipe, str(ckpt_dir), **kwargs
            )
            self.run_train_steps(args, model, optimizer, num_steps=2)
            force_param_sync(model, optimizer=optimizer)
            saved_state = self.quantized_param_state(model[0])
            sync_checkpoint_config(get_args())
            save_checkpoint(2, model, optimizer, opt_param_scheduler, 0)
            torch.distributed.barrier()

            loaded_states = {}
            for stream_ckpt_dequant in (False, True):
                self.cleanup_between_runs()
                # Build the model without loading (load=None), then load weights-only like
                # `megatron.inference.utils.get_model_for_inference` does.
                args, model, _, _ = self.setup_checkpoint_case(
                    tp_size,
                    recipe,
                    None,
                    no_load_optim=True,
                    stream_ckpt_dequant=stream_ckpt_dequant,
                    **kwargs,
                )
                args.load = str(ckpt_dir)
                try:
                    sync_checkpoint_config(get_args())
                    iteration, _ = load_checkpoint(model, None, None, strict=True)
                except Exception:
                    # A failure on one rank only leaves the other rank waiting in a collective
                    # and pytest's report is never reached: print the traceback right away.
                    print(
                        f"[rank {Utils.rank}] load_checkpoint(stream_ckpt_dequant="
                        f"{stream_ckpt_dequant}) failed:\n{traceback.format_exc()}",
                        flush=True,
                    )
                    raise
                assert iteration == 2
                loaded_states[stream_ckpt_dequant] = self.quantized_param_state(model[0])

        assert len(saved_state) == 4 * args.num_layers, sorted(saved_state)
        upfront_state, stream_state = loaded_states[False], loaded_states[True]
        assert saved_state.keys() == upfront_state.keys() == stream_state.keys()
        mismatches = []
        for name, upfront_tensors in upfront_state.items():
            for attr, upfront_tensor in upfront_tensors.items():
                stream_tensor = stream_state[name][attr]
                num_differing = int((upfront_tensor != stream_tensor).sum())
                if num_differing:
                    mismatches.append(
                        f"{name}{attr}: streaming differs from upfront in "
                        f"{num_differing}/{upfront_tensor.numel()}"
                    )
            # The BF16 checkpoint round trip preserves the values (block scales may re-encode).
            num_differing = int(
                (stream_state[name]["dequantized"] != saved_state[name]["dequantized"]).sum()
            )
            if num_differing:
                mismatches.append(f"{name}: values differ from the saved ones in {num_differing}")
        if mismatches:
            print(f"[rank {Utils.rank}] " + "\n".join(mismatches), flush=True)
        assert not mismatches, "\n".join(mismatches)


# ---------------------------------------------------------------------------
# LayerWise (Muon) MXFP8 sync: real GDP-AdamW + grouped-MoE model
# ---------------------------------------------------------------------------


def _is_gtp_param(param):
    from megatron.core.tensor_parallel.gtp_api import HAVE_GTP

    if not HAVE_GTP:
        return False
    from megatron.core.tensor_parallel.gtp_api import is_gtp_param

    return is_gtp_param(param)


HAVE_FLA_SEQUENCE_PACKING, FLA_SEQUENCE_PACKING_REASON = check_fla_sequence_packing_support()
HAVE_GDP_DEPS = all(
    (HAVE_MAMBA_SSM, HAVE_EINOPS, HAVE_FLA, causal_conv1d_fn is not None, HAVE_FLA_SEQUENCE_PACKING)
)


def _gdp_moe_test_args(overlap, *, num_weight_shards):
    """Return the shared real-GDP + grouped-MoE training configuration."""

    return dict(
        tp_size=1,
        num_steps=4,
        num_layers=2,
        hybrid_layer_pattern="ME",
        spec=["megatron.core.models.hybrid.hybrid_layer_specs", "gated_delta_product_stack_spec"],
        position_embedding_type="none",
        padded_vocab_size=512,
        hidden_size=256,
        num_attention_heads=8,
        ffn_hidden_size=256,
        normalization="RMSNorm",
        # Keep d_inner=256 while making GDP's logical in-proj/dgrad K
        # 4 * (d_inner + groups * state_dim + heads) = 2080, which is
        # divisible by the MXFP8 block size (32). GTP storage remains padded when enabled.
        mamba_num_heads=8,
        mamba_head_dim=32,
        mamba_num_groups=2,
        mamba_state_dim=128,
        gdp_num_householder=3,
        gdp_cutedsl_kernel=False,
        num_experts=2,
        moe_grouped_gemm=True,
        moe_single_grouped_weight=False,
        moe_ffn_hidden_size=256,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=1,
        expert_tensor_parallel_num_weight_shards=1,
        moe_token_dispatcher_type="alltoall",
        moe_router_topk=2,
        moe_router_pre_softmax=True,
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        add_bias_linear=False,
        optimizer="muon",
        muon_scalar_optimizer="adam",
        muon_momentum=0.9,
        muon_scale_mode="spectral",
        muon_num_ns_steps=5,
        muon_coefficient_type="quintic",
        muon_tp_mode="duplicated",
        lr=1e-3,
        clip_grad=0.0,
        global_batch_size=4,
        tensor_parallel_num_weight_shards=num_weight_shards,
        use_layer_wise_param_layout=True,
        untie_embeddings_and_output_weights=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        overlap_param_gather=overlap,
        overlap_grad_reduce=overlap,
    )


class _GDPAdamWMuonHarness(TestFP8Param):
    """Validate the real GDP-AdamW + MoE-Muon ownership and sync paths."""

    @staticmethod
    def _dequantize(param):
        with torch.no_grad():
            if is_mxfp8tensor(param):
                if _is_gtp_param(param):
                    from megatron.core.tensor_parallel.gtp_api import dequantize_gtp_native_fp8

                    value = dequantize_gtp_native_fp8(param)
                else:
                    value = dequantize_fp8_tensor(param)
            else:
                value = param.float()
            return value.detach().clone()

    @staticmethod
    def _find_buffer(model, param):
        matches = [
            buffer
            for buffer in model.buffers + model.expert_parallel_buffers
            if any(buffer_param is param for buffer_param in buffer.params)
        ]
        assert len(matches) == 1, f"Expected one DDP buffer for parameter, found {len(matches)}"
        return matches[0]

    def _on_model_built(self, model_chunks, optimizer, args):
        assert args.use_layer_wise_param_layout
        assert args.expert_gtp_weight_remat_size == 1
        gtp_enabled = args.gtp_weight_remat_size > 1
        native_mxfp8 = args.fp8_param_gather
        assert args.fp8_recipe == "mxfp8"
        assert args.fp8 is not None
        if gtp_enabled:
            assert native_mxfp8, "GTP with the MXFP8 recipe requires FP8 parameter gather"
        if native_mxfp8:
            assert args.reuse_grad_buf_for_mxfp8_param_ag
        else:
            assert not args.reuse_grad_buf_for_mxfp8_param_ag
        assert args.muon_scalar_optimizer == "adam"
        assert optimizer.config.decoupled_weight_decay, "Scalar Adam must use AdamW semantics"

        model = model_chunks[0]
        core_model = unwrap_model(model)
        assert core_model.decoder.layer_type_list == ["M", "E"]

        gdp_mixers = [
            module for module in core_model.modules() if isinstance(module, GatedDeltaProductMixer)
        ]
        assert len(gdp_mixers) == 1, f"Expected one GDP mixer, found {len(gdp_mixers)}"
        gdp = gdp_mixers[0]
        in_proj_width = (1 + gdp.num_householder) * (
            gdp.d_inner + gdp.ngroups * gdp.d_state + gdp.nheads
        )
        assert (
            in_proj_width % 32 == 0
        ), f"GDP in-proj width {in_proj_width} is incompatible with MXFP8's 32-element blocks"
        in_proj = gdp.in_proj.weight
        out_proj = gdp.out_proj.weight

        named_params = dict(core_model.named_parameters())
        expert_name = "decoder.layers.1.mlp.experts.linear_fc1.weight0"
        assert expert_name in named_params, f"Missing grouped-MoE parameter {expert_name}"
        expert_weight = named_params[expert_name]

        # These are production attributes: the test does not inject optimizer routing.
        assert getattr(in_proj, "use_muon", True) is False
        assert _is_muon_excluded(in_proj)
        assert not is_managed_by_layer_wise_optimizer(in_proj)
        assert getattr(in_proj, "is_managed_by_layer_wise_optimizer", None) is False

        for label, param in (("GDP out_proj", out_proj), ("MoE expert", expert_weight)):
            assert not _is_muon_excluded(param), f"{label} must remain on Muon"
            assert is_managed_by_layer_wise_optimizer(param)
            assert getattr(param, "is_managed_by_layer_wise_optimizer", None) is True

        assert _is_gtp_param(in_proj) == gtp_enabled
        assert _is_gtp_param(out_proj) == gtp_enabled
        assert not _is_gtp_param(expert_weight)
        assert getattr(expert_weight, "allreduce", True) is False
        for label, param in (
            ("GDP in_proj", in_proj),
            ("GDP out_proj", out_proj),
            ("MoE expert", expert_weight),
        ):
            assert (
                is_mxfp8tensor(param) == native_mxfp8
            ), f"{label} storage does not match fp8_param_gather={args.fp8_param_gather}"

        layerwise_optimizers = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, LayerWiseDistributedOptimizer)
        ]
        distributed_optimizers = [
            child
            for child in optimizer.chained_optimizers
            if isinstance(child, DistributedOptimizer)
        ]
        assert len(layerwise_optimizers) == 1
        assert len(distributed_optimizers) == 1
        layerwise_optimizer = layerwise_optimizers[0]
        distributed_optimizer = distributed_optimizers[0]

        assert any(
            param is in_proj
            for group in distributed_optimizer.model_float16_groups
            for param in group
        ), "GDP in_proj must be owned by the scalar AdamW DistributedOptimizer"
        assert any(
            param is out_proj
            for owner_params in (layerwise_optimizer.dp_cp_params_list or [])
            for param in owner_params
        ), "GDP out_proj must be owned by LayerWise/Muon"
        assert any(
            param is expert_weight
            for owner_params in (layerwise_optimizer.expt_dp_params_list or [])
            for param in owner_params
        ), "MoE expert weight must be owned by LayerWise/Muon"

        in_proj_buffer = self._find_buffer(model, in_proj)
        out_proj_buffer = self._find_buffer(model, out_proj)
        expert_buffer = self._find_buffer(model, expert_weight)
        expected_buffer_dtype = torch.uint8 if native_mxfp8 else torch.bfloat16
        assert in_proj_buffer.param_dtype == expected_buffer_dtype
        assert out_proj_buffer.param_dtype == expected_buffer_dtype
        assert expert_buffer.param_dtype == expected_buffer_dtype
        assert (
            in_proj_buffer is not out_proj_buffer
        ), "AdamW GDP in_proj and Muon GDP out_proj require distinct DDP buffers"
        assert in_proj_buffer.data_parallel_group.size() == args.data_parallel_size
        assert out_proj_buffer.data_parallel_group.size() == args.data_parallel_size
        assert expert_buffer.data_parallel_group.size() == 2

        # Subset sync classifies a whole bucket group from its first bucket, so mixed ownership is
        # never legal. This is the invariant guarded by the partition_buckets change.
        for bucket_group in model.bucket_groups + model.expert_parallel_bucket_groups:
            owners = {
                getattr(param, "is_managed_by_layer_wise_optimizer", False)
                for bucket in bucket_group.buckets
                for param in bucket.params
            }
            assert len(owners) == 1, f"Bucket group mixes optimizer ownership: {owners}"

        in_proj_group_index, in_proj_group_order = (
            distributed_optimizer.model_param_group_index_map[in_proj]
        )
        in_proj_main = distributed_optimizer.optimizer.param_groups[in_proj_group_index]["params"][
            in_proj_group_order
        ]

        tracked = [
            ("GDP in_proj", in_proj, in_proj_buffer),
            ("GDP out_proj", out_proj, out_proj_buffer),
            ("MoE expert", expert_weight, expert_buffer),
        ]
        self._tracked_params = [
            (label, param, buffer, self._dequantize(param)) for label, param, buffer in tracked
        ]
        self._tracked_masters = [
            ("GDP in_proj AdamW master", in_proj_main, in_proj_main.detach().clone())
        ]
        for label, param, _ in tracked[1:]:
            main_param = getattr(param, "main_param", None)
            if main_param is not None:
                self._tracked_masters.append(
                    (f"{label} Muon master", main_param, main_param.detach().clone())
                )
        self._runtime_validation_ran = False

    def _on_forward_complete(self, model_chunks, optimizer, args, step, num_steps):
        if args.overlap_param_gather:
            hooks_enabled = bool(model_chunks[0].remove_forward_pre_hook_handles)
            assert hooks_enabled == (step > 0), (
                "Parameter-gather pre-hook lifecycle differs from production: "
                f"step={step}, hooks_enabled={hooks_enabled}"
            )

        if step != num_steps - 1:
            return

        failures = []
        for label, main_param, initial_main in self._tracked_masters:
            if torch.equal(main_param, initial_main):
                failures.append(f"{label} did not update")

        for label, param, buffer, initial_value in self._tracked_params:
            current_value = self._dequantize(param)
            if not torch.isfinite(current_value).all():
                failures.append(f"{label} contains NaN/Inf")
            if torch.equal(current_value, initial_value):
                failures.append(f"{label} forward weight did not update")

            initial_norm = initial_value.float().norm()
            current_norm = current_value.float().norm()
            if not (current_norm > initial_norm * 0.1 and current_norm < initial_norm * 10.0):
                failures.append(
                    f"{label} norm changed implausibly: {initial_norm.item():.4f} -> "
                    f"{current_norm.item():.4f}"
                )

            replicas = [
                torch.empty_like(current_value) for _ in range(buffer.data_parallel_group.size())
            ]
            torch.distributed.all_gather(
                replicas, current_value.contiguous(), group=buffer.data_parallel_group
            )
            if any(not torch.equal(replicas[0], replica) for replica in replicas[1:]):
                failures.append(f"{label} differs across its data-parallel replicas")

        # Make every rank fail together so teardown barriers cannot hang on a rank-local error.
        failure_flag = torch.tensor(bool(failures), dtype=torch.int32, device="cuda")
        torch.distributed.all_reduce(failure_flag, op=torch.distributed.ReduceOp.MAX)
        if failure_flag.item():
            if not failures:
                failures.append("runtime validation failed on another rank")
            raise AssertionError("; ".join(failures))
        self._runtime_validation_ran = True


class TestFP8ParamLayerWiseMxfp8:
    """LayerWise (Muon) MXFP8 parameter sync on grouped-MoE and GDP + grouped-MoE models.

    _GDPAdamWMuonHarness is also driven with GTP weight remat on by test_gtp_fp8_param_gather.py.
    """

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    def test_moe_grouped_tensor_op_fuser_layerwise_mxfp8_sync_parity(self, monkeypatch):
        """Grouped-tensor op fuser must match regular TEGroupedMLP with LayerWise MXFP8 sync.

        The reference disables grouped-tensor execution and the TE op fuser; the target enables
        both. Everything else is identical: MXFP8 primary weights, parameter gather, grad-buffer
        reuse, padded LayerWise layout, and overlap. The target therefore reaches
        ``_stage_layerwise_mxfp8_params`` through the fused implementation's forwarded pre-hooks.
        """
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for EP2 x EDP2")

        monkeypatch.setenv("NVTE_CUTEDSL_FUSED_GROUPED_MLP", "1")

        harness = TestFP8Param()
        harness.setup_method(None)
        harness.seq_length = 128
        harness.micro_batch_size = 1
        try:
            common = dict(
                tp_size=1,
                num_steps=6,
                num_layers=1,
                padded_vocab_size=512,
                hidden_size=256,
                num_attention_heads=8,
                ffn_hidden_size=256,
                global_batch_size=4,
                num_experts=2,
                moe_grouped_gemm=True,
                moe_single_grouped_weight=False,
                moe_ffn_hidden_size=256,
                expert_model_parallel_size=2,
                expert_tensor_parallel_size=1,
                expert_tensor_parallel_num_weight_shards=1,
                moe_token_dispatcher_type="alltoall",
                moe_router_topk=2,
                moe_router_pre_softmax=True,
                moe_router_load_balancing_type="none",
                moe_aux_loss_coeff=0.0,
                add_bias_linear=False,
                optimizer="muon",
                muon_scalar_optimizer="adam",
                muon_momentum=0.9,
                muon_scale_mode="spectral",
                muon_num_ns_steps=5,
                muon_coefficient_type="quintic",
                muon_tp_mode="duplicated",
                lr=1e-3,
                clip_grad=0.0,
                tensor_parallel_num_weight_shards=1,
                use_layer_wise_param_layout=True,
                untie_embeddings_and_output_weights=True,
                hidden_dropout=0.0,
                attention_dropout=0.0,
                overlap_param_gather=True,
                overlap_grad_reduce=True,
            )

            loss_reference = harness._run_test_helper(
                recipe="mxfp8",
                fp8_param_gather=True,
                moe_use_grouped_tensor=False,
                use_transformer_engine_op_fuser=False,
                **common,
            )

            loss_target = harness._run_test_helper(
                recipe="mxfp8",
                fp8_param_gather=True,
                moe_use_grouped_tensor=True,
                use_transformer_engine_op_fuser=True,
                **common,
            )

            local_trajectories = torch.stack((loss_target, loss_reference)).cuda()
            gathered_trajectories = [
                torch.empty_like(local_trajectories)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_trajectories, local_trajectories)
            trajectories = torch.stack(gathered_trajectories)
            torch.testing.assert_close(trajectories[:, 0], trajectories[:, 1], atol=1e-4, rtol=1e-3)
        finally:
            harness.teardown_method(None)

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(
        not HAVE_EMERGING_OPTIMIZERS, reason="emerging-optimizers package is required"
    )
    @pytest.mark.skipif(
        not HAVE_GDP_DEPS,
        reason=(
            FLA_SEQUENCE_PACKING_REASON or "GDP requires mamba-ssm, einops, FLA, and causal-conv1d"
        ),
    )
    @pytest.mark.parametrize("overlap", [False, True])
    def test_gdp_adamw_moe_muon_mxfp8_param_gather(self, overlap, monkeypatch):
        """GDP-AdamW + MoE-Muon must agree with MXFP8 parameter gather ON and OFF.

        ``ME`` builds a GDP mixer followed by a grouped-MoE layer. GDP itself marks its
        ``in_proj.weight`` ``use_muon=False``, which routes it to scalar AdamW; GDP ``out_proj``
        and the two-dimensional expert weights remain on LayerWise/Muon. The padded layout then
        creates separate AdamW-owned and Muon-owned buffers. For each overlap mode, the native
        MXFP8 parameter-gather/reuse trajectory must remain close to the same model using BF16
        primary weights, while both runs use the same MXFP8 compute recipe. In particular, the
        first post-update loss verifies that LayerWise/Muon initialized its FP32 masters from
        TE's preserved high-precision values instead of dequantized MXFP8 weights.

        Weight rematerialization is intentionally disabled: active GTP requires MXFP8 parameter
        gather and cannot form this exact ON/OFF comparison. The original reused-grad-buffer bug
        does not require GTP, and remains exercised by the ON leg's LayerWise DP all-gather.
        """
        if Utils.world_size != 4:
            pytest.skip("Requires exactly 4 torchrun ranks for DP4 and EP2 x EDP2")

        # GDP's channels-last causal-conv backward normally accumulates dweight with atomicAdd.
        # Its launch-order-dependent rounding becomes visible when gradient reduce is overlapped,
        # even between two otherwise identical BF16 runs. causal-conv1d 1.6.2 reads this switch on
        # every backward and uses a deterministic workspace reduction, keeping this test focused
        # on fp8-param-gather/reuse instead of an unrelated convolution-kernel scheduling effect.
        monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")

        harness = _GDPAdamWMuonHarness()
        harness.setup_method(None)
        harness.seq_length = 128
        harness.micro_batch_size = 1
        try:
            common = _gdp_moe_test_args(overlap, num_weight_shards=1)

            loss_fp8_param_gather_on = harness._run_test_helper(
                recipe="mxfp8", fp8_param_gather=True, **common
            )
            assert harness._runtime_validation_ran

            loss_fp8_param_gather_off = harness._run_test_helper(
                recipe="mxfp8", fp8_param_gather=False, **common
            )
            assert harness._runtime_validation_ran

            local_trajectories = torch.stack(
                (loss_fp8_param_gather_on, loss_fp8_param_gather_off)
            ).cuda()
            gathered_trajectories = [
                torch.empty_like(local_trajectories)
                for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather(gathered_trajectories, local_trajectories)
            trajectories = torch.stack(gathered_trajectories)
            per_rank_step_diff = (trajectories[:, 0] - trajectories[:, 1]).abs()
            per_rank_step_diff.nan_to_num_(
                nan=float("inf"), posinf=float("inf"), neginf=float("inf")
            )
            atol = 1e-4
            rtol = 1e-3
            allowed_diff = atol + rtol * trajectories[:, 1].abs()
            error_ratio = per_rank_step_diff / allowed_diff
            error_ratio.nan_to_num_(nan=float("inf"), posinf=float("inf"), neginf=float("inf"))
            worst_index = int(error_ratio.argmax().item())
            worst_rank, worst_step = divmod(worst_index, per_rank_step_diff.shape[1])
            diff = per_rank_step_diff[worst_rank, worst_step].item()
            tolerance = allowed_diff[worst_rank, worst_step].item()
            assert torch.isfinite(trajectories).all() and diff <= tolerance, (
                "GDP-AdamW + MoE-Muon loss differs with MXFP8 parameter gather ON versus OFF "
                f"(overlap={overlap}, |diff|={diff:.6f}, allowed={tolerance:.6f}, "
                f"atol={atol}, rtol={rtol}, worst_rank={worst_rank}, "
                f"worst_step={worst_step}; fp8-param-gather-ON: "
                f"{trajectories[worst_rank, 0].tolist()}, fp8-param-gather-OFF: "
                f"{trajectories[worst_rank, 1].tolist()})."
            )
        finally:
            harness.teardown_method(None)
