# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import gc
import os
import sys

import pytest
import torch
from packaging.version import Version
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig, finalize_model_grads
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import (
    is_float8tensor,
    is_grouped_mxfp8tensor,
    is_layerwise_fp8_param,
    is_mxfp8tensor,
    uses_grad_buffer_for_fp8_param_gather,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    initialize_runtime_services,
    set_args,
    set_run_config,
)
from megatron.training.training import (
    force_param_sync,
    setup_model_and_optimizer,
    should_disable_forward_pre_hook,
)
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.a2a_overlap.utils import deterministic_mode
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_fp8_param_gather_policy import (
    assert_param_storage_policy,
    require_fp8_recipe,
)
from tests.unit_tests.test_utilities import Utils

_SEED = 1234
fp8_available, reason_for_no_fp8 = check_fp8_support()

cuda_graph_supported = False
reason_for_no_cuda_graph = ""
try:
    from transformer_engine.pytorch.tensor.utils import post_all_gather_processing

    if callable(post_all_gather_processing) and is_te_min_version("2.10.0"):
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
        config = core_transformer_config_from_args(args)
        transformer_layer_spec = layer_spec_fn(
            num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
        )
        return GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
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
        set_run_config(pretrain_cfg_container_from_args(args))
        initialize_runtime_services(args, build_tokenizer=False)
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
        # Explicit sync delegates staging to all optimizer leaves, including nested
        # LayerWise optimizers and implicit blockwise staging.
        for model_chunk in model_chunks:
            model_chunk.zero_grad_buffer()
        optimizer.prepare_model_params_for_param_sync()

    def run_eval_transition(self, args, model_chunks, optimizer, batch):
        input_ids, labels, position_ids, attention_mask, loss_mask = batch

        if args.overlap_param_gather:
            self.copy_main_params_to_param_buffer(model_chunks, optimizer)

        if should_disable_forward_pre_hook(args):
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

        if should_disable_forward_pre_hook(args):
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
        model_class = "hybrid" if args.hybrid_layer_pattern is not None else "gpt"
        cfg_container = Utils.pretrain_config_from_global_args(args, model_class)
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
                self.model_provider,
                cfg_container=cfg_container,
                pg_collection=pg_collection,
            )
        assert len(gpt_model) == 1  # Assume only one model in the model provider.
        if getattr(args, "use_layer_wise_distributed_optimizer", False):
            has_param_layout = getattr(gpt_model[0], "full_param_layout", None) is not None
            assert (
                has_param_layout == args.use_layer_wise_param_layout
            ), "Only padded Muon uses a full parameter layout and separate Adam DistOpt"
            assert_param_storage_policy(gpt_model[0], args)
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

        if not inference:
            assert_param_storage_policy(gpt_model[0], args)

        loss_list = []
        eval_loss_list = []

        # Production starts overlapped training with parameter-gather pre-hooks disabled: model
        # initialization/checkpoint loading has already populated the forward weights, and the
        # reused grad buffer must stay empty for the first backward. Enable the hooks only after
        # the first successful optimizer step. CUDA-graph tests retain their existing capture-time
        # hook lifecycle, which handles this transition separately.
        first_iteration_pre_hook_disabled = (
            not inference and not use_cuda_graph and should_disable_forward_pre_hook(args)
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
                if should_disable_forward_pre_hook(args):
                    force_param_sync(gpt_model, optimizer=optimizer)
                _ = gpt_model[0].sharded_state_dict()

            # Capture CUDA graphs after warmup if helper is provided.
            # Hard coded cuda_graph_warmup_steps = 0.
            cuda_graph_warmup_steps = 0
            if self.cuda_graph_helper is not None and i == cuda_graph_warmup_steps:
                if should_disable_forward_pre_hook(args):
                    disable_forward_pre_hook(gpt_model, param_sync=False)
                self.cuda_graph_helper.create_cudagraphs()
                if should_disable_forward_pre_hook(args):
                    enable_forward_pre_hook(gpt_model)
                    self.cuda_graph_helper.cuda_graph_set_manual_hooks()

            # For the mxfp8_param with reuse_grad_buf_for_mxfp8_param_ag and dp_ag_overlap,
            # we need to call the _copy_main_params_to_param_buffer() after the grad buffer
            # is zeroed by zero_grad_buffer() because param and grad buffer are shared.
            forward_pre_hook_enabled = bool(
                getattr(gpt_model[0], 'remove_forward_pre_hook_handles', {})
            )
            if args.overlap_param_gather and forward_pre_hook_enabled:
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

            # Match finalize_model_grads in both sync and overlap modes.
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
        self.run_test(tp_size=tp_size, recipe="blockwise", **kwargs)

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
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            ModelType.encoder_or_decoder,
            self.model_provider,
            cfg_container=cfg_container,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
        )
        assert len(model) == 1
        return args, model, optimizer, opt_param_scheduler

    def run_train_steps(self, args, model, optimizer, num_steps, opt_param_scheduler=None):
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        input_ids, labels, position_ids, attention_mask, loss_mask = batch
        losses = []
        for _ in range(num_steps):
            model[0].zero_grad_buffer()
            optimizer.zero_grad()
            if args.overlap_param_gather:
                optimizer.prepare_model_params_for_param_sync()
            model[0].set_is_first_microbatch()
            output = model[0].forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            loss = output.mean()
            losses.append(loss.detach().clone())
            loss.backward()
            # Include TP reductions for sequence-parallel layernorms. DDP alone leaves
            # replicated Adam moments inconsistent across TP ranks before checkpointing.
            finalize_model_grads(
                model, pg_collection=ProcessGroupCollection.use_mpu_process_groups()
            )
            update_successful, _, _ = optimizer.step()
            assert update_successful
            if opt_param_scheduler is not None:
                opt_param_scheduler.step(increment=args.global_batch_size)
        return torch.stack(losses)

    @pytest.mark.launch_on_gb200
    @pytest.mark.parametrize("kind", ["float8", "nvfp4", "grouped_bf16", "grouped_mxfp8"])
    def test_native_compact_layerwise_rejects_unsupported_storage(self, monkeypatch, kind):
        """Reject real TE storage before DDP can replace payloads or quantization metadata."""
        import transformer_engine.pytorch as te

        # The launch marker selects GB200 tests; H100 CI also collects this file.
        # Skip only hardware that cannot construct the requested native storage.
        if kind in ("nvfp4", "grouped_mxfp8") and torch.cuda.get_device_capability()[0] < 10:
            pytest.skip(f"{kind} tensor construction requires compute capability 10.0 or newer")
        if kind == "float8":
            if not fp8_available:
                pytest.skip(reason_for_no_fp8)
            from transformer_engine.common.recipe import Float8CurrentScaling
            from transformer_engine.pytorch.tensor.float8_tensor import Float8Tensor

            recipe = Float8CurrentScaling()
            expected_class = Float8Tensor
        elif kind == "nvfp4":
            from transformer_engine.common.recipe import NVFP4BlockScaling
            from transformer_engine.pytorch.fp8 import check_nvfp4_support
            from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

            supported, reason = check_nvfp4_support()
            assert supported, reason
            recipe = NVFP4BlockScaling()
            expected_class = NVFP4Tensor
        else:
            from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor

            expected_class = GroupedTensor
            if kind == "grouped_mxfp8":
                from transformer_engine.common.recipe import MXFP8BlockScaling
                from transformer_engine.pytorch.fp8 import check_mxfp8_support

                supported, reason = check_mxfp8_support()
                assert supported, reason
                recipe = MXFP8BlockScaling()

        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        config_kwargs = {}
        if kind == "grouped_bf16":
            grouped = GroupedTensor.make_grouped_tensor_from_rowwise_data(
                num_tensors=2,
                tensor_shape=(128, 128),
                rowwise_data=torch.zeros(2, 128, 128, dtype=torch.bfloat16, device="cuda"),
            )
            module = torch.nn.Module()
            module.register_parameter("weight", torch.nn.Parameter(grouped))
        else:
            with te.fp8_model_init(enabled=True, recipe=recipe):
                if kind == "grouped_mxfp8":
                    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
                    module = te.GroupedLinear(
                        2,
                        128,
                        128,
                        bias=False,
                        params_dtype=torch.bfloat16,
                        device="cuda",
                        single_grouped_weight=True,
                    )
                else:
                    module = te.Linear(
                        128, 128, bias=False, params_dtype=torch.bfloat16, device="cuda"
                    )
            if kind == "nvfp4":
                config_kwargs = dict(fp4="e2m1", fp4_param=True)
            else:
                config_kwargs = dict(
                    fp8="e4m3",
                    fp8_param=True,
                    fp8_recipe="mxfp8" if kind == "grouped_mxfp8" else "tensorwise",
                )

        param = module.weight
        assert isinstance(param, expected_class)
        param.is_managed_by_layer_wise_optimizer = True
        config = DistributedDataParallelConfig(
            use_distributed_optimizer=True,
            use_layer_wise_param_layout=False,
            fp8_param_gather=True,
            reuse_grad_buf_for_mxfp8_param_ag=True,
        )
        assert not is_layerwise_fp8_param(param)
        assert is_grouped_mxfp8tensor(param) is (kind == "grouped_mxfp8")
        # Grouped MXFP8 obeys the global opt-in but has no compact whole-param copy-back.
        assert uses_grad_buffer_for_fp8_param_gather(param, config) is (kind == "grouped_mxfp8")
        if kind == "float8":
            attributes = ("_data", "_scale_inv", "_transpose")
        elif kind == "nvfp4":
            attributes = (
                "_rowwise_data",
                "_columnwise_data",
                "_rowwise_scale_inv",
                "_columnwise_scale_inv",
                "_amax_rowwise",
                "_amax_columnwise",
            )
        else:
            attributes = (
                "rowwise_data",
                "columnwise_data",
                "scale_inv",
                "columnwise_scale_inv",
                "amax",
                "columnwise_amax",
                "scale",
            )
        assert torch.is_tensor(getattr(param, attributes[0]))
        before = {}
        for name in attributes:
            tensor = getattr(param, name)
            before[name] = (
                None
                if tensor is None
                else (
                    tensor.data_ptr(),
                    tensor.dtype,
                    tensor.shape,
                    tensor.stride(),
                    tensor.detach().contiguous().reshape(-1).view(torch.uint8).clone(),
                )
            )
        with pytest.raises(TypeError) as error:
            DDP(
                TransformerConfig(
                    num_layers=1, hidden_size=128, num_attention_heads=4, bf16=True, **config_kwargs
                ),
                config,
                module,
                pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
            )
        assert str(error.value) == (
            "Compact LayerWise parameter gather supports only plain MXFP8Tensor "
            "and Float8BlockwiseQTensor quantized parameters; "
            f"got {type(param).__name__}. GroupedTensor is not supported."
        )
        assert config.use_distributed_optimizer  # The shared configuration was not changed.
        for name, original in before.items():
            tensor = getattr(param, name)
            if original is None:
                assert tensor is None, name
                continue
            pointer, dtype, shape, stride, values = original
            assert (tensor.data_ptr(), tensor.dtype, tensor.shape, tensor.stride()) == (
                pointer,
                dtype,
                shape,
                stride,
            ), name
            assert torch.equal(
                tensor.detach().contiguous().reshape(-1).view(torch.uint8), values
            ), name

    @pytest.mark.launch_on_gb200
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.parametrize("optimizer_name", ["adam", "muon"])
    @pytest.mark.parametrize("use_param_layout", [False, True])
    @pytest.mark.parametrize("reuse_grad_buf", [False, True])
    def test_native_mxfp8_reuse_transport(
        self, tmp_path_dist_ckpt, optimizer_name, use_param_layout, reuse_grad_buf
    ):
        """Exercise real MXFP8 storage: four training cases and four refusal boundaries.

        Reuse ON trains and completes a pending gather without retaining scratch data.
        With reuse OFF, TE 2.14/2.16 cannot remap plain MXFP8 storage for native gather;
        Muon reports its explicit requirement and Adam preserves TE's exact rejection.
        These negative cases do not count as successful training with reuse disabled.
        """
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import MXFP8BlockScaling
        from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Tensor

        require_fp8_recipe("mxfp8")
        if Utils.world_size != 8:
            pytest.skip("Native MXFP8 transport matrix requires eight ranks (TP1 x DP8)")
        Utils.initialize_model_parallel(tensor_model_parallel_size=1)
        owner = optimizer_name == "muon"
        if not reuse_grad_buf:
            # Construct real TE parameters without the training CLI, so both layouts
            # reach DDP's storage boundary rather than only testing argument validation.
            with te.fp8_model_init(enabled=True, recipe=MXFP8BlockScaling()):
                module = te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16, device="cuda")
            param = module.weight
            assert isinstance(param, MXFP8Tensor) and is_mxfp8tensor(param)
            param.is_managed_by_layer_wise_optimizer = owner
            config = DistributedDataParallelConfig(
                use_distributed_optimizer=True,
                use_layer_wise_param_layout=use_param_layout,
                fp8_param_gather=True,
                reuse_grad_buf_for_mxfp8_param_ag=False,
            )
            assert not uses_grad_buffer_for_fp8_param_gather(param, config)
            raw_storage = param._rowwise_data.untyped_storage().data_ptr()
            error = ValueError if owner else NotImplementedError
            message = (
                "LayerWise MXFP8 parameter gather requires"
                if owner
                else r"^replace_raw_data for MXFP8Tensor is not supported yet$"
            )
            with pytest.raises(error, match=message):
                DDP(
                    TransformerConfig(
                        num_layers=1,
                        hidden_size=128,
                        num_attention_heads=4,
                        bf16=True,
                        fp8="e4m3",
                        fp8_recipe="mxfp8",
                        fp8_param=True,
                    ),
                    config,
                    module,
                    pg_collection=ProcessGroupCollection.use_mpu_process_groups(),
                )
            assert not config.reuse_grad_buf_for_mxfp8_param_ag
            assert not uses_grad_buffer_for_fp8_param_gather(param, config)
            assert param._rowwise_data.untyped_storage().data_ptr() == raw_storage
            return

        self.seq_length = 128
        self.micro_batch_size = 1
        kwargs = dict(
            num_layers=2,
            padded_vocab_size=512,
            ffn_hidden_size=256,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            clip_grad=1.0,
            attention_backend="unfused",
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            optimizer=optimizer_name,
            use_layer_wise_param_layout=use_param_layout,
            reuse_grad_buf_for_mxfp8_param_ag=True,
            tensor_parallel_num_weight_shards=1,
        )
        if owner:
            kwargs["muon_tp_mode"] = "duplicated"
        with (
            deterministic_mode(),
            TempNamedDir(tmp_path_dist_ckpt / "native_mxfp8_transport", sync=True) as ckpt_dir,
        ):
            args, model, optimizer, scheduler = self.setup_checkpoint_case(
                1, "mxfp8", str(ckpt_dir), **kwargs
            )
            ddp = model[0]
            assert args.tensor_model_parallel_size == 1
            assert args.tensor_parallel_num_weight_shards == 1
            assert_param_storage_policy(ddp, args)
            native_params = [param for param in ddp.parameters() if is_mxfp8tensor(param)]
            assert native_params, "The matrix must exercise native MXFP8 parameters"
            assert all(isinstance(param, MXFP8Tensor) for param in native_params)
            assert all(
                bool(getattr(param, "is_managed_by_layer_wise_optimizer", False)) == owner
                for param in native_params
            )
            assert all(
                uses_grad_buffer_for_fp8_param_gather(param, ddp.ddp_config)
                for param in native_params
            )
            losses = self.run_train_steps(args, model, optimizer, 2, scheduler)
            assert torch.isfinite(losses).all()

            # Match the next iteration's staging order, then inspect real receive views
            # while the gather is pending. Force completion must release those views.
            ddp.zero_grad_buffer()
            optimizer.zero_grad()
            optimizer.prepare_model_params_for_param_sync()
            ddp.start_param_sync()
            groups = ddp.bucket_groups + ddp.expert_parallel_bucket_groups
            assert any(group.param_gather_handle is not None for group in groups)
            reused_buckets = []
            for buffer in ddp.buffers + ddp.expert_parallel_buffers:
                for bucket in buffer.buckets:
                    if not bucket.reuse_grad_buffer_for_param_ag:
                        continue
                    reused_buckets.append(bucket)
                    grad_ptr = bucket.grad_data.untyped_storage().data_ptr()
                    if bucket.param_data is not None:
                        assert bucket.param_data.dtype == torch.bfloat16
                        assert bucket.param_data.untyped_storage().data_ptr() == grad_ptr
                    else:
                        assert owner and not use_param_layout
                        transports = [
                            views for _, views, reuse in bucket.layerwise_gather_list if reuse
                        ]
                        assert transports
                        assert all(
                            view.dtype == torch.bfloat16 for views in transports for view in views
                        )
                        assert all(
                            view.untyped_storage().data_ptr() == grad_ptr
                            for views in transports
                            for view in views
                        )
            assert reused_buckets
            ddp.start_param_sync(force_sync=True)
            assert all(group.param_gather_handle is None for group in groups)
            assert all(bucket.layerwise_gather_list is None for bucket in reused_buckets)
            assert all(torch.count_nonzero(bucket.grad_data) == 0 for bucket in reused_buckets)
            state = self.quantized_param_state(ddp)
            assert state
            for name, values in state.items():
                for kind, value in values.items():
                    replicas = [torch.empty_like(value) for _ in range(Utils.world_size)]
                    torch.distributed.all_gather(replicas, value.contiguous())
                    assert all(
                        torch.equal(replicas[0], other) for other in replicas[1:]
                    ), f"{name}.{kind} differs across DP replicas after MXFP8 gather"

    def cleanup_between_runs(self):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def snapshot_optimizer_state(value):
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: TestFP8Param.snapshot_optimizer_state(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(TestFP8Param.snapshot_optimizer_state(item) for item in value)
        return value

    @staticmethod
    def complete_optimizer_state(optimizer):
        """Include rank-local Adam moments and FP32 masters omitted by DistOpt.state_dict."""
        children = getattr(optimizer, "chained_optimizers", None)
        if children is not None:
            return [TestFP8Param.complete_optimizer_state(child) for child in children]
        inner = optimizer.optimizer
        state = {"wrapper": optimizer.state_dict()}
        if inner is not None:
            state["inner"] = inner.state_dict()
            state["masters"] = [
                [param for param in group["params"]] for group in inner.param_groups
            ]
        state = TestFP8Param.snapshot_optimizer_state(state)
        if inner is not None:
            # Checkpoints align TE FusedAdam's step across groups, while FusedAdam
            # never advances empty groups. Normalize only that field in copied raw
            # state, including Float16Optimizer's duplicate payload. Populated groups,
            # canonical DistOpt metadata, moments and masters remain strict.
            raw_states = [state["inner"]]
            wrapped_inner = state["wrapper"].get("optimizer", {})
            if "state" in wrapped_inner:
                raw_states.append(wrapped_inner)
            for raw_state in raw_states:
                for group in raw_state["param_groups"]:
                    if not group["params"]:
                        group.pop("step", None)
        return state

    @staticmethod
    def assert_state_equal(actual, expected, path="optimizer"):
        if torch.is_tensor(expected):
            assert actual.dtype == expected.dtype and actual.shape == expected.shape, path
            assert torch.equal(actual.cpu(), expected.cpu()), path
        elif isinstance(expected, dict):
            assert actual.keys() == expected.keys(), path
            for key in expected:
                TestFP8Param.assert_state_equal(actual[key], expected[key], f"{path}.{key}")
        elif isinstance(expected, (list, tuple)):
            assert len(actual) == len(expected), path
            for index, (got, want) in enumerate(zip(actual, expected)):
                TestFP8Param.assert_state_equal(got, want, f"{path}[{index}]")
        else:
            assert actual == expected, path

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    @pytest.mark.parametrize("overlap", [False, True])
    @pytest.mark.parametrize(
        ("recipe", "reuse_grad_buf", "optimizer_name", "use_param_layout"),
        [
            ("mxfp8", True, "adam", True),
            ("tensorwise", False, "adam", True),
            ("delayed", False, "adam", True),
            ("blockwise", False, "adam", True),
            # The MXFP8 opt-in must not redirect Adam's blockwise native FP8 gather.
            ("blockwise", True, "adam", True),
            ("mxfp8", True, "muon", False),
            ("mxfp8", True, "muon", True),
            ("blockwise", False, "muon", False),
            ("blockwise", False, "muon", True),
        ],
    )
    @pytest.mark.launch_on_gb200
    def test_fp8_param_checkpoint_resume_is_bitwise_exact(
        self,
        tmp_path_dist_ckpt,
        monkeypatch,
        tp_size,
        overlap,
        recipe,
        reuse_grad_buf,
        optimizer_name,
        use_param_layout,
    ):
        """Save/load preserves codes, then resumes the same losses and optimizer state.

        Native FP8 checkpoint values are dequantized to BF16. Loading must derive
        the live weights from restored FP32 masters, rather than quantizing an
        already lossy dequantized value again. Both step-time and deferred gathers
        must reproduce uninterrupted training, including optimizer moments and LR.
        """
        require_fp8_recipe(recipe)
        if optimizer_name == "muon" and Version(
            os.getenv('NVIDIA_PYTORCH_VERSION', "24.01")
        ) <= Version("25.05"):
            pytest.skip("Layer-wise optimizer is not supported on LTS")
        monkeypatch.setenv("NVTE_ALLOW_UNSAFE_PICKLE_EXTRA_STATE", "1")
        self.seq_length = 256  # TP2 blockwise sequence-parallel partitions need 128 tokens.
        self.micro_batch_size = 1
        kwargs = {
            "num_layers": 2,
            "padded_vocab_size": 512,
            "ffn_hidden_size": 256,
            "hidden_dropout": 0.0,
            "attention_dropout": 0.0,
            "clip_grad": 1.0,
            "attention_backend": "unfused",
            "overlap_param_gather": overlap,
            "overlap_grad_reduce": overlap,
            "reuse_grad_buf_for_mxfp8_param_ag": reuse_grad_buf,
            "optimizer": optimizer_name,
            "use_layer_wise_param_layout": use_param_layout,
        }
        if optimizer_name == "muon":
            kwargs["muon_tp_mode"] = "duplicated"
        # deterministic_mode seeds the model-parallel RNG, so its groups must exist first.
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        with (
            deterministic_mode(),
            TempNamedDir(tmp_path_dist_ckpt / "test_fp8_ckpt_resume", sync=True) as ckpt_dir,
        ):
            args, model, optimizer, scheduler = self.setup_checkpoint_case(
                tp_size, recipe, str(ckpt_dir), **kwargs
            )
            if optimizer_name == "muon" or recipe == "blockwise":
                counts = assert_param_storage_policy(model[0], args)
                if optimizer_name == "adam" and recipe == "blockwise":
                    assert counts["adam_blockwise"]
            self.run_train_steps(args, model, optimizer, 3, scheduler)
            force_param_sync(model, optimizer=optimizer)
            saved_state = self.quantized_param_state(model[0])
            saved_optimizer = self.complete_optimizer_state(optimizer)
            saved_scheduler = self.snapshot_optimizer_state(scheduler.state_dict())
            save_checkpoint(3, model, optimizer, scheduler, 0)
            torch.distributed.barrier()

            # Continue the saving run as the reference; all state snapshots are copies.
            model[0].reset_param_sync_dispatch_state()
            reference_losses = self.run_train_steps(args, model, optimizer, 3, scheduler)
            force_param_sync(model, optimizer=optimizer)
            reference_params = self.quantized_param_state(model[0])
            reference_optimizer = self.complete_optimizer_state(optimizer)
            reference_scheduler = self.snapshot_optimizer_state(scheduler.state_dict())
            del model, optimizer, scheduler
            self.cleanup_between_runs()

            args, model, optimizer, scheduler = self.setup_checkpoint_case(
                tp_size, recipe, str(ckpt_dir), **kwargs
            )
            # setup_model_and_optimizer performs the single production load.
            assert args.iteration == 3
            loaded_state = self.quantized_param_state(model[0])
            self.assert_state_equal(loaded_state, saved_state, "loaded.quantized_params")
            self.assert_state_equal(
                self.complete_optimizer_state(optimizer), saved_optimizer, "loaded.optimizer"
            )
            self.assert_state_equal(scheduler.state_dict(), saved_scheduler, "loaded.scheduler")
            model[0].reset_param_sync_dispatch_state()
            resumed_losses = self.run_train_steps(args, model, optimizer, 3, scheduler)
            force_param_sync(model, optimizer=optimizer)
            self.assert_state_equal(resumed_losses, reference_losses, "resumed.losses")
            self.assert_state_equal(
                self.quantized_param_state(model[0]), reference_params, "resumed.quantized_params"
            )
            self.assert_state_equal(
                self.complete_optimizer_state(optimizer), reference_optimizer, "resumed.optimizer"
            )
            self.assert_state_equal(
                scheduler.state_dict(), reference_scheduler, "resumed.scheduler"
            )

        assert len(saved_state) == 4 * args.num_layers, sorted(saved_state)
