# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import gc
import os
import sys

import pytest
import torch
from transformer_engine.pytorch.fp8 import check_fp8_support

from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.enums import ModelType
from megatron.core.fp8_utils import dequantize_fp8_tensor, is_float8tensor, is_mxfp8tensor
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import force_param_sync, get_model, setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.test_utilities import Utils

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


def should_disable_forward_pre_hook(args):
    """Block forward pre-hook for certain configurations."""
    return (
        not args.use_megatron_fsdp and args.use_distributed_optimizer and args.overlap_param_gather
    )


class TestFP8Param:

    def setup_method(self, method):
        self.seq_length = 512
        self.micro_batch_size = 2
        self.cuda_graph_helper = None
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()

    def model_provider(
        self,
        pre_process=True,
        post_process=True,
        layer_spec_fn=get_gpt_layer_with_transformer_engine_spec,
        **config_kwargs,
    ):
        args = get_args()
        model_parallel_cuda_manual_seed(_SEED, te_rng_tracker=args.te_rng_tracker)
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

    def assert_precision_aware_optimizer_state(self, args, optimizer):
        """Check the precision-aware Adam layout used with a non-delayed FP8 recipe.

        With fp32 main params and a non-delayed FP8 recipe (with or without FP8 primary
        weights), MCore keeps the fp32 master shards, from which FP8 primary weights are
        re-quantized, and TE FusedAdam stores only the two moments, which must be held in
        exactly the requested dtypes.
        """
        expected_dtypes = {"exp_avg": args.exp_avg_dtype, "exp_avg_sq": args.exp_avg_sq_dtype}
        num_checked = 0
        for optim_instance in optimizer.chained_optimizers:
            assert isinstance(optim_instance, DistributedOptimizer)
            assert optim_instance.config.use_precision_aware_optimizer
            assert not optim_instance.config.use_precision_aware_optimizer_no_fp8_or_ds_fp8
            for main_param, state in optim_instance.optimizer.state.items():
                assert main_param.dtype == args.main_params_dtype
                assert "master_param" not in state
                for state_name, dtype in expected_dtypes.items():
                    assert state[state_name].dtype == dtype, (state_name, state[state_name].dtype)
                    num_checked += 1
        assert num_checked > 0

    @staticmethod
    def dequantize_fp8_params(model_chunk):
        """Return an fp32 copy of every FP8 primary weight of a model chunk, keyed by name."""
        with torch.no_grad():
            return {
                name: dequantize_fp8_tensor(param).float()
                for name, param in model_chunk.named_parameters()
                if is_float8tensor(param)
            }

    @staticmethod
    def assert_fp8_params_gathered(model_chunk, pg_collection, initial_values):
        """Check the gathered FP8 primary weights exactly; call right after a forward pass.

        By the end of a forward pass every bucket's param all-gather has completed: without
        overlap it runs synchronously in optimizer.step(), and with overlap the forward
        pre-hooks wait for it. Each rank re-quantizes only its own fp32 master shard. The
        per-block scales come from an amax that is all-reduced over the bucket's
        data-parallel group, and the other shards arrive by all-gather. The checks are:

        * Every replica of a weight in its data-parallel group (expt_dp for routed experts,
          dp_cp for dense weights) holds bitwise-identical dequantized values. If the gather
          of any bucket were skipped, stale or misplaced, each rank would hold its freshly
          re-quantized shard next to older values of the other shards, and the replicas
          would differ.
        * With 2-D block scaling, the columnwise copy (used for dgrad on Hopper) is exactly
          the transpose of the rowwise data and scales. A columnwise copy that was not
          rebuilt after the gather fails this.
        * Every weight differs from its value at the first forward pass. Otherwise the
          replica agreement could just be the trivial agreement of untouched initial
          weights.

        The verdict is all-reduced over all ranks before asserting, so a failure in one
        expert-DP group cannot leave the other ranks waiting in a collective.
        """
        failures = []
        with torch.no_grad():
            for name, param in model_chunk.named_parameters():
                if not is_float8tensor(param):
                    continue
                is_expert = not getattr(param, 'allreduce', True)
                group = pg_collection.expt_dp if is_expert else pg_collection.dp_cp
                value = dequantize_fp8_tensor(param).float()
                lowest = value.clone()
                highest = value.clone()
                torch.distributed.all_reduce(lowest, op=torch.distributed.ReduceOp.MIN, group=group)
                torch.distributed.all_reduce(
                    highest, op=torch.distributed.ReduceOp.MAX, group=group
                )
                if not torch.equal(lowest, highest):
                    failures.append(f"{name}: data-parallel replicas differ after the gather")
                if torch.equal(value, initial_values[name]):
                    failures.append(f"{name}: FP8 values never changed during training")

                data = param.data
                if getattr(data, '_is_2D_scaled', False) and data._columnwise_data is not None:
                    # Same region that TE's _create_columnwise copies: the unpadded
                    # (column tiles, row tiles) part of the columnwise scale_inv.
                    rowwise_scale_inv_t = data._rowwise_scale_inv.t()
                    columnwise_scale_inv = data._columnwise_scale_inv
                    tiles = (
                        min(columnwise_scale_inv.shape[0], rowwise_scale_inv_t.shape[0]),
                        min(columnwise_scale_inv.shape[1], rowwise_scale_inv_t.shape[1]),
                    )
                    if not (
                        torch.equal(data._columnwise_data, data._rowwise_data.t())
                        and torch.equal(
                            columnwise_scale_inv[: tiles[0], : tiles[1]],
                            rowwise_scale_inv_t[: tiles[0], : tiles[1]],
                        )
                    ):
                        failures.append(f"{name}: columnwise copy is not the rowwise transpose")
            num_failures = torch.tensor([len(failures)], dtype=torch.int64, device='cuda')
            torch.distributed.all_reduce(num_failures, op=torch.distributed.ReduceOp.MAX)
        assert num_failures.item() == 0, failures

    def _run_test_helper(
        self,
        tp_size,
        recipe,
        inference: bool = False,
        fp8_param_gather: bool = True,
        use_cuda_graph: bool = False,
        eval_transition: bool = False,
        num_steps: int = 100,
        **kwargs,
    ):
        """Test fp8_param with a small GPT model."""
        # Test-only knob: not a model arg, so pop before create_test_args (which asserts every
        # kwarg is a real arg attribute).
        save_at_steps_kw = kwargs.pop("save_at_steps", ())
        check_precision_aware_optimizer_state = kwargs.pop(
            "check_precision_aware_optimizer_state", False
        )
        # Test-only knob, too. It applies to the FP8-param run only, because the BF16-param
        # reference holds no FP8 weights (asserted below).
        check_fp8_param_replicas = (
            kwargs.pop("check_fp8_param_replicas", False) and fp8_param_gather
        )
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
            # Enable GTP weight-remat when the test requested it (default 1 => no GTP, so
            # non-GTP fp8 tests are unaffected).
            gtp_remat_size=getattr(args, "gtp_weight_remat_size", 1),
        )

        input_ids, labels, position_ids, attention_mask, loss_mask = self.get_batch(
            self.seq_length, self.micro_batch_size
        )
        model_parallel_cuda_manual_seed(
            _SEED, te_rng_tracker=args.te_rng_tracker, force_reset_rng=True
        )
        cfg_container = Utils.pretrain_config_from_global_args(args, "gpt")
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
        if not fp8_param_gather:
            # Negative control for the loss-parity reference: it must keep BF16 primary
            # weights, or parity would compare the FP8-param configuration with itself.
            assert num_fp8_params == 0
        elif fp8_layers > 0:
            if args.num_experts is None:
                # Each dense layer has 4 GEMM weights: qkv, proj, fc1, fc2.
                assert num_fp8_params == 4 * fp8_layers
            else:
                # Each MoE layer has qkv and proj, the shared expert's fc1 and fc2 (all in the
                # dense DP buffers), and one fc1 and one fc2 weight per local expert in the
                # grouped GEMMs (in the expert-DP buffers). The router weight stays BF16.
                num_shared_expert_weights = 2 if args.moe_shared_expert_intermediate_size else 0
                num_local_experts = args.num_experts // args.expert_model_parallel_size
                assert num_fp8_params == fp8_layers * (
                    2 + num_shared_expert_weights + 2 * num_local_experts
                ), num_fp8_params
                assert any(
                    not getattr(param, 'allreduce', True) for param in gpt_model[0].parameters()
                )
                # Routed-expert GEMM weights (allreduce=False) live in the expert-DP buffers;
                # they must be FP8 primary weights too, so the expert-DP all-gather moves
                # quantized data rather than only the dense attention weights doing so.
                expert_weights = [
                    param
                    for param in gpt_model[0].parameters()
                    if not getattr(param, 'allreduce', True) and param.dim() == 2
                ]
                assert expert_weights
                assert all(is_float8tensor(param) for param in expert_weights)
                if not inference:
                    assert len(optimizer.chained_optimizers) >= 2
        if check_fp8_param_replicas:
            # The check compares the first and last forward passes of a training run.
            assert not inference and num_steps > 1 and num_fp8_params > 0
        initial_fp8_values = None

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
        delay_master_offload = (
            not inference
            and args.chunked_optimizer_state_offload
            and args.reuse_grad_buf_for_mxfp8_param_ag
            and args.overlap_param_gather
        )

        # Optional: generate the sharded_state_dict (the checkpoint-save metadata path) at these
        # steps to catch save side-effects on the live weights — a correct save must not perturb
        # the subsequent training step (regression guard for GTP native-FP8 save corruption).
        save_at_steps = set(save_at_steps_kw or ())

        for i in range(num_steps):
            if not inference:
                if args.chunked_optimizer_state_offload:
                    optimizer.offload_optimizer_state_for_forward(
                        offload_master=not delay_master_offload
                    )
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
            if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
                # Each DistOpt staging entry restores its own offloaded master window.
                self.copy_main_params_to_param_buffer(gpt_model, optimizer)

            if delay_master_offload:
                optimizer.offload_optimizer_state_for_forward()

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

            # After a forward pass no param all-gather is in flight (see
            # assert_fp8_params_gathered), and the grad reduce-scatter has not started yet.
            if check_fp8_param_replicas:
                if i == 0:
                    initial_fp8_values = self.dequantize_fp8_params(gpt_model[0])
                elif i == num_steps - 1:
                    self.assert_fp8_params_gathered(gpt_model[0], pg_collection, initial_fp8_values)

            # Verify gradients
            loss = output.mean()
            loss.backward()

            if args.overlap_grad_reduce:
                gpt_model[0].finish_grad_sync()

            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

            if args.chunked_optimizer_state_offload:
                optimizer.prefetch_optimizer_state_for_gradient_finalization()
            update_successful, _, _ = optimizer.step()
            assert update_successful

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

        if not inference and args.chunked_optimizer_state_offload:
            optimizer.offload_optimizer_state_for_forward()
            managers = [
                child._optimizer_state_offloader
                for child in optimizer.chained_optimizers
                if child._optimizer_state_offloader is not None
            ]
            assert managers
            for manager in managers:
                for param in manager.selected_params:
                    for value in manager.optimizer.state[param].values():
                        if isinstance(value, torch.Tensor) and value.numel() == param.numel():
                            assert value.device.type == 'cpu'
            assert any(
                param.device.type == 'cpu'
                or (
                    isinstance(manager.optimizer.state[param].get('master_param'), torch.Tensor)
                    and manager.optimizer.state[param]['master_param'].device.type == 'cpu'
                )
                for manager in managers
                for param in manager.selected_params
            )

        if self.cuda_graph_helper is not None and self.cuda_graph_helper.graphs_created():
            self.cuda_graph_helper.delete_cuda_graphs()
            self.cuda_graph_helper = None

        if check_precision_aware_optimizer_state:
            assert not inference
            self.assert_precision_aware_optimizer_state(args, optimizer)

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
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("recipe", ["blockwise", "mxfp8"])
    def test_chunked_optimizer_state_offload(self, recipe):
        """Adam DistOpt keeps FP8-gather updates equivalent while state/master are offloaded."""

        arch = get_device_arch_version()
        if recipe == "blockwise" and arch != 9:
            pytest.skip("blockwise FP8 is Hopper-only")
        if recipe == "mxfp8" and arch < 10:
            pytest.skip("mxfp8 requires Blackwell architecture or newer")

        common = dict(overlap_param_gather=True, overlap_grad_reduce=True, num_steps=3)
        offloaded = self._run_test_helper(
            1,
            recipe,
            fp8_param_gather=True,
            chunked_optimizer_state_offload=True,
            optimizer_state_offload_chunk_size_mb=1,
            optimizer_state_offload_fraction=1.0,
            ckpt_format='torch_dist',
            **common,
        )
        reference = self._run_test_helper(1, recipe, fp8_param_gather=True, **common)
        torch.testing.assert_close(offloaded, reference, atol=1e-4, rtol=1e-4)

    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    def test_precision_aware_mxfp8_chunked_optimizer_state_offload(self):
        """Precision-aware FusedAdam masters and moments remain CPU canonical with MXFP8."""

        if get_device_arch_version() < 10:
            pytest.skip("mxfp8 requires Blackwell architecture or newer")

        common = dict(
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            num_steps=3,
            use_precision_aware_optimizer=True,
            exp_avg_dtype='bf16',
            exp_avg_sq_dtype='bf16',
        )
        offloaded = self._run_test_helper(
            1,
            "mxfp8",
            fp8_param_gather=True,
            chunked_optimizer_state_offload=True,
            optimizer_state_offload_chunk_size_mb=1,
            optimizer_state_offload_fraction=1.0,
            ckpt_format='torch_dist',
            **common,
        )
        reference = self._run_test_helper(1, "mxfp8", fp8_param_gather=True, **common)
        torch.testing.assert_close(offloaded, reference, atol=1e-4, rtol=1e-4)

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
    @pytest.mark.parametrize("dp_overlap", [(False, False), (True, True)])
    def test_blockwise_scaling(self, tp_size, dp_overlap):
        """
        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
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
            "vocal_size": 128800,
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

    @pytest.mark.skipif(
        get_device_arch_version() < 10, reason="MXFP8 is supported since Blackwell architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.3.0.dev0"), reason="TE 2.3.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [2])
    def test_mxfp8_eval_transition(self, tp_size):
        kwargs = {"overlap_param_gather": True, "overlap_grad_reduce": True}
        self.run_test_with_eval_transition(tp_size=tp_size, recipe="mxfp8", **kwargs)

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

    @staticmethod
    def blockwise_moe_kwargs(dp_overlap):
        """MoE settings for the blockwise FP8 param-gather tests.

        Mirrors test_mxfp8_moe (EP2, grouped GEMM, AllToAll, no aux loss), but with four
        experts and top-2 routing. Each EP rank then has two local experts, so its fc1 and
        fc2 grouped GEMMs each hold two FP8 expert weights in the expert-DP buffers, and
        AllToAll sends each token to two experts. A shared expert adds FP8 weights to the
        dense DP buffers.

        The GLM-5.3 MoE settings (SwiGLU clamped at 10.0, FP32 routing) are compatibility
        coverage only: TransformerConfig accepts them under FP8 and they run. Both arms of
        run_test use them, so loss parity cannot detect errors in them. At this scale the
        clamp never binds either: fc1 pre-activations have a std of about
        0.02 * sqrt(128) ~= 0.23. moe_apply_probs_after_fc2=False is the default and only
        pins the standard probability placement. The rejection of True under fp8/fp8_param
        is tested in transformer/test_glm53_config.py and
        transformer/moe/test_moe_probs_after_fc2.py, not here.

        check_fp8_param_replicas is a test-only knob that runs assert_fp8_params_gathered
        in the FP8-param run.
        """
        return {
            "overlap_param_gather": dp_overlap[0],
            "overlap_grad_reduce": dp_overlap[1],
            "expert_model_parallel_size": 2,
            "num_experts": 4,
            "moe_grouped_gemm": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_router_topk": 2,
            "moe_router_pre_softmax": True,
            "moe_router_load_balancing_type": "none",
            "moe_aux_loss_coeff": 0.0,
            "moe_ffn_hidden_size": 128,
            "moe_shared_expert_intermediate_size": 128,
            "moe_router_dtype": "fp32",
            "activation_func_clamp_value": 10.0,
            "moe_apply_probs_after_fc2": False,
            "check_fp8_param_replicas": True,
        }

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [1])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (True, True)])
    def test_blockwise_moe(self, tp_size, dp_overlap):
        """Blockwise FP8 param gather for routed-expert (expert-DP) and dense (DP) weights.

        This test checks four things:

        * The FP8 primary-weight layout: the exact count, and every routed-expert weight
          is FP8.
        * After the param all-gather, with and without overlap, every FP8 weight is
          bitwise identical across its data-parallel replicas, and its columnwise copy is
          rebuilt from the gathered rowwise data.
        * Loss parity with the BF16-param reference.
        * The BF16-param reference holds no FP8 weights.

        Every rank sees the same batch, so this does not check that the gradient
        reduce-scatter or the EP token routing is correct. Both arms also share those paths.

        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        self.run_test(tp_size=tp_size, recipe="blockwise", **self.blockwise_moe_kwargs(dp_overlap))

    @pytest.mark.skipif(
        get_device_arch_version() != 9, reason="blockwise is only supported on Hopper architecture"
    )
    @pytest.mark.skipif(not fp8_available, reason=reason_for_no_fp8)
    @pytest.mark.skipif(not is_te_min_version("2.4.0.dev0"), reason="TE 2.4.0.dev0 is required")
    @pytest.mark.parametrize("tp_size", [1])
    @pytest.mark.parametrize("dp_overlap", [(False, False), (True, True)])
    def test_blockwise_moe_precision_aware_optimizer(self, tp_size, dp_overlap):
        """Blockwise FP8 MoE param gather with the precision-aware optimizer (BF16 moments).

        Runs the same checks as test_blockwise_moe, plus the optimizer-state layout. The
        FP8-param run and its BF16-param reference share this optimizer configuration, so
        run_test's loss-parity tolerance applies unchanged.

        dp_overlap: (overlap_param_gather, overlap_grad_reduce)
        """
        kwargs = self.blockwise_moe_kwargs(dp_overlap)
        kwargs.update(
            use_precision_aware_optimizer=True,
            main_params_dtype="fp32",
            main_grads_dtype="fp32",
            exp_avg_dtype="bf16",
            exp_avg_sq_dtype="bf16",
            check_precision_aware_optimizer_state=True,
        )
        self.run_test(tp_size=tp_size, recipe="blockwise", **kwargs)
