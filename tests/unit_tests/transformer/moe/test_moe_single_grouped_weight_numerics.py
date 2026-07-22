# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import gc
import inspect
import os
import sys
import traceback

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.fp4_utils import is_grouped_nvfp4tensor
from megatron.core.fp8_utils import (
    is_grouped_mxfp8tensor,
    is_grouped_tensor,
    is_grouped_tensor_with_quantized_storage,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import force_param_sync, setup_model_and_optimizer
from megatron.training.utils import get_device_arch_version
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    from transformer_engine.pytorch.fp8 import check_fp8_support, check_nvfp4_support

    _FP8_AVAILABLE, _NO_FP8_REASON = check_fp8_support()
    _NVFP4_AVAILABLE, _NO_NVFP4_REASON = check_nvfp4_support()
except ImportError:
    _FP8_AVAILABLE = False
    _NO_FP8_REASON = "Transformer Engine FP8 support is unavailable"
    _NVFP4_AVAILABLE = False
    _NO_NVFP4_REASON = "Transformer Engine NVFP4 support is unavailable"


_SEED = 1234
_BLACKWELL_AVAILABLE = torch.cuda.is_available() and get_device_arch_version() >= 10
try:
    from transformer_engine.pytorch import GroupedLinear as TEGroupedLinear

    _TE_GROUPED_LINEAR_SUPPORTS_SINGLE_PARAM = (
        "single_grouped_weight" in inspect.signature(TEGroupedLinear.__init__).parameters
    )
    _TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR_BACKEND = (
        "grouped_gemm_backend" in inspect.signature(TEGroupedLinear.__init__).parameters
    )
except (ImportError, AttributeError):
    _TE_GROUPED_LINEAR_SUPPORTS_SINGLE_PARAM = False
    _TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR_BACKEND = False

pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        not is_te_min_version("2.14.0"),
        reason="moe_single_grouped_weight requires Transformer Engine >= 2.14.0",
    ),
    pytest.mark.skipif(
        not _TE_GROUPED_LINEAR_SUPPORTS_SINGLE_PARAM,
        reason="Installed TE GroupedLinear does not expose single_grouped_weight",
    ),
]


def _skip_if_unsupported(precision: str) -> None:
    if Utils.world_size < 2:
        pytest.skip("distributed optimizer parity test requires torchrun with at least 2 ranks")

    if precision in ("mxfp8", "nvfp4") and not _BLACKWELL_AVAILABLE:
        pytest.skip(f"{precision} single grouped weight parity requires Blackwell (SM >= 10)")
    if precision == "mxfp8" and not _FP8_AVAILABLE:
        pytest.skip(_NO_FP8_REASON)
    if precision == "nvfp4" and not _NVFP4_AVAILABLE:
        pytest.skip(_NO_NVFP4_REASON)


class TestMoESingleGroupedWeightNumerics:
    """Numerical parity tests for MoE single grouped weights under DistOpt."""

    seq_length = 128
    micro_batch_size = 2
    num_train_steps = 4

    def setup_method(self, method):
        self._old_single_param_env = os.environ.get("NVTE_GROUPED_LINEAR_SINGLE_PARAM")
        self._old_cutedsl_fused_grouped_mlp_env = os.environ.get("NVTE_CUTEDSL_FUSED_GROUPED_MLP")
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = "1"
        os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = "1"

    def teardown_method(self, method):
        try:
            self._cleanup()
        finally:
            if self._old_single_param_env is None:
                os.environ.pop("NVTE_GROUPED_LINEAR_SINGLE_PARAM", None)
            else:
                os.environ["NVTE_GROUPED_LINEAR_SINGLE_PARAM"] = self._old_single_param_env
            if self._old_cutedsl_fused_grouped_mlp_env is None:
                os.environ.pop("NVTE_CUTEDSL_FUSED_GROUPED_MLP", None)
            else:
                os.environ["NVTE_CUTEDSL_FUSED_GROUPED_MLP"] = (
                    self._old_cutedsl_fused_grouped_mlp_env
                )

    def _cleanup(self):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def model_provider(
        self, pre_process=True, post_process=True, config=None, pg_collection=None, vp_stage=None
    ):
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        if config is None:
            config = core_transformer_config_from_args(args)
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            num_experts=args.num_experts, moe_grouped_gemm=args.moe_grouped_gemm
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
            pg_collection=pg_collection,
            vp_stage=vp_stage,
        )

    def create_test_args(
        self,
        precision: str,
        primary_param_gather: bool,
        single_weight: bool,
        gradient_accumulation_fusion: bool,
        use_transformer_engine_op_fuser: bool,
        overlap_param_gather: bool = False,
        overlap_grad_reduce: bool = False,
        grad_reduce_in_fp32: bool = False,
    ):
        self._cleanup()

        sys.argv = ["test_moe_single_grouped_weight_numerics.py"]
        args = parse_args()
        args.num_layers = 1
        args.vocab_size = 1024
        args.hidden_size = 256
        args.ffn_hidden_size = 256
        args.num_attention_heads = 8
        args.max_position_embeddings = self.seq_length
        args.seq_length = self.seq_length
        args.micro_batch_size = self.micro_batch_size
        args.global_batch_size = self.micro_batch_size * Utils.world_size
        args.create_attention_mask_in_dataloader = True
        args.tensor_model_parallel_size = 1
        args.pipeline_model_parallel_size = 1
        args.context_parallel_size = 1
        args.expert_model_parallel_size = 1
        args.train_iters = self.num_train_steps
        args.lr = 3e-5
        args.bf16 = True
        args.attention_backend = "unfused"
        args.add_bias_linear = False
        args.hidden_dropout = 0.0
        args.attention_dropout = 0.0
        args.swiglu = True
        args.gradient_accumulation_fusion = gradient_accumulation_fusion
        args.use_distributed_optimizer = True
        args.use_transformer_engine_op_fuser = use_transformer_engine_op_fuser
        args.overlap_param_gather = overlap_param_gather
        args.overlap_grad_reduce = overlap_grad_reduce
        args.accumulate_allreduce_grads_in_fp32 = grad_reduce_in_fp32
        args.ddp_bucket_size = 40960

        args.num_experts = 2
        args.moe_layer_freq = 1
        args.moe_grouped_gemm = True
        args.moe_grouped_gemm_backend = "grouped_tensor"
        args.moe_single_grouped_weight = single_weight
        args.moe_token_dispatcher_type = "alltoall"
        args.moe_router_topk = 1
        args.moe_router_pre_softmax = True
        args.moe_router_load_balancing_type = "none"
        args.moe_aux_loss_coeff = 0.0
        args.moe_ffn_hidden_size = 256
        args.moe_mlp_glu_interleave_size = 32

        if precision == "mxfp8":
            args.fp8 = "e4m3"
            args.fp8_recipe = "mxfp8"
            args.fp8_param_gather = primary_param_gather
            args.reuse_grad_buf_for_mxfp8_param_ag = primary_param_gather
        elif precision == "nvfp4":
            args.fp4 = "e2m1"
            args.fp4_recipe = "nvfp4"
            args.fp4_param_gather = primary_param_gather
        elif precision != "bf16":
            raise ValueError(f"Unknown precision test case: {precision}")

        validate_args(args)
        set_global_variables(args, False)
        return args

    def get_batch(self):
        data = torch.arange(self.seq_length, dtype=torch.int64, device="cuda")
        input_ids = data.repeat((self.micro_batch_size, 1))
        labels = (data + 1).repeat((self.micro_batch_size, 1))
        position_ids = data.repeat((self.micro_batch_size, 1))
        attention_mask = torch.ones(
            (self.micro_batch_size, 1, self.seq_length, self.seq_length), dtype=bool, device="cuda"
        )
        loss_mask = torch.ones(
            (self.micro_batch_size, self.seq_length), dtype=torch.float32, device="cuda"
        )
        return input_ids, labels, position_ids, attention_mask, loss_mask

    def assert_storage_path_is_exercised(
        self, model, precision: str, primary_param_gather: bool, single_weight: bool
    ):
        params = list(model.named_parameters())
        if not single_weight:
            assert not any(is_grouped_tensor(param) for _, param in params)
            return

        grouped_params = [param for _, param in params if is_grouped_tensor(param)]
        assert grouped_params, "Expected at least one TE GroupedTensor MoE parameter"

        if not primary_param_gather or precision == "bf16":
            assert any(
                not is_grouped_tensor_with_quantized_storage(param) for param in grouped_params
            ), "Expected high-precision grouped primary weights"
            return

        if precision == "mxfp8":
            assert any(is_grouped_mxfp8tensor(param) for param in grouped_params)
        elif precision == "nvfp4":
            assert any(is_grouped_nvfp4tensor(param) for param in grouped_params)

    @staticmethod
    def iter_distopt_buffers(optimizer):
        optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
        for optim_instance in optimizers:
            for buffer in getattr(optim_instance, "buffers", []):
                yield buffer

    def assert_grouped_params_remapped_to_ddp_param_data(self, optimizer, precision: str):
        """Grouped BF16/NVFP4 params must point at the live DDP param_data slice."""
        num_checked = 0
        for buffer in self.iter_distopt_buffers(optimizer):
            for bucket in buffer.buckets:
                if bucket.param_data is None:
                    continue
                for param in bucket.params:
                    if not is_grouped_tensor(param):
                        continue

                    rowwise_data = getattr(param, "rowwise_data", None)
                    assert rowwise_data is not None, "GroupedTensor is missing rowwise_data"

                    if precision == "bf16":
                        if is_grouped_tensor_with_quantized_storage(param):
                            continue
                        start, end = bucket.param_to_index[param]
                        expected = bucket.param_data.view(-1)[start:end]
                    elif precision == "nvfp4":
                        if not is_grouped_nvfp4tensor(param):
                            continue
                        packed_start, packed_end, bucket_id = buffer.nvfp4_packed_param_index_map[
                            param
                        ]
                        assert bucket_id == bucket.bucket_id
                        bucket_start, _ = buffer.nvfp4_packed_bucket_indices[bucket_id]
                        expected = bucket.param_data.view(-1)[
                            packed_start - bucket_start : packed_end - bucket_start
                        ]
                    else:
                        raise ValueError(f"Unsupported remap precision: {precision}")

                    rowwise_flat = rowwise_data.view(-1)
                    assert rowwise_flat.numel() == expected.numel()
                    assert rowwise_flat.dtype == expected.dtype
                    assert rowwise_flat.data_ptr() == expected.data_ptr(), (
                        "Live grouped parameter rowwise_data is not mapped to the DDP "
                        f"param_data slice for precision={precision}"
                    )
                    num_checked += 1

        assert num_checked > 0, f"Did not find any {precision} grouped params to verify"

    def assert_execution_path_is_exercised(
        self, model, use_transformer_engine_op_fuser: bool, after_forward: bool = False
    ):
        grouped_mlps = [
            module for module in model.modules() if module.__class__.__name__ == "TEGroupedMLP"
        ]
        assert grouped_mlps, "Expected at least one TEGroupedMLP module"
        assert all(
            module._with_fused_impl == use_transformer_engine_op_fuser for module in grouped_mlps
        ), "Unexpected TEGroupedMLP execution path"
        if after_forward:
            assert all(
                (module._fused_ops is not None) == use_transformer_engine_op_fuser
                for module in grouped_mlps
            ), "Unexpected TEGroupedMLP fused-op construction state"

    def run_training_case(
        self,
        precision: str,
        primary_param_gather: bool,
        single_weight: bool,
        gradient_accumulation_fusion: bool,
        use_transformer_engine_op_fuser: bool,
    ):
        args = self.create_test_args(
            precision,
            primary_param_gather,
            single_weight,
            gradient_accumulation_fusion,
            use_transformer_engine_op_fuser,
        )
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=args.expert_model_parallel_size
        )

        batch = self.get_batch()
        model, optimizer, _ = setup_model_and_optimizer(
            model_type=ModelType.encoder_or_decoder, model_provider_func=self.model_provider
        )
        assert len(model) == 1
        self.assert_storage_path_is_exercised(
            model[0], precision, primary_param_gather, single_weight
        )
        self.assert_execution_path_is_exercised(model[0], use_transformer_engine_op_fuser)

        losses = []
        for _ in range(self.num_train_steps):
            model[0].zero_grad_buffer()
            optimizer.zero_grad()
            model[0].set_is_first_microbatch()
            output = model[0].forward(
                input_ids=batch[0],
                labels=batch[1],
                position_ids=batch[2],
                attention_mask=batch[3],
                loss_mask=batch[4],
            )
            loss = output.mean()
            assert torch.isfinite(loss)
            loss.backward()

            # Wait for an overlapped reduction, or launch it synchronously when overlap is off.
            model[0].finish_grad_sync()

            update_successful, _, _ = optimizer.step()
            assert update_successful
            losses.append(loss.detach().float().cpu())

        self.assert_execution_path_is_exercised(
            model[0], use_transformer_engine_op_fuser, after_forward=True
        )
        return torch.stack(losses)

    def run_one_mxfp8_overlap_train_step(self, args, model, optimizer, batch):
        model[0].zero_grad_buffer()
        optimizer.zero_grad()
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            optimizer.prepare_model_params_for_param_sync()
        model[0].set_is_first_microbatch()
        output = model[0].forward(
            input_ids=batch[0],
            labels=batch[1],
            position_ids=batch[2],
            attention_mask=batch[3],
            loss_mask=batch[4],
        )
        loss = output.mean()
        assert torch.isfinite(loss)
        loss.backward()

        # Wait for an overlapped reduction, or launch it synchronously when overlap is off.
        model[0].finish_grad_sync()

        update_successful, _, _ = optimizer.step()
        assert update_successful
        return loss.detach().float().cpu()

    def run_mxfp8_eval_step(self, args, model, optimizer, batch):
        if args.reuse_grad_buf_for_mxfp8_param_ag and args.overlap_param_gather:
            optimizer.prepare_model_params_for_param_sync()

        model[0].disable_forward_pre_hook(param_sync=True)
        model[0].eval()
        with torch.no_grad():
            output = model[0].forward(
                input_ids=batch[0],
                labels=batch[1],
                position_ids=batch[2],
                attention_mask=batch[3],
                loss_mask=batch[4],
            )
            assert torch.isfinite(output.mean())
        model[0].train()
        model[0].enable_forward_pre_hook()

    def setup_mxfp8_overlap_case(self, single_weight: bool, checkpoint_dir=None):
        args = self.create_test_args(
            precision="mxfp8",
            primary_param_gather=True,
            single_weight=single_weight,
            gradient_accumulation_fusion=True,
            use_transformer_engine_op_fuser=True,
            overlap_param_gather=True,
            overlap_grad_reduce=True,
            grad_reduce_in_fp32=True,
        )
        if checkpoint_dir is not None:
            args.save = checkpoint_dir
            args.load = checkpoint_dir
            args.ckpt_format = "torch_dist"
            args.use_dist_ckpt = True
            args.auto_detect_ckpt_format = False
            args.async_save = False
            args.ckpt_assume_constant_structure = False
            args.ckpt_load_validate_sharding_integrity = True
            args.dist_ckpt_strictness = "assume_ok_unexpected"
            args.no_save_optim = True
            args.no_load_optim = True
            args.no_save_rng = True
            args.no_load_rng = True
            args.load_main_params_from_ckpt = True
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, expert_model_parallel_size=args.expert_model_parallel_size
        )

        model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            model_type=ModelType.encoder_or_decoder, model_provider_func=self.model_provider
        )
        assert len(model) == 1
        self.assert_storage_path_is_exercised(model[0], "mxfp8", True, single_weight)
        self.assert_execution_path_is_exercised(model[0], True)

        batch = self.get_batch()
        return args, model, optimizer, opt_param_scheduler, batch

    def run_mxfp8_training_losses_with_optional_eval(
        self, eval_after_step: int | None, single_weight: bool = True
    ):
        args, model, optimizer, _, batch = self.setup_mxfp8_overlap_case(
            single_weight=single_weight
        )
        losses = []
        for step in range(4):
            if eval_after_step is not None and step == eval_after_step:
                self.run_mxfp8_eval_step(args, model, optimizer, batch)
            losses.append(self.run_one_mxfp8_overlap_train_step(args, model, optimizer, batch))
        return torch.stack(losses)

    def run_mxfp8_training_losses_with_optional_checkpoint(
        self, checkpoint_dir, checkpoint_before_step: int | None
    ):
        args, model, optimizer, opt_param_scheduler, batch = self.setup_mxfp8_overlap_case(
            single_weight=True, checkpoint_dir=checkpoint_dir
        )
        losses = []
        for step in range(4):
            if checkpoint_before_step is not None and step == checkpoint_before_step:
                force_param_sync(model, optimizer=optimizer)
                save_checkpoint(step, model, optimizer, opt_param_scheduler, 0)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
            losses.append(self.run_one_mxfp8_overlap_train_step(args, model, optimizer, batch))
        return torch.stack(losses)

    def run_mxfp8_checkpoint_save_load_next_loss(
        self, checkpoint_dir, save_single_weight: bool, load_single_weight: bool
    ):
        args, model, optimizer, opt_param_scheduler, batch = self.setup_mxfp8_overlap_case(
            single_weight=save_single_weight, checkpoint_dir=checkpoint_dir
        )

        for _ in range(2):
            self.run_one_mxfp8_overlap_train_step(args, model, optimizer, batch)
        force_param_sync(model, optimizer=optimizer)
        save_checkpoint(2, model, optimizer, opt_param_scheduler, 0)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        self._cleanup()

        args, model, optimizer, opt_param_scheduler, batch = self.setup_mxfp8_overlap_case(
            single_weight=load_single_weight, checkpoint_dir=checkpoint_dir
        )
        loaded_iteration, _ = load_checkpoint(model, optimizer, opt_param_scheduler, strict=True)
        assert loaded_iteration == 2
        return self.run_one_mxfp8_overlap_train_step(args, model, optimizer, batch)

    @staticmethod
    def assert_loss_parity(precision: str, single_weight_losses, discrete_weight_losses):
        if precision == "bf16":
            atol = rtol = 5e-3
        else:
            atol = rtol = 2e-2
        torch.testing.assert_close(
            single_weight_losses, discrete_weight_losses, atol=atol, rtol=rtol
        )

    @staticmethod
    def assert_all_ranks_passed(local_passed: bool, local_error: str) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if not local_passed:
                pytest.fail(local_error)
            return

        pass_flag = torch.tensor(
            [1 if local_passed else 0], dtype=torch.int32, device=torch.cuda.current_device()
        )
        torch.distributed.all_reduce(pass_flag, op=torch.distributed.ReduceOp.MIN)
        if pass_flag.item() == 1:
            return

        rank = torch.distributed.get_rank()
        if local_passed:
            pytest.fail("At least one distributed rank failed this parity case.")
        pytest.fail(f"Rank {rank} failed this parity case:\n{local_error}")

    def run_parity_case(
        self,
        precision: str,
        primary_param_gather: bool,
        gradient_accumulation_fusion: bool,
        use_transformer_engine_op_fuser: bool,
    ) -> None:
        local_passed = True
        local_error = ""
        try:
            single_losses = self.run_training_case(
                precision=precision,
                primary_param_gather=primary_param_gather,
                single_weight=True,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
            )
            discrete_losses = self.run_training_case(
                precision=precision,
                primary_param_gather=primary_param_gather,
                single_weight=False,
                gradient_accumulation_fusion=gradient_accumulation_fusion,
                use_transformer_engine_op_fuser=use_transformer_engine_op_fuser,
            )
            self.assert_loss_parity(precision, single_losses, discrete_losses)
        except Exception:
            local_passed = False
            local_error = traceback.format_exc()

        self.assert_all_ranks_passed(local_passed, local_error)

    def run_remap_case(self, precision: str) -> None:
        local_passed = True
        local_error = ""
        try:
            primary_param_gather = precision == "nvfp4"
            args = self.create_test_args(
                precision=precision,
                primary_param_gather=primary_param_gather,
                single_weight=True,
                gradient_accumulation_fusion=True,
                use_transformer_engine_op_fuser=True,
            )
            set_args(args)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(
                tensor_model_parallel_size=1,
                expert_model_parallel_size=args.expert_model_parallel_size,
            )

            model, optimizer, _ = setup_model_and_optimizer(
                model_type=ModelType.encoder_or_decoder, model_provider_func=self.model_provider
            )
            assert len(model) == 1
            self.assert_storage_path_is_exercised(
                model[0], precision, primary_param_gather, single_weight=True
            )
            self.assert_grouped_params_remapped_to_ddp_param_data(optimizer, precision)
        except Exception:
            local_passed = False
            local_error = traceback.format_exc()

        self.assert_all_ranks_passed(local_passed, local_error)

    @pytest.mark.parametrize("precision", ["bf16", "nvfp4"])
    def test_single_grouped_weight_ddp_param_data_remap_data_ptr(self, precision):
        """BF16/NVFP4 single grouped weights must alias DDP param_data after buffer setup."""
        _skip_if_unsupported(precision)
        self.run_remap_case(precision)

    def test_single_grouped_mxfp8_train_eval_train_matches_train_only(self):
        """Eval should not change subsequent MXFP8 single grouped weight training losses."""
        _skip_if_unsupported("mxfp8")
        local_passed = True
        local_error = ""
        try:
            train_only_losses = self.run_mxfp8_training_losses_with_optional_eval(
                eval_after_step=None
            )
            train_eval_train_losses = self.run_mxfp8_training_losses_with_optional_eval(
                eval_after_step=2
            )
            torch.testing.assert_close(
                train_eval_train_losses, train_only_losses, atol=1e-4, rtol=1e-4
            )
        except Exception:
            local_passed = False
            local_error = traceback.format_exc()

        self.assert_all_ranks_passed(local_passed, local_error)

    @pytest.mark.parametrize(
        "checkpoint_case, save_single_weight, load_single_weight",
        [
            # Save-only: checkpointing should not perturb live training state.
            pytest.param("save_only", True, None, id="save-only-single"),
            # Layout interchange: torch_dist saves grouped MoE weights as per-expert keys.
            pytest.param("save_load", True, False, id="save-single-load-discrete"),
            # Reverse interchange: per-expert checkpoint keys must fold into one grouped param.
            pytest.param("save_load", False, True, id="save-discrete-load-single"),
        ],
    )
    def test_mxfp8_single_weight_torch_dist_checkpoint_matches_discrete_baseline(
        self, tmp_path_dist_ckpt, checkpoint_case, save_single_weight, load_single_weight
    ):
        """torch_dist checkpoint save/load should preserve MXFP8 discrete baseline numerics."""
        _skip_if_unsupported("mxfp8")
        local_passed = True
        local_error = ""
        try:
            discrete_train_only_losses = self.run_mxfp8_training_losses_with_optional_eval(
                eval_after_step=None, single_weight=False
            )
            with TempNamedDir(
                tmp_path_dist_ckpt / "test_mxfp8_single_weight_torch_dist_checkpoint", sync=True
            ) as checkpoint_dir:
                if checkpoint_case == "save_only":
                    # This catches forced-param-sync/checkpoint side effects without reload.
                    checkpoint_losses = self.run_mxfp8_training_losses_with_optional_checkpoint(
                        checkpoint_dir=checkpoint_dir, checkpoint_before_step=2
                    )
                    self.assert_loss_parity("mxfp8", checkpoint_losses, discrete_train_only_losses)
                else:
                    # This catches checkpoint key/layout conversion bugs across single/discrete.
                    loaded_next_loss = self.run_mxfp8_checkpoint_save_load_next_loss(
                        checkpoint_dir,
                        save_single_weight=save_single_weight,
                        load_single_weight=load_single_weight,
                    )
                    torch.testing.assert_close(
                        loaded_next_loss, discrete_train_only_losses[2], atol=2e-2, rtol=2e-2
                    )
        except Exception:
            local_passed = False
            local_error = traceback.format_exc()

        self.assert_all_ranks_passed(local_passed, local_error)

    @pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
    @pytest.mark.parametrize("gradient_accumulation_fusion", [False, True])
    def test_single_grouped_weight_parity_with_primary_param_gather(
        self, precision, gradient_accumulation_fusion
    ):
        """Compare single vs discrete MoE weights with primary param gather enabled if applicable."""
        _skip_if_unsupported(precision)
        self.run_parity_case(
            precision=precision,
            primary_param_gather=True,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            use_transformer_engine_op_fuser=True,
        )

    @pytest.mark.parametrize("precision", ["bf16", "mxfp8", "nvfp4"])
    @pytest.mark.parametrize("gradient_accumulation_fusion", [False, True])
    def test_single_grouped_weight_parity_without_primary_param_gather(
        self, precision, gradient_accumulation_fusion
    ):
        """Compare single vs discrete MoE weights when primary weights stay BF16."""
        _skip_if_unsupported(precision)
        self.run_parity_case(
            precision=precision,
            primary_param_gather=False,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            use_transformer_engine_op_fuser=True,
        )

    @pytest.mark.parametrize(
        "precision,primary_param_gather", [("bf16", False), ("mxfp8", False), ("mxfp8", True)]
    )
    @pytest.mark.parametrize("gradient_accumulation_fusion", [False, True])
    def test_single_grouped_weight_parity_module_grouped_linear(
        self, precision, primary_param_gather, gradient_accumulation_fusion
    ):
        """Compare native TE GroupedLinear single and discrete parameter layouts."""
        if not _TE_GROUPED_LINEAR_SUPPORTS_GROUPED_TENSOR_BACKEND:
            pytest.skip("Installed TE GroupedLinear does not expose grouped_gemm_backend")
        _skip_if_unsupported(precision)
        self.run_parity_case(
            precision=precision,
            primary_param_gather=primary_param_gather,
            gradient_accumulation_fusion=gradient_accumulation_fusion,
            use_transformer_engine_op_fuser=False,
        )
