# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import os
import sys

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from megatron.training.training import get_model, setup_model_and_optimizer
from megatron.training.utils import get_batch_on_this_cp_rank, unwrap_model
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

try:
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

_SEED = 42


class TestMultiTokenPredictionLayer:
    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    def _create_config_and_mtp_block_spec(self, tp, cp, use_te=False):
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config = TransformerConfig(
            mtp_num_layers=2,
            num_layers=4,
            hidden_size=64,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True if tp > 1 else False,
            context_parallel_size=cp,  # Enable CP for MTP testing
        )
        if use_te:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec()
        else:
            transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=use_te
        )
        return config, mtp_block_spec

    @pytest.mark.parametrize(('tp'), [(1), (2), (4)])
    def test_constructor_local(self, tp):
        """Test basic construction of MTP module."""

        torch.manual_seed(_SEED)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp=1)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].transformer_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(('tp', 'cp'), [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_constructor_ues_te(self, tp, cp):
        """Test basic construction of MTP module."""
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        config, mtp_block_spec = self._create_config_and_mtp_block_spec(tp, cp, use_te=True)
        mtp = MultiTokenPredictionBlock(config=config, spec=mtp_block_spec)

        assert isinstance(mtp, MultiTokenPredictionBlock)
        assert len(mtp.layers) == config.mtp_num_layers
        for i in range(config.mtp_num_layers):
            assert mtp.layers[i].layer_number == i + 1
            assert mtp.layers[i].enorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].hnorm.weight.shape[0] == config.hidden_size
            assert mtp.layers[i].eh_proj.weight.shape[0] == config.hidden_size / tp
            assert mtp.layers[i].eh_proj.weight.shape[1] == config.hidden_size * 2
            assert mtp.layers[i].transformer_layer is not None
        num_weights = sum([p.numel() for p in mtp.parameters()])
        if tp == 1:
            assert num_weights == 58560 * config.mtp_num_layers
        elif tp == 2:
            assert num_weights == 29664 * config.mtp_num_layers
        elif tp == 4:
            assert num_weights == 15216 * config.mtp_num_layers


class TestMultiTokenPrediction:
    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

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
            args.num_experts, args.moe_grouped_gemm, args.qk_layernorm
        )
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=True
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_block_spec,
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

        return model

    def create_test_args(
        self, tp, cp, sequence_length, micro_batch_size, fp8=None, full_recompute=False
    ):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_multi_token_predictioin.py']
        args = parse_args()
        args.num_layers = 2
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.vocal_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.num_experts = 8
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.moe_router_topk = 2
        args.moe_router_pre_softmax = False
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.async_tensor_model_parallel_allreduce = False
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        if HAVE_TE:
            # only use grouped gemm if there is TE
            args.moe_grouped_gemm = True
        else:
            args.moe_grouped_gemm = False
        args.bf16 = True
        if fp8 is not None:
            args.fp8 = 'e4m3'
        if full_recompute:
            args.recompute_granularity = 'full'
            args.recompute_method = 'uniform'
            args.recompute_num_layers = 1
        else:
            args.recompute_granularity = None
        args.add_bias_linear = False
        args.swiglu = True

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
        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
        }
        return batch

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (1, 2), (2, 1), (2, 2)])
    def test_sharded_state_dict(self, tp, cp):
        """Test MTP with different tensor parallel sizes."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        gpt_model = get_model(self.model_provider, ModelType.encoder_or_decoder)
        gpt_model = unwrap_model(gpt_model)
        sharded_state_dict = gpt_model[0].sharded_state_dict()
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    @pytest.mark.parametrize(
        ("tp", "cp"), [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2), (2, 4), (4, 1), (4, 2)]
    )
    def test_forward_backward(self, tmp_path_dist_ckpt, tp, cp, full_recompute):
        """Test MTP forward and backward with gptmodel."""
        tp_ref = 1
        cp_ref = 1
        args = self.create_test_args(tp_ref, cp_ref, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=tp_ref, context_parallel_size=cp_ref
        )
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )
        output_ref = gpt_model_ref[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )
        tracker = MTPLossLoggingHelper.tracker
        mtp_loss_ref = None
        assert "values" in tracker
        mtp_loss_ref = tracker['values'].clone()
        MTPLossLoggingHelper.clean_loss_in_tracker()

        iteration = 123
        num_floating_point_operations_so_far = 456

        def set_ckpt_path(ckpt_path):
            args.save = ckpt_path
            args.load = ckpt_path

        with TempNamedDir(
            tmp_path_dist_ckpt / 'test_mtp_model_reconfiguration_model_A'
        ) as ckpt_dir_A:
            set_ckpt_path(ckpt_dir_A)
            save_checkpoint(
                iteration,
                gpt_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            # Test with different TP/CP configuration
            Utils.destroy_model_parallel()
            args = self.create_test_args(
                tp, cp, self.seq_length, self.micro_batch_size, full_recompute=full_recompute
            )
            set_args(args)
            set_ckpt_path(ckpt_dir_A)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
            gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                self.model_provider, ModelType.encoder_or_decoder
            )
            load_checkpoint(gpt_model, optimizer, opt_param_scheduler, strict=False)
            batch["output_ref"] = output_ref
            # Get batch for current CP rank (handles CP tensor splitting)
            batch = get_batch_on_this_cp_rank(batch)
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = gpt_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            tracker = MTPLossLoggingHelper.tracker
            assert "values" in tracker
            mtp_loss = tracker['values'].clone()
            # Average MTP loss across CP ranks for comparison with reference
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
            torch.distributed.all_reduce(
                mtp_loss, group=pg_collection.cp, op=torch.distributed.ReduceOp.AVG
            )
            MTPLossLoggingHelper.clean_loss_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            # Check output shapes - sequence length is divided by CP size
            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            # Verify gradients
            loss = output.mean()
            loss.backward()
            # for param in gpt_model[0].parameters():
            for name, param in gpt_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("1.7.0"),
        reason="Only transformer-engine>=1.7.0 supports MoE FP8 training",
    )
    @pytest.mark.parametrize("full_recompute", [False, True])
    def test_fp8_support(self, full_recompute):
        """Test MTP with FP8 training enabled."""
        tp = 1
        cp = 1
        fp8 = 'e4m3'
        args = self.create_test_args(
            tp, cp, self.seq_length, self.micro_batch_size, fp8, full_recompute=full_recompute
        )
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        batch = self.get_batch(self.seq_length, self.micro_batch_size)
        tokens, labels, loss_mask, attention_mask, position_ids = batch.values()
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )

        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
        )

        assert output.dtype == torch.float32  # Output should be converted back to float32

        loss = output.mean()
        loss.backward()


class TestMTPLossLoggingHelper:
    def setup_method(self, method):
        self.num_layers = 4
        # Reset the tracker before each test
        MTPLossLoggingHelper.tracker = {}

    def teardown_method(self, method):
        # Clean up the tracker after each test
        MTPLossLoggingHelper.tracker = {}

    def test_save_loss_to_tracker(self):
        """Test saving loss to tracker."""
        # Create a dummy loss tensor
        loss = torch.tensor(1.3)
        layer_number = 2
        num_layers = self.num_layers

        # Test saving loss
        MTPLossLoggingHelper.save_loss_to_tracker(
            loss=loss, layer_number=layer_number, num_layers=num_layers
        )

        # Verify tracker state
        assert "values" in MTPLossLoggingHelper.tracker
        assert MTPLossLoggingHelper.tracker["values"].shape == (num_layers,)
        assert MTPLossLoggingHelper.tracker["values"][layer_number] == loss
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None

    def test_track_mtp_metrics(self):
        """Test tracking MTP metrics."""
        # First save some losses
        loss = torch.tensor(2.3)
        num_layers = self.num_layers
        for i in range(num_layers):
            MTPLossLoggingHelper.save_loss_to_tracker(
                loss=loss, layer_number=i, num_layers=num_layers
            )

        # Create dummy writer and loss dict
        class DummyWriter:
            def add_scalar(self, name, value, iteration):
                pass

        class DummyWandBWriter:
            def log(self, metrics, iteration):
                pass

        loss_scale = 1.5
        iteration = 2
        writer = DummyWriter()
        wandb_writer = DummyWandBWriter()
        total_loss_dict = {}

        # Test tracking metrics
        MTPLossLoggingHelper.track_mtp_metrics(
            loss_scale=loss_scale,
            iteration=iteration,
            writer=writer,
            wandb_writer=wandb_writer,
            total_loss_dict=total_loss_dict,
        )

        # Verify total_loss_dict is populated
        for i in range(num_layers):
            assert f"mtp_{i+1} loss" in total_loss_dict
            assert total_loss_dict[f"mtp_{i+1} loss"] == loss * loss_scale

        # Verify tracker is cleaned
        assert torch.all(MTPLossLoggingHelper.tracker["values"] == 0)
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None
