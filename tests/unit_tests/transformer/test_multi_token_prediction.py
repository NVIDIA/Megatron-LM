# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import sys

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.extensions.transformer_engine import HAVE_TE
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.parallel_state import get_context_parallel_group
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.multi_token_prediction import (
    MTPLossLoggingHelper,
    MultiTokenPredictionBlock,
    roll_tensor,
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

if HAVE_TE:
    from megatron.core.extensions.transformer_engine import TEColumnParallelGroupedLinear
else:
    TEColumnParallelGroupedLinear = None

from megatron.core.tensor_parallel import gather_from_sequence_parallel_region

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
            assert mtp.layers[i].mtp_model_layer is not None
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
            assert mtp.layers[i].mtp_model_layer is not None
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
            vocab_size=args.vocab_size,
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
        args.vocab_size = 128800
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

    def get_packed_batch(self, seq_lengths, micro_batch_size):
        """
        Create a packed sequence batch with multiple sequences of varying lengths.

        Args:
            seq_lengths: List of sequence lengths (e.g., [10, 15, 8] for 3 sequences)
            micro_batch_size: Batch size (typically 1 for packed sequences)

        Returns:
            batch: Dictionary containing packed sequences and PackedSeqParams
        """
        total_seq_length = sum(seq_lengths)

        # Create packed input_ids, labels, and position_ids
        input_ids_list = []
        labels_list = []
        position_ids_list = []

        for seq_len in seq_lengths:
            data = list(range(seq_len))
            input_ids_list.extend(data)
            labels_list.extend([x + 1 for x in data])
            position_ids_list.extend(data)

        # Convert to tensors with shape [batch, total_seq_length]
        input_ids = torch.tensor(input_ids_list, dtype=torch.int64).unsqueeze(0).cuda()
        labels = torch.tensor(labels_list, dtype=torch.int64).unsqueeze(0).cuda()
        position_ids = torch.tensor(position_ids_list, dtype=torch.int64).unsqueeze(0).cuda()

        # Create attention mask for packed sequences (all ones for simplicity)
        attention_mask = torch.ones(
            (micro_batch_size, 1, total_seq_length, total_seq_length), dtype=bool
        ).cuda()

        # Create loss mask with shape [batch, total_seq_length]
        loss_mask = torch.ones(micro_batch_size, total_seq_length).cuda()

        # Create cumulative sequence lengths for PackedSeqParams
        cu_seqlens = torch.tensor(
            [0] + [sum(seq_lengths[: i + 1]) for i in range(len(seq_lengths))], dtype=torch.int32
        ).cuda()

        packed_seq_params = PackedSeqParams(
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(seq_lengths),
            max_seqlen_kv=max(seq_lengths),
            qkv_format='thd',
        )

        batch = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'packed_seq_params': packed_seq_params,
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

    @pytest.mark.skipif(
        not HAVE_TE or not is_te_min_version("2.1.0"),
        reason="grouped_gemm requires TransformerEngine >= 2.1.0",
    )
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1), (2, 2)])
    def test_packed_sequences(self, tp, cp):
        """Test MTP with packed sequences."""
        # Create args with packed sequences support
        seq_lengths = [16, 24, 12]  # Three sequences of different lengths
        total_seq_length = sum(seq_lengths)

        args = self.create_test_args(tp, cp, total_seq_length, micro_batch_size=1)
        set_args(args)

        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)

        # Get packed batch
        batch = self.get_packed_batch(seq_lengths, micro_batch_size=1)
        tokens = batch['tokens']
        labels = batch['labels']
        loss_mask = batch['loss_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        packed_seq_params = batch['packed_seq_params']

        # Create model
        gpt_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )

        # Forward pass with packed sequences
        output = gpt_model[0].forward(
            input_ids=tokens,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=packed_seq_params,
        )

        # Verify output shape
        assert output.shape[0] == 1  # batch size
        assert output.shape[1] == total_seq_length

        # Verify MTP loss was computed
        tracker = MTPLossLoggingHelper.tracker
        assert "values" in tracker
        mtp_loss = tracker['values'].clone()
        assert mtp_loss.shape[0] == args.mtp_num_layers
        MTPLossLoggingHelper.clean_loss_in_tracker()

        # Backward pass
        loss = output.mean()
        loss.backward()

        # Verify gradients exist
        for name, param in gpt_model[0].named_parameters():
            assert param.main_grad is not None, f"Gradient missing for {name}"

    @pytest.mark.parametrize("cp", [1, 2])
    def test_roll_tensor_with_packed_sequences(self, cp):
        """Test roll_tensor function with packed sequences, with and without CP.

        For CP=1: Tests standard packed sequence rolling with verified expected values
        For CP=2: Tests CP-enabled rolling executes without errors
        """
        Utils.initialize_model_parallel(tensor_model_parallel_size=1, context_parallel_size=cp)
        cp_group = get_context_parallel_group() if cp > 1 else None
        cp_rank = torch.distributed.get_rank(group=cp_group) if cp_group is not None else 0

        if cp == 1:
            # Test case: Simple packed sequences (CP disabled)
            tensor = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32).cuda()
            cu_seqlens = torch.tensor([0, 3, 5], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=3,
                max_seqlen_kv=3,
                qkv_format='thd',
            )

            # Roll by -1 (shift left)
            rolled, sum_val = roll_tensor(
                tensor, shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )

            # Expected: [2, 3, 0, 5, 0] - boundaries at indices 2 and 4 are zeroed
            expected = torch.tensor([2, 3, 0, 5, 0], dtype=torch.float32).cuda()
            assert torch.equal(rolled, expected), f"Expected {expected}, got {rolled}"
        else:
            # Test case: Packed sequences with CP=2
            # Two sequences:
            #   seq1 = [1, 2, 3, 4, 5, 6, 7, 8]
            #   seq2 = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

            if cp_rank == 0:
                # CP Rank 0: first half of each sequence
                tensor = torch.tensor(
                    [1, 2, 7, 8, 11, 12, 13, 20, 21, 22], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [2, 3, 8, 0, 12, 13, 14, 21, 22, 0], dtype=torch.float32
                ).cuda()
            else:
                # CP Rank 1: second half of each sequence
                tensor = torch.tensor(
                    [3, 4, 5, 6, 14, 15, 16, 17, 18, 19], dtype=torch.float32
                ).cuda()
                expected = torch.tensor(
                    [4, 5, 6, 7, 15, 16, 17, 18, 19, 20], dtype=torch.float32
                ).cuda()

            cu_seqlens = torch.tensor([0, 8, 20], dtype=torch.int32).cuda()

            packed_seq_params = PackedSeqParams(
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_kv=cu_seqlens,
                max_seqlen_q=6,  # max(4, 6) - max local seq length per sequence
                max_seqlen_kv=6,
                qkv_format='thd',
            )

            # Roll by -1 (shift left) with CP communication
            rolled, sum_val = roll_tensor(
                tensor, shifts=-1, dims=0, cp_group=cp_group, packed_seq_params=packed_seq_params
            )

            # Verify the rolled tensor matches expected values
            assert (
                rolled.shape == expected.shape
            ), f"Shape mismatch: expected {expected.shape}, got {rolled.shape}"
            assert torch.equal(
                rolled, expected
            ), f"CP Rank {cp_rank}: Expected\n{expected}\nbut got\n{rolled}\nDiff:\n{rolled - expected}"

            # Verify sum is correct
            assert sum_val.numel() == 1, "Sum should be a scalar"

        Utils.destroy_model_parallel()


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
            assert f"mtp_{i + 1} loss" in total_loss_dict
            assert total_loss_dict[f"mtp_{i + 1} loss"] == loss * loss_scale

        # Verify tracker is cleaned
        assert torch.all(MTPLossLoggingHelper.tracker["values"] == 0)
        assert MTPLossLoggingHelper.tracker["reduce_group"] is None
        assert MTPLossLoggingHelper.tracker["avg_group"] is None


class TestMultiTokenPredictionMamba:
    """Test Multi-Token Prediction with Mamba hybrid models."""

    def setup_method(self, method):
        self.seq_length = 32
        self.micro_batch_size = 2
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()
        MTPLossLoggingHelper.tracker = {}

    def model_provider(self, pre_process=True, post_process=True, **config_kwargs):
        """Model provider for Mamba hybrid models with MTP.

        Uses the unified pattern syntax where MTP is configured via hybrid_layer_pattern:
        Format: "<main_pattern>/<mtp_pattern>/<mtp_pattern>/..."
        Example: "M*M*/M*/M*" = main decoder "M*M*", MTP pattern "M*" with 2 depths
        """
        model_parallel_cuda_manual_seed(_SEED)
        args = get_args()
        config = core_transformer_config_from_args(args)

        # MTP is configured via unified pattern in hybrid_layer_pattern
        # MambaModel creates the MTP block internally based on the parsed pattern
        model = MambaModel(
            config=config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            hybrid_layer_pattern=args.hybrid_layer_pattern,
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

        sys.argv = ['test_multi_token_prediction_mamba.py']
        args = parse_args()
        args.mtp_num_layers = 2
        args.mtp_loss_scaling_factor = 0.1
        args.vocab_size = 128800
        args.hidden_size = 128
        args.num_attention_heads = 8
        args.num_query_groups = 8
        args.mamba_num_groups = 4
        args.max_position_embeddings = 256
        args.micro_batch_size = micro_batch_size
        args.create_attention_mask_in_dataloader = True
        args.seq_length = sequence_length
        args.tensor_model_parallel_size = tp
        args.sequence_parallel = True if tp > 1 else False
        args.context_parallel_size = cp
        args.position_embedding_type = 'rope'
        args.train_iters = 1
        args.ckpt_format = 'torch_dist'
        args.lr = 3e-5
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.no_save_optim = True
        args.no_load_optim = True
        args.no_load_rng = True
        args.bf16 = True
        # Unified pattern: "main/mtp/mtp" - main decoder "M*M*", MTP pattern "M*" with 2 depths
        args.hybrid_layer_pattern = "M*M*/M*/M*"
        args.spec = "megatron.core.models.mamba.mamba_layer_specs.mamba_stack_spec"

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

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_sharded_state_dict_mamba(self, tp, cp):
        """Test MTP with Mamba hybrid model - sharded state dict."""
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        mamba_model = get_model(self.model_provider, ModelType.encoder_or_decoder)
        mamba_model = unwrap_model(mamba_model)
        sharded_state_dict = mamba_model[0].sharded_state_dict()

        # Verify MTP layers are in the state dict
        for i in range(args.mtp_num_layers):
            assert f"mtp.layers.{i}.enorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.hnorm.weight" in sharded_state_dict.keys()
            assert f"mtp.layers.{i}.eh_proj.weight" in sharded_state_dict.keys()

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    @pytest.mark.parametrize(("tp", "cp"), [(1, 1), (2, 1)])
    def test_forward_backward_mamba(self, tmp_path_dist_ckpt, tp, cp):
        """Test MTP forward and backward with Mamba hybrid model."""
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

        mamba_model_ref, optimizer, opt_param_scheduler = setup_model_and_optimizer(
            self.model_provider, ModelType.encoder_or_decoder
        )

        output_ref = mamba_model_ref[0].forward(
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

        with TempNamedDir(tmp_path_dist_ckpt / 'test_mtp_mamba_model_reconfiguration') as ckpt_dir:
            set_ckpt_path(ckpt_dir)
            save_checkpoint(
                iteration,
                mamba_model_ref,
                optimizer,
                opt_param_scheduler,
                num_floating_point_operations_so_far,
            )

            expected_ckpt_path = args.save / "iter_0000123" / ".metadata"
            assert os.path.exists(expected_ckpt_path)

            Utils.destroy_model_parallel()
            args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
            set_args(args)
            set_ckpt_path(ckpt_dir)
            torch.manual_seed(_SEED)
            Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
            mamba_model, optimizer, opt_param_scheduler = setup_model_and_optimizer(
                self.model_provider, ModelType.encoder_or_decoder
            )
            load_checkpoint(mamba_model, optimizer, opt_param_scheduler, strict=False)

            batch["output_ref"] = output_ref
            batch = get_batch_on_this_cp_rank(batch)
            tokens, labels, loss_mask, attention_mask, position_ids, output_ref = batch.values()
            output = mamba_model[0].forward(
                input_ids=tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                loss_mask=loss_mask,
            )
            tracker = MTPLossLoggingHelper.tracker
            assert "values" in tracker
            mtp_loss = tracker['values'].clone()
            pg_collection = ProcessGroupCollection.use_mpu_process_groups(required_pgs=['cp'])
            torch.distributed.all_reduce(
                mtp_loss, group=pg_collection.cp, op=torch.distributed.ReduceOp.AVG
            )
            MTPLossLoggingHelper.clean_loss_in_tracker()
            assert torch.allclose(output_ref, output, rtol=1e-03, atol=1e-03)
            assert torch.allclose(mtp_loss, mtp_loss_ref, rtol=1e-02, atol=1e-02)

            assert output.shape[0] == self.micro_batch_size
            assert output.shape[1] == self.seq_length / cp

            loss = output.mean()
            loss.backward()
            for name, param in mamba_model[0].named_parameters():
                assert param.main_grad is not None

    @pytest.mark.skipif(not HAVE_TE, reason="transformer_engine not available")
    def test_attention_mask_validation_mamba(self):
        """Test that attention mask type validation works for Mamba hybrid models."""
        tp = 1
        cp = 1
        args = self.create_test_args(tp, cp, self.seq_length, self.micro_batch_size)
        set_args(args)
        torch.manual_seed(_SEED)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp, context_parallel_size=cp)
        try:
            mamba_model = get_model(self.model_provider, ModelType.encoder_or_decoder)
            mamba_model = unwrap_model(mamba_model)
            assert isinstance(mamba_model[0], MambaModel)
            assert mamba_model[0].mtp is not None
        except AssertionError as e:
            if "Multi-Token Prediction (MTP) is not yet supported" in str(e):
                pytest.fail(f"Attention mask validation failed for Mamba hybrid model: {e}")
            else:
                raise


class TestMTPSequenceParallelDisableRestore:
    """Test that sequence parallelism is correctly disabled during MTP inference
    and restored afterwards.

    These tests verify the _disable_sp_for_mtp / _restore_sp_for_mtp cycle
    and end-to-end compute_mtp_single_step with non-TP-aligned input sizes.
    """

    def setup_method(self, method):
        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'

    def teardown_method(self, method):
        Utils.destroy_model_parallel()
        destroy_global_vars()
        destroy_num_microbatches_calculator()

    def _build_gpt_model(self, tp, hidden_size=64, vocab_size=128, mtp_num_layers=2):
        """Build a small GPTModel with MTP and sequence parallelism."""
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        config = TransformerConfig(
            mtp_num_layers=mtp_num_layers,
            num_layers=2,
            hidden_size=hidden_size,
            num_attention_heads=8,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp,
            sequence_parallel=True if tp > 1 else False,
        )
        transformer_layer_spec = get_gpt_layer_local_spec()
        mtp_block_spec = get_gpt_mtp_block_spec(
            config=config, spec=transformer_layer_spec, use_transformer_engine=False
        )
        model_parallel_cuda_manual_seed(_SEED)
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            mtp_block_spec=mtp_block_spec,
            vocab_size=vocab_size,
            max_sequence_length=256,
            pre_process=True,
            post_process=True,
            share_embeddings_and_output_weights=True,
            position_embedding_type='rope',
        )
        model.cuda()
        model.eval()
        return model, config

    def _collect_sp_flags(self, model, depth=0):
        """Collect all SP-related flags from the model for verification."""
        flags = {}
        flags['embedding_scatter'] = model.embedding.scatter_to_sequence_parallel
        flags['embedding_reduce_scatter'] = model.embedding.reduce_scatter_embeddings
        flags['word_emb_reduce_scatter'] = model.embedding.word_embeddings.reduce_scatter_embeddings
        layer_idx = 0 if model.mtp.mtp_use_repeated_layer else depth
        layer = model.mtp.layers[layer_idx]
        flags['mtp_layer_sp'] = layer.sequence_parallel
        flags['mtp_submodule_sp'] = {}
        for name, module in layer.named_modules():
            if hasattr(module, 'sequence_parallel'):
                flags['mtp_submodule_sp'][name] = module.sequence_parallel
        flags['output_layer_sp'] = model.output_layer.sequence_parallel
        return flags

    @pytest.mark.parametrize('tp', [2, 4])
    def test_disable_sp_sets_all_flags_false(self, tp):
        """_disable_sp_for_mtp must set all SP flags to False."""
        model, config = self._build_gpt_model(tp)
        assert config.sequence_parallel is True

        # Verify SP is initially enabled on the relevant modules.
        initial_flags = self._collect_sp_flags(model)
        assert initial_flags['embedding_scatter'] is True
        assert initial_flags['mtp_layer_sp'] is True
        assert initial_flags['output_layer_sp'] is True
        # At least some submodules (e.g. ColumnParallelLinear) should have SP=True.
        sp_true_submodules = [
            name for name, val in initial_flags['mtp_submodule_sp'].items() if val is True
        ]
        assert len(sp_true_submodules) > 0, "Expected some submodules with SP=True"

        # Disable and check.
        saved = model._disable_sp_for_mtp(depth=0)
        disabled_flags = self._collect_sp_flags(model)
        assert disabled_flags['embedding_scatter'] is False
        assert disabled_flags['embedding_reduce_scatter'] is False
        assert disabled_flags['word_emb_reduce_scatter'] is False
        assert disabled_flags['mtp_layer_sp'] is False
        assert disabled_flags['output_layer_sp'] is False
        for name, val in disabled_flags['mtp_submodule_sp'].items():
            assert val is False, f"Expected SP=False on submodule {name} after disable"

        # Saved state should be non-empty.
        assert len(saved) > 0

    @pytest.mark.parametrize('tp', [2, 4])
    def test_restore_sp_restores_all_flags(self, tp):
        """_restore_sp_for_mtp must restore all SP flags to their original values."""
        model, config = self._build_gpt_model(tp)
        initial_flags = self._collect_sp_flags(model)

        saved = model._disable_sp_for_mtp(depth=0)
        model._restore_sp_for_mtp(depth=0, saved=saved)

        restored_flags = self._collect_sp_flags(model)
        assert restored_flags == initial_flags

    @pytest.mark.parametrize('tp', [2, 4])
    def test_disable_restore_cycle_multiple_depths(self, tp):
        """Disable/restore should work correctly for each MTP depth independently."""
        model, config = self._build_gpt_model(tp, mtp_num_layers=2)

        for depth in range(config.mtp_num_layers):
            initial_flags = self._collect_sp_flags(model, depth=depth)

            saved = model._disable_sp_for_mtp(depth=depth)
            # All SP flags should be False while disabled.
            disabled_flags = self._collect_sp_flags(model, depth=depth)
            assert disabled_flags['mtp_layer_sp'] is False

            model._restore_sp_for_mtp(depth, saved)
            restored_flags = self._collect_sp_flags(model, depth=depth)
            assert restored_flags == initial_flags

    def test_no_op_when_sp_disabled(self):
        """When SP is not enabled (tp=1), _disable_sp_for_mtp returns empty dict."""
        model, config = self._build_gpt_model(tp=1)
        assert config.sequence_parallel is False

        saved = model._disable_sp_for_mtp(depth=0)
        assert saved == {}

        # Restore with empty dict should be a no-op.
        model._restore_sp_for_mtp(depth=0, saved=saved)

    @pytest.mark.parametrize('tp', [2, 4])
    def test_restore_happens_on_exception(self, tp):
        """SP flags must be restored even if the MTP forward raises an exception."""
        model, config = self._build_gpt_model(tp)
        initial_flags = self._collect_sp_flags(model)

        # Monkey-patch to force an exception inside compute_mtp_single_step.
        original_forward = model.mtp.layers[0].forward_single_position

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated failure")

        model.mtp.layers[0].forward_single_position = _raise

        with pytest.raises(RuntimeError, match="simulated failure"):
            dummy_hidden = torch.randn(3, 1, config.hidden_size, device='cuda')
            dummy_ids = torch.zeros(1, 3, dtype=torch.long, device='cuda')
            dummy_pos = torch.arange(3, device='cuda').unsqueeze(0)
            model.compute_mtp_single_step(
                hidden_states=dummy_hidden,
                next_token_ids=dummy_ids,
                position_ids=dummy_pos,
                depth=0,
            )

        # Restore original to allow flag collection.
        model.mtp.layers[0].forward_single_position = original_forward
        restored_flags = self._collect_sp_flags(model)
        assert restored_flags == initial_flags

    @pytest.mark.parametrize(
        ('tp', 'num_positions'),
        [
            (2, 3),  # 3 is not divisible by TP=2
            (2, 5),  # 5 is not divisible by TP=2
            (4, 3),  # 3 is not divisible by TP=4
            (4, 5),  # 5 is not divisible by TP=4
            (4, 7),  # 7 is not divisible by TP=4
        ],
    )
    def test_compute_mtp_single_step_non_aligned_sizes(self, tp, num_positions):
        """compute_mtp_single_step must succeed with sequence lengths not divisible by TP.

        When SP is active, the input tensors are replicated (not scattered) for MTP
        inference.  If SP were not disabled, scatter operations would fail or produce
        wrong shapes for non-TP-aligned sizes.  This test verifies that the disable/
        restore logic allows the computation to complete and produces correct shapes.
        """
        hidden_size = 64
        vocab_size = 128
        model, config = self._build_gpt_model(tp, hidden_size=hidden_size, vocab_size=vocab_size)
        assert config.sequence_parallel is True
        assert num_positions % tp != 0, "Test requires non-TP-aligned input size"

        # Replicated hidden states (as produced by gather_from_sequence_parallel_region).
        hidden_states = torch.randn(num_positions, 1, hidden_size, device='cuda')
        next_token_ids = torch.randint(0, vocab_size, (1, num_positions), device='cuda')
        position_ids = torch.arange(num_positions, device='cuda').unsqueeze(0)

        with torch.inference_mode():
            new_hidden, logits = model.compute_mtp_single_step(
                hidden_states=hidden_states,
                next_token_ids=next_token_ids,
                position_ids=position_ids,
                depth=0,
                runtime_gather_output=True,
            )

        # Shape checks: output should match the non-aligned input size.
        assert new_hidden.shape == (num_positions, 1, hidden_size)
        assert logits.shape[0] == num_positions
        assert logits.shape[1] == 1
        # Vocab dim should be the full (gathered) vocab size.
        assert logits.shape[2] == vocab_size

        # Verify SP was restored after the call.
        restored_flags = self._collect_sp_flags(model)
        assert restored_flags['embedding_scatter'] is True
        assert restored_flags['mtp_layer_sp'] is True
        assert restored_flags['output_layer_sp'] is True

    @pytest.mark.parametrize('tp', [2, 4])
    def test_compute_mtp_single_step_all_depths(self, tp):
        """Run compute_mtp_single_step for every depth, chaining hidden states."""
        hidden_size = 64
        vocab_size = 128
        num_positions = 3  # Non-aligned for both TP=2 and TP=4
        mtp_num_layers = 2
        model, config = self._build_gpt_model(
            tp, hidden_size=hidden_size, vocab_size=vocab_size, mtp_num_layers=mtp_num_layers
        )

        hidden_states = torch.randn(num_positions, 1, hidden_size, device='cuda')
        next_token_ids = torch.randint(0, vocab_size, (1, num_positions), device='cuda')
        position_ids = torch.arange(num_positions, device='cuda').unsqueeze(0)

        with torch.inference_mode():
            for depth in range(mtp_num_layers):
                hidden_states, logits = model.compute_mtp_single_step(
                    hidden_states=hidden_states,
                    next_token_ids=next_token_ids,
                    position_ids=position_ids,
                    depth=depth,
                    runtime_gather_output=True,
                )
                assert hidden_states.shape == (num_positions, 1, hidden_size)
                assert logits.shape == (num_positions, 1, vocab_size)

        # After all depths, SP should be fully restored.
        restored_flags = self._collect_sp_flags(model)
        assert restored_flags['embedding_scatter'] is True
        assert restored_flags['mtp_layer_sp'] is True
        assert restored_flags['output_layer_sp'] is True

    @pytest.mark.parametrize('tp', [2, 4])
    def test_gather_before_mtp_cache(self, tp):
        """Verify that gather_from_sequence_parallel_region produces the correct
        full-sequence tensor from a partitioned input, simulating what GPTModel.forward
        does before caching hidden states for MTP.
        """
        hidden_size = 64
        # Use a sequence length that IS divisible by TP (as the main decoder produces).
        seq_len = 12
        model, config = self._build_gpt_model(tp, hidden_size=hidden_size)
        pg_collection = ProcessGroupCollection.use_mpu_process_groups()

        # Simulate partitioned hidden states: each rank holds seq_len/tp tokens.
        local_seq = seq_len // tp
        partitioned = torch.randn(local_seq, 1, hidden_size, device='cuda')

        gathered = gather_from_sequence_parallel_region(partitioned, group=pg_collection.tp)
        assert gathered.shape == (seq_len, 1, hidden_size)
