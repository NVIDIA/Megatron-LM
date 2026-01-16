# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit tests for chunked_pipeline_parallel_utils.py"""

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.chunked_pipeline_parallel_utils import (
    ChunkedPipelineParallelDataIterator,
    ChunkedPipelineParallelParams,
    ChunkedPipelineParallelQueue,
)
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec as gpt_te_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.num_microbatches_calculator import (
    init_num_microbatches_calculator,
    unset_num_microbatches_calculator,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import ModelType
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import get_attr_wrapped_model
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.global_vars import set_args
from megatron.training.utils import is_first_or_last_pipeline_stage
from tests.unit_tests.dist_checkpointing import TempNamedDir
from tests.unit_tests.test_utilities import Utils

############################################################################
# End-to-end tests for chunked pipeline model parallel with forward/backward
############################################################################


def initialize_gpt_model(seed, **config_kwargs):
    """Initialize GPT model for chunked pipeline parallel testing."""
    torch.manual_seed(seed)
    model_parallel_cuda_manual_seed(seed)

    default_config_kwargs = dict(
        num_layers=8,
        hidden_size=128,
        num_attention_heads=8,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    default_config_kwargs.update(**config_kwargs)
    transformer_config = TransformerConfig(**default_config_kwargs)

    model = []
    for i in range(transformer_config.virtual_pipeline_model_parallel_size or 1):
        layer_spec = gpt_te_spec()

        mtp_block_spec = None

        pre_process = parallel_state.is_pipeline_first_stage(ignore_virtual=False, vp_stage=i)
        post_process = parallel_state.is_pipeline_last_stage(ignore_virtual=False, vp_stage=i)
        this_model = (
            GPTModel(
                config=transformer_config,
                transformer_layer_spec=layer_spec,
                vocab_size=262144,
                max_sequence_length=256,
                pre_process=pre_process,
                post_process=post_process,
                position_embedding_type="rope",
                vp_stage=i,
                mtp_block_spec=mtp_block_spec,
                share_embeddings_and_output_weights=False,
            )
            .bfloat16()
            .cuda()
        )
        this_model.model_type = ModelType.encoder_or_decoder
        model.append(this_model)

    if len(model) == 1:
        model = model[0]
    return model


@pytest.fixture
def create_args():
    """Setup dummy args."""
    args = SimpleNamespace()
    args.finetune = False
    args.non_persistent_global_ckpt_dir = None
    args.non_persistent_ckpt_type = None
    args.non_persistent_save_interval = None
    args.exit_on_missing_checkpoint = True
    args.async_save = False
    args.data_parallel_random_init = False
    args.log_progress = False
    args.ckpt_fully_parallel_save = False
    args.ckpt_fully_parallel_load = False
    args.auto_detect_ckpt_format = False
    args.retro_add_retriever = False
    args.ckpt_convert_update_legacy_dist_opt_format = False
    args.ckpt_step = None
    args.use_dist_ckpt = True
    args.consumed_train_samples = 0
    args.skipped_train_samples = 0
    args.consumed_valid_samples = 0
    args.vocab_file = None
    args.add_position_embedding = False
    args.ckpt_assume_constant_structure = True
    args.dist_ckpt_strictness = "assume_ok_unexpected"
    args.fp16 = False
    args.bf16 = True
    args.no_save_optim = True
    args.no_save_rng = True
    args.no_load_optim = True
    args.no_load_rng = True
    args.use_distributed_optimizer = True
    args.use_megatron_fsdp = False
    args.dist_ckpt_save_pre_mcore_014 = False
    args.dist_ckpt_optim_fully_reshardable = False
    args.distrib_optim_fully_reshardable_mem_efficient = False

    yield args


@pytest.mark.parametrize(
    ('tp_pp_vpp', 'chunked_pp_splits'),
    [((1, 2, 1), 4), ((1, 2, 2), 4), ((2, 4, 1), 2), ((2, 4, 2), 2)],
)
def test_forward_chunked_pipeline_parallel(
    create_args, tmp_path_dist_ckpt, tp_pp_vpp, chunked_pp_splits
):
    """Test forward pass with chunked pipeline model parallel (splits > 1).

    This test verifies that the forward pass produces correct results when
    chunked_pipeline_model_parallel_splits > 1, by comparing against a
    baseline run with PP=1.
    """
    from megatron.core.pipeline_parallel import get_forward_backward_func

    args = create_args
    # Model config
    args.num_layers = 8
    args.hidden_size = 128
    args.num_attention_heads = 8
    # Ckpt format
    args.ckpt_format = "torch_dist"
    args.multi_latent_attention = False
    set_args(args)

    def set_tp_pp_vpp(tp, pp, vpp, chunked_pp_splits=None, destroy_first=True):
        if destroy_first:
            Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tp, pp, vpp)
        args.tensor_model_parallel_size = tp
        args.pipeline_model_parallel_size = pp
        args.virtual_pipeline_model_parallel_size = vpp
        args.chunked_pipeline_model_parallel_splits = chunked_pp_splits

    set_tp_pp_vpp(*tp_pp_vpp, chunked_pp_splits=chunked_pp_splits, destroy_first=False)
    init_num_microbatches_calculator(0, None, 1, 1, 1)

    seq_length = 256
    num_microbatches = 4
    micro_batch_size = 1

    def forward_step_func(data_iterator, model: GPTModel):
        """Forward training step."""
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        if not is_first_or_last_pipeline_stage(vp_stage):
            if model.config.chunked_pipeline_model_parallel_splits > 1:
                data_iterator.mock_next(seq_length)
            tokens, labels, position_ids, attention_mask, loss_mask = None, None, None, None, None
        else:
            data = next(data_iterator)
            tokens = data["tokens"]
            labels = data["labels"]
            position_ids = data["position_ids"]
            attention_mask = data.get("attention_mask", None)
            loss_mask = data["loss_mask"]

        if model.config.chunked_pipeline_model_parallel_splits > 1:
            chunked_pp_params = data_iterator.get_current_chunked_pp_params()
        else:
            chunked_pp_params = None

        output_tensor = model(
            tokens,
            position_ids,
            attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            packed_seq_params=None,
            chunked_pp_params=chunked_pp_params,
        )

        def loss_func(output_tensor: torch.Tensor):
            loss = output_tensor.mean()
            return output_tensor, loss

        return output_tensor, loss_func

    iteration = 123
    model = initialize_gpt_model(
        seed=1,
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        chunked_pipeline_model_parallel_splits=chunked_pp_splits,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
    )
    model = model if isinstance(model, list) else [model]

    forward_backward_func = get_forward_backward_func()
    data_iterator = []
    for vp_stage in range(args.virtual_pipeline_model_parallel_size):
        if not is_first_or_last_pipeline_stage(vp_stage):
            curr_data_iterator = None
        else:
            curr_data_iterator = get_batch_iterator(
                seq_length=seq_length, micro_batch_size=micro_batch_size
            )
        data_iterator.append(curr_data_iterator)
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=data_iterator,
        model=model,
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=micro_batch_size,
        forward_only=True,
    )

    optimizer = None
    opt_param_scheduler = None
    num_floating_point_operations_so_far = 456

    with TempNamedDir(tmp_path_dist_ckpt / 'test_chunked_pp_forward') as ckpt_dir:
        args.save = ckpt_dir
        args.load = ckpt_dir
        save_checkpoint(
            iteration, model, optimizer, opt_param_scheduler, num_floating_point_operations_so_far
        )

        # Load with PP=1 as baseline
        set_tp_pp_vpp(1, 1, None, chunked_pp_splits=1)
        model_baseline = initialize_gpt_model(
            seed=1,
            num_layers=args.num_layers,
            hidden_size=args.hidden_size,
            num_attention_heads=args.num_attention_heads,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            chunked_pipeline_model_parallel_splits=1,  # Baseline uses splits=1
        )
        load_checkpoint([model_baseline], optimizer, opt_param_scheduler, strict=False)

        forward_backward_func = get_forward_backward_func()
        if not is_first_or_last_pipeline_stage(vp_stage=None):
            data_iterator = None
        else:
            data_iterator = get_batch_iterator(
                seq_length=seq_length, micro_batch_size=micro_batch_size
            )
        losses_reduced_baseline = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=data_iterator,
            model=[model_baseline],
            num_microbatches=num_microbatches,
            seq_length=seq_length,
            micro_batch_size=micro_batch_size,
            forward_only=True,
        )

        for loss, loss_baseline in zip(losses_reduced, losses_reduced_baseline):
            torch.testing.assert_close(loss, loss_baseline, atol=1e-2, rtol=1e-2)

    Utils.destroy_model_parallel()
    unset_num_microbatches_calculator()


def get_batch_iterator(seq_length, micro_batch_size, num_batches=None):
    """
    Generator function that yields batches indefinitely or for a specified number of batches.

    Args:
        seq_length: Length of the sequence
        micro_batch_size: Size of each micro batch
        num_batches: Optional number of batches to generate. If None, generates indefinitely.
    """
    batch_count = 0
    while num_batches is None or batch_count < num_batches:
        # Generate different data for each batch by adding batch_count offset
        data = list(range(batch_count, batch_count + seq_length))
        tokens = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        labels = 1 + torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = (
            torch.tensor(list(range(seq_length)), dtype=torch.int64)
            .repeat((micro_batch_size, 1))
            .cuda()
        )
        attention_mask = None  # Chunked PP does not support attention mask
        loss_mask = torch.ones(seq_length).repeat((micro_batch_size, 1)).cuda()

        yield {
            "tokens": tokens,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }
        batch_count += 1


############
# Test chunked pipeline parallel utilities
############


def create_mock_config(chunked_pp_splits: int):
    """Create a mock TransformerConfig."""
    config = MagicMock()
    config.chunked_pipeline_model_parallel_splits = chunked_pp_splits
    return config


def create_sample_data(batch_size: int, seq_length: int):
    """Create sample data dict with sequential tokens for easy verification."""
    tokens = torch.arange(seq_length).unsqueeze(0).expand(batch_size, -1).clone()
    return {
        "tokens": tokens,
        "labels": tokens.clone(),
        "loss_mask": torch.ones(batch_size, seq_length),
        "position_ids": tokens.clone(),
        "attention_mask": None,
    }


class TestChunkedPipelineParallelDataIterator:
    """Tests for ChunkedPipelineParallelDataIterator."""

    def test_get_span(self):
        """Test span calculation for various configurations."""
        chunked_pp_splits = 4
        seq_length = 256
        expected_spans = [64, 64, 64, 64]
        config = create_mock_config(chunked_pp_splits)
        iterator = ChunkedPipelineParallelDataIterator(None, config)
        assert iterator._get_span(seq_length) == expected_spans

    def test_mock_next(self):
        """Test mock_next correctly cycles through spans and micro batches."""
        num_micro_batches = 2
        chunked_pp_splits = 4
        seq_length = 256
        expected_spans = [64, 64, 64, 64]
        config = create_mock_config(chunked_pp_splits)
        iterator = ChunkedPipelineParallelDataIterator(None, config)

        for mb_idx in range(num_micro_batches):
            for span_idx in range(chunked_pp_splits):
                iterator.mock_next(seq_length)
                assert iterator.count == mb_idx + 1
                assert iterator.current_span_idx == span_idx

                params = iterator.get_current_chunked_pp_params()
                assert isinstance(params, ChunkedPipelineParallelParams)
                assert params.micro_batch_idx == mb_idx
                assert params.span_idx_in_micro == span_idx
                assert params.spans == expected_spans

    def test_next_slices_data_correctness(self):
        """Test __next__ produces correctly sliced data."""
        num_micro_batches = 2
        chunked_pp_splits = 4
        seq_length = 256
        expected_spans = [64, 64, 64, 64]
        micro_batch_size = 3
        config = create_mock_config(chunked_pp_splits)

        sample_data = [
            create_sample_data(micro_batch_size, seq_length) for _ in range(num_micro_batches)
        ]
        mock_iter = MagicMock()
        mock_iter.__next__ = Mock(side_effect=sample_data)

        iterator = ChunkedPipelineParallelDataIterator(mock_iter, config)

        # Verify each slice has correct content
        for mb_idx in range(num_micro_batches):
            for span_idx in range(chunked_pp_splits):
                slice_data = next(iterator)
                expected_start = span_idx * (seq_length // chunked_pp_splits)
                expected_end = expected_start + (seq_length // chunked_pp_splits)
                expected_tokens = (
                    torch.arange(expected_start, expected_end)
                    .unsqueeze(0)
                    .expand(micro_batch_size, -1)
                )
                assert torch.equal(slice_data["tokens"], expected_tokens)

                params = iterator.get_current_chunked_pp_params()
                assert isinstance(params, ChunkedPipelineParallelParams)
                assert params.micro_batch_idx == mb_idx
                assert params.span_idx_in_micro == span_idx
                assert params.spans == expected_spans

    def test_get_current_chunked_pp_params(self):
        """Test params generation matches iterator state."""

        config = create_mock_config(4)
        iterator = ChunkedPipelineParallelDataIterator(None, config)

        # Iterate and verify params at each step
        for mb in range(2):
            for span in range(4):
                iterator.mock_next(256)
                params = iterator.get_current_chunked_pp_params()

                assert isinstance(params, ChunkedPipelineParallelParams)
                assert params.micro_batch_idx == mb
                assert params.span_idx_in_micro == span
                assert params.spans == [64, 64, 64, 64]


class TestChunkedPipelineParallelQueue:
    """Tests for ChunkedPipelineParallelQueue."""

    def test_append_and_pop(self):
        """Test append correctly manages inner queues and length."""
        chunked_pp_splits = 3
        num_items = 7
        queue = ChunkedPipelineParallelQueue(chunked_pp_splits)

        # Push and verify length and tail
        for i in range(num_items):
            queue.append(i)
            assert len(queue) == i + 1
            assert queue[-1] == i

        # Verify internal state
        expected_outer_idx = num_items // chunked_pp_splits
        expected_inner_cnt = num_items % chunked_pp_splits
        assert queue._outer_idx == expected_outer_idx
        assert queue._inner_cnt == expected_inner_cnt

        # Pop and verify order: LIFO within inner queues, FIFO across queues
        popped = []
        while len(queue) > 0:
            popped.append(queue.pop())

        # Reconstruct expected order
        expected = []
        for outer_start in range(0, num_items, chunked_pp_splits):
            inner_items = list(range(outer_start, min(outer_start + chunked_pp_splits, num_items)))
            expected.extend(reversed(inner_items))

        assert popped == expected

    def test_getitem_and_pop_assertions(self):
        """Test that __getitem__ and pop only support specific indices."""
        queue = ChunkedPipelineParallelQueue(4)
        queue.append(torch.tensor(1))

        with pytest.raises(AssertionError):
            _ = queue[0]  # Only -1 supported

        with pytest.raises(AssertionError):
            queue.pop(idx=1)  # Only 0 supported
