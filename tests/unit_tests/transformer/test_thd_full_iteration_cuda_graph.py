# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Unit tests for the THD full-iteration CUDA graph static-input contract.

Covers the fixed-``num_microbatches`` contract from the THD full-iteration
plan: batch canonicalization outside the captured region, the captured
``get_batch`` fast path, capture-signature enforcement, static PP
communication shapes, and eager-vs-graph loss equality on replay.
"""

import functools
import itertools
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from megatron.core import ModelParallelConfig, parallel_state
from megatron.core.datasets.data_schedule import (
    THD_FULL_ITERATION_STATIC_BATCH_KEY,
    get_batch_on_this_rank_for_sequence_packing,
    prepare_thd_static_batch_for_full_iteration_cuda_graph,
)
from megatron.core.full_cuda_graph import FullCudaGraphWrapper, StaticBufferLoader
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.schedules import get_tensor_shapes
from megatron.core.tensor_parallel.random import HAVE_TE, initialize_rng_tracker
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

TOKEN_CAPACITY = 64
MAX_PACKED_SEQS = 8
VOCAB_SIZE = 1024
HIDDEN_SIZE = 32


def _make_thd_full_iter_config(**overrides):
    """Minimal config namespace satisfying the THD full-iteration batch path."""
    config = SimpleNamespace(
        cuda_graph_impl='full_iteration',
        sequence_packing_scheduler='dp_balanced',
        pad_packed_seq_alignment='max',
        max_seqlen_per_dp_cp_rank=TOKEN_CAPACITY,
        thd_max_packed_sequences=MAX_PACKED_SEQS,
        pad_packed_seq_by_appending_dummy_seq=True,
        cp_partition_mode='zigzag',
        dynamic_context_parallel=False,
        pipeline_model_parallel_layout=None,
        mtp_num_layers=None,
        virtual_pipeline_model_parallel_size=None,
        # Derived field; TransformerConfig.__post_init__ sets it for
        # full_iteration + sequence packing.
        thd_static_pp_communication=True,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _make_raw_packed_batch(sequence_lengths, seed, device='cuda', with_tokens=True):
    """Build one raw scheduler-style packed batch with the given seqlens."""
    generator = torch.Generator(device='cpu').manual_seed(seed)
    total = sum(sequence_lengths)
    tokens = torch.randint(0, VOCAB_SIZE - 2, (total,), generator=generator, dtype=torch.int64).to(
        device
    )
    position_ids = torch.cat(
        [torch.arange(length, dtype=torch.int64) for length in sequence_lengths]
    ).to(device)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(sequence_lengths)), dtype=torch.int32, device=device
    )
    batch = {
        'tokens': tokens if with_tokens else None,
        'position_ids': position_ids if with_tokens else None,
        'labels': tokens + 1 if with_tokens else None,
        'loss_mask': torch.ones(total, dtype=torch.float32, device=device) if with_tokens else None,
        'cu_seqlens': cu_seqlens,
        'cu_seqlens_padded': cu_seqlens.clone(),
        'max_seqlen': torch.tensor([max(sequence_lengths)], dtype=torch.int32, device=device),
    }
    return batch


def _reset_full_cuda_graph_state():
    FullCudaGraphWrapper.reset_cuda_graph()
    StaticBufferLoader.reset()


@pytest.fixture(autouse=True)
def _clean_wrapper_state():
    """Class-level graph/buffer state must not leak across tests."""
    _reset_full_cuda_graph_state()
    yield
    _reset_full_cuda_graph_state()


class TestThdStaticBatchPreparation:
    """Canonicalization outside the graph produces one fixed static spec."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize(
        'sequence_lengths', [[10, 20, 5], [TOKEN_CAPACITY], [3, 3, 3, 3], [1, TOKEN_CAPACITY - 1]]
    )
    def test_different_seq_distributions_same_static_spec(self, sequence_lengths):
        config = _make_thd_full_iter_config()
        batch = prepare_thd_static_batch_for_full_iteration_cuda_graph(
            iter([_make_raw_packed_batch(sequence_lengths, seed=17)]), config=config
        )

        assert batch[THD_FULL_ITERATION_STATIC_BATCH_KEY]
        for key in ('tokens', 'labels', 'loss_mask', 'position_ids', 'padding_mask'):
            assert batch[key].shape == (1, TOKEN_CAPACITY), key
        assert batch['cu_seqlens'].numel() == MAX_PACKED_SEQS + 1
        assert batch['cu_seqlens_padded'].numel() == MAX_PACKED_SEQS + 1
        # Unused tail entries must repeat the final cumulative length.
        num_real = len(sequence_lengths)
        total = sum(sequence_lengths)
        cu = batch['cu_seqlens'].cpu()
        assert cu[num_real].item() == total
        assert (cu[num_real + 1 :] == TOKEN_CAPACITY).all()
        # max_seqlen must be the static config upper bound, not the batch max.
        assert batch['max_seqlen'] == TOKEN_CAPACITY

        # Real tokens keep their positions; the tail is masked as padding.
        padding_mask = batch['padding_mask'][0].cpu()
        assert not padding_mask[:total].any()
        assert padding_mask[total:].all()

    def test_fast_path_reads_static_tensors_without_rebuilding(self):
        config = _make_thd_full_iter_config()
        batch = prepare_thd_static_batch_for_full_iteration_cuda_graph(
            iter([_make_raw_packed_batch([7, 9], seed=3)]), config=config
        )

        tokens, labels, loss_mask, attention_mask, position_ids, psp, padding_mask = (
            get_batch_on_this_rank_for_sequence_packing(iter([batch]), config=config)
        )

        # The captured path must return the static buffer tensors themselves.
        assert tokens is batch['tokens']
        assert labels is batch['labels']
        assert loss_mask is batch['loss_mask']
        assert position_ids is batch['position_ids']
        assert padding_mask is batch['padding_mask']
        assert attention_mask is None
        assert psp.qkv_format == 'thd'
        assert psp.cu_seqlens_q is batch['cu_seqlens']
        assert psp.cu_seqlens_kv is batch['cu_seqlens']
        assert psp.cu_seqlens_q_padded is batch['cu_seqlens_padded']
        assert psp.cu_seqlens_kv_padded is batch['cu_seqlens_padded']
        assert psp.max_seqlen_q == TOKEN_CAPACITY
        assert psp.max_seqlen_kv == TOKEN_CAPACITY
        assert psp.pad_between_seqs is False

    def test_oversized_pack_is_rejected(self):
        config = _make_thd_full_iter_config()
        too_many_seqs = [4] * (MAX_PACKED_SEQS + 1)
        with pytest.raises(AssertionError):
            prepare_thd_static_batch_for_full_iteration_cuda_graph(
                iter([_make_raw_packed_batch(too_many_seqs, seed=5)]), config=config
            )

    def test_prepare_is_repeatable_on_same_raw_batch(self):
        """The PagedStashRunner overflow fallback replays the same raw batches,
        so canonicalizing one raw batch twice must work and be deterministic."""
        config = _make_thd_full_iter_config()
        raw = _make_raw_packed_batch([7, 9], seed=3)
        first = prepare_thd_static_batch_for_full_iteration_cuda_graph(iter([raw]), config=config)
        second = prepare_thd_static_batch_for_full_iteration_cuda_graph(iter([raw]), config=config)
        for key in (
            'tokens',
            'labels',
            'loss_mask',
            'position_ids',
            'padding_mask',
            'cu_seqlens',
            'cu_seqlens_padded',
        ):
            assert torch.equal(first[key], second[key]), key
        assert first['max_seqlen'] == second['max_seqlen']


class TestThdStaticPPCommunicationShapes:
    """Full-iteration THD mode replaces the P2P shape handshake with static shapes."""

    def setup_method(self, method):
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_get_tensor_shapes_uses_static_thd_width(self):
        config = _make_thd_full_iter_config(
            variable_seq_lengths=True,
            sequence_parallel=False,
            hidden_size=HIDDEN_SIZE,
            enable_hyper_connections=False,
        )
        shapes = get_tensor_shapes(
            seq_length=8192,  # nominal value; must be ignored in THD static mode
            micro_batch_size=4,
            decoder_seq_length=None,
            config=config,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            cp_group=parallel_state.get_context_parallel_group(),
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
        )
        assert shapes == [(TOKEN_CAPACITY, 1, HIDDEN_SIZE)]

        # Without full-iteration graphs, variable-seq-length negotiation remains.
        config.thd_static_pp_communication = False
        shapes = get_tensor_shapes(
            seq_length=8192,
            micro_batch_size=4,
            decoder_seq_length=None,
            config=config,
            tp_group=parallel_state.get_tensor_model_parallel_group(),
            cp_group=parallel_state.get_context_parallel_group(),
            pp_group=parallel_state.get_pipeline_model_parallel_group(),
        )
        assert shapes == [()]


class _ThdToyModel(torch.nn.Module):
    """Deterministic THD consumer: one-hot projection + head (no atomics)."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(VOCAB_SIZE, HIDDEN_SIZE, bias=False)
        self.head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.model_type = 'unit-test'

    def set_input_tensor(self, input_tensor):
        pass

    def forward(self, tokens):
        one_hot = F.one_hot(tokens, VOCAB_SIZE).to(self.proj.weight.dtype)
        return self.head(self.proj(one_hot))


def _make_masked_loss_func(labels, loss_mask, padding_mask):
    """Padding-aware cross-entropy loss over the static [1, T] batch tensors."""

    def loss_func(output_tensor):
        losses = F.cross_entropy(
            output_tensor.view(-1, VOCAB_SIZE), labels.view(-1), reduction='none'
        )
        mask = loss_mask.view(-1) * (~padding_mask.view(-1)).to(loss_mask.dtype)
        loss = (losses * mask).sum() / mask.sum().clamp(min=1.0)
        return loss, {'loss': loss.detach().clone()}

    return loss_func


def _forward_backward_kwargs(forward_step, data_iterator, model, num_microbatches, seq_length):
    return dict(
        forward_step_func=forward_step,
        data_iterator=data_iterator,
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=seq_length,
        micro_batch_size=1,
        forward_only=False,
    )


def _make_forward_step(config, call_counter):
    def forward_step(data_iterator, model):
        call_counter['calls'] += 1
        tokens, labels, loss_mask, _, _, packed_seq_params, padding_mask = (
            get_batch_on_this_rank_for_sequence_packing(data_iterator, config=config)
        )
        assert packed_seq_params.max_seqlen_q == TOKEN_CAPACITY
        logits = model(tokens)
        return logits, _make_masked_loss_func(labels, loss_mask, padding_mask)

    return forward_step


def _step_sequence_lengths(step, microbatch):
    """A different real packing layout per (step, microbatch)."""
    layouts = [
        [10, 20, 5],
        [TOKEN_CAPACITY],
        [3, 3, 3, 3],
        [1, 40],
        [30, 30],
        [8, 8, 8, 8, 8],
        [50],
        [2, 60, 2],
    ]
    return layouts[(2 * step + microbatch) % len(layouts)]


def _raw_batches_for_step(step, num_microbatches, seed_base=1000):
    return [
        _make_raw_packed_batch(_step_sequence_lengths(step, mb), seed=seed_base + 10 * step + mb)
        for mb in range(num_microbatches)
    ]


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="full-iteration capture uses the TE RNG tracker (requires TE >= 1.5)",
)
class TestThdFullIterationWrapper:
    """End-to-end capture/replay with a fixed num_microbatches."""

    NUM_MICROBATCHES = 2
    NUM_STEPS = 4

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel()

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_models(self):
        torch.manual_seed(1234 + Utils.rank)
        model_graph = _ThdToyModel().cuda()
        model_eager = _ThdToyModel().cuda()
        model_eager.load_state_dict(model_graph.state_dict())
        mp_config = ModelParallelConfig(pipeline_model_parallel_size=1)
        model_graph.config = mp_config
        model_eager.config = mp_config
        return model_graph, model_eager

    def _run_kwargs(self, forward_step, data_iterator, model, num_microbatches):
        return _forward_backward_kwargs(
            forward_step, data_iterator, model, num_microbatches, seq_length=None
        )

    def test_capture_replay_matches_eager(self):
        thd_config = _make_thd_full_iter_config()
        model_graph, model_eager = self._make_models()

        graph_counter = {'calls': 0}
        eager_counter = {'calls': 0}
        forward_step_graph = _make_forward_step(thd_config, graph_counter)
        forward_step_eager = _make_forward_step(thd_config, eager_counter)

        prepare_fn = functools.partial(
            prepare_thd_static_batch_for_full_iteration_cuda_graph, config=thd_config
        )
        wrapped = FullCudaGraphWrapper(
            get_forward_backward_func(), cuda_graph_warmup_steps=1, batch_preparation_fn=prepare_fn
        )
        eager_fbf = get_forward_backward_func()

        graph_losses, eager_losses = [], []
        for step in range(self.NUM_STEPS):
            raw = _raw_batches_for_step(step, self.NUM_MICROBATCHES)
            losses = wrapped(
                **self._run_kwargs(
                    forward_step_graph, [iter(raw)], model_graph, self.NUM_MICROBATCHES
                )
            )
            graph_losses.append([d['loss'].item() for d in losses])

            raw = _raw_batches_for_step(step, self.NUM_MICROBATCHES)
            static_batches = [prepare_fn(iter([b])) for b in raw]
            losses = eager_fbf(
                **self._run_kwargs(
                    forward_step_eager, [iter(static_batches)], model_eager, self.NUM_MICROBATCHES
                )
            )
            eager_losses.append([d['loss'].item() for d in losses])

        # Graph must be captured after the warmup step and replayed afterwards:
        # Python-side forward_step stops running once replay begins.
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None
        assert graph_counter['calls'] == 2 * self.NUM_MICROBATCHES
        assert eager_counter['calls'] == self.NUM_STEPS * self.NUM_MICROBATCHES

        # Per-step, per-microbatch losses must match eager execution bitwise,
        # including replay steps that saw packing layouts never captured.
        assert graph_losses == eager_losses

        # Gradients accumulate identically inside and outside the graph.
        for p_graph, p_eager in zip(model_graph.parameters(), model_eager.parameters()):
            assert torch.equal(p_graph.grad, p_eager.grad)

    def test_num_microbatches_change_after_capture_raises(self):
        thd_config = _make_thd_full_iter_config()
        model_graph, _ = self._make_models()

        prepare_fn = functools.partial(
            prepare_thd_static_batch_for_full_iteration_cuda_graph, config=thd_config
        )
        wrapped = FullCudaGraphWrapper(
            get_forward_backward_func(), cuda_graph_warmup_steps=1, batch_preparation_fn=prepare_fn
        )
        forward_step = _make_forward_step(thd_config, {'calls': 0})

        for step in range(2):
            raw = _raw_batches_for_step(step, self.NUM_MICROBATCHES)
            wrapped(
                **self._run_kwargs(forward_step, [iter(raw)], model_graph, self.NUM_MICROBATCHES)
            )
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None

        raw = _raw_batches_for_step(2, self.NUM_MICROBATCHES + 1)
        with pytest.raises(RuntimeError, match='signature mismatch'):
            wrapped(
                **self._run_kwargs(
                    forward_step, [iter(raw)], model_graph, self.NUM_MICROBATCHES + 1
                )
            )


class _ThdToyFirstStage(torch.nn.Module):
    """First PP stage: embeds tokens into the static [T, 1, H] PP shape."""

    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(VOCAB_SIZE, HIDDEN_SIZE, bias=False)
        self.model_type = 'unit-test'

    def set_input_tensor(self, input_tensor):
        pass

    def forward(self, tokens):
        one_hot = F.one_hot(tokens, VOCAB_SIZE).to(self.proj.weight.dtype)
        return self.proj(one_hot).transpose(0, 1).contiguous()


class _ThdToyLastStage(torch.nn.Module):
    """Last PP stage: consumes the received [T, 1, H] activation."""

    def __init__(self):
        super().__init__()
        self.head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        self.model_type = 'unit-test'
        self.input_tensor = None

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, list):
            input_tensor = input_tensor[0]
        self.input_tensor = input_tensor

    def forward(self):
        return self.head(self.input_tensor.transpose(0, 1))


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="full-iteration capture uses the TE RNG tracker (requires TE >= 1.5)",
)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="PP2 requires at least 2 GPUs")
class TestThdFullIterationWrapperPP2:
    """PP2 1F1B with graph-captured NCCL P2P and static THD shapes."""

    NUM_MICROBATCHES = 4
    NUM_STEPS = 4

    def setup_method(self, method):
        initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
        Utils.initialize_model_parallel(1, 2)

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def _make_config(self):
        config = ModelParallelConfig(
            pipeline_model_parallel_size=2, pipeline_dtype=torch.float32, variable_seq_lengths=True
        )
        for key, value in vars(_make_thd_full_iter_config()).items():
            setattr(config, key, value)
        config.hidden_size = HIDDEN_SIZE
        return config

    def _make_stage_models(self, config):
        torch.manual_seed(4321)  # identical init across the DP group per stage
        if parallel_state.is_pipeline_first_stage():
            model_graph, model_eager = _ThdToyFirstStage().cuda(), _ThdToyFirstStage().cuda()
        else:
            model_graph, model_eager = _ThdToyLastStage().cuda(), _ThdToyLastStage().cuda()
        model_eager.load_state_dict(model_graph.state_dict())
        model_graph.config = config
        model_eager.config = config
        return model_graph, model_eager

    def _raw_batches(self, step):
        # The two PP ranks of one pipeline must see identical raw batches.
        dp_rank = parallel_state.get_data_parallel_rank()
        return [
            _make_raw_packed_batch(
                _step_sequence_lengths(step, mb), seed=5000 + 100 * step + 10 * mb + dp_rank
            )
            for mb in range(self.NUM_MICROBATCHES)
        ]

    def _make_forward_step(self, config):
        def forward_step(data_iterator, model):
            tokens, labels, loss_mask, _, _, _, padding_mask = (
                get_batch_on_this_rank_for_sequence_packing(data_iterator, config=config)
            )
            if parallel_state.is_pipeline_first_stage():
                output = model(tokens)
            else:
                output = model()
            return output, _make_masked_loss_func(labels, loss_mask, padding_mask)

        return forward_step

    def test_pp2_capture_replay_matches_eager(self):
        config = self._make_config()
        model_graph, model_eager = self._make_stage_models(config)
        forward_step = self._make_forward_step(config)

        prepare_fn = functools.partial(
            prepare_thd_static_batch_for_full_iteration_cuda_graph, config=config
        )
        wrapped = FullCudaGraphWrapper(
            get_forward_backward_func(), cuda_graph_warmup_steps=1, batch_preparation_fn=prepare_fn
        )
        eager_fbf = get_forward_backward_func()

        def run_kwargs(data_iterator, model):
            return _forward_backward_kwargs(
                forward_step, data_iterator, model, self.NUM_MICROBATCHES, seq_length=TOKEN_CAPACITY
            )

        graph_losses, eager_losses = [], []
        for step in range(self.NUM_STEPS):
            losses = wrapped(**run_kwargs([iter(self._raw_batches(step))], model_graph))
            graph_losses.append([d['loss'].item() for d in losses])

            static_batches = [prepare_fn(iter([b])) for b in self._raw_batches(step)]
            losses = eager_fbf(**run_kwargs([iter(static_batches)], model_eager))
            eager_losses.append([d['loss'].item() for d in losses])

        # The PP2 1F1B schedule, including its NCCL P2P, must have been captured.
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None
        # Losses exist on the last stage only and must match eager bitwise.
        if parallel_state.is_pipeline_last_stage():
            assert all(len(step_losses) == self.NUM_MICROBATCHES for step_losses in graph_losses)
            assert graph_losses == eager_losses
        for p_graph, p_eager in zip(model_graph.parameters(), model_eager.parameters()):
            assert torch.equal(p_graph.grad, p_eager.grad)


# =============================================================================
# E2E eager vs full-iteration bitwise loss/grad_norm match.
#    Subprocess-launches `torchrun pretrain_gpt.py` twice (same recipe style as
#    test_thd_cuda_graph.py::TestE2EBitwise) and asserts per-iteration metrics
#    are byte-identical. Data uses fixed-length sequences so the packing
#    scheduler emits a constant num_microbatches per step, as required by the
#    full-iteration graph contract.
# =============================================================================

_E2E_TRAIN_ITERS = 8

# Every sequence is exactly max_seqlen_per_dp_cp_rank tokens: one sequence per
# pack, GBS 16 / dp 4 = 4 microbatches per rank per step, constant.
_FIXED_LEN_VARLEN_JSON = (
    '{"mode":"distribution","type":"lognormal",'
    '"format":"thd","min_seq_len":4096,"max_seq_len":4096,'
    '"mean_seq_len":4096,"lognormal_sigma":0.1}'
)

_E2E_COMMON_ARGS = [
    "--seq-length",
    "4096",
    "--max-position-embeddings",
    "8192",
    "--micro-batch-size",
    "1",
    "--global-batch-size",
    "16",
    "--train-iters",
    str(_E2E_TRAIN_ITERS),
    "--lr",
    "1e-5",
    "--min-lr",
    "1e-6",
    "--lr-decay-style",
    "cosine",
    "--lr-warmup-iters",
    "1",
    "--weight-decay",
    "0.01",
    "--clip-grad",
    "1.0",
    "--seed",
    "1234",
    "--te-rng-tracker",
    "--bf16",
    "--tensor-model-parallel-size",
    "1",
    "--pipeline-model-parallel-size",
    "2",
    "--swiglu",
    "--disable-bias-linear",
    "--use-varlen-dataset",
    "--mock-data",
    "--tokenizer-type",
    "NullTokenizer",
    "--varlen-mock-dataset-config-json",
    _FIXED_LEN_VARLEN_JSON,
    "--sequence-packing-scheduler",
    "dp_balanced",
    "--max-seqlen-per-dp-cp-rank",
    "4096",
    "--pad-packed-seq-alignment",
    "max",
    "--thd-max-packed-sequences",
    "8",
    "--calculate-per-token-loss",
    "--transformer-impl",
    "transformer_engine",
    "--attention-dropout",
    "0",
    "--hidden-dropout",
    "0",
    "--no-bias-swiglu-fusion",
    "--no-gradient-accumulation-fusion",
    "--no-save-optim",
    "--no-save-rng",
    "--save-interval",
    "999999",
    "--eval-interval",
    "999999",
    "--eval-iters",
    "1",
    "--log-interval",
    "1",
    "--no-check-for-nan-in-loss-and-grad",
    "--deterministic-mode",
]

_DENSE_E2E_ARGS = _E2E_COMMON_ARGS + [
    "--num-layers",
    "8",
    "--hidden-size",
    "512",
    "--ffn-hidden-size",
    "1376",
    "--num-attention-heads",
    "8",
    "--vocab-size",
    "32000",
]

# Moonlight-style MLA + MoE, downsized, with STATIC expert shapes: alltoall
# dispatcher + capacity factor + pad-to-capacity (the v1 full-iteration MoE
# contract). Router fusion exercises TE's fused_moe_aux_loss with a tensor
# total_num_tokens inside the captured region. Native dropless HybridEP has
# dynamic expert shapes and stays out of full-iteration scope.
_MOONLIGHT_STATIC_E2E_ARGS = _E2E_COMMON_ARGS + [
    "--num-layers",
    "13",
    "--hidden-size",
    "1024",
    "--ffn-hidden-size",
    "5632",
    "--num-attention-heads",
    "16",
    "--decoder-first-pipeline-num-layers",
    "6",
    "--decoder-last-pipeline-num-layers",
    "7",
    "--expert-model-parallel-size",
    "4",
    "--expert-tensor-parallel-size",
    "1",
    "--multi-latent-attention",
    "--kv-lora-rank",
    "512",
    "--qk-head-dim",
    "128",
    "--qk-pos-emb-head-dim",
    "64",
    "--v-head-dim",
    "128",
    "--num-experts",
    "64",
    "--moe-ffn-hidden-size",
    "704",
    "--moe-router-topk",
    "6",
    "--moe-shared-expert-intermediate-size",
    "1408",
    "--moe-layer-freq",
    "([0]+[1]*12)",
    "--moe-token-dispatcher-type",
    "alltoall",
    "--moe-expert-capacity-factor",
    "1.5",
    "--moe-pad-expert-input-to-capacity",
    "--moe-router-fusion",
    "--moe-router-score-function",
    "sigmoid",
    "--moe-router-topk-scaling-factor",
    "2.446",
    "--moe-router-load-balancing-type",
    "aux_loss",
    "--moe-aux-loss-coeff",
    "0.001",
    "--normalization",
    "RMSNorm",
    "--norm-epsilon",
    "1e-5",
    "--rotary-base",
    "50000",
    "--vocab-size",
    "32000",
]

_FULL_ITER_ARGS = ["--cuda-graph-impl", "full_iteration", "--cuda-graph-warmup-steps", "3"]

# Subprocess timeout: two full pretrain runs of ~2 min each plus slack.
_E2E_RUN_TIMEOUT = 1500


def _wait_for_idle_gpus(timeout_s=90):
    """Wait until no compute process holds a GPU.

    Back-to-back parametrized cases each spawn 8-rank torchrun subprocesses;
    the previous case's workers may still be tearing down when the next case
    launches, which fails CUDA initialization.
    """
    import subprocess
    import time

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return
        time.sleep(2)


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="requires 8 GPUs")
@pytest.mark.skipif(
    os.environ.get("WORLD_SIZE", "1") != "1",
    reason="spawns its own torchrun; run under plain pytest, not torch.distributed.run",
)
@pytest.mark.parametrize(
    "model_name,model_args,base_port",
    [
        ("dense", _DENSE_E2E_ARGS, 29720),
        ("moonlight_static_capacity", _MOONLIGHT_STATIC_E2E_ARGS, 29724),
    ],
)
class TestE2EFullIterationBitwise:
    """End-to-end bitwise comparison: pretrain_gpt.py eager vs full_iteration.

    Fixed num_microbatches per step (fixed-length packed data), THD sequence
    packing, PP2. The full-iteration run enables the CUDA allocator's
    graph_capture_record_stream_reuse option. Asserts per-iteration
    `lm loss / grad norm` metrics are byte-identical and that the graph was
    actually captured and replayed.

    Slow (several minutes per model). Marked `internal` so CI can opt-in.
    """

    def test_eager_vs_full_iteration(self, model_name, model_args, base_port):
        from tests.unit_tests.transformer.test_thd_cuda_graph import _extract_metrics, _run_pretrain

        _wait_for_idle_gpus()
        r1 = _run_pretrain(model_args, [], master_port=base_port, timeout=_E2E_RUN_TIMEOUT)
        assert r1.returncode == 0, (
            f"[{model_name}] eager pretrain failed (rc={r1.returncode})\n"
            f"--- stdout (tail) ---\n{r1.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r1.stderr[-2000:]}"
        )
        metrics_eager = _extract_metrics(r1.stdout)
        assert len(metrics_eager) == _E2E_TRAIN_ITERS, (
            f"[{model_name}] eager: expected {_E2E_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_eager)}\n--- stdout (tail) ---\n{r1.stdout[-2000:]}"
        )

        r2 = _run_pretrain(
            model_args,
            _FULL_ITER_ARGS,
            master_port=base_port + 1,
            extra_env={"PYTORCH_CUDA_ALLOC_CONF": "graph_capture_record_stream_reuse:True"},
            timeout=_E2E_RUN_TIMEOUT,
        )
        assert r2.returncode == 0, (
            f"[{model_name}] full_iteration pretrain failed (rc={r2.returncode})\n"
            f"--- stdout (tail) ---\n{r2.stdout[-4000:]}\n"
            f"--- stderr (tail) ---\n{r2.stderr[-2000:]}"
        )
        assert "Capture CUDA graph for training" in (
            r2.stdout + r2.stderr
        ), f"[{model_name}] full_iteration run never captured a CUDA graph"
        metrics_graph = _extract_metrics(r2.stdout)
        assert len(metrics_graph) == _E2E_TRAIN_ITERS, (
            f"[{model_name}] full_iteration: expected {_E2E_TRAIN_ITERS} metric lines, "
            f"got {len(metrics_graph)}\n--- stdout (tail) ---\n{r2.stdout[-2000:]}"
        )

        assert metrics_eager == metrics_graph, (
            f"[{model_name}] eager vs full_iteration metrics diverge\n"
            f"--- eager ---\n" + "\n".join(metrics_eager) + "\n"
            f"--- full_iteration ---\n" + "\n".join(metrics_graph)
        )
