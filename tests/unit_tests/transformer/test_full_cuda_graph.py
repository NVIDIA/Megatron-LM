# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import contextlib
import gc

import pytest
import torch
from pytest_mock import mocker

import megatron.core.pipeline_parallel.schedules as schedule
from megatron.core import ModelParallelConfig
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.tensor_parallel.random import (
    HAVE_TE,
    initialize_rng_tracker,
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.experimental_attention_variant.dsa import DSAIndexerLossLoggingHelper
from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import is_te_min_version
from tests.unit_tests.test_utilities import Utils

rank = Utils.rank


def _reset_full_cuda_graph_state():
    """Drop process-global graph inputs and outputs between unit tests."""
    FullCudaGraphWrapper.reset_cuda_graph()


@pytest.fixture(autouse=True)
def reset_full_cuda_graph_state():
    """Keep the full-iteration wrapper's class state isolated per test."""
    _reset_full_cuda_graph_state()
    MTPLossLoggingHelper.tracker = {}
    DSAIndexerLossLoggingHelper.tracker = {}
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    _reset_full_cuda_graph_state()
    MTPLossLoggingHelper.tracker = {}
    DSAIndexerLossLoggingHelper.tracker = {}
    Utils.destroy_model_parallel()
    gc.collect()


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="StaticBufferLoader stages inputs on the GPU"
)
def test_static_buffer_loader_isolates_cached_batch_structure():
    loader = StaticBufferLoader()
    first_inputs = {
        'tokens': torch.ones(2, 4),
        'labels': torch.ones(2, 4),
        'loss_mask': torch.ones(2, 4),
    }

    first_batch = loader(first_inputs, 'training', 0)
    cached_batch = StaticBufferLoader.static_buffers['training'][0]

    assert first_batch is not cached_batch
    assert first_batch['labels'] is cached_batch['labels']
    assert first_batch['loss_mask'] is cached_batch['loss_mask']

    # Pipeline stages replace unused fields in place. This must not replace the
    # corresponding tensors in the loader's static buffer.
    first_batch['labels'] = None
    first_batch['loss_mask'] = None

    second_inputs = {
        'tokens': torch.full((2, 4), 2.0),
        'labels': torch.full((2, 4), 3.0),
        'loss_mask': torch.full((2, 4), 4.0),
    }
    second_batch = loader(second_inputs, 'training', 0)

    assert second_batch is not cached_batch
    assert second_batch['labels'] is cached_batch['labels']
    assert second_batch['loss_mask'] is cached_batch['loss_mask']
    torch.testing.assert_close(second_batch['tokens'], second_inputs['tokens'].cuda())
    torch.testing.assert_close(second_batch['labels'], second_inputs['labels'].cuda())
    torch.testing.assert_close(second_batch['loss_mask'], second_inputs['loss_mask'].cuda())


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_forward_backward_func_with_full_cuda_graph(mocker):
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=2, pipeline_model_parallel_size=1)

    def forward_step_func(data_iterator, model):
        import os

        rank = int(os.environ['LOCAL_RANK'])
        dummy_data = torch.ones(1, 4)

        def loss_func(output_tensor):
            return rank, {'loss_reduced': rank}

        return model(dummy_data), loss_func

    model = torch.nn.Linear(4, 1)

    model.model_type = 'unit-test'

    def set_input_tensor(input_tensor):
        return None

    model.set_input_tensor = set_input_tensor

    forward_backward_func = get_forward_backward_func()
    assert schedule.get_forward_backward_func() == schedule.forward_backward_no_pipelining

    # Wrapping the forward_backward_func with FullCudaGraphWrapper enables full iteration CUDA graphs.
    forward_backward_func = FullCudaGraphWrapper(forward_backward_func)
    mocker.patch("megatron.core.pipeline_parallel.schedules.custom_backward", return_value=2)
    config = ModelParallelConfig(pipeline_model_parallel_size=1)
    model.config = config

    num_microbatches = 4

    # CUDA graph warmup
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    # CUDA graph capture and replay
    losses_reduced = forward_backward_func(
        forward_step_func=forward_step_func,
        data_iterator=[iter([{'input': torch.ones(1, 4)}] * num_microbatches)],
        model=[model],
        num_microbatches=num_microbatches,
        seq_length=None,
        micro_batch_size=None,
        forward_only=True,
    )
    loss_reduced_expected = [
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
        {'loss_reduced': rank},
    ]

    for i, j in zip(losses_reduced, loss_reduced_expected):
        print(losses_reduced)
        assert i['loss_reduced'] == j['loss_reduced']
    Utils.destroy_model_parallel()


def test_full_cuda_graph_capture_counts_dsa_metric_once():
    """Discard capture-time DSA metric writes before replaying the real iteration."""
    initialize_rng_tracker(force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    values = torch.zeros(1, device="cuda")
    DSAIndexerLossLoggingHelper.tracker = {
        "values": values,
        "reduce_group": None,
        "avg_group": None,
    }
    reduce_group = object()
    avg_group = object()

    def forward_backward_func(**kwargs):
        del kwargs
        DSAIndexerLossLoggingHelper.tracker["reduce_group"] = reduce_group
        DSAIndexerLossLoggingHelper.tracker["avg_group"] = avg_group
        DSAIndexerLossLoggingHelper.tracker["values"].add_(1.0)
        return [DSAIndexerLossLoggingHelper.tracker["values"]]

    model = torch.nn.Linear(1, 1).cuda()
    wrapped = FullCudaGraphWrapper(forward_backward_func, cuda_graph_warmup_steps=0)
    result = wrapped(
        data_iterator=[iter([{"tokens": torch.ones(1)}])],
        model=[model],
        num_microbatches=1,
        seq_length=1,
        forward_only=True,
    )

    assert DSAIndexerLossLoggingHelper.tracker["values"] is values
    torch.testing.assert_close(values, torch.ones_like(values))
    torch.testing.assert_close(result[0], torch.ones_like(result[0]))
    assert DSAIndexerLossLoggingHelper.tracker["reduce_group"] is reduce_group
    assert DSAIndexerLossLoggingHelper.tracker["avg_group"] is avg_group


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="full-iteration MTP test requires TransformerEngine RNG tracking",
)
def test_repeated_mtp_bshd_training_with_full_cuda_graph():
    """Capture and replay a full iteration with one MTP layer repeated to depth three."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    sequence_length = 16
    micro_batch_size = 1
    vocab_size = 128
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        mtp_num_layers=3,
        mtp_use_repeated_layer=True,
        mtp_loss_scaling_factor=0.1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        cuda_graph_impl="full_iteration",
        cuda_graph_modules=[],
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=layer_spec, use_transformer_engine=True
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        position_embedding_type="rope",
    ).cuda()
    model.train()
    assert model.mtp_process

    tokens = torch.arange(sequence_length, dtype=torch.int64).repeat(micro_batch_size, 1)
    batch = {
        'tokens': tokens,
        'labels': (tokens + 1) % vocab_size,
        'loss_mask': torch.ones_like(tokens, dtype=torch.float32),
        'position_ids': torch.arange(sequence_length, dtype=torch.int64).repeat(
            micro_batch_size, 1
        ),
    }

    def forward_step_func(data_iterator, model):
        microbatch = next(data_iterator)
        output = model(
            input_ids=microbatch['tokens'],
            position_ids=microbatch['position_ids'],
            attention_mask=None,
            labels=microbatch['labels'],
            loss_mask=microbatch['loss_mask'],
            packed_seq_params=None,
        )

        def loss_func(output_tensor):
            loss_mask = microbatch['loss_mask']
            loss = (output_tensor.float() * loss_mask).sum() / loss_mask.sum()
            return loss, {'lm loss': loss.detach()}

        return output, loss_func

    forward_backward_func = FullCudaGraphWrapper(
        get_forward_backward_func(), cuda_graph_warmup_steps=1
    )
    losses = []
    for _ in range(3):
        model.zero_grad(set_to_none=False)
        reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[iter([batch])],
            model=[model],
            num_microbatches=1,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        assert len(reduced) == 1
        losses.append(reduced[0]['lm loss'].detach().clone())

    assert FullCudaGraphWrapper.cuda_graph['training'] is not None
    assert all(torch.isfinite(loss) for loss in losses)
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(losses[2], losses[0], rtol=1e-3, atol=1e-3)

    mtp_grads = [
        parameter.grad for parameter in model.mtp.parameters() if parameter.grad is not None
    ]
    assert mtp_grads
    assert all(torch.isfinite(grad).all() for grad in mtp_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in mtp_grads)


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="full-iteration MTP test requires TransformerEngine RNG tracking",
)
def test_mtp_bshd_training_with_full_cuda_graph():
    """Capture and replay a full forward-backward iteration containing MTP BSHD rolls."""
    from megatron.core.pipeline_parallel import get_forward_backward_func

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)
    model_parallel_cuda_manual_seed(123)

    sequence_length = 16
    micro_batch_size = 1
    vocab_size = 128
    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        ffn_hidden_size=128,
        num_attention_heads=4,
        mtp_num_layers=1,
        mtp_loss_scaling_factor=0.1,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        use_cpu_initialization=True,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        cuda_graph_impl="full_iteration",
        cuda_graph_modules=[],
    )
    layer_spec = get_gpt_layer_with_transformer_engine_spec()
    mtp_block_spec = get_gpt_mtp_block_spec(
        config=config, spec=layer_spec, use_transformer_engine=True
    )
    model = GPTModel(
        config=config,
        transformer_layer_spec=layer_spec,
        mtp_block_spec=mtp_block_spec,
        vocab_size=vocab_size,
        max_sequence_length=sequence_length,
        position_embedding_type="rope",
    ).cuda()
    model.train()
    assert model.mtp_process

    tokens = torch.arange(sequence_length, dtype=torch.int64).repeat(micro_batch_size, 1)
    batch = {
        'tokens': tokens,
        'labels': (tokens + 1) % vocab_size,
        'loss_mask': torch.ones_like(tokens, dtype=torch.float32),
        'position_ids': torch.arange(sequence_length, dtype=torch.int64).repeat(
            micro_batch_size, 1
        ),
    }

    def forward_step_func(data_iterator, model):
        microbatch = next(data_iterator)
        output = model(
            input_ids=microbatch['tokens'],
            position_ids=microbatch['position_ids'],
            attention_mask=None,
            labels=microbatch['labels'],
            loss_mask=microbatch['loss_mask'],
            packed_seq_params=None,
        )

        def loss_func(output_tensor):
            loss_mask = microbatch['loss_mask']
            loss = (output_tensor.float() * loss_mask).sum() / loss_mask.sum()
            return loss, {'lm loss': loss.detach()}

        return output, loss_func

    forward_backward_func = FullCudaGraphWrapper(
        get_forward_backward_func(), cuda_graph_warmup_steps=1
    )
    losses = []
    for _ in range(3):
        model.zero_grad(set_to_none=False)
        reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=[iter([batch])],
            model=[model],
            num_microbatches=1,
            seq_length=sequence_length,
            micro_batch_size=micro_batch_size,
            forward_only=False,
        )
        assert len(reduced) == 1
        losses.append(reduced[0]['lm loss'].detach().clone())

    assert FullCudaGraphWrapper.cuda_graph['training'] is not None
    assert all(torch.isfinite(loss) for loss in losses)
    torch.testing.assert_close(losses[1], losses[0], rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(losses[2], losses[0], rtol=1e-3, atol=1e-3)

    mtp_grads = [
        parameter.grad for parameter in model.mtp.parameters() if parameter.grad is not None
    ]
    assert mtp_grads
    assert all(torch.isfinite(grad).all() for grad in mtp_grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in mtp_grads)


@pytest.mark.skipif(
    not (HAVE_TE and is_te_min_version("1.5.0")),
    reason="use_te_rng_tracker requires TransformerEngine version >= 1.5",
)
def test_full_cuda_graph_training_with_mhc_recompute():
    """Full-iteration capture must swallow eager mHC selective recompute.

    Trains a small hyper-connected GPT (selective recompute_modules=["mhc"],
    dropout disabled) through FullCudaGraphWrapper with forward_only=False:
    one eager warmup iteration, one capture iteration, then replays on fresh
    data. Per-iteration losses must match an identically-initialized eager
    twin fed the same batches — proving checkpoint discard, backward-time
    recompute, and storage rebind captured inside the graph replay correctly.
    """
    from megatron.core.enums import ModelType
    from megatron.core.full_cuda_graph import get_shared_capture_stream
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.pipeline_parallel import get_forward_backward_func
    from megatron.core.transformer.transformer_config import TransformerConfig

    seq_len, micro_batch, vocab = 32, 2, 128
    num_microbatches = 2
    num_iterations = 4  # 1 warmup + 1 capture + 2 replays

    initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)

    def reset_wrapper_state():
        FullCudaGraphWrapper.reset_cuda_graph()

    def build_model():
        model_parallel_cuda_manual_seed(123, te_rng_tracker=True, force_reset_rng=True)
        torch.manual_seed(456)
        config = TransformerConfig(
            num_layers=2,
            hidden_size=64,
            num_attention_heads=4,
            use_cpu_initialization=True,
            enable_hyper_connections=True,
            num_residual_streams=4,
            mhc_sinkhorn_iterations=5,
            mhc_init_gating_factor=0.01,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            recompute_granularity="selective",
            recompute_modules=["mhc"],
            cuda_graph_impl="full_iteration",
            # Match the real training path (arguments.py forces this True):
            # backward under capture must go through custom_backward — the
            # torch.autograd.backward wrapper trips a legacy-stream capture
            # dependency (cudaErrorStreamCaptureImplicit).
            deallocate_pipeline_outputs=True,
        )
        model = GPTModel(
            config=config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
                enable_hyper_connection=True
            ),
            vocab_size=vocab,
            max_sequence_length=seq_len,
        ).cuda()
        model.model_type = ModelType.encoder_or_decoder
        return model

    def forward_step_func(data_iterator, model):
        data = next(data_iterator)

        def loss_func(output_tensor):
            loss = output_tensor.float().mean()
            return loss, {'lm loss': loss.detach()}

        output = model(data['tokens'], data['position_ids'], None, labels=data['labels'])
        return output, loss_func

    def make_batches(iteration):
        gen = torch.Generator(device='cpu').manual_seed(1000 + iteration)
        batches = []
        for _ in range(num_microbatches):
            tokens = torch.randint(0, vocab, (micro_batch, seq_len), generator=gen).cuda()
            position_ids = (
                torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(micro_batch, -1).cuda()
            )
            batches.append(
                {'tokens': tokens, 'position_ids': position_ids, 'labels': tokens.clone()}
            )
        return batches

    def zero_grads(model):
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    def run(model, fbf, iteration, on_capture_stream=False):
        zero_grads(model)
        if on_capture_stream:
            # Raw (non-DDP) modules cache AccumulateGrad nodes with the stream
            # they were first executed on. The wrapper's eager warmup iteration
            # must therefore run on the same stream the capture will use, or
            # the captured backward trips a legacy-stream dependency
            # (cudaErrorStreamCaptureImplicit). Megatron's real training path
            # avoids this via the DDP main_grad hooks; this mirrors the
            # documented torch.cuda.graph warmup recipe instead.
            stream = get_shared_capture_stream()
            stream.wait_stream(torch.cuda.current_stream())
            ctx = torch.cuda.stream(stream)
        else:
            stream = None
            ctx = contextlib.nullcontext()
        with ctx:
            losses = fbf(
                forward_step_func=forward_step_func,
                data_iterator=[iter(make_batches(iteration))],
                model=[model],
                num_microbatches=num_microbatches,
                seq_length=seq_len,
                micro_batch_size=micro_batch,
                forward_only=False,
            )
        if stream is not None:
            torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        return [float(d['lm loss']) for d in losses]

    reset_wrapper_state()
    try:
        graphed_model = build_model()
        eager_model = build_model()
        for p_g, p_e in zip(graphed_model.parameters(), eager_model.parameters()):
            assert torch.equal(p_g, p_e), "twin models must start identical"

        graphed_fbf = FullCudaGraphWrapper(get_forward_backward_func(), cuda_graph_warmup_steps=1)
        eager_fbf = get_forward_backward_func()

        for iteration in range(num_iterations):
            eager_losses = run(eager_model, eager_fbf, iteration)
            graphed_losses = run(graphed_model, graphed_fbf, iteration, on_capture_stream=True)
            assert len(eager_losses) == len(graphed_losses) == num_microbatches
            for e, g in zip(eager_losses, graphed_losses):
                assert (
                    abs(e - g) < 1e-5
                ), f"iteration {iteration}: graphed loss {g} != eager loss {e}"

        # The capture happened after warmup; make sure replays actually ran.
        assert FullCudaGraphWrapper.cuda_graph['training'] is not None

        # Loss parity only proves the captured forward; gradients prove the
        # captured backward *including the recorded mHC recompute*. The last
        # iteration's grads were zeroed before each twin ran, so they are
        # directly comparable.
        compared = 0
        for p_g, p_e in zip(graphed_model.parameters(), eager_model.parameters()):
            if p_e.grad is not None:
                assert p_g.grad is not None
                torch.testing.assert_close(p_g.grad, p_e.grad)
                compared += 1
        assert compared > 0, "no gradients were compared"
    finally:
        reset_wrapper_state()
        Utils.destroy_model_parallel()
