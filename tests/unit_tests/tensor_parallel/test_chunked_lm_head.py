# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Chunked LM head: ``ColumnParallelLinear.forward_chunked_loss`` / ``LinearCrossEntropyChunked``.

The chunked head projects and scores the sequence one chunk at a time, so the logits of the
whole sequence are never allocated. Every test here compares it with the plain path
(``output_layer`` followed by the cross entropy) on the same data: the per-token loss, the
gradient of the head input and the gradient of the output-layer weight must agree.

Launch with ``torchrun --nproc-per-node=N -m pytest <this file>``. The TP=2 cases, with and
without sequence parallelism, are skipped unless the world size is even, so N=2 covers
everything and a plain single-process run covers TP=1.

The GTP-sharded output weight is covered in
``tests/unit_tests/generalized_tensor_parallel/test_gtp_chunked_lm_head.py``.
"""

import types

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state as ps
from megatron.core import tensor_parallel
from megatron.core.extensions.transformer_engine import te_general_gemm
from megatron.core.models.common.language_module.language_module import LanguageModule
from megatron.core.ops.loss.backends import select_cross_entropy
from megatron.core.tensor_parallel import layers
from megatron.core.transformer import multi_token_prediction as mtp
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

SEQ = 48
BATCH = 2
HIDDEN = 64
VOCAB = 1024
MTP_LAYERS = 2

# (input dtype, logits dtype). The second is the production setting: bf16 activations with
# --output-logit-dtype fp32, which needs Transformer Engine's mixed-precision GEMM.
FP32 = (torch.float32, None)
BF16_FP32_LOGITS = (torch.bfloat16, torch.float32)
DTYPES = [
    pytest.param(FP32, id="fp32"),
    pytest.param(
        BF16_FP32_LOGITS,
        id="bf16-fp32logits",
        marks=pytest.mark.skipif(
            te_general_gemm is None, reason="requires Transformer Engine general_gemm"
        ),
    ),
]
requires_te_gemm = pytest.mark.skipif(
    te_general_gemm is None, reason="requires Transformer Engine general_gemm"
)

SEQUENCE_PARALLEL = [pytest.param(False, id="no-sp"), pytest.param(True, id="sp")]


# ---------------------------------------------------------------------------
# Helpers (also used by the GTP variant of this test)
# ---------------------------------------------------------------------------


def set_lm_head_loss_chunks(config, num_chunks: int) -> None:
    """Switch an existing config between the plain (1) and the chunked head."""
    config.lm_head_loss_num_chunks = num_chunks


def make_cross_entropy(tp_group):
    """``LanguageModule.compute_language_model_loss`` bound to the native fused cross entropy.

    Like the real method it accepts ``return_argmax`` and returns None for the argmax, which the
    native implementation does not produce.
    """
    cross_entropy = select_cross_entropy(
        cross_entropy_loss_fusion=True, cross_entropy_fusion_impl="native"
    )

    def compute_language_model_loss(labels, logits, return_argmax=False):
        loss = cross_entropy(logits, labels.transpose(0, 1).contiguous(), tp_group)
        loss = loss.transpose(0, 1).contiguous()
        return (loss, None) if return_argmax else loss

    return compute_language_model_loss


def make_head_config(tp_size: int, sequence_parallel: bool, **overrides) -> TransformerConfig:
    """Config for a bare output layer plus the MTP loss settings the chunked MTP head needs."""
    kwargs = dict(
        num_layers=2,
        hidden_size=HIDDEN,
        num_attention_heads=4,
        use_cpu_initialization=True,
        gradient_accumulation_fusion=False,
        sequence_parallel=sequence_parallel,
        tensor_model_parallel_size=tp_size,
        mtp_num_layers=MTP_LAYERS,
        mtp_loss_scaling_factor=0.1,
        calculate_per_token_loss=True,
        mtp_loss_early_backward=True,
        mtp_normalize_loss_by_main_tokens=True,
    )
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def make_batch(seed: int, num_sequences: int, tp_size: int, sequence_parallel: bool, dtype):
    """Head input, labels and loss mask for ``num_sequences`` stacked sequences.

    ``seed`` must be the same on every rank of a TP group. With sequence parallelism each rank
    keeps its own slice of every sequence, as the model would hand it to the output layer.
    """
    generator = torch.Generator().manual_seed(seed)
    hidden = torch.randn(num_sequences * SEQ, BATCH, HIDDEN, generator=generator)
    labels = torch.randint(0, VOCAB, (BATCH, SEQ), generator=generator).cuda()
    mask = (torch.rand(BATCH, SEQ, generator=generator) > 0.3).float().cuda()
    if sequence_parallel and tp_size > 1:
        tp_rank = ps.get_tensor_model_parallel_rank()
        local = SEQ // tp_size
        hidden = torch.cat(
            [
                hidden[i * SEQ + tp_rank * local : i * SEQ + (tp_rank + 1) * local]
                for i in range(num_sequences)
            ],
            dim=0,
        )
    return hidden.cuda().to(dtype).requires_grad_(True), labels, mask


def run_head_sequence(
    layer, config, compute_loss, hidden, labels, mask, tp_group, before_backward=None
):
    """One microbatch of the model's head: every MTP head, then the main head, then backward.

    ``before_backward`` runs between the main head's forward and the backward, where the
    schedule does its own bookkeeping.

    ``hidden`` holds ``1 + MTP_LAYERS`` stacked sequences, as ``process_mtp_loss`` expects.
    Whether the heads are chunked is decided by ``config.lm_head_loss_num_chunks``, as in the
    model.
    Returns the main per-token loss.
    """
    main_hidden = mtp.process_mtp_loss(
        hidden_states=hidden * 1.5,
        labels=labels,
        loss_mask=mask,
        output_layer=layer,
        output_weight=None,
        is_training=True,
        compute_language_model_loss=compute_loss,
        config=config,
        cp_group=None,
        tp_group=tp_group,
        # Only handed to the metrics tracker, which the fixture below replaces.
        metric_avg_group=object(),
    )
    module = types.SimpleNamespace(
        config=config,
        output_layer=layer,
        _scale_logits=lambda logits: logits,
        compute_language_model_loss=compute_loss,
    )
    if LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None):
        loss = LanguageModule.compute_chunked_lm_head_loss(module, main_hidden, labels, mask, None)
    else:
        logits, _ = layer(main_hidden)
        loss = compute_loss(labels, logits)
    total = torch.sum(loss.view(-1).float() * mask.view(-1))
    if before_backward is not None:
        before_backward()
    total.backward()
    return loss


def assert_matches(reference, result, dtype, what: str) -> None:
    """Chunk size changes which GEMM kernel runs, so allow rounding in the compute dtype."""
    atol, rtol = (1e-5, 1e-4) if dtype == torch.float32 else (2e-3, 2e-3)
    torch.testing.assert_close(
        result.float(), reference.float(), atol=atol, rtol=rtol, msg=lambda m: f"{what}: {m}"
    )


@pytest.fixture
def mtp_metrics(monkeypatch):
    """Capture what the MTP heads would log, and give the early backward a loss scale."""
    logged = []

    def save_metrics_to_tracker(loss, correct, total, layer_number, num_layers, **kwargs):
        # Left on the GPU: a copy to the host is not allowed inside a CUDA graph capture.
        logged.append(
            torch.stack(
                [loss.detach().float(), correct.float(), total.float(), kwargs["ce_loss"].float()]
            )
        )

    monkeypatch.setattr(
        mtp.MTPLossLoggingHelper, "save_metrics_to_tracker", staticmethod(save_metrics_to_tracker)
    )
    monkeypatch.setattr(
        mtp.MTPLossAutoScaler, "main_loss_backward_scale", torch.ones(1, device="cuda")
    )
    return logged


@pytest.fixture(autouse=True)
def _reset_chunked_head_state(monkeypatch):
    # Every test gets the full budget of upstream-gradient checks.
    monkeypatch.setattr(layers.LinearCrossEntropyChunked, "checks_left", 8)


@pytest.fixture(scope="module", params=[1, 2], ids=["tp1", "tp2"])
def tp_env(request):
    """Model-parallel state for one TP size: ``(tp_size, tp_group, data_lane)``.

    Module-scoped so the process groups are built once per TP size. Building them for every
    test leaves enough NCCL communicators behind to exhaust device memory.
    """
    tp_size = request.param
    world_size = Utils.world_size
    if world_size % tp_size != 0:
        pytest.skip(f"TP={tp_size} needs a world size divisible by {tp_size}, got {world_size}")
    Utils.initialize_model_parallel(tp_size, 1)
    tp_group = ps.get_tensor_model_parallel_group()
    # Ranks of one TP group must see the same data; ranks of different groups need not.
    data_lane = dist.get_process_group_ranks(tp_group)[0]
    yield tp_size, tp_group, data_lane
    Utils.destroy_model_parallel()


def _skip_unless_sequence_parallel_applies(tp_size: int, sequence_parallel: bool) -> None:
    if sequence_parallel and tp_size == 1:
        pytest.skip("sequence parallelism needs TP > 1")


def _make_layer(config, in_dtype, out_dtype, seed: int = 100):
    # Each TP rank owns a different vocabulary partition, so seed the weight per rank.
    torch.manual_seed(seed + ps.get_tensor_model_parallel_rank())
    layer = tensor_parallel.ColumnParallelLinear(
        HIDDEN,
        VOCAB,
        config=config,
        init_method=config.init_method,
        bias=False,
        skip_bias_add=False,
        gather_output=False,
        output_dtype=out_dtype,
    )
    return layer.cuda().to(in_dtype)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seq_len,num_chunks,expected",
    [
        (8, 1, [(0, 8)]),
        (8, 4, [(0, 2), (2, 4), (4, 6), (6, 8)]),
        (10, 4, [(0, 3), (3, 6), (6, 9), (9, 10)]),
        (3, 8, [(0, 1), (1, 2), (2, 3)]),
        (5, 0, [(0, 5)]),
    ],
)
def test_chunk_bounds_cover_the_sequence(seq_len, num_chunks, expected):
    assert layers.lm_head_chunk_bounds(seq_len, num_chunks) == expected


def _plain_config(**overrides) -> TransformerConfig:
    kwargs = dict(num_layers=2, hidden_size=HIDDEN, num_attention_heads=4)
    kwargs.update(overrides)
    return TransformerConfig(**kwargs)


def test_config_defaults_to_the_plain_head():
    assert _plain_config().lm_head_loss_num_chunks == 1


def test_config_accepts_a_supported_chunked_head():
    _plain_config(lm_head_loss_num_chunks=8, calculate_per_token_loss=True)
    make_head_config(1, False, lm_head_loss_num_chunks=8)


_CHUNKED_WITH_MTP = dict(
    lm_head_loss_num_chunks=8,
    calculate_per_token_loss=True,
    mtp_num_layers=MTP_LAYERS,
    mtp_loss_scaling_factor=0.1,
    mtp_loss_early_backward=True,
    mtp_normalize_loss_by_main_tokens=True,
)


@pytest.mark.parametrize(
    "kwargs,message",
    [
        (dict(lm_head_loss_num_chunks=0), "at least 1"),
        # The main head: only the per-token loss is a reduction the forward can know.
        (dict(lm_head_loss_num_chunks=8), "calculate_per_token_loss"),
        # With MTP every head has to be chunkable, or the unchunked ones keep the peak memory.
        (dict(_CHUNKED_WITH_MTP, mtp_loss_early_backward=False), "mtp_loss_early_backward"),
        (
            dict(_CHUNKED_WITH_MTP, mtp_normalize_loss_by_main_tokens=False),
            "mtp_normalize_loss_by_main_tokens",
        ),
        (dict(_CHUNKED_WITH_MTP, mtp_kl_loss_weight=0.5), "only MTP loss"),
        (dict(_CHUNKED_WITH_MTP, mtp_detach_heads=True), "mtp_detach_heads"),
    ],
)
def test_config_rejects_what_the_chunked_head_cannot_do(kwargs, message):
    with pytest.raises(ValueError, match=message):
        _plain_config(**kwargs)


@pytest.mark.parametrize("sequence_parallel", SEQUENCE_PARALLEL)
@pytest.mark.parametrize("dtypes", DTYPES)
@pytest.mark.parametrize("fusion", [False, True], ids=["autograd-wgrad", "fused-wgrad"])
@pytest.mark.parametrize("num_chunks", [3, 8])
def test_layer_matches_unchunked(tp_env, sequence_parallel, dtypes, fusion, num_chunks):
    """One head: loss, input gradient and weight gradient equal the plain path's."""
    tp_size, tp_group, data_lane = tp_env
    _skip_unless_sequence_parallel_applies(tp_size, sequence_parallel)
    in_dtype, out_dtype = dtypes
    config = make_head_config(tp_size, sequence_parallel, gradient_accumulation_fusion=fusion)
    layer = _make_layer(config, in_dtype, out_dtype)
    assert layer.sequence_parallel == sequence_parallel
    assert layer.weight.shape[0] == VOCAB // tp_size
    assert layer.supports_chunked_loss(None, False)
    if fusion:
        layer.weight.main_grad = torch.zeros_like(layer.weight, dtype=torch.float32)
    compute_loss = make_cross_entropy(tp_group)
    hidden, labels, mask = make_batch(7 + data_lane, 1, tp_size, sequence_parallel, in_dtype)

    def run(chunked: bool):
        layer.weight.grad = None
        hidden.grad = None
        if fusion:
            layer.weight.main_grad.zero_()
        head_input = hidden * 1.5
        if chunked:
            loss = layer.forward_chunked_loss(
                head_input,
                lambda logits, start, end: compute_loss(labels[:, start:end], logits),
                mask,
                num_chunks,
            )
        else:
            logits, _ = layer(head_input, runtime_gather_output=False)
            loss = compute_loss(labels, logits)
        torch.sum(loss.view(-1).float() * mask.view(-1)).backward()
        weight_grad = layer.weight.main_grad if fusion else layer.weight.grad
        return loss.detach().clone(), hidden.grad.clone(), weight_grad.clone()

    reference = run(chunked=False)
    result = run(chunked=True)

    # The loss covers the full sequence even when the input is a sequence-parallel slice.
    assert result[0].shape == (BATCH, SEQ)
    assert result[1].shape == hidden.shape
    assert result[2].shape == layer.weight.shape
    for name, ref, res in zip(("loss", "input gradient", "weight gradient"), reference, result):
        assert_matches(ref, res, in_dtype if name != "loss" else torch.float32, name)


@requires_te_gemm
@pytest.mark.parametrize("sequence_parallel", SEQUENCE_PARALLEL)
@pytest.mark.parametrize("num_chunks", [4, 5, 8])
def test_head_sequence_matches_unchunked(tp_env, sequence_parallel, num_chunks, mtp_metrics):
    """Both MTP heads and the main head, wired the way the model wires them."""
    tp_size, tp_group, data_lane = tp_env
    _skip_unless_sequence_parallel_applies(tp_size, sequence_parallel)
    config = make_head_config(tp_size, sequence_parallel)
    layer = _make_layer(config, *BF16_FP32_LOGITS, seed=200)
    compute_loss = make_cross_entropy(tp_group)
    hidden, labels, mask = make_batch(
        11 + data_lane, 1 + MTP_LAYERS, tp_size, sequence_parallel, torch.bfloat16
    )

    chunked_calls = []
    forward_chunked_loss = layer.forward_chunked_loss

    def counting_forward_chunked_loss(*args, **kwargs):
        chunked_calls.append(1)
        return forward_chunked_loss(*args, **kwargs)

    layer.forward_chunked_loss = counting_forward_chunked_loss

    def run(chunks: int):
        set_lm_head_loss_chunks(config, chunks)
        mtp_metrics.clear()
        chunked_calls.clear()
        layer.weight.grad = None
        hidden.grad = None
        loss = run_head_sequence(layer, config, compute_loss, hidden, labels, mask, tp_group)
        return (
            (loss.detach().clone(), hidden.grad.clone(), layer.weight.grad.clone()),
            torch.stack(mtp_metrics),
            len(chunked_calls),
        )

    reference, reference_metrics, reference_calls = run(1)
    result, result_metrics, result_calls = run(num_chunks)

    # Otherwise this compares the plain path with itself.
    assert reference_calls == 0
    assert result_calls == 1 + MTP_LAYERS
    # Each chunked head had its assumed upstream gradient verified in backward.
    assert layers.LinearCrossEntropyChunked.checks_left == 8 - (1 + MTP_LAYERS)

    assert_matches(reference[0], result[0], torch.float32, "main loss")
    assert_matches(reference[1], result[1], torch.bfloat16, "input gradient")
    assert_matches(reference[2], result[2], torch.bfloat16, "weight gradient")
    # Loss sum, correct count, token count and CE sum of each MTP head.
    torch.testing.assert_close(result_metrics, reference_metrics, atol=1e-2, rtol=1e-5)


@requires_te_gemm
def test_no_grad_scores_without_keeping_a_gradient(tp_env):
    """Evaluation: the chunks are only scored, and nothing is kept for a backward."""
    tp_size, tp_group, data_lane = tp_env
    config = make_head_config(tp_size, False)
    layer = _make_layer(config, *BF16_FP32_LOGITS)
    compute_loss = make_cross_entropy(tp_group)
    hidden, labels, mask = make_batch(3 + data_lane, 1, tp_size, False, torch.bfloat16)
    module = types.SimpleNamespace(
        config=config,
        output_layer=layer,
        _scale_logits=lambda logits: logits,
        compute_language_model_loss=compute_loss,
    )
    set_lm_head_loss_chunks(config, 4)

    with torch.no_grad():
        logits, _ = layer(hidden)
        reference = compute_loss(labels, logits)
        assert LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)
        result = LanguageModule.compute_chunked_lm_head_loss(module, hidden, labels, mask, None)

    assert not result.requires_grad
    assert_matches(reference, result, torch.float32, "loss")


@requires_te_gemm
def test_backward_rejects_a_loss_reduced_differently(tp_env):
    """The gradient to the logits is computed in forward from an assumed upstream gradient.

    A loss function that is not the masked sum sends back something else; training on with
    the forward's gradient would silently use the wrong one, so backward must raise.
    """
    tp_size, tp_group, data_lane = tp_env
    config = make_head_config(tp_size, False)
    layer = _make_layer(config, *BF16_FP32_LOGITS)
    compute_loss = make_cross_entropy(tp_group)
    hidden, labels, mask = make_batch(5 + data_lane, 1, tp_size, False, torch.bfloat16)

    loss = layer.forward_chunked_loss(
        hidden * 1.5, lambda logits, start, end: compute_loss(labels[:, start:end], logits), mask, 4
    )
    with pytest.raises(RuntimeError, match="Chunked LM head"):
        loss.float().mean().backward()


def test_falls_back_when_the_configuration_is_not_covered(tp_env, monkeypatch):
    """Unsupported set-ups must take the plain path rather than a wrong chunked one."""
    tp_size, _, _ = tp_env
    config = make_head_config(tp_size, False)
    layer = _make_layer(config, torch.float32, None)
    labels = torch.zeros(BATCH, SEQ, dtype=torch.long, device="cuda")
    mask = torch.ones(BATCH, SEQ, device="cuda")
    module = types.SimpleNamespace(config=config, output_layer=layer)

    assert layer.supports_chunked_loss(None, False)
    # Gathering the output across TP ranks applies to logits, not to a loss.
    assert layer.supports_chunked_loss(None, True) == (tp_size == 1)
    # A frozen (or detached) weight takes linear_with_frozen_weight, which is not chunked.
    assert not layer.supports_chunked_loss(layer.weight.detach(), False)

    # Off by default, and then it has nothing to report.
    monkeypatch.setattr(layers, "_lm_head_chunking_messages", set())
    messages = layers._lm_head_chunking_messages
    set_lm_head_loss_chunks(config, 1)
    assert not LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)
    assert not messages

    # On: the log says so, once however often the head runs.
    set_lm_head_loss_chunks(config, 4)
    assert LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)
    assert LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)
    assert len(messages) == 1 and "in 4 chunks" in next(iter(messages))

    # No labels: the caller wants logits, which is not a fallback worth reporting.
    assert not LanguageModule.can_chunk_lm_head_loss(module, None, mask, None, None)
    assert len(messages) == 1
    # A requested chunked head that cannot run names the reason.
    assert not LanguageModule.can_chunk_lm_head_loss(module, labels, None, None, None)
    assert any("NOT chunked" in m and "loss mask" in m for m in messages)
    assert not LanguageModule.can_chunk_lm_head_loss(
        module, labels, mask, layer.weight.detach(), None
    )
    assert any("NOT chunked" in m and "output layer" in m for m in messages)

    # Without the per-token loss the upstream gradient depends on counts only the schedule
    # has, so training falls back; evaluation needs no gradient and may still chunk.
    config.calculate_per_token_loss = False
    assert not LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)
    with torch.no_grad():
        assert LanguageModule.can_chunk_lm_head_loss(module, labels, mask, None, None)


@requires_te_gemm
@pytest.mark.parametrize("sequence_parallel", SEQUENCE_PARALLEL)
@pytest.mark.parametrize("num_chunks", [4, 8])
def test_cuda_graph_replay_matches_eager_unchunked(
    tp_env, sequence_parallel, num_chunks, mtp_metrics
):
    """Capture the chunked head sequence in a CUDA graph and replay it on new data.

    The chunk loop differentiates each chunk inside the forward, and under tensor parallelism
    the cross entropy issues its collectives once per chunk; all of it has to be capturable.
    """
    tp_size, tp_group, data_lane = tp_env
    _skip_unless_sequence_parallel_applies(tp_size, sequence_parallel)
    config = make_head_config(tp_size, sequence_parallel, cuda_graph_impl="full_iteration")
    layer = _make_layer(config, *BF16_FP32_LOGITS, seed=200)
    compute_loss = make_cross_entropy(tp_group)
    # Static inputs: the graph reads these addresses on every replay.
    hidden, labels, mask = make_batch(0, 1 + MTP_LAYERS, tp_size, sequence_parallel, torch.bfloat16)

    def load(seed: int):
        new_hidden, new_labels, new_mask = make_batch(
            seed + data_lane, 1 + MTP_LAYERS, tp_size, sequence_parallel, torch.bfloat16
        )
        with torch.no_grad():
            hidden.copy_(new_hidden)
        labels.copy_(new_labels)
        mask.copy_(new_mask)

    chunked_calls = []
    forward_chunked_loss = layer.forward_chunked_loss

    def counting_forward_chunked_loss(*args, **kwargs):
        chunked_calls.append(1)
        return forward_chunked_loss(*args, **kwargs)

    layer.forward_chunked_loss = counting_forward_chunked_loss

    def step():
        hidden.grad = None
        layer.weight.grad = None
        loss = run_head_sequence(layer, config, compute_loss, hidden, labels, mask, tp_group)
        return loss, hidden.grad, layer.weight.grad

    capture_stream = torch.cuda.Stream()
    if hasattr(torch.autograd.graph, "set_override_stale_capture_stream"):
        torch.autograd.graph.set_override_stale_capture_stream(True)

    # Eager warmup on the capture stream, as the full-iteration wrapper does.
    set_lm_head_loss_chunks(config, num_chunks)
    for seed in (100, 200):
        load(seed)
        with torch.cuda.stream(capture_stream):
            step()
    torch.cuda.synchronize()
    dist.barrier()

    load(300)
    chunked_calls.clear()
    graph = torch.cuda.CUDAGraph()
    with (
        torch.compiler.set_stance("eager_on_recompile"),
        torch.cuda.graph(graph, stream=capture_stream, capture_error_mode="thread_local"),
    ):
        captured = step()
    assert len(chunked_calls) == 1 + MTP_LAYERS, "the captured step did not use the chunked head"
    torch.cuda.synchronize()
    dist.barrier()

    for seed in (400, 500, 600):
        load(seed)
        set_lm_head_loss_chunks(config, 1)
        reference = [tensor.detach().clone() for tensor in step()]
        set_lm_head_loss_chunks(config, num_chunks)
        torch.cuda.synchronize()
        dist.barrier()

        graph.replay()
        torch.cuda.synchronize()
        dist.barrier()

        assert_matches(reference[0], captured[0], torch.float32, f"replayed loss (seed {seed})")
        assert_matches(
            reference[1], captured[1], torch.bfloat16, f"replayed input gradient (seed {seed})"
        )
        assert_matches(
            reference[2], captured[2], torch.bfloat16, f"replayed weight gradient (seed {seed})"
        )
