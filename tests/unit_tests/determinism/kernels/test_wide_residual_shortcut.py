# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise determinism of the fused wide-residual shortcut boundaries.

Bit-exact output/gradient replay and poisoned CUDA-graph replay of the forward, inference,
backward, final-reduction, checkpoint-interval and deferred MXFP8 weight-gradient paths. Shared
inputs, runners and FP64 references live in ``tests/unit_tests/fusions/test_wide_residual_shortcut.py``.
"""

import pytest
import torch

from megatron.core.fusions.wide_residual_shortcut import (
    backward_kernel,
    forward_kernel,
    kernel_utils,
)
from megatron.core.fusions.wide_residual_shortcut.mxfp8_handoff import Reader, reader_for
from megatron.core.fusions.wide_residual_shortcut.replay_kernel import replay_interval
from megatron.core.fusions.wide_residual_shortcut.residual_replay import residual_replay_scope
from megatron.core.inference.utils import InferenceMode
from megatron.core.transformer.streamwise_residual_ops import streamwise_sigmoid_writeback
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact, seeded
from tests.unit_tests.fusions.test_wide_residual_shortcut import (  # noqa: F401 (reader_work)
    GPU,
    MX_SHAPES,
    QUALIFIED,
    F,
    T,
    all_bytes_equal,
    assert_columns_equal,
    backward_args,
    boundary,
    buffers,
    consume,
    forbid_backward_state,
    fwd,
    inputs,
    invoke,
    leaf_grads,
    linear,
    mx_inputs,
    mxfp8_test,
    native_backward,
    ordered_reductions,
    poison_bytes,
    reader_work,
    reference,
    replay_config,
    replay_poisoned,
    run,
    run_with_scratch,
    standard_backward,
    workspace_tensors,
)

pytestmark = GPU


def offset_view(tensor, offset=1):
    """A copy of ``tensor`` whose base is ``offset`` elements past an allocation start."""
    shifted = tensor.new_empty(tensor.numel() + offset)[offset:].view_as(tensor)
    with torch.no_grad():
        shifted.copy_(tensor)
    return shifted


def assert_poisoned_graph(execute, expected, fill=poison_bytes):
    """Capture execute(); every replay must rewrite each poisoned output byte for byte."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = list(execute())
    replay_poisoned(graph, outputs, outputs, expected, fill)


def forward_state(current, saved, operands):
    """Forward outputs and every saved-tape tensor, for byte comparisons."""
    data, scales = kernel_utils.operand_buffers(operands, columnwise=False)
    tape = ((saved.update,) if saved.reader_count == 2 else ()) + (saved.coefficients, *saved.rstd)
    return (current, *tape, *data, *(scale for scale in scales if scale is not None))


# ---------------------------------------------------------------- forward and inference


@pytest.mark.parametrize(
    "streams,readers,retention,rows,hidden",
    [
        (s, n, t, 1024, h)
        for h in (96, 128, 288)
        for s in range(1, 7)
        for n in (1, 2)
        for t in (F, T)
    ]
    # Production shapes cover the fused forward layouts and the learned one-reader profile.
    + [
        pytest.param(3, n, t, m, h, marks=QUALIFIED)
        for m, h in kernel_utils.FLAGSHIPS
        for n in (1, 2)
        for t in (F, T)
    ]
    # Production-size dispatch for other stream counts (three streams is listed above).
    + [pytest.param(s, 2, T, 8192, 4096, marks=QUALIFIED) for s in (2, 5, 6)],
)
def test_standard_boundaries_replay(streams, readers, retention, rows, hidden):
    raw = inputs(streams, readers, hidden, rows, retention=retention)
    assert_replays_bit_exact(lambda *args: run(args, streams), raw, replays=3)


@pytest.mark.parametrize("streams", [1, 2, 3, 5, 6])
@pytest.mark.parametrize("retention", [False, True])
@pytest.mark.parametrize("hidden", [96, 128, 288])
def test_checkpoint_chain_replay(streams, retention, hidden):
    def execute(residual, updates, controllers, gammas):
        with residual_replay_scope(replay_config(), True):
            current, q0 = run((residual, updates, controllers, gammas), streams)
            current = streamwise_sigmoid_writeback(
                current,
                updates[0],
                controllers[0],
                streams,
                retention_logits=controllers[1],
                retention_max_forget=0.1,
            )
            current, q1 = run((current, updates, controllers, gammas), streams)
        # q0 is also an output so its independent reader gradient participates.
        return current, q0, q1

    raw = inputs(streams, 1, hidden, 1024, retention=retention)
    assert_replays_bit_exact(execute, raw, replays=3)


@QUALIFIED
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("retention", [False, True])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize(
    "rows,hidden",
    [*kernel_utils.FLAGSHIPS, (2048, 2048), (3072, 5120), (4096, 8192), (12288, 13312)]
    + [(2048, 16384), (32, 6144)],
)
@pytest.mark.parametrize("epsilon", [1e-5, 1e-3])
def test_packet_forward_poisoned_replay(quantized, retention, readers, rows, hidden, epsilon):
    """Production and heuristic forward schedules overwrite all outputs on replay."""
    raw = inputs(3, readers, hidden, rows, retention=retention)

    def execute():
        return forward_state(*fwd(raw, mxfp8_handoff=quantized, epsilon=epsilon))

    expected, replays = [value.clone() for value in execute()], iter(range(3))

    def refresh(actual):
        if replay := next(replays):
            # Mixed register/shared-memory fragments must use the new inputs, and
            # normalization replays at different activation magnitudes.
            with torch.no_grad():
                raw[0].mul_(0.125 if replay == 1 else -8.0)
                for value in (*raw[1], *raw[2]):
                    if value is not None:
                        value.add_(0.125)
                for gamma in raw[3]:
                    gamma.mul_(0.75)
            expected[:] = [value.clone() for value in execute()]
        poison_bytes(actual, 0xA5 if replay % 2 else 0x5A)

    assert_poisoned_graph(execute, expected, refresh)


@QUALIFIED
@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("readers,operand", [(n, i) for n in (1, 2) for i in range(1 + 2 * n)])
@pytest.mark.parametrize("hidden,streams", [(3072, 3), (4096, 3), (10240, 3), (128, 1), (288, 6)])
# Element offsets: 1 (FP32 4 B, BF16 2 B), 2 (8 B / 4 B) and 4 (16 B / 8 B).
# Offset 3 repeats offset 1's alignment class for both dtypes.
@pytest.mark.parametrize("offset", [1, 2, 4])
def test_packet_forward_offset_views(quantized, readers, operand, hidden, streams, offset):
    """Only explicit residual packet loads require 16-byte alignment."""
    residual, updates, controllers, gammas = inputs(streams, readers, hidden, 128)
    values = [residual, *updates, *gammas]

    def execute(args):
        raw = args[0], args[1 : readers + 1], controllers, tuple(args[readers + 1 :])
        return forward_state(*fwd(raw, streams, mxfp8_handoff=quantized))

    expected = execute(values)
    values[operand] = offset_view(values[operand], offset)
    if operand == 0 and values[0].data_ptr() % 16:
        with pytest.raises(AssertionError, match="16-byte aligned"):
            execute(values)
    else:
        all_bytes_equal(execute(values), expected)


# BF16: hidden 1 (<32) and 1025 (odd) use scalar loads; 48, 288, 2052 masked packets; 18432
# and 32768 the widest profiles and column partitions (32768 is unmasked). Widths that are not
# a multiple of four use an offset residual; odd stream counts drop retention. MXFP8 covers
# generic streams and initializes both row and column scale padding.
GENERAL_FORWARD = [
    (33 if inf else 64, h, s, n, inf, F)
    for h in (1, 48, 288, 1025, 2052, 18432, 32768)
    for s in range(1, 7)
    for n in (1, 2)
    for inf in (F, T)
] + [
    pytest.param(m, h, s, 2, inf, T, marks=QUALIFIED)
    for m, h in ((32, 32), (64, 96), (96, 288), (160, 18432))
    for s in (1, 2, 4, 6)
    for inf in (F, T)
]


@pytest.mark.parametrize("tokens,hidden,streams,readers,inference,quantized", GENERAL_FORWARD)
def test_general_forward_replay(tokens, hidden, streams, readers, inference, quantized):
    """Arbitrary widths/streams match FP64; replay restores every poisoned saved/output byte."""
    raw = inputs(streams, readers, hidden, tokens, retention=quantized or not streams % 2)
    if not quantized and hidden % 4:
        raw = (offset_view(raw[0]), *raw[1:])

    def execute():
        current, saved, outputs = fwd(raw, streams, inference=inference, mxfp8_handoff=quantized)
        tape = () if saved is None else (saved.coefficients, *saved.rstd)
        if saved is not None and readers == 2:
            tape += (saved.update,)
        return current, *outputs, *tape

    with torch.no_grad():
        actual = execute()
        expected = reference(raw, streams)
        torch.testing.assert_close(actual[0], expected[0], rtol=2e-6, atol=2e-6)
        if quantized:
            relative = (actual[1].dequantize().float() - expected[1].float()).norm()
            assert relative / expected[1].float().norm() < 0.04
        for a, b in zip(
            actual[1 + quantized : 1 + readers], expected[1 + quantized :], strict=True
        ):
            torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
        assert_poisoned_graph(lambda: buffers(execute()), [t.clone() for t in buffers(actual)])


# Tokens: replica classes 4/2/1 and padded single-/multi-tile or unpadded MXFP8 scales.
# Widths: masked/unmasked row profiles, both mid-width profiles (13312 is the upper bound),
# forced retention (10240), and the widest profile.
@pytest.mark.parametrize("tokens", [32, 64, 96, 320, 1024])
@pytest.mark.parametrize("hidden", [3072, 4096, 7168, 10240, 13312, 14336])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("quantized", [False, pytest.param(True, marks=QUALIFIED)])
def test_inference_poisoned_replay(monkeypatch, tokens, hidden, readers, quantized):
    """Replay inference launches (including split two-reader schedules) without backward
    state; every poisoned output byte is restored."""
    raw = inputs(3, readers, hidden, tokens)
    observed = forbid_backward_state(monkeypatch)
    with torch.inference_mode(), InferenceMode.active():
        expected = [t.clone() for t in buffers(invoke(raw, quantized))]
        assert_poisoned_graph(lambda: buffers(invoke(raw, quantized)), expected)
    assert len(observed) == 2


# ---------------------------------------------------------------- backward and reductions

# The backward kernel does not take column routes; they only prepend _replay_mx launches,
# specialized by stream count, retention, width and reader index, not by reader count. Two
# readers with columns replay both reader indices; one reader without columns keeps the
# backward-only graph. Explicit grids include idle groups that must skip bootstrap transfers.
EXPLICIT_GRIDS = [(32, 1), (96, 1), (128, 64), (288, 4), (4128, 34), (4096, 32)]
EXPLICIT_GRIDS += [(4096, 960), (3072, 24), (10240, 80)]
ADMISSION_SHAPES = MX_SHAPES[:10] + [(4096, 13312)]
POISONED_BACKWARD = (
    # Partial-width reductions (4128 columns) with reader-zero columns.
    [(s, n, 4128, 1024, T, "auto", (T, F)[:n]) for s in (3, 6) for n in (1, 2)]
    # Partial row tiles and padded MXFP8 scales with every reader's columns.
    + [(s, n, 96, 160, T, "auto", (T,) * n) for s in (2, 3, 6) for n in (1, 2)]
    + [
        (s, n, h, 1024, t, grid, (n == 2,) * n)
        for n in (1, 2)
        for s in (1, 2, 4, 5, 6)
        for t in (F, T)
        for h, grid in EXPLICIT_GRIDS
    ]
    # Dynamic row admission at production and expanded shapes resets all queue epochs.
    + [
        (3, n, h, m, t, "auto", (c,) + (F,) * (n - 1))
        for m, h in ADMISSION_SHAPES
        for n in (1, 2)
        for t in (F, T)
        for c in (F, T)
    ]
)


@QUALIFIED
@pytest.mark.parametrize("streams,readers,hidden,rows,retention,grid,routes", POISONED_BACKWARD)
def test_poisoned_backward_replay(streams, readers, hidden, rows, retention, grid, routes):
    """A captured backward restores its eager outputs after all scratch is poisoned."""
    _, args = backward_args(inputs(streams, readers, hidden, rows, retention=retention), streams)
    options = dict(config=grid, prepare_wgrad=routes)
    execute = lambda: run_with_scratch(backward_kernel.backward, *args, **options)[1]
    eager = execute()
    assert eager.kernel.metadata.launch_cooperative_grid is False
    assert [column is not None for column in eager.columns] == list(routes)
    # The admission and row-tile counters must be poisoned and reset on replay.
    assert any(t.dtype == torch.int32 and t.numel() == 2 for t in eager.allocations)
    expected = [t.clone() for t in workspace_tensors((eager.grads, eager.columns))]
    # Warm up and capture on a side stream, as production graph capture does.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        execute()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        captured = execute()
    outputs = workspace_tensors((captured.grads, captured.columns))
    replay_poisoned(graph, captured.allocations, outputs, expected)


@pytest.mark.parametrize("explicit_grid", [False, True])
@pytest.mark.parametrize("rows,hidden,columns", [(1024, 4096, False), (8192, 10240, True)])
def test_backward_owns_independent_scratch(explicit_grid, rows, hidden, columns):
    """Independent calls and captured graphs cannot alias inputs or each other's scratch."""
    current, args = backward_args(inputs(3, 2, hidden, rows))
    saved, _, dys, carry = args
    routes = (columns, False)
    options = (
        dict(config=hidden // 128, save_columns=routes)
        if explicit_grid
        else dict(prepare_wgrad=routes)
    )
    backward = native_backward if explicit_grid else backward_kernel.backward
    protected = (current, carry, *dys, saved.update, saved.coefficients, *saved.rstd, *saved.gammas)
    originals = [tensor.clone() for tensor in protected]
    _, first = run_with_scratch(backward, *args, **options)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, second = run_with_scratch(backward, *args, **options)
    storage = lambda tensors: {tensor.untyped_storage().data_ptr() for tensor in tensors}
    first_storage, second_storage = (storage(workspace_tensors(ws)) for ws in (first, second))
    assert not first_storage & second_storage
    assert not (first_storage | second_storage) & storage(protected)
    for workspace in (first, second):
        tensors = list(workspace_tensors((workspace.grads, workspace.columns)))
        assert len(tensors) - len(storage(tensors)) == 1
        assert workspace.grads[1] is workspace.grads[2]
    outputs = lambda ws: [*workspace_tensors((ws.grads, ws.columns))]
    replay_poisoned(
        graph,
        workspace_tensors(second),
        outputs(second) + list(protected),
        outputs(first) + originals,
    )


@pytest.mark.parametrize("streams", [1, 3, 6])
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("retention", [False, True])
def test_final_reduction_modes_match_exact_oracle(streams, readers, retention):
    """Standalone and gamma-fused reductions preserve scalars, padding and replay."""
    rows, hidden, chunks = 1024, 128, 5
    active = (True, retention, True, readers == 2)
    lengths = tuple(streams + 3 if on else 0 for on in active)
    # Dyadic values keep every sum and product exactly representable in FP32,
    # so the independent torch formula is a bitwise oracle for either tree.
    partial = torch.arange((1 + readers + retention) * streams * chunks, device="cuda")
    partial = (partial.remainder(7).float() / 8).reshape(-1, chunks)
    coefficients = (torch.arange(8 * streams, device="cuda").remainder(3).float() + 1) / 8
    gamma_partial = tuple(
        (
            (torch.arange(rows // 32 * hidden, device="cuda").remainder(7).float() + reader) / 8
        ).reshape(rows // 32, hidden)
        for reader in range(readers)
    )
    planes = (0, 1, 1 + retention, 2 + retention)
    for fused_gamma in (False, True):
        outputs = tuple(
            torch.empty(length, device="cuda") if length else None for length in lengths
        )
        gamma_grads = tuple(torch.empty(hidden, device="cuda") for _ in range(readers))
        jobs, expected = [], []
        for controller, output in enumerate(outputs):
            if output is not None:
                start = planes[controller] * streams
                factors = coefficients[(4 + controller) * streams : (5 + controller) * streams]
                jobs.append((partial[start : start + streams], output, 1, 1, 0, factors))
                reference = torch.zeros_like(output)
                reference[:streams] = partial[start : start + streams].sum(1) * factors
                expected.append((output, reference))
        if fused_gamma:
            jobs = [(p, out, 0, 8, 0, None) for p, out in zip(gamma_partial, gamma_grads)] + jobs
            expected.extend((out, part.sum(0)) for out, part in zip(gamma_grads, gamma_partial))
        kernel_utils.reduce_jobs(jobs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            kernel_utils.reduce_jobs(jobs)
        actual, references = zip(*expected)
        replay_poisoned(graph, actual, actual, references)


# ---------------------------------------------------------------- checkpoint replay


@pytest.mark.parametrize("depth", [1, 3, 16])
@pytest.mark.parametrize("streams", [1, 3, 6])
@pytest.mark.parametrize("columns", [False, True])
def test_interval_replay_matches_forward_bits(depth, streams, columns):
    """Mixed one-/two-reader intervals replay original states and overwrite all MX padding."""
    seeded()
    checkpoint = torch.randn(128, streams * 96, device="cuda")
    current, states, tapes = checkpoint, [checkpoint], []
    for step in range(depth):
        _, updates, controllers, gammas = inputs(streams, 1 + step % 2, 96, 128, seed=917 + step)
        retain = controllers[1] if step % 3 else None
        current, saved, _ = forward_kernel.forward(
            current, updates, controllers[0], retain, controllers[2:], gammas, num_streams=streams
        )
        tapes.append(saved)
        states.append(current)
    terminal = torch.empty_like(checkpoint)
    previous, operands = replay_interval(
        checkpoint, tapes, prepare_columns=columns, terminal_output=terminal
    )
    assert previous[0] is checkpoint
    all_bytes_equal((*previous, terminal), states)
    if columns:
        # TE independently quantizes the matched FP32 forward values.
        for before, tape, operand in zip(states, tapes, operands):
            routes = (True,) + (False,) * (tape.reader_count - 1)
            none = (None,) * tape.reader_count
            reference = standard_backward(tape, before, none, None, prepare_wgrad=routes)[1][0]
            assert_columns_equal((operand,), (reference,))
    expected = [*states[1:], *(t.clone() for t in workspace_tensors(operands))]

    def execute():
        replay_interval(
            checkpoint,
            tapes,
            outputs=previous[1:],
            columns=operands,
            prepare_columns=columns,
            terminal_output=terminal,
        )
        return (*previous[1:], terminal, *workspace_tensors(operands))

    assert_poisoned_graph(execute, expected)


# ---------------------------------------------------------------- deferred MXFP8 weight gradient


@mxfp8_test
@pytest.mark.parametrize("readers", [1, 2])
@pytest.mark.parametrize("other_stream", [False, True])
def test_deferred_wgrad_order_and_replay(monkeypatch, readers, other_stream):
    """dW is submitted before parameter reductions but publishes weight readiness only after
    it completed; moving it ahead preserves every gradient bit across replays."""
    expected = None
    for _ in range(3):
        raw = mx_inputs(2048, 2048, readers)
        layer = linear(2048, mcore=True)
        out = boundary(raw, layer)
        provider = reader_for(out[1])
        consumer = torch.cuda.Stream() if other_stream else torch.cuda.current_stream()
        consumer.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(consumer):
            y = consume(layer, out[1])
        torch.cuda.current_stream().wait_stream(consumer)
        order, observed = [], []
        compute_wgrad = Reader.compute_wgrad

        def submit(self, current, column):
            assert self is provider and column is not None
            assert self.fused_calls == 1 and not order
            compute_wgrad(self, current, column)
            order.append("wgrad")

        def ready(weight):
            assert order == ["wgrad", "controller_chunks", "parameters"]
            order.append("ready")
            observed.append(weight.main_grad.clone())

        outputs = (y, out[0], *out[2:])
        with monkeypatch.context() as patcher:
            if readers == 2:
                layer.weight.register_post_accumulate_grad_hook(ready)
                patcher.setattr(Reader, "compute_wgrad", submit)
                ordered_reductions(patcher, order, "wgrad")
            torch.autograd.backward(outputs, tuple(torch.ones_like(t) for t in outputs))
        if readers == 2:
            assert order == ["wgrad", "controller_chunks", "parameters", "ready"]
            assert torch.equal(observed[0], layer.weight.main_grad)
            assert torch.count_nonzero(observed[0]) > 0
        assert provider.reconstructions == 1 and not provider.pending
        actual = leaf_grads(raw, layer.weight.main_grad)
        if expected is None:
            expected = [t.clone() for t in actual]
        all_bytes_equal(actual, expected)
