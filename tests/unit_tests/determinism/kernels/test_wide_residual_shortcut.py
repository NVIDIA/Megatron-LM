# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Bitwise determinism of the fused wide-residual shortcut seams.

Poisoned CUDA-graph replay of the forward, inference, backward and final reductions, backward
scratch ownership, the checkpoint interval replay kernel, and the deferred MXFP8 weight-gradient
order. Shared inputs, runners and the FP64 reference live in
``tests/unit_tests/fusions/test_wide_residual_shortcut.py``.
"""

import itertools
from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from megatron.core.fusions.wide_residual_shortcut import (
    api,
    backward_kernel,
    forward_kernel,
    kernel_utils,
)
from megatron.core.fusions.wide_residual_shortcut.mxfp8_handoff import Reader, eligible, reader_for
from megatron.core.fusions.wide_residual_shortcut.replay_kernel import replay_interval
from megatron.core.inference.utils import InferenceMode
from tests.unit_tests.determinism.kernels.harness import seeded
from tests.unit_tests.fusions.test_wide_residual_shortcut import (  # noqa: F401 (reader_work)
    BF16,
    F32,
    FLAGSHIPS,
    GPU,
    MX_SHAPES,
    QUALIFIED,
    REQUIRES_FUSED_BACKWARD,
    SEAMS,
    F,
    T,
    all_bytes_equal,
    assert_columns_equal,
    assert_matches_fp64,
    backward_args,
    boundary,
    buffers,
    consume,
    folded_reader,
    fp64_backward,
    fwd,
    inputs,
    invoke,
    leaf_grads,
    linear,
    mx_inputs,
    mxfp8_test,
    offset_view,
    ordered_reductions,
    poison,
    poison_bytes,
    reader_work,
    reference,
    reference_columns,
    replay_poisoned,
    run_with_scratch,
    te,
    workspace_tensors,
    written,
)

pytestmark = GPU


def graph_replay(execute, expected, fill=poison_bytes, stream=None):
    """Capture ``execute() -> (scratch, outputs)``; every replay after poisoning ``scratch``
    must rewrite ``outputs`` to ``expected`` byte for byte."""
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        scratch, outputs = execute()
    replay_poisoned(graph, scratch, outputs, expected, fill)


def taped_forward(forward, *args, **kwargs):
    """Materialize the tape with the inference schedule (one CTA per row), as an oracle."""
    choose = forward_kernel.choose_forward_config

    def same_schedule(m, h, streams, readers=1, **options):
        cfg = choose(m, h, streams, readers, **(options | {"inference": True}))
        paired = cfg.PAIRED_REDUCE and not cfg.SPLIT
        return cfg._replace(PARTS=1, SPLIT=False, PAIRED_REDUCE=paired)

    with patch.object(forward_kernel, "choose_forward_config", side_effect=same_schedule):
        return forward(*args, **(kwargs | {"inference": False}))


def forbid_backward_state(monkeypatch):
    """Fail if inference enters autograd or allocates backward state; returns a list that
    gains one entry per non-warmup ``_forward`` launch."""

    def forbidden(*args, **kwargs):
        raise AssertionError("inference entered autograd or allocated backward state")

    monkeypatch.setattr(api._Boundary, "apply", forbidden)
    monkeypatch.setattr(api, "current_manager", forbidden)
    monkeypatch.setattr(forward_kernel, "SavedWrite", forbidden)
    launch = forward_kernel._forward.run
    observed = []

    def inspect(*args, **kwargs):
        if not kwargs.get("warmup"):
            write, readers = args[0], kwargs["read_tensors"]
            assert kwargs["saved_coefficients"] is None and write.summed_update is None
            assert all(stat is None for stat in readers.rstd)
            observed.append(True)
        return launch(*args, **kwargs)

    monkeypatch.setattr(forward_kernel._forward, "run", inspect)
    return observed


# ---------------------------------------------------------------- poisoned graph replay

# fmt: off
# forward: (tokens, hidden, streams, readers, updates, inference, quantized, retention, dtype,
# epsilon, offset residual, MXFP8 reader checked against FP64)
FORWARD = [
    # Production and heuristic training schedules.
    pytest.param(m, h, 3, n, u, F, q, t, F32, 1e-5 if q else 1e-3, F, F, marks=QUALIFIED)
    for m, h, n, u, q, t in [
        (16384, 3072, 1, 1, T, F), (8192, 4096, 2, 1, T, T), (32, 6144, 1, 1, T, F),
        (8192, 10240, 1, 2, F, T), (2048, 2048, 1, 2, F, F), (12288, 13312, 2, 1, T, F),
        (2048, 16384, 1, 2, F, T)]
] + [
    # Widths 1 (< 32) and 1025 (odd) use scalar loads; 48, 288, 2052 masked packets; 18432 and
    # 32768 the widest profiles and column partitions. Widths that are not a multiple of four
    # use an offset residual; odd stream counts drop retention.
    (33 if inf else 64, h, s, *SEAMS[(i + s) % 4], inf, F, not s % 2, F32, 1e-5, h % 4 > 0, F)
    for i, h in enumerate((1, 48, 288, 1025, 2052, 18432, 32768))
    for s in range(1, 7)
    for inf in [(i + s) % 2 == 0]
] + [
    # MXFP8 covers generic streams and initializes both row and column scale padding.
    pytest.param(m, h, s, 2, u, inf, T, T, F32, 1e-5, F, T, marks=QUALIFIED)
    for i, (m, h) in enumerate(((32, 32), (64, 96), (96, 288), (160, 18432)))
    for s, u, inf in [((1, 2, 4, 6)[i], 1 + i % 2, i % 2 == 0), ((6, 4, 2, 1)[i], 2 - i % 2, i < 2)]
] + [
    # Every tuned training table entry: FP32 streams, and BF16 streams with BF16 logits.
    pytest.param(2048, h, 3, n, n, F, q, T, F32, 1e-5, F, T, marks=QUALIFIED if q else ())
    for h, n, q in sorted(forward_kernel._FORWARD_CONFIGS)
] + [
    pytest.param(2048, h, 3, n, u, F, q, T, BF16, 1e-5, F, T, marks=QUALIFIED if q else ())
    for h, n, u, q in sorted(forward_kernel._FORWARD_CONFIGS_BF16)
]
# inference: (tokens, hidden, quantized, retention, readers, updates, dtype, TE consumer).
# Tokens: cluster/tree schedules, the MXFP8 scale padding of the last 128-row tile (one or
# several tiles, or none), and the inference_mode (tokens % 64 == 0) or no_grad entry. Decode
# rows at 128/7168/14336 widths exercise fallback profiles; 10240 forces runtime retention.
INFERENCE = [
    (m, h, q, t, *SEAMS[i % 4], F32, F)
    for i, (m, h, q, t) in enumerate(
        [(m, h, q, (m // 32 + q) % 3 > 0)
         for m, h, q in itertools.product((32, 96, 160, 992, 1024), (3072, 4096, 10240), (F, T))]
        + [(m, h, F, T) for m in (1, 17, 65) for h in (128, 4096, 7168, 14336)]
    )
] + [
    # Replica classes 4/2/1, split two-reader schedules, BF16 streams, mid-width profiles.
    (m, h, q, T, *SEAMS[(i + j) % 4], d, F)
    for i, h in enumerate((3072, 4096, 7168, 10240, 13312, 14336))
    for j, (m, q, d) in enumerate(
        [((32, 64, 96)[i % 3], i % 2 == 0, F32), ((320, 1024)[i % 2], i % 2 == 1, BF16)]
        + [((64, 96, 32)[i % 3], T, (F32, BF16)[i % 2])]
    )
] + [
    # A Transformer Engine MXFP8 linear consumes the handoff like the taped one.
    (m, h, T, T, n, u, F32, T)
    for m, h, n, u in [(32, 3072, 1, 1), (64, 4096, 2, 2), (96, 10240, 2, 1), (1024, 3072, 1, 2)]
    + [(1024, 10240, 2, 2)]
]
# backward: (streams, readers, hidden, rows, retention, grid, routes, updates, BF16 residual).
# Column routes only prepend _replay_mx launches (per reader index); explicit grids include
# idle groups that must skip their bootstrap transfers.
EXPLICIT_GRIDS = [(32, 1), (96, 1), (128, 64), (288, 4), (4128, 34), (4096, 32)]
EXPLICIT_GRIDS += [(4096, 960), (3072, 24), (10240, 80)]
_BF16_BACKWARD = [(3, 4128, 1024, "auto"), (6, 96, 160, "auto"), (2, 3072, 1024, 24)]
_BF16_BACKWARD += [(3, h, m, "auto") for m, h in FLAGSHIPS]
BACKWARD = (
    [
        ((1, 2, 4, 5, 6)[i % 5], n, h, 1024, i % 3 > 0, grid, (n == 2,) * n, n, F)
        for i, (h, grid) in enumerate(EXPLICIT_GRIDS)
        for n in [1 + i % 2]
    ]
    # Dynamic row admission at production and expanded shapes resets all queue epochs.
    + [
        (3, 1 + i % 2, h, m, i % 3 > 0, "auto", (i % 2 == 0, F)[: 1 + i % 2], 1 + i % 2, F)
        for i, (m, h) in enumerate(MX_SHAPES + [(4096, 13312)])
    ]
    # BF16 residual streams (BF16 tiles and gradients), partial widths (4128 columns), partial
    # row tiles with padded MXFP8 scales, and two readers of a single write.
    + [
        (s, SEAMS[i % 4][0], h, m, T, grid, (T, F)[: SEAMS[i % 4][0]], SEAMS[i % 4][1], T)
        for i, (s, h, m, grid) in enumerate(_BF16_BACKWARD)
    ]
)
# fmt: on
# reduction: (streams, readers, retention)
REDUCTION = [(1, 1, F), (3, 2, T), (6, 1, T), (6, 2, F)]


def _cases(kind, cases, marks=()):
    for i, case in enumerate(cases):
        values, case_marks = (case.values, case.marks) if hasattr(case, "values") else (case, ())
        if kind == "inference" and values[2]:
            case_marks = QUALIFIED
        yield pytest.param(kind, values, id=f"{kind}-{i}", marks=case_marks or marks)


@pytest.mark.parametrize(
    "kind,case",
    [
        *_cases("forward", FORWARD),
        *_cases("inference", INFERENCE),
        *_cases("backward", BACKWARD, QUALIFIED),
        *_cases("reduction", REDUCTION),
    ],
)
def test_graph_replay_restores_poisoned_outputs(monkeypatch, kind, case):
    """Each captured launch rewrites every poisoned output (and scratch) byte on replay: the
    forward (against FP64, with new inputs between replays), Megatron inference (bit for bit
    against the taped forward, without backward state), the backward and final reductions."""
    if kind == "forward":
        tokens, hidden, streams, readers, updates, inference, quantized, *rest = case
        retention, dtype, epsilon, offset, dequantize = rest
        raw = inputs(
            streams, readers, hidden, tokens, retention=retention, updates=updates,
            residual_dtype=dtype, logit_dtype=dtype,
        )  # fmt: skip
        if offset:
            raw = (offset_view(raw[0]), *raw[1:])
        options = dict(inference=inference, mxfp8_handoff=quantized, epsilon=epsilon)
        with torch.no_grad():
            current, _, operands = fwd(raw, streams, **options)
            fp64 = reference(raw, streams, epsilon)[0]
        exact = (2e-6, 2e-6) if dtype == F32 else (1 / 128, 1e-6)
        torch.testing.assert_close(current, fp64[0], rtol=exact[0], atol=exact[1])
        for a, b in zip(operands, fp64[1:], strict=True):
            if type(a) is torch.Tensor:
                torch.testing.assert_close(a, b, rtol=1 / 128, atol=1 / 128)
        execute = lambda: written(*fwd(raw, streams, **options))
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

        graph_replay(lambda: (lambda outputs: (outputs, outputs))(execute()), expected, refresh)
        if quantized and dequantize:
            relative = (operands[0].dequantize().float() - fp64[1].float()).norm()
            assert relative / fp64[1].float().norm() < 0.04
    elif kind == "inference":
        tokens, hidden, quantized, retention, readers, updates, dtype, consumer = case
        raw = inputs(
            3, readers, hidden, tokens, retention=retention, updates=updates,
            residual_dtype=dtype, logit_dtype=dtype,
        )  # fmt: skip
        current, _, operands = taped_forward(fwd, raw, mxfp8_handoff=quantized)
        taped = [t.clone() for t in buffers((current, *operands))]
        layer = linear(hidden, mcore=True).eval() if consumer else None
        observed = forbid_backward_state(monkeypatch)
        with ExitStack() as stack:
            stack.enter_context((torch.inference_mode if tokens % 64 == 0 else torch.no_grad)())
            stack.enter_context(InferenceMode.active())
            if consumer:
                from transformer_engine.common.recipe import MXFP8BlockScaling

                stack.enter_context(te.fp8_autocast(fp8_recipe=MXFP8BlockScaling()))
                assert eligible(layer, raw[1][0], 3)
            actual = invoke(raw, quantized, layer)
            assert all(not t.requires_grad and t.grad_fn is None for t in actual)
            assert all(reader_for(t) is None for t in actual)
            assert actual[0].shape == (tokens, 1, 3 * hidden)
            all_bytes_equal(
                (a.reshape(-1) for a in buffers(actual)), (b.reshape(-1) for b in taped)
            )
            if consumer:
                result = consume(layer, actual[1])
                assert torch.equal(result, consume(layer, operands[0].view(tokens, 1, hidden)))
                assert not result.requires_grad and result.grad_fn is None
                assert torch.count_nonzero(layer.weight.main_grad) == 0
            execute = lambda: list(buffers(invoke(raw, quantized, layer)))
            graph_replay(
                lambda: (lambda outputs: (outputs, outputs))(execute()),
                [t.clone() for t in buffers(actual)],
            )
        assert observed == [True, True]
    elif kind == "backward":
        streams, readers, hidden, rows, retention, grid, routes, updates, bf16 = case
        dtype = BF16 if bf16 else F32
        raw = inputs(
            streams, readers, hidden, rows, retention=retention, updates=updates,
            residual_dtype=dtype, logit_dtype=dtype,
        )  # fmt: skip
        _, args = backward_args(raw, streams)
        options = dict(config=grid, save_columns=routes)
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

        def captured():
            ws = execute()
            return ws.allocations, workspace_tensors((ws.grads, ws.columns))

        graph_replay(captured, expected, poison, side)
    else:
        streams, readers, retention = case
        rows, hidden, chunks = 1024, 128, 5
        active = (True, retention, True, readers == 2)
        lengths = tuple(streams + 3 if on else 0 for on in active)
        # Dyadic values keep every sum and product exactly representable in FP32,
        # so the independent torch formula is a bitwise oracle for either tree.
        partial = torch.arange((1 + readers + retention) * streams * chunks, device="cuda")
        partial = (partial.remainder(7).float() / 8).reshape(-1, chunks)
        coefficients = (torch.arange(8 * streams, device="cuda").remainder(3).float() + 1) / 8
        base = torch.arange(rows // 32 * hidden, device="cuda").remainder(7).float()
        gamma_partial = tuple(((base + r) / 8).reshape(rows // 32, hidden) for r in range(readers))
        planes = (0, 1, 1 + retention, 2 + retention)
        for fused_gamma in (False, True):
            outputs = tuple(torch.empty(n, device="cuda") if n else None for n in lengths)
            gamma_grads = tuple(torch.empty(hidden, device="cuda") for _ in range(readers))
            jobs, expected = [], []
            for controller, output in enumerate(outputs):
                if output is not None:
                    start = planes[controller] * streams
                    factors = coefficients[(4 + controller) * streams : (5 + controller) * streams]
                    jobs.append((partial[start : start + streams], output, 1, 1, 0, factors))
                    reference_ = torch.zeros_like(output)
                    reference_[:streams] = partial[start : start + streams].sum(1) * factors
                    expected.append((output, reference_))
            if fused_gamma:
                jobs = [
                    (p, out, 0, 8, 0, None) for p, out in zip(gamma_partial, gamma_grads)
                ] + jobs
                expected.extend((out, part.sum(0)) for out, part in zip(gamma_grads, gamma_partial))
            kernel_utils.reduce_jobs(jobs)
            actual, references = zip(*expected)
            execute = lambda: (kernel_utils.reduce_jobs(jobs), (actual, actual))[1]
            graph_replay(execute, references, poison)


@REQUIRES_FUSED_BACKWARD
@pytest.mark.parametrize("explicit_grid", [False, True])
@pytest.mark.parametrize("rows,hidden,columns", [(1024, 4096, False), (8192, 10240, True)])
def test_backward_owns_independent_scratch(explicit_grid, rows, hidden, columns):
    """Independent calls and captured graphs cannot alias inputs or each other's scratch."""
    current, args = backward_args(inputs(3, 2, hidden, rows))
    saved, _, dys, carry = args
    grid = hidden // 128 if explicit_grid else "auto"
    options = dict(save_columns=(columns, False), config=grid)
    protected = (current, carry, *dys, saved.update, saved.coefficients, *saved.rstd, *saved.gammas)
    originals = [tensor.clone() for tensor in protected]
    _, first = run_with_scratch(backward_kernel.backward, *args, **options)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, second = run_with_scratch(backward_kernel.backward, *args, **options)
    storage = lambda tensors: {tensor.untyped_storage().data_ptr() for tensor in tensors}
    first_storage, second_storage = (storage(workspace_tensors(ws)) for ws in (first, second))
    assert not first_storage & second_storage
    assert not (first_storage | second_storage) & storage(protected)
    for workspace in (first, second):
        tensors = list(workspace_tensors((workspace.grads, workspace.columns)))
        assert len(tensors) - len(storage(tensors)) == 1
        assert workspace.grads[1] is workspace.grads[2]
    outputs = lambda ws: [*workspace_tensors((ws.grads, ws.columns))]
    expected = outputs(first) + originals
    replay_poisoned(graph, workspace_tensors(second), outputs(second) + list(protected), expected)


# ---------------------------------------------------------------- checkpoint interval replay


@pytest.mark.parametrize(
    "depth,streams,columns",
    [(3, 3, F), (16, 1, F)]
    + [pytest.param(d, s, T, marks=QUALIFIED) for d, s in [(1, 3), (3, 6), (16, 6)]],
)
def test_interval_replay_matches_forward_bits(depth, streams, columns):
    """Mixed one-/two-reader intervals replay original states and overwrite all MX padding."""
    seeded()
    checkpoint = torch.randn(128, streams * 96, device="cuda")
    current, states, tapes = checkpoint, [checkpoint], []
    for step in range(depth):
        _, updates, controllers, gammas = inputs(streams, 1 + step % 2, 96, 128, seed=917 + step)
        retain = controllers[1] if step % 3 else None
        current, saved, _ = fwd(
            (current, updates, (controllers[0], retain, *controllers[2:]), gammas), streams
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
            assert_columns_equal((operand,), (reference_columns(tape, before),))
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
        outputs = (*previous[1:], terminal, *workspace_tensors(operands))
        return outputs, outputs

    graph_replay(execute, expected)


def _advance(previous, tape):
    """Emulate the forward's RN/FMA residual write in FP64."""
    m, h = tape.update.shape
    c = tape.coefficients.double().view(8, tape.num_streams)
    rows = torch.arange(m, device=previous.device)[:, None]
    result = []
    for stream, value in enumerate(previous.view(m, tape.num_streams, h).unbind(1)):
        retained = (value.double() * c[1, stream]).float().double()
        update = tape.update.double() * c[0, stream]
        fused = (stream == 0) | ((stream == 1) & (rows % 32 < 12))
        result.append(
            torch.where(
                fused, (retained + update).float(), (retained + update.float().double()).float()
            )
        )
    return torch.stack(result, 1).flatten(1)


@pytest.mark.parametrize(
    "value,depth,retain,readers,columns",
    [
        pytest.param(v, 3, 0.3, 1, c, marks=QUALIFIED if c else ())
        for v in (float("nan"), float("inf"), -float("inf"), 3e38, "subnormal")
        for c in (F, T)
    ]
    # 16-step intervals: two full plus a partial tail.
    + [
        pytest.param(None, 40, r, n, F, marks=REQUIRES_FUSED_BACKWARD)
        for r, n in [(0.0, 1), (0.3, 2)]
    ],
)
def test_interval_replay_matches_emulated_writes(value, depth, retain, readers, columns):
    """Replay matches the FP64-emulated forward write: exceptional values keep their class,
    finite values (and IEEE subnormals) their bits. Deep intervals (retain 0, FP32 and
    zero updates) give FP64 gradients from the replayed predecessors (no inverse reference)."""
    torch.manual_seed(1317 if value is None else 29)
    m, h, streams = 32, 96, 3
    coefficients = torch.ones(8, streams, device="cuda")
    coefficients[0].fill_(0.75)
    coefficients[1].fill_(retain)
    if value is None:
        previous = torch.randn(m, streams * h, device="cuda")
    elif value == "subnormal":
        # Normal FP32 residuals near the subnormal boundary; retention pushes them below it.
        previous = (torch.rand(m, streams * h, device="cuda") + 1) * 2.0**-126
        update = torch.rand(m, h, device="cuda") * 2.0**-130
        update[:, ::2] = 0
    else:
        previous = torch.zeros(m, streams * h, device="cuda")
        previous[0, 0] = previous[31, h + 31] = value
        update = torch.ones(m, h, device="cuda")
        update[16, 64] = value
        coefficients[2].copy_(torch.tensor([0.3, 0.2, -0.1], device="cuda"))
    states, tapes = [previous], []
    for step in range(depth):
        if value is None:
            update = torch.randn(m, h, device="cuda", dtype=BF16 if step % 2 == 0 else F32)
            if step % 7 == 0:
                update.zero_()
            coefficients = torch.rand(8, streams, device="cuda")
            coefficients[0].mul_(2)
            coefficients[1].fill_(retain)
            gammas = tuple(torch.randn(h, device="cuda", dtype=BF16) for _ in range(readers))
            rstd = tuple(torch.empty(m, device="cuda") for _ in range(readers))
        elif not tapes:
            gammas, rstd = (torch.ones(h, device="cuda", dtype=BF16),), (
                torch.ones(m, device="cuda"),
            )
        tape = kernel_utils.SavedWrite(
            update,
            coefficients.flatten(),
            gammas,
            rstd,
            (streams,) * (2 + readers),
            num_streams=streams,
        )
        for reader in range(readers if value is None else 0):
            x = folded_reader(tape, states[-1], reader)
            tape.rstd[reader].copy_((x.double().square().mean(-1) + 1e-5).rsqrt().float())
        tapes.append(tape)
        states.append(_advance(states[-1], tape))
    if value == "subnormal":
        assert (states[-1] != 0).any() and (states[-1].abs() < 2.0**-126).all()
    carry = torch.randn_like(previous)
    for begin in reversed(range(0, depth, 16)):
        end = min(begin + 16, depth)
        terminal = torch.empty_like(previous)
        predecessors, _ = replay_interval(
            states[begin], tapes[begin:end], prepare_columns=columns, terminal_output=terminal
        )
        for actual, expected in zip((*predecessors, terminal), states[begin : end + 1]):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
            finite = torch.isfinite(expected)
            assert torch.equal(actual[finite].view(torch.uint8), expected[finite].view(torch.uint8))
        for offset in reversed(range(end - begin) if value is None else ()):
            tape, previous = tapes[begin + offset], predecessors[offset]
            dys = tuple(torch.randn(m, h, device="cuda", dtype=BF16) for _ in range(readers))
            raw = (previous, (tape.update,), (), tape.gammas)
            expected, bounds = fp64_backward(raw, streams, dys, carry, tape=tape)
            actual, _ = backward_kernel.backward(tape, previous, dys, carry)
            assert_matches_fp64(actual, expected, bounds, streams, first=(5e-4, 2e-4))
            carry = actual[0]


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
