from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest
import torch

import megatron.lite.primitive.parallel.pipeline as pipeline_mod
from megatron.lite.primitive.parallel.pipeline import _1f1b_schedule
from megatron.lite.primitive.parallel.state import ParallelState

pytestmark = pytest.mark.mlite


class _DummyStage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.current_input: torch.Tensor | None = None
        self.forward_inputs: list[torch.Tensor] = []

    def set_input_tensor(self, tensor: torch.Tensor) -> None:
        self.current_input = tensor
        self.forward_inputs.append(tensor)


class _FakePipelineP2P:
    """Stand-in for `_send_recv_pipeline` that fills every fwd receive with a
    monotonically increasing value so the test can tell which microbatch's
    activation ends up in a retained input tensor."""

    def __init__(self, *, force_clone: bool | None = None):
        self.force_clone = force_clone
        self.counts: Counter[str] = Counter()
        self.clone_recv_args: list[bool] = []

    def __call__(
        self,
        send_fwd: torch.Tensor | None,
        send_bwd: torch.Tensor | None,
        recv_fwd: bool,
        recv_bwd: bool,
        ps: ParallelState,
        tensor_shape: tuple[int, ...],
        *,
        fwd_recv_buf: torch.Tensor | None = None,
        bwd_recv_buf: torch.Tensor | None = None,
        batch_p2p: bool = True,
        clone_recv: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        del ps, batch_p2p
        self.clone_recv_args.append(clone_recv)
        clone = self.force_clone if self.force_clone is not None else clone_recv

        if send_fwd is not None:
            self.counts["send_fwd"] += 1
        if send_bwd is not None:
            self.counts["send_bwd"] += 1

        fwd_buf = None
        bwd_buf = None
        if recv_fwd:
            self.counts["recv_fwd"] += 1
            fwd_buf = fwd_recv_buf if fwd_recv_buf is not None else torch.empty(tensor_shape)
            with torch.no_grad():
                fwd_buf.fill_(float(self.counts["recv_fwd"]))
            if clone:
                fwd_buf = fwd_buf.clone()
            fwd_buf.grad = None
            fwd_buf.requires_grad_()
        if recv_bwd:
            self.counts["recv_bwd"] += 1
            bwd_buf = bwd_recv_buf if bwd_recv_buf is not None else torch.empty(tensor_shape)
            with torch.no_grad():
                bwd_buf.fill_(1.0)
            if clone:
                bwd_buf = bwd_buf.clone()

        return fwd_buf, bwd_buf


def _patch_pipeline_for_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    orig_empty = torch.empty

    def _cpu_empty(*args, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("device", None)
        return orig_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", _cpu_empty)
    monkeypatch.setattr(pipeline_mod, "_PIPELINE_TENSOR_DTYPE", torch.float32)


def _parallel_state(pp_size: int, pp_rank: int) -> ParallelState:
    return ParallelState(
        pp_size=pp_size,
        pp_rank=pp_rank,
        pp_is_first=(pp_rank == 0),
        pp_is_last=(pp_rank == pp_size - 1),
        pp_prev_rank=pp_rank - 1,
        pp_next_rank=pp_rank + 1,
    )


def _forward_step(model: _DummyStage, batch: dict[str, float]) -> dict:
    assert model.current_input is not None
    hidden = model.current_input * batch["scale"]
    return {"hidden_states": hidden, "model_output": batch["id"]}


def _run_1f1b_schedule(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pp_size: int,
    pp_rank: int,
    num_microbatches: int,
    force_clone: bool | None = None,
) -> tuple[_DummyStage, _FakePipelineP2P, list[dict]]:
    _patch_pipeline_for_cpu(monkeypatch)
    transport = _FakePipelineP2P(force_clone=force_clone)
    monkeypatch.setattr(pipeline_mod, "_send_recv_pipeline", transport)

    model = _DummyStage()
    batches = iter({"id": i, "scale": float(i + 2)} for i in range(num_microbatches))
    outputs = _1f1b_schedule(
        _forward_step,
        model,
        batches,
        num_microbatches,
        SimpleNamespace(),
        _parallel_state(pp_size, pp_rank),
        tensor_shape=(2, 3),
    )
    return model, transport, outputs


# (pp_size, pp_rank, num_microbatches).  num_warmup W = pp_size - pp_rank - 1.
# Cross-covers:
#   * the boundary case the prefetch fix addressed (W=1);
#   * the historically silent-corruption case (W>=2);
#   * num_microbatches < pp_size, == pp_size, and > pp_size;
#   * multiple pp_ranks per pp_size and multiple pp_sizes.
# Every case targets a middle (non-first, non-last) stage so all four p2p
# directions must transfer exactly num_microbatches tensors.
_SCHEDULE_CASES: list[tuple[int, int, int]] = [
    # ── pp_size=4 ──
    (4, 2, 1),  # W=1, num_microbatches < pp_size (warmup only, no steady)
    (4, 2, 2),  # W=1, exercises warmup->steady transition
    (4, 2, 5),  # W=1, num_microbatches > pp_size
    (4, 1, 2),  # W=2, num_microbatches == W (all warmup, no steady)
    (4, 1, 3),  # W=2, three retained inputs; alias would silently corrupt grads
    (4, 1, 4),  # W=2, num_microbatches == pp_size
    (4, 1, 5),  # W=2, num_microbatches > pp_size
    # ── pp_size=5 (deeper pipeline) ──
    (5, 1, 3),  # W=3, num_microbatches < pp_size
    (5, 1, 5),  # W=3, num_microbatches == pp_size
    (5, 1, 7),  # W=3, num_microbatches > pp_size
    (5, 2, 4),  # W=2 mid-rank
    (5, 3, 6),  # W=1 penultimate rank, num_microbatches > pp_size
]

_MULTI_MB_CASES = [c for c in _SCHEDULE_CASES if c[2] >= 2]


@pytest.mark.parametrize(("pp_size", "pp_rank", "num_microbatches"), _SCHEDULE_CASES)
def test_1f1b_schedule_balances_p2p_and_never_forwards_none(
    monkeypatch, pp_size: int, pp_rank: int, num_microbatches: int
):
    model, transport, outputs = _run_1f1b_schedule(
        monkeypatch,
        pp_size=pp_size,
        pp_rank=pp_rank,
        num_microbatches=num_microbatches,
    )

    # Every microbatch reaches forward with a non-None input, i.e. the
    # warmup->steady transition never leaves fwd_input dangling even when
    # num_microbatches < pp_size.
    assert len(model.forward_inputs) == num_microbatches
    assert all(tensor is not None for tensor in model.forward_inputs)

    # Adjacent forward and backward send/receive counts stay balanced.
    for op in ("send_fwd", "recv_fwd", "send_bwd", "recv_bwd"):
        assert transport.counts[op] == num_microbatches, (
            f"{op}={transport.counts[op]} != num_microbatches={num_microbatches} "
            f"(pp_size={pp_size}, pp_rank={pp_rank})"
        )

    # The local `_p2p` wrapper must propagate clone_recv=True on every call so
    # that retained inputs get fresh storage instead of aliasing _fwd_recv_buf.
    assert transport.clone_recv_args and all(transport.clone_recv_args)

    assert len(outputs) == num_microbatches


@pytest.mark.parametrize(("pp_size", "pp_rank", "num_microbatches"), _MULTI_MB_CASES)
def test_1f1b_schedule_clones_retained_forward_inputs(
    monkeypatch, pp_size: int, pp_rank: int, num_microbatches: int
):
    model, _transport, _outputs = _run_1f1b_schedule(
        monkeypatch,
        pp_size=pp_size,
        pp_rank=pp_rank,
        num_microbatches=num_microbatches,
    )

    assert len(model.forward_inputs) == num_microbatches

    # Every simultaneously-retained forward input must live in its own storage,
    # not alias into the shared _fwd_recv_buf.
    ptrs = [t.data_ptr() for t in model.forward_inputs]
    assert len(set(ptrs)) == len(ptrs), f"aliased retained inputs: {ptrs}"

    # The i-th retained input must still carry the value written by the i-th
    # forward receive (1, 2, 3, ...), i.e. no later receive overwrote it.
    for i, tensor in enumerate(model.forward_inputs):
        expected = float(i + 1)
        assert torch.all(tensor == expected), (
            f"forward_inputs[{i}] expected fill={expected}, got {tensor.tolist()}"
        )


@pytest.mark.parametrize(("pp_size", "pp_rank", "num_microbatches"), _MULTI_MB_CASES)
def test_1f1b_schedule_no_clone_negative_control_aliases_inputs(
    monkeypatch, pp_size: int, pp_rank: int, num_microbatches: int
):
    """Force clone_recv=False in the fake transport to reproduce the alias:
    without the clone every retained input references the same _fwd_recv_buf,
    so by the time the last receive completes they all read the last value."""
    model, transport, _outputs = _run_1f1b_schedule(
        monkeypatch,
        pp_size=pp_size,
        pp_rank=pp_rank,
        num_microbatches=num_microbatches,
        force_clone=False,
    )

    assert len(model.forward_inputs) == num_microbatches

    ptrs = {t.data_ptr() for t in model.forward_inputs}
    assert len(ptrs) == 1, f"expected all retained inputs to alias, got ptrs={ptrs}"

    last_recv_value = float(transport.counts["recv_fwd"])
    for tensor in model.forward_inputs:
        assert torch.all(tensor == last_recv_value), (
            f"expected aliased buf to hold last recv value {last_recv_value}, "
            f"got {tensor.tolist()}"
        )
