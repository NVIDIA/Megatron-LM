# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real forward schedules with CPU transport/stream doubles, not GPU parity."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

# Initialize Core before installing the CPU-only TE import double.
import megatron.core  # noqa: F401


@pytest.mark.parametrize("chunk_count", [2, 3, 4])
@pytest.mark.parametrize("rows", [0, 1, 3])
@pytest.mark.parametrize("saved_context", [False, True])
@pytest.mark.parametrize("peer_rows", [0, 2])
def test_small_rank_forward_matches_ordinary_ep(
    chunk_count, rows, saved_context, peer_rows, monkeypatch, transformer_engine_import_stub
):
    transformer_engine_import_stub()
    from megatron.lite.primitive.modules import moe_ep_chunk_overlap as overlap

    calls = []

    class Router(torch.nn.Module):
        def forward(self, x):
            return x[:, :1].sigmoid(), torch.zeros((len(x), 1), dtype=torch.long)

    class Experts(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[2.0, 1.0], [-1.0, 3.0]]))

        def forward(self, x, counts, probs, **kwargs):
            return (x @ self.weight) * probs

    class Dispatcher:
        moe_permute_fusion = False

        def submit_deepep_dispatch(self, x, scores, indices, **kwargs):
            calls.append(len(x))
            self.local_rows = len(x)
            return {
                "recv_hidden": torch.cat((x, torch.ones(peer_rows, 2))),
                "recv_probs": torch.cat((scores, torch.ones(peer_rows, 1))),
                "handle": len(x),
            }

        def finish_deepep_dispatch(self, state, **kwargs):
            return (state["recv_hidden"], [len(state["recv_hidden"])], state["recv_probs"])

        def finish_deepep_dispatch_for_backward(self, state):
            x, counts, scores = self.finish_deepep_dispatch(state)
            ids = torch.arange(len(x))
            return (
                x,
                counts,
                scores,
                {
                    "local_tpe_list": counts,
                    "manual_row_id_map": ids,
                    "manual_prob_flat_indices": ids,
                },
            )

        def prepare_deepep_combine(self, output):
            return output, self.local_rows

        def submit_deepep_combine_prepared(self, output, handle, **kwargs):
            return output[:handle]

        def finish_deepep_combine(self, state):
            return state

    profile = overlap.EPChunkShapeProfile(
        max_input_rows=8, hidden_size=2, topk=1, ep_size=2, chunk_count=chunk_count
    )
    workspace = SimpleNamespace(
        key=SimpleNamespace(op="forward", shape_profile=profile),
        acquire=lambda *args, **kwargs: SimpleNamespace(
            dispatcher=Dispatcher(), check_active=lambda: None, release=lambda event: None
        ),
        acquire_expert_activation=lambda **kwargs: SimpleNamespace(
            tensor=lambda name, shape, **kw: torch.empty(shape, **kw),
            allocate=nullcontext,
            release=lambda event: None,
        ),
    )

    def event():
        return SimpleNamespace(record=lambda stream: None)

    stream = SimpleNamespace(wait_event=lambda event: None, record_event=event)
    monkeypatch.setattr(overlap.torch.cuda, "Event", event)
    monkeypatch.setattr(overlap.torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(overlap.torch.cuda, "stream", lambda *args: nullcontext())
    monkeypatch.setattr(overlap._EPChunkOperationBase, "_streams", lambda *args: (stream, stream))
    monkeypatch.setattr(overlap, "unpermute", lambda output, *args, **kwargs: output)
    router, experts = Router(), Experts()
    backward = overlap.EPChunkBackwardOp(router=router, experts=experts, workspace=workspace)
    op = overlap.EPChunkForwardOp(
        router=router, experts=experts, workspace=workspace, backward_op=backward
    )
    x = torch.arange(rows * 2, dtype=torch.float32).reshape(rows, 2).requires_grad_(True)
    # Ordinary EP: one router -> dispatch -> expert -> combine, without chunks.
    native = Dispatcher()
    scores, indices = router(x)
    state = native.submit_deepep_dispatch(x, scores, indices)
    received, counts, probs = native.finish_deepep_dispatch(state)
    output, handle = native.prepare_deepep_combine(experts(received, counts, probs))
    expected = native.finish_deepep_combine(native.submit_deepep_combine_prepared(output, handle))
    calls.clear()
    with torch.set_grad_enabled(saved_context):
        actual = op.forward(x)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert actual.shape == x.shape
    # Empty ranks still participate in every collective, even receiving peer work.
    assert len(calls) == profile.chunk_count
    assert sum(calls) == rows
