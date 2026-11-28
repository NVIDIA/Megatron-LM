# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Sequence-chunked MLP/MoE calls (``mlp_chunks_for_training`` / ``mlp_chunks_for_prefill``).

``TransformerLayer._run_mlp_in_chunks`` only reads ``self.is_moe_layer`` and ``self.mlp``, so
these tests drive it with a stand-in layer and a toy MoE on CPU.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.transformer.moe.router_replay import (
    ChunkedRouterReplay,
    RouterReplay,
    RouterReplayAction,
)
from megatron.core.transformer.transformer_layer import TransformerLayer

H, E, K = 8, 6, 2


def setup_function():
    RouterReplay.global_router_replay_instances.clear()


def teardown_function():
    RouterReplay.global_router_replay_instances.clear()


class ToyRouter(torch.nn.Module):
    """Linear router that picks its top-k through ``RouterReplay`` like ``TopKRouter``."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Linear(H, E, bias=False)
        self.router_replay = RouterReplay()

    def forward(self, x):
        scores = torch.softmax(self.weight(x.reshape(-1, H)), dim=-1)

        def default(scores, topk, num_groups=None, group_topk=None):
            return torch.topk(scores, topk, dim=-1)

        return self.router_replay.get_replay_topk(scores, K, None, None, default)


class ToyMoE(torch.nn.Module):
    """Per-token MoE stand-in: output = x * (sum of the selected experts' probs), masked to 0.

    Like ``MoELayer``, it replays ``replay_topk_idx`` around its router call and, with
    ``moe_layer_recompute``, checkpoints that call (``calls`` then also logs the recompute).
    """

    def __init__(self, moe_layer_recompute=False):
        super().__init__()
        self.router = ToyRouter()
        self.moe_layer_recompute = moe_layer_recompute
        self.calls = []

    def forward(
        self, x, padding_mask=None, packed_seq_params=None, input_ids=None, replay_topk_idx=None
    ):
        def custom_forward(x):
            with RouterReplay.replay_indices_for_call(self.router.router_replay, replay_topk_idx):
                probs, indices = self.router(x)
            self.calls.append(
                {
                    "tokens": x.shape[0],
                    "x": x.detach().clone(),
                    "padding_mask": padding_mask,
                    "indices": indices.clone(),
                }
            )
            out = x * probs.sum(dim=-1).reshape(x.shape[0], x.shape[1], 1)
            if padding_mask is not None:  # [b, s], True = padding
                out = out * (~padding_mask).transpose(0, 1).unsqueeze(-1)
            return out

        if self.moe_layer_recompute and self.training:
            out = torch.utils.checkpoint.checkpoint(custom_forward, x, use_reentrant=True)
        else:
            out = custom_forward(x)
        return out, None


def _run_in_chunks(mlp, x, padding_mask, num_chunks, is_moe_layer=True, input_ids=None):
    layer = SimpleNamespace(is_moe_layer=is_moe_layer, mlp=mlp)
    return TransformerLayer._run_mlp_in_chunks(layer, x, padding_mask, input_ids, None, num_chunks)


@pytest.mark.parametrize("seq_len", [32, 30, 10, 5, 1])
@pytest.mark.parametrize("with_mask", [False, True])
def test_moe_makes_exactly_num_chunks_calls(seq_len, with_mask):
    """Every EP rank must make the same number of MoE calls, whatever its sequence length."""
    torch.manual_seed(0)
    num_chunks, batch_size = 8, 2
    mlp = ToyMoE()
    x = torch.randn(seq_len, batch_size, H)
    mask = (torch.rand(batch_size, seq_len) < 0.3) if with_mask else None

    ref, _ = mlp(x, padding_mask=mask)
    mlp.calls.clear()
    out, bias = _run_in_chunks(mlp, x, mask, num_chunks)

    torch.testing.assert_close(out, ref, rtol=0, atol=1e-6)
    assert bias is None
    assert len(mlp.calls) == num_chunks
    assert all(call["tokens"] >= 1 for call in mlp.calls)
    assert sum(call["tokens"] for call in mlp.calls) == max(seq_len, num_chunks)

    pad = max(0, num_chunks - seq_len)
    if pad:
        # The dummy tokens are masked, also when the caller passed no mask.
        real = mask if mask is not None else torch.zeros(batch_size, seq_len, dtype=torch.bool)
        expected = torch.cat([real, torch.ones(batch_size, pad, dtype=torch.bool)], dim=1)
        assert torch.equal(torch.cat([c["padding_mask"] for c in mlp.calls], dim=1), expected)
    elif mask is None:
        assert all(call["padding_mask"] is None for call in mlp.calls)


def test_dense_mlp_is_not_padded():
    """A dense MLP runs no EP collectives, so a short sequence just gets fewer chunks."""
    mlp = ToyMoE()
    out, _ = _run_in_chunks(mlp, torch.randn(5, 2, H), None, 8, is_moe_layer=False)
    assert out.shape == (5, 2, H)
    assert [call["tokens"] for call in mlp.calls] == [1] * 5
    assert all(call["padding_mask"] is None for call in mlp.calls)


def test_input_ids_are_split_and_padded_with_the_tokens():
    seen = []

    class IdMoE(ToyMoE):
        def forward(self, x, padding_mask=None, packed_seq_params=None, input_ids=None):
            seen.append(input_ids.clone())
            return super().forward(x, padding_mask=padding_mask)

    ids = torch.arange(10).reshape(2, 5)  # [b, s]
    _run_in_chunks(IdMoE(), torch.randn(5, 2, H), None, 8, input_ids=ids)
    assert len(seen) == 8
    assert torch.equal(torch.cat(seen, dim=1)[:, :5], ids)


@pytest.mark.parametrize("seq_len", [30, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_routing_replay_per_chunk(batch_size, seq_len):
    """Each chunk replays its own tokens' indices, forward and recompute; recording is whole."""
    torch.manual_seed(0)
    num_chunks = 4
    real_tokens = seq_len * batch_size
    x = torch.randn(seq_len, batch_size, H)
    mlp = ToyMoE()
    rr = mlp.router.router_replay
    target = torch.randint(0, E, (real_tokens, K))

    # Forward replay: the concatenated per-chunk indices are the micro-batch's target.
    rr.set_target_indices(target)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _run_in_chunks(mlp, x, None, num_chunks)
    assert len(mlp.calls) == num_chunks
    assert torch.equal(torch.cat([c["indices"] for c in mlp.calls])[:real_tokens], target)
    assert rr.target_topk_idx is target

    # Recompute pops the micro-batch's entry, split per chunk.
    mlp.calls.clear()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    _run_in_chunks(mlp, x, None, num_chunks)
    assert torch.equal(torch.cat([c["indices"] for c in mlp.calls])[:real_tokens], target)
    assert rr.replay_backward_list == []

    # Recording keeps every chunk's indices, not just the last chunk's.
    mlp.calls.clear()
    rr.clear_indices()
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    _run_in_chunks(mlp, x, None, num_chunks)
    recorded = rr.get_recorded_indices()
    assert recorded.shape[0] == real_tokens
    assert torch.equal(recorded, torch.cat([c["indices"] for c in mlp.calls])[:real_tokens])


def test_routing_replay_record_into_static_buffer():
    """With a static (CUDA-graph) buffer, the chunks' indices end up in it in order."""
    torch.manual_seed(0)
    x = torch.randn(12, 1, H)
    mlp = ToyMoE()
    rr = mlp.router.router_replay
    buffer = torch.full((16, K), -1, dtype=torch.long)
    rr.set_static_buffer(buffer)
    rr.set_router_replay_action(RouterReplayAction.RECORD)
    _run_in_chunks(mlp, x, None, 4)
    expected = torch.cat([c["indices"] for c in mlp.calls])
    assert torch.equal(buffer[:12], expected)
    assert torch.equal(rr.get_recorded_indices(), expected)


def test_chunked_router_replay_without_replay_is_a_no_op():
    replay = ChunkedRouterReplay(None, [3, 3], 0)
    assert replay.chunk_indices(0) is None
    replay.after_chunk()
    replay.finish()


def test_pad_without_real_tokens():
    # No real token to copy from: the dummies replay distinct experts 0..topk-1.
    padded = ChunkedRouterReplay._pad(torch.empty(0, K, dtype=torch.long), 3)
    assert torch.equal(padded, torch.arange(K).expand(3, -1))


def _forward_calls_by_input(calls):
    return {tuple(c["x"].flatten().tolist()): c["indices"] for c in calls}


@pytest.mark.parametrize("seq_len", [30, 3])
@pytest.mark.parametrize("batch_size", [1, 2])
def test_routing_replay_with_moe_recompute(batch_size, seq_len):
    """With selective "moe" recompute, backward reruns each chunk's checkpointed MoE call alone,
    not the chunk loop. Each rerun still replays its own chunk's slice."""
    torch.manual_seed(0)
    num_chunks = 4
    real_tokens = seq_len * batch_size
    mlp = ToyMoE(moe_layer_recompute=True).train()
    rr = mlp.router.router_replay
    target = torch.randint(0, E, (real_tokens, K))
    x = torch.randn(seq_len, batch_size, H, requires_grad=True)

    rr.set_target_indices(target)
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    out, _ = _run_in_chunks(mlp, x, None, num_chunks)
    forward_calls = list(mlp.calls)
    assert len(forward_calls) == num_chunks
    assert torch.equal(torch.cat([c["indices"] for c in forward_calls])[:real_tokens], target)
    # No rerun pops the micro-batch's entry, so it is dropped up front.
    assert rr.replay_backward_list == []

    mlp.calls.clear()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    out.sum().backward()
    assert len(mlp.calls) == num_chunks
    expected = _forward_calls_by_input(forward_calls)
    for rerun in mlp.calls:
        assert torch.equal(rerun["indices"], expected[tuple(rerun["x"].flatten().tolist())])
    assert rr.call_topk_idx is None


def test_replay_slice_is_bound_to_its_call():
    """A rerun replays the slice its call was given, whatever the backward order."""
    torch.manual_seed(0)
    mlp = ToyMoE(moe_layer_recompute=True).train()
    rr = mlp.router.router_replay
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    xs = [torch.randn(3, 1, H, requires_grad=True) for _ in range(3)]
    slices = [torch.randint(0, E, (3, K)) for _ in range(3)]
    outs = [mlp(x, replay_topk_idx=idx)[0] for x, idx in zip(xs, slices)]

    mlp.calls.clear()
    rr.set_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)
    order = (1, 2, 0)
    for i in order:
        outs[i].sum().backward()
    assert len(mlp.calls) == 3
    for call, i in zip(mlp.calls, order):
        assert torch.equal(call["indices"], slices[i])


def test_moe_recompute_keeps_earlier_micro_batches_entries():
    """Only this micro-batch's entry is dropped; earlier ones (e.g. pipeline 1F1B) stay queued."""
    mlp = ToyMoE(moe_layer_recompute=True).train()
    rr = mlp.router.router_replay
    earlier = torch.randint(0, E, (5, K))
    rr.set_target_indices(earlier)
    rr.set_target_indices(torch.randint(0, E, (8, K)))
    rr.set_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    _run_in_chunks(mlp, torch.randn(8, 1, H, requires_grad=True), None, 4)
    assert len(rr.replay_backward_list) == 1 and rr.replay_backward_list[0] is earlier


def test_quantile_balancing_rejects_padded_chunks():
    """Dummy tokens need a padding mask, which quantile balancing does not support yet."""
    mlp = ToyMoE()
    mlp.router.routing_type = "quantile_balancing"
    with pytest.raises(NotImplementedError, match="quantile-balancing"):
        _run_in_chunks(mlp, torch.randn(3, 2, H), None, 4)
    assert mlp.calls == []
    # A sequence of at least num_chunks tokens needs no padding and runs as before.
    _run_in_chunks(mlp, torch.randn(8, 2, H), None, 4)
    assert len(mlp.calls) == 4


@pytest.mark.parametrize(
    "action", [RouterReplayAction.REPLAY_FORWARD, RouterReplayAction.REPLAY_BACKWARD]
)
def test_replay_indices_must_match_the_token_count(action):
    """Extra or missing rows raise, as the unchunked gather does, instead of being sliced off."""
    rr = RouterReplay()
    rr.set_router_replay_action(action)
    for rows in (10, 6):
        rr.clear_indices()
        rr.set_target_indices(torch.randint(0, E, (rows, K)))
        with pytest.raises(ValueError, match=f"have {rows} rows, but this MoE layer routes 8"):
            ChunkedRouterReplay(rr, [4, 4])
    rr.clear_indices()
    rr.set_target_indices(torch.randint(0, E, (8, K)))
    assert ChunkedRouterReplay(rr, [4, 4]).chunk_indices(1).shape == (4, K)
    # Dummy tokens are not counted: 6 real tokens padded with 2.
    rr.clear_indices()
    rr.set_target_indices(torch.randint(0, E, (6, K)))
    assert ChunkedRouterReplay(rr, [4, 4], pad_tokens=2).chunk_indices(1).shape == (4, K)
