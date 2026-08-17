# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""CPU coverage for MLite R3 router replay."""

from types import SimpleNamespace

import torch
import torch.nn as nn


def test_replay_mask_uses_live_scores_and_keeps_native_unmasked_row():
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    dense = torch.tensor(
        [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], requires_grad=True
    )
    native_indices = torch.tensor([[2, 1], [0, 1]])
    native_scores = dense.gather(1, native_indices)
    target = torch.tensor([[0, 1], [2, 1]])
    RouterReplay.set_replay_data([target], replay_mask=torch.tensor([True, False]))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)

    scores, indices = replay.apply(dense, native_scores, native_indices)

    assert torch.equal(indices, torch.tensor([[0, 1], [0, 1]]))
    torch.testing.assert_close(scores, dense.gather(1, indices))
    scores.square().sum().backward()
    assert dense.grad is not None
    assert dense.grad.abs().sum() > 0


def test_replayed_sqrtsoftplus_scores_are_live_normalized_and_nonzero():
    from megatron.lite.primitive.modules.router_replay import (
        gather_replayed_router_scores,
    )

    logits = torch.tensor([[4.0, -3.0, 0.5, -1.0]], requires_grad=True)
    # Select experts that are not the natural top-2; a sparse native routing
    # tensor would contain zeros here, which was the PR49 half-finished bug.
    indices = torch.tensor([[1, 3]])
    scores = gather_replayed_router_scores(
        logits,
        indices,
        score_function="sqrtsoftplus",
        scaling_factor=2.5,
    )
    assert torch.all(scores > 0)
    torch.testing.assert_close(scores.sum(dim=-1), torch.tensor([2.5]))
    scores.square().sum().backward()
    assert logits.grad is not None
    assert logits.grad[0, 1].abs() > 0
    assert logits.grad[0, 3].abs() > 0


def test_backward_replay_keeps_fifo_across_pipeline_warmup_forwards():
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    first = torch.tensor([[10, 11]])
    second = torch.tensor([[20, 21]])
    native = torch.tensor([[0, 1]])

    # Simulate two PP warmup forwards before either checkpoint recomputation.
    RouterReplay.set_replay_data([first])
    RouterReplay.set_replay_data([second])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_BACKWARD)

    assert torch.equal(replay.select_indices(native), first)
    assert torch.equal(replay.select_indices(native), second)


def test_ds4_hash_router_records_and_replays_its_layer_column():
    import pytest

    pytest.importorskip("transformer_engine")
    from megatron.lite.model.deepseek_v4.lite.moe import DeepseekV4MoE
    from megatron.lite.primitive.modules.router_replay import RouterReplay, RouterReplayAction

    RouterReplay.clear_global_router_replay_instances()
    module = DeepseekV4MoE.__new__(DeepseekV4MoE)
    nn.Module.__init__(module)
    gate = nn.Module()
    gate.gate = nn.Linear(2, 4, bias=False)
    gate.score_function = "sigmoid"
    gate.num_experts = 4
    gate.register_buffer("tid2eid", torch.tensor([[0, 1], [2, 3]], dtype=torch.long))
    gate.router_replay = RouterReplay()
    module.gate = gate
    module.topk = 2
    module.route_scale = 1.0
    x = torch.tensor([[1.0, -1.0], [0.5, 0.25]])
    token_ids = torch.tensor([0, 1])

    RouterReplay.set_global_router_replay_action(RouterReplayAction.RECORD)
    _, recorded = module._hash_route(x, token_ids)
    assert torch.equal(gate.router_replay.recorded_topk_idx, recorded)

    target = torch.tensor([[1, 0], [3, 2]])
    RouterReplay.set_replay_data([target], replay_mask=torch.tensor([True, True]))
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    weights, replayed = module._hash_route(x, token_ids)
    assert torch.equal(replayed, target)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(2))


def test_r3_mask_replays_every_causal_row_except_last():
    from megatron.lite.model.protocol_utils import pack_r3_replay_mask
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.primitive.parallel.thd import thd_pack_meta
    from megatron.lite.runtime.contracts import PackedBatch

    lengths = torch.tensor([3, 5], dtype=torch.long)
    batch = PackedBatch(
        input_ids=torch.arange(8),
        labels=torch.arange(8),
        seq_lens=lengths,
        loss_mask=torch.tensor([0, 1, 1, 0, 0, 1, 1, 1], dtype=torch.float32),
    )
    model = SimpleNamespace(ps=ParallelState())
    mask = pack_r3_replay_mask(model, batch)
    meta = thd_pack_meta(lengths, tp_size=1, cp_size=1)

    expected = torch.zeros_like(mask)
    for idx, length in enumerate(lengths.tolist()):
        start = int(meta.cu_seqlens_padded[idx].item())
        expected[start : start + length - 1] = True
    assert torch.equal(mask, expected)


def test_ds4_replay_roots_exclude_mtp_layers():
    from megatron.lite.model.deepseek_v4.lite.protocol import router_replay_roots

    main_layers = nn.ModuleDict({"0": nn.Linear(2, 2), "1": nn.Linear(2, 2)})
    mtp_layers = nn.ModuleList([nn.Linear(2, 2)])
    model = nn.Module()
    model.layers = main_layers
    model.mtp = mtp_layers
    chunk = nn.Module()
    chunk.model = model

    assert router_replay_roots(chunk) == list(main_layers.values())


def test_r3_driver_replays_layer_order_and_causal_rows_end_to_end():
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
    from megatron.lite.runtime.contracts import PackedBatch

    class FakeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.router_replay = None

        def forward(self, native):
            return self.router_replay.select_indices(native)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = ParallelState()
            self.routers = nn.ModuleList([FakeRouter(), FakeRouter()])

    model = FakeModel()
    handle = SimpleNamespace(
        _model=model,
        _extras={"model_chunks": [model], "protocol": None},
    )
    batch = PackedBatch(
        input_ids=torch.arange(5),
        labels=torch.arange(5),
        seq_lens=torch.tensor([3, 2]),
        loss_mask=torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32),
        routed_experts=torch.nested.as_nested_tensor(
            [
                torch.tensor(
                    [
                        [[10, 11], [20, 21]],
                        [[12, 13], [22, 23]],
                        [[14, 15], [24, 25]],
                    ]
                ),
                torch.tensor(
                    [
                        [[16, 17], [26, 27]],
                        [[18, 19], [28, 29]],
                    ]
                ),
            ],
            layout=torch.jagged,
        ),
    )
    native = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    driver = RouterReplayDriver.maybe_create(handle, {"action": "replay"})
    assert driver is not None
    driver.begin()
    try:
        stepped = driver.wrap(
            lambda active_model, _batch: [router(native) for router in active_model.routers]
        )
        layer0, layer1 = stepped(model, batch)
    finally:
        driver.end()

    # Every row that can causally affect a response token uses rollout routes.
    # The final row of each sequence stays native because it predicts no token.
    assert torch.equal(
        layer0,
        torch.tensor([[10, 11], [12, 13], [4, 5], [16, 17], [8, 9]]),
    )
    assert torch.equal(
        layer1,
        torch.tensor([[20, 21], [22, 23], [4, 5], [26, 27], [8, 9]]),
    )
    assert all(router.router_replay is None for router in model.routers)


def test_r3_driver_accepts_next_token_routes_without_final_input_row():
    from megatron.lite.primitive.parallel import ParallelState
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver
    from megatron.lite.runtime.contracts import PackedBatch

    class FakeRouter(nn.Module):
        def __init__(self):
            super().__init__()
            self.router_replay = None

        def forward(self, native):
            return self.router_replay.select_indices(native)

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ps = ParallelState()
            self.routers = nn.ModuleList([FakeRouter()])

    model = FakeModel()
    handle = SimpleNamespace(
        _model=model,
        _extras={"model_chunks": [model], "protocol": None},
    )
    batch = PackedBatch(
        input_ids=torch.arange(5),
        labels=torch.arange(5),
        seq_lens=torch.tensor([3, 2]),
        loss_mask=torch.tensor([0, 1, 1, 0, 1], dtype=torch.float32),
        routed_experts=torch.nested.as_nested_tensor(
            [
                torch.tensor([[[10, 11]], [[12, 13]]]),
                torch.tensor([[[20, 21]]]),
            ],
            layout=torch.jagged,
        ),
    )
    native = torch.tensor([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]])

    driver = RouterReplayDriver.maybe_create(handle, {"action": "replay"})
    assert driver is not None
    driver.begin()
    try:
        stepped = driver.wrap(lambda active_model, _batch: active_model.routers[0](native))
        actual = stepped(model, batch)
    finally:
        driver.end()

    # vLLM supplies routes for causal rows only.  The absent final row of each
    # input sequence remains native and therefore needs no synthetic route.
    assert torch.equal(
        actual,
        torch.tensor([[10, 11], [12, 13], [4, 5], [20, 21], [8, 9]]),
    )


def test_r3_driver_slices_global_layer_axis_for_pipeline_stage():
    from megatron.lite.runtime.backends.mlite.router_replay import RouterReplayDriver

    driver = RouterReplayDriver(
        SimpleNamespace(_model=nn.Module(), _extras={}),
        "replay",
    )
    driver._ps = SimpleNamespace(pp_size=2)
    driver._num_routers = 2
    driver._pp_offset = 2
    driver._pp_total = 4
    routed = torch.nested.as_nested_tensor(
        [
            torch.arange(3 * 4 * 2).reshape(3, 4, 2),
            torch.arange(2 * 4 * 2).reshape(2, 4, 2) + 100,
        ],
        layout=torch.jagged,
    )

    local = driver._select_local_layers(routed)

    assert getattr(local, "is_nested", False)
    for actual, source in zip(local.unbind(0), routed.unbind(0), strict=True):
        assert torch.equal(actual, source[:, 2:4, :])
