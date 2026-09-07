import pytest
import torch
from types import SimpleNamespace

pytestmark = pytest.mark.gpus(1)


def test_r3_official_route_is_exact_vllm_impl_noop() -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and the official vLLM DS4 router")

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.model.deepseek_v4.vllm.primitive.moe.module import (
        DeepseekV4MoE,
        _learned_route,
    )
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    config = DeepseekV4Config(hidden_size=256)
    base = torch.linspace(-4.0, 4.0, 256, device="cuda")
    logits = torch.stack(
        (base, base.flip(0), base.roll(73), torch.sin(torch.arange(256, device="cuda")))
    )
    bias = torch.linspace(0.125, -0.125, 256, device="cuda")
    _, rollout_ids = dsv4_topk(
        logits, bias, torch.int64, config.routed_scaling_factor
    )
    native_scores, native_ids = _learned_route(
        logits, bias, config.routed_scaling_factor
    )
    assert torch.equal(native_ids, rollout_ids)
    assert not torch.equal(native_ids, native_ids.sort(dim=-1).values)

    RouterReplay.clear_global_router_replay_instances()
    replay = RouterReplay()
    RouterReplay.set_replay_data([rollout_ids])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    RouterReplay.reset_replay_stats()
    owner = SimpleNamespace(gate=SimpleNamespace(router_replay=replay), config=config)
    replay_scores, replay_ids = DeepseekV4MoE._replay_route(
        owner, logits, native_scores, native_ids
    )

    assert torch.equal(replay_ids, native_ids)
    assert torch.equal(replay_scores, native_scores)
    assert RouterReplay.replay_stats()["changed"] == 0
    RouterReplay.clear_global_router_replay_instances()
