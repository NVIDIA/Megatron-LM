from unittest.mock import Mock

import pytest
import torch

pytestmark = pytest.mark.gpus(1)


def test_r3_official_route_is_exact_mlite_native_noop() -> None:
    """Preserve vLLM's token/layer/slot ordering through exact R3 replay."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA and the official vLLM DS4 router")

    import torch.nn as nn

    from megatron.lite.model.deepseek_v4.config import DeepseekV4Config
    from megatron.lite.primitive.modules.router import SigmoidTopKRouter
    from megatron.lite.primitive.modules.router_replay import (
        RouterReplay,
        RouterReplayAction,
    )
    from vllm.model_executor.layers.fused_moe.router.dsv4_topk import dsv4_topk

    config = DeepseekV4Config(hidden_size=256)
    router = SigmoidTopKRouter(
        config, Mock(tp_size=1, tp_group=None), compute_aux_loss=False
    ).cuda()
    router.gate = nn.Identity()
    base = torch.linspace(-4.0, 4.0, 256, device="cuda")
    logits = torch.stack(
        (base, base.flip(0), base.roll(73), torch.sin(torch.arange(256, device="cuda")))
    )
    bias = torch.linspace(0.125, -0.125, 256, device="cuda")
    router.expert_bias.copy_(bias)

    _, rollout_ids = dsv4_topk(
        logits, bias, torch.int64, config.routed_scaling_factor
    )
    native_scores, native_ids = router(logits)
    assert torch.equal(native_ids, rollout_ids)
    assert not torch.equal(native_ids, native_ids.sort(dim=-1).values)

    RouterReplay.clear_global_router_replay_instances()
    router.router_replay = RouterReplay()
    RouterReplay.set_replay_data([rollout_ids])
    RouterReplay.set_global_router_replay_action(RouterReplayAction.REPLAY_FORWARD)
    RouterReplay.reset_replay_stats()
    replay_scores, replay_ids = router(logits)

    assert torch.equal(replay_ids, native_ids)
    assert torch.equal(replay_scores, native_scores)
    assert RouterReplay.replay_stats()["changed"] == 0
    RouterReplay.clear_global_router_replay_instances()
