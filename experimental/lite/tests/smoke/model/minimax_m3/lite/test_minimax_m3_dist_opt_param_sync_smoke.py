# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""dist_opt overlapped param all-gather reaches every trainable param of MiniMax-M3 lite, 2 GPUs.

mcore DDP finishes the all-gather in each module's forward pre-hook, so every trainable param must be owned by a
module whose forward runs, and after the next forward the dense params must be identical on all DP ranks.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = [
    pytest.mark.gpus(2, min_architecture="blackwell"),
    pytest.mark.env(CUDA_DEVICE_MAX_CONNECTIONS="1"),
]

S, STEPS = 2048, 3
_CASES = {"dp2": dict(tp=1, ep=1, etp=1, pp=1, cp=1), "ep2": dict(tp=1, ep=2, etp=1, pp=1, cp=1)}


def _chunks(handle):
    from megatron.lite.primitive.ckpt.hf_weights import unwrap_model

    chunks = handle._model if isinstance(handle._model, list) else [handle._model]
    return [unwrap_model(c) for c in chunks]


def _dense_params_differing_across_dp(handle, dist):
    from megatron.lite.model.minimax_m3.lite.checkpoint import EXPERT_CLASSIFIER

    group = handle._parallel_state.dp_group
    src = dist.get_global_rank(group, 0)
    bad = []
    for model in _chunks(handle):
        for name, p in model.named_parameters():
            if not p.requires_grad or EXPERT_CLASSIFIER(name):
                continue
            ref = p.detach().clone()
            dist.broadcast(ref, src, group=group)
            diff = torch.tensor([int(not torch.equal(ref, p.detach()))], device="cuda")
            dist.all_reduce(diff, op=dist.ReduceOp.MAX, group=group)
            if diff.item():
                bad.append(name)
    return bad


@pytest.mark.parametrize("name", list(_CASES))
def test_param_gather_reaches_all_trainable_params(magi_cfg, magi_source, dist, runtime_tools, name):
    pytest.importorskip("megatron.core", reason="dist_opt needs megatron-core")
    pytest.importorskip("magi_attn_extensions.MSA")
    if dist.get_world_size() != 2:
        pytest.skip("expects exactly 2 ranks")
    from megatron.lite.runtime.backends.mlite.runtime import MegatronLiteRuntime
    from megatron.lite.runtime.contracts.config import ParallelConfig

    runtime = MegatronLiteRuntime.__new__(MegatronLiteRuntime)
    handle = runtime_tools.build_handle(magi_cfg, magi_source, ParallelConfig(**_CASES[name]))
    ps = handle._parallel_state
    called = set()
    hook = torch.nn.modules.module.register_module_forward_pre_hook(lambda m, _: called.add(id(m)))
    try:
        for step in range(STEPS):
            runtime.zero_grad(handle)
            runtime.forward_backward(handle, iter([runtime_tools.batch(magi_cfg, step, S)]), None, num_microbatches=1)
            if step == 0:
                orphans = [f"{mn}.{pn}" for model in _chunks(handle) for mn, m in model.named_modules()
                           for pn, p in m.named_parameters(recurse=False) if p.requires_grad and id(m) not in called]
                assert not orphans, f"trainable params whose owner module never runs forward: {orphans}"
            else:
                bad = _dense_params_differing_across_dp(handle, dist)
                assert not bad, f"{name} step {step}: dense params differ across DP ranks: {bad[:8]}"
            ok, _, _ = runtime.optimizer_step(handle)
            assert ok
    finally:
        hook.remove()
        runtime.zero_grad(handle)
        del handle, runtime
        runtime_tools.reset_parallel_state(ps)
