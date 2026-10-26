# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A backward prefetch must not satisfy the next optimizer-step forward hooks."""

from types import MethodType, SimpleNamespace

from megatron.core.transformer.moe.experts import TEGroupedMLP
from megatron.core.transformer.moe.replica_expert_runtime import ReplicaExpertRuntime
from megatron.core.transformer.moe.replica_expert_runtime import _WeightDirection as Direction


def check(prepare, forward_hook, prepare_source):
    events = []
    generation = {'published': 0, 'ready': -1}

    def gather(*_):
        generation['ready'] = generation['published']
        events.append(('gather', generation['ready']))

    module = SimpleNamespace(_forward_pre_hooks={0: gather})
    linear = SimpleNamespace(modules=lambda: [module])
    experts = SimpleNamespace(
        linear_fc1=linear,
        linear_fc2=linear,
        _fused_impl_parameters_prepared=False,
        _replica_expert_runtime=None,
        _ensure_main_grad_for_fused_impl=lambda: None,
    )
    experts.prepare_fused_impl_parameters = MethodType(prepare, experts)
    runtime = SimpleNamespace(
        _experts_ref=lambda: experts, projections=(), prepare_runtime_parameters=lambda: None
    )
    hook = forward_hook(experts)
    for step in range(2):
        generation['published'] = step
        for microbatch in range(2):
            prepare_source(runtime, Direction.FORWARD)
            assert generation['ready'] == step, (
                f'step {step + 1} microbatch {microbatch}: prefetched stale weight generation '
                f"{generation['ready']} instead of {step}; forward gather hook was skipped"
            )
            count = len(events)
            hook(None)
            assert len(events) == count, 'Forward prefetch must not duplicate fused-op hooks'
            prepare_source(runtime, Direction.BACKWARD)
    print('PASS: forward/backward/optimizer/forward re-runs gather hooks before weight reads')


def test_replica_prefetch_hook_lifetime():
    check(
        TEGroupedMLP.prepare_fused_impl_parameters,
        TEGroupedMLP._make_fused_impl_pre_forward_hook,
        ReplicaExpertRuntime.prepare_source_weights,
    )
