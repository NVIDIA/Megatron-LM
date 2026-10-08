# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest

from megatron.core.models.common.model_chunk_schedule_plan import TransformerModelChunkSchedulePlan


@pytest.mark.parametrize("accepts_tensor_release", [False, True])
def test_custom_layer_factory_receives_hook_metadata_and_compatible_constructor(
    accepts_tensor_release,
):
    calls = []

    class MainSignature:
        def __init__(self, layer, event, state, comp, comm, extra):
            calls.append((layer, extra, None))

    class ReleaseSignature:
        def __init__(self, layer, event, state, comp, comm, extra, *, tensor_release):
            calls.append((layer, extra, tensor_release))

    class CustomChunkPlan(TransformerModelChunkSchedulePlan):
        LAYER_SCHEDULE_PLAN_CLASS = ReleaseSignature if accepts_tensor_release else MainSignature

        def _extra_args_for_layer(self, module, index, num_layers):
            return {
                **super()._extra_args_for_layer(module, index, num_layers),
                "custom_layer": index,
            }

    plan = CustomChunkPlan.__new__(CustomChunkPlan)
    plan._event = None
    plan._model_chunk_state = SimpleNamespace()
    plan._transformer_layers = []
    plan._tensor_release = object()
    config = SimpleNamespace(
        enable_hyper_connections=False,
        recompute_granularity=None,
        recompute_modules=[],
        mhc_recompute_layer_num=None,
    )
    module = SimpleNamespace(training=False, config=config, layers=[object(), object()])
    plan._build_layer_schedule_plan(module, None, None, module_tag="decoder")
    assert len(plan._transformer_layers) == 2
    assert [x[0] for x in calls] == module.layers
    assert [x[1]["custom_layer"] for x in calls] == [0, 1]
    assert [x[1]["is_first_layer"] for x in calls] == [True, False]
    assert [x[1]["is_last_layer"] for x in calls] == [False, True]
    assert all(x[1]["mhc_recompute_module_tag"] == "decoder" for x in calls)
    assert all(x[1]["mhc_recompute_manager"] is None for x in calls)
    assert all(x[2] is (plan._tensor_release if accepts_tensor_release else None) for x in calls)
