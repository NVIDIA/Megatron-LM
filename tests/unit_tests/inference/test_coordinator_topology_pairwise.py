# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Real coordinator routing over TP, PP and TP2xPP2xDP2 layouts, which need eight ranks.

The other routed-feature tests run at two ranks (tests/unit_tests/inference/two_rank/).
"""

from collections import Counter

import pytest

from tests.unit_tests.inference.coordinator_pairwise_utils import greedy_params, routed_model
from tests.unit_tests.inference.engines.test_dynamic_engine_async_sched import (
    _instrument_parallel_runtime,
)
from tests.unit_tests.inference.two_rank.test_coordinator_features_pairwise import (
    _exercise_all_owners,
)

pytestmark = [pytest.mark.internal, pytest.mark.asyncio]


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"tensor_model_parallel_size": 2}, id="tp2-dp4"),
        pytest.param({"pipeline_model_parallel_size": 2}, id="pp2-dp4"),
        pytest.param(
            {
                "tensor_model_parallel_size": 2,
                "pipeline_model_parallel_size": 2,
                "sequence_parallel": True,
            },
            id="tp2-pp2-sp-dp2",
        ),
    ],
)
async def test_routed_topology_matches_direct_on_every_owner(monkeypatch, overrides):
    """TP, PP and the full eight-rank TP2xPP2xDP2 layout execute routed targets."""
    prompt = list(range(4, 16))
    params = greedy_params(num_tokens_to_generate=4, return_prompt_tokens=True)
    runtime = Counter()
    async with routed_model(monkeypatch, transformer_impl="local", **overrides) as harness:
        direct = await harness.direct(prompt, params)

        def install(_target):
            if harness.config.sequence_parallel:
                from megatron.core.tensor_parallel import layers as tp_layers

                all_gather = tp_layers.dist_all_gather_func

                def traced_all_gather(output, input_, *args, **kwargs):
                    result = all_gather(output, input_, *args, **kwargs)
                    runtime["tp-collective:gather_from_sequence_parallel_region"] += 1
                    runtime["tp-sp-gather-dimensions"] += int(
                        output.shape[0]
                        == input_.shape[0] * harness.config.tensor_model_parallel_size
                    )
                    return result

                monkeypatch.setattr(tp_layers, "dist_all_gather_func", traced_all_gather)
            for module in harness.engine.controller.inference_wrapped_model.model.modules():
                if "ParallelLinear" in type(module).__name__:
                    _instrument_parallel_runtime(module, runtime)

        target = await _exercise_all_owners(
            harness, prompt, params, direct, runtime=runtime, install=install
        )
        if harness.config.tensor_model_parallel_size > 1:
            assert runtime["tp-column-partitions-installed"] > 0
            assert runtime["tp-row-partitions-installed"] > 0
            if harness.config.sequence_parallel:
                assert target["tp-collective:gather_from_sequence_parallel_region"] > 0
                assert target["tp-collective:reduce_scatter_to_sequence_parallel_region"] > 0
                assert target["tp-sp-gather-dimensions"] > 0
                assert target["tp-sp-reduce-scatter-dimensions"] > 0
            else:
                assert target["tp-collective:reduce_from_tensor_model_parallel_region"] > 0
