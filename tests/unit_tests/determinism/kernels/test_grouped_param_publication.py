# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Replay grouped master initialization and local/gathered MXFP8 publication."""

import pytest
import torch

from megatron.core.fp8_utils import (
    prepare_grouped_mxfp8_weight_,
    requantize_whole_params_from_high_precision,
)
from megatron.core.optimizer import OptimizerConfig
from megatron.core.optimizer.layer_wise_optimizer import LayerWiseDistributedOptimizer
from tests.unit_tests.determinism.kernels.harness import assert_replays_bit_exact
from tests.unit_tests.test_utilities import Utils

te = pytest.importorskip("transformer_engine.pytorch")
from transformer_engine.common.recipe import MXFP8BlockScaling

pytestmark = [pytest.mark.internal, pytest.mark.launch_on_gb200]


@pytest.mark.parametrize(
    "gather_publishes,preserve_init",
    [(False, True), (True, True), (False, False)],
    ids=["local", "gather", "local-model-values"],
)
def test_grouped_mxfp8_publication_replays_bit_exact(monkeypatch, gather_publishes, preserve_init):
    monkeypatch.setenv("NVTE_GROUPED_LINEAR_SINGLE_PARAM", "1")
    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    Utils.initialize_model_parallel()
    try:
        generator = torch.Generator(device="cuda").manual_seed(9814)
        # 128x256 needs no padding in either scale layout. Every compared scale
        # byte belongs to a real block, including under side-stream contention.
        initial = torch.randn(3, 128, 256, device="cuda", generator=generator)
        gradients = torch.randn(2, *initial.shape, device="cuda", generator=generator)

        @torch.no_grad()
        def publish(initial, gradients):
            with (
                torch.enable_grad(),
                te.fp8_model_init(
                    recipe=MXFP8BlockScaling(), preserve_high_precision_init_val=preserve_init
                ),
            ):
                layer = te.GroupedLinear(
                    initial.shape[0],
                    initial.shape[2],
                    initial.shape[1],
                    init_method=torch.nn.init.zeros_,
                    bias=False,
                    params_dtype=torch.bfloat16,
                    device="cuda",
                    single_grouped_weight=True,
                    use_grouped_tensor=True,
                )
            param = layer.weight
            prepare_grouped_mxfp8_weight_(param)
            requantize_whole_params_from_high_precision([param], [initial.to(torch.bfloat16)])
            optimizer = LayerWiseDistributedOptimizer(
                [torch.optim.SGD([param], lr=0.125)], OptimizerConfig(optimizer="sgd", bf16=True)
            )
            # Replay the master created from the saved initializer or model values,
            # before resetting it to the shared FP32 source for publication checks.
            history = [param.main_param.clone()]
            param.main_param.copy_(initial)
            param.requantized_by_layer_wise_gather = gather_publishes
            storage_fields = (
                "rowwise_data",
                "columnwise_data",
                "scale_inv",
                "columnwise_scale_inv",
            )
            for gradient in gradients:
                if gather_publishes:
                    before_gather = [getattr(param, field).clone() for field in storage_fields]
                param.main_param.grad = gradient.clone()
                assert optimizer.step_with_ready_grads()
                if gather_publishes:
                    # An owner defers publication until the BF16 gather completes. Its
                    # local optimizer step must leave both quantized layouts untouched.
                    for field, before in zip(storage_fields, before_gather):
                        torch.testing.assert_close(getattr(param, field), before, rtol=0, atol=0)
                    requantize_whole_params_from_high_precision(
                        [param], [param.main_param.to(torch.bfloat16)]
                    )
                assert not param._with_gemm_swizzled_scales
                assert all(
                    not member._with_gemm_swizzled_scales for member in param.quantized_tensors
                )
                history.append(
                    tuple(
                        value.clone()
                        for value in (
                            param.main_param,
                            param.rowwise_data,
                            param.columnwise_data,
                            param.scale_inv,
                            param.columnwise_scale_inv,
                        )
                    )
                )
            return history

        assert_replays_bit_exact(
            publish,
            (initial, gradients),
            replays=3,
            backward=False,
            contention=True,
            what="grouped MXFP8 FP32 masters and row/column payloads/scales",
        )
    finally:
        Utils.destroy_model_parallel()
