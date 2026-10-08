# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native embedding gradients must retain the configured FP32 DP accumulation."""

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F

from megatron.core.tensor_parallel.generalized_tensor_parallelism import (
    GTP_CONFIG,
    GTPEmbeddingWeight,
    reset_gtp_state,
    update_gtp_config,
    wrap_module_params_gtp,
)
from tests.unit_tests.generalized_tensor_parallel.gtp_test_utils import (  # noqa: F401
    _torchrun_dist_init,
)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("async_reduction", [False, True])
@pytest.mark.parametrize("fp32_accum", [False, True])
def test_native_embedding_preserves_fp32_main_grad(dtype, async_reduction, fp32_accum):
    if dist.get_world_size() != 4:
        pytest.skip("Requires four ranks")
    keys = (
        "pad_for_alignment",
        "async_reduction",
        "calculate_per_token_loss",
        "reduce_scatter_with_fp32_accumulation",
    )
    old = {key: getattr(GTP_CONFIG, key) for key in keys}
    update_gtp_config(
        pad_for_alignment=1,
        async_reduction=async_reduction,
        calculate_per_token_loss=False,
        reduce_scatter_with_fp32_accumulation=fp32_accum,
    )
    try:
        rank = dist.get_rank()
        initial = torch.arange(16 * 8, device="cuda", dtype=torch.float32).view(16, 8)
        initial = (initial / 100).to(dtype)
        indices = torch.arange(16, device="cuda")
        # Each local gradient is representable in BF16/FP16, but the global
        # mean is not. A downcast after reducing therefore loses real bits.
        coefficient = [1.0, 2**-7, 2**-8, 2**-9][rank]
        upstream = torch.arange(1, 17, device="cuda").float().view(16, 1) * coefficient
        reference = [torch.nn.Parameter(initial.clone()) for _ in range(2)]
        reference_out = [F.embedding(indices, p) for p in reference]
        (
            reference_out[0].float() * upstream - reference_out[1].float() * upstream / 2
        ).sum().backward()
        expected = []
        for parameter in reference:
            gradient = parameter.grad.float()
            dist.all_reduce(gradient, group=dist.group.WORLD)
            expected.append(gradient / dist.get_world_size())

        module = torch.nn.Module()
        module.first = torch.nn.Parameter(initial.clone())
        module.second = torch.nn.Parameter(initial.clone())
        wrap_module_params_gtp(module, ["first", "second"], dist.group.WORLD)
        weights = [module.first, module.second]
        for p in weights:
            p.main_grad = torch.zeros_like(p, dtype=torch.float32)
        outputs = [F.embedding(indices, GTPEmbeddingWeight.apply(p)) for p in weights]
        for actual, baseline in zip(outputs, reference_out):
            torch.testing.assert_close(actual, baseline, rtol=0, atol=0)
        (outputs[0].float() * upstream - outputs[1].float() * upstream / 2).sum().backward()
        for p, full_gradient in zip(weights, expected):
            wanted = full_gradient.chunk(4, dim=0)[rank]
            assert p.main_grad.dtype == torch.float32
            torch.testing.assert_close(p.main_grad, wanted, rtol=0, atol=0)
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        reset_gtp_state()
        update_gtp_config(**old)
