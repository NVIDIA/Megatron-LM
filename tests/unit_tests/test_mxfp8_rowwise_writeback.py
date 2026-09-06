# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Run with the standard distributed unit-test launcher on Blackwell GPUs."""

import pytest
import torch

from megatron.core import fp8_utils
from tests.unit_tests.test_utilities import Utils


@pytest.mark.skipif(not fp8_utils.HAVE_TE_MXFP8TENSOR, reason="Requires TE MXFP8")
@pytest.mark.parametrize('replicated_shards', [False, True])
def test_rowwise_writeback_preserves_storage_and_matches_full_quantization(replicated_shards):
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    Utils.initialize_model_parallel(1, 1)
    try:
        group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group)
        size = torch.distributed.get_world_size(group)
        # Crosses the 128-row tile/padding boundary; split inside a 32-element block.
        shape = (160, 8192)
        source = torch.linspace(-3, 5, 160 * 8192, device='cuda').reshape(shape)
        quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False)
        param = quantizer(source.to(torch.bfloat16))
        pointers = (param._rowwise_data.data_ptr(), param._rowwise_scale_inv.data_ptr())
        for step in range(2):
            master = source + step * 0.25
            if size == 1 or replicated_shards:
                shard, offset = master.flatten(), 0
            elif rank == 0:
                shard, offset = master.flatten()[:17], 0
            elif rank == 1:
                shard, offset = master.flatten()[17:], 17
            else:
                shard, offset = None, None
            fp8_utils.quantize_param_shard([param], [shard], [offset], group)
            expected = quantizer(master.to(torch.bfloat16))
            torch.testing.assert_close(param._rowwise_data, expected._rowwise_data, rtol=0, atol=0)
            # Padding bytes are not part of the quantized value and may be uninitialized.
            torch.testing.assert_close(
                param._rowwise_scale_inv[:160, :256],
                expected._rowwise_scale_inv[:160, :256],
                rtol=0,
                atol=0,
            )
            assert param._columnwise_data is None
            assert param._columnwise_scale_inv is None
            assert pointers == (param._rowwise_data.data_ptr(), param._rowwise_scale_inv.data_ptr())
            fp8_utils.post_all_gather_processing([param])
            assert param._columnwise_data is None
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not fp8_utils.HAVE_TE_MXFP8TENSOR, reason="Requires TE MXFP8")
@pytest.mark.parametrize('columnwise', [False, True])
def test_bf16_copy_back_preserves_mxfp8_directions(columnwise):
    import transformer_engine.pytorch as te
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    available, reason = te.is_mxfp8_available(return_reason=True)
    if not available:
        pytest.skip(reason)
    source = torch.randn(128, 128, dtype=torch.bfloat16, device='cuda')
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=columnwise)
    param = torch.nn.Parameter(quantizer(source))
    updated = source + 1
    fp8_utils.copy_back_gathered_bf16_into_fp8_param(param, updated)
    expected = quantizer(updated)
    torch.testing.assert_close(param._rowwise_data, expected._rowwise_data, rtol=0, atol=0)
    assert (param._columnwise_data is not None) == columnwise
    if columnwise:
        torch.testing.assert_close(
            param._columnwise_data, expected._columnwise_data, rtol=0, atol=0
        )
