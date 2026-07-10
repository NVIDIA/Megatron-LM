# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from megatron.core.tensor_parallel.inference_layers import (
    inference_all_gather_from_tensor_model_parallel_region,
)


def test_inference_all_gather_barriers_before_reusing_symmetric_buffer():
    """The consecutive MTP all-gather must synchronize before reusing TP storage."""
    tp_group = MagicMock()
    rank_chunks = [
        torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]]]),
        torch.tensor([[[5.0, 6.0]], [[7.0, 8.0]]]),
    ]
    gathered_storage = torch.cat(rank_chunks, dim=0)
    symmetric_storage = torch.empty_like(gathered_storage)

    buffer = MagicMock()
    buffer.maybe_get_tensor.return_value = {
        "tensor": symmetric_storage,
        "handle": MagicMock(),
    }

    def mock_all_gather(output, _input, _handle, barrier_before=False):
        assert barrier_before
        output.copy_(gathered_storage)

    with (
        patch(
            "megatron.core.tensor_parallel.inference_layers.dist.get_world_size",
            return_value=2,
        ),
        patch(
            "megatron.core.tensor_parallel.inference_layers.are_tensors_nvls_eligible",
            return_value=True,
        ),
        patch(
            "megatron.core.tensor_parallel.inference_layers.SymmetricMemoryManager.get_buffer",
            return_value=buffer,
        ) as get_buffer,
        patch(
            "megatron.core.tensor_parallel.inference_layers.SymmetricMemoryManager.is_initialized",
            return_value=True,
        ),
        patch(
            "megatron.core.tensor_parallel.inference_layers.multimem_all_gather",
            side_effect=mock_all_gather,
        ),
    ):
        output = inference_all_gather_from_tensor_model_parallel_region(
            rank_chunks[0],
            tp_group,
            SimpleNamespace(inference_disable_triton_nvls_kernels=False),
            barrier_before=True,
        )

    get_buffer.assert_called_once_with("tp", process_group=tp_group)
    torch.testing.assert_close(output, torch.cat(rank_chunks, dim=-1))


def test_inference_all_gather_disable_nvls_uses_standard_collective():
    tp_group = MagicMock()
    x = torch.randn(4, 1, 8)
    expected = torch.randn(4, 1, 16)

    with (
        patch(
            "megatron.core.tensor_parallel.inference_layers.dist.get_world_size",
            return_value=2,
        ),
        patch(
            "megatron.core.tensor_parallel.inference_layers.SymmetricMemoryManager.get_buffer"
        ) as get_buffer,
        patch(
            "megatron.core.tensor_parallel.inference_layers.gather_from_tensor_model_parallel_region",
            return_value=expected,
        ) as standard_gather,
    ):
        output = inference_all_gather_from_tensor_model_parallel_region(
            x,
            tp_group,
            SimpleNamespace(inference_disable_triton_nvls_kernels=True),
        )

    get_buffer.assert_not_called()
    standard_gather.assert_called_once_with(x, group=tp_group)
    assert output is expected
