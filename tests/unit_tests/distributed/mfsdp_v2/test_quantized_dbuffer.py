# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MFSDP v2 MXFP8 buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import BlockAtomic

QuantizedDBuffer = pytest.importorskip(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer",
    reason="Transformer Engine MXFP8 support is required.",
).QuantizedDBuffer

import transformer_engine_torch as tex
from transformer_engine.pytorch import is_mxfp8_available
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

if not is_mxfp8_available():
    pytest.skip("MXFP8 quantization is not available.", allow_module_level=True)


def test_quantized_dbuffer_quantization_matches_te(distributed_setup):
    """Multiple tensors retain their data and scale ownership after quantization."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    shapes = [(128, 128), (128, 128), (64, 64), (32, 64)]
    grouped = QuantizedDBuffer(mesh, [BlockAtomic(32)], shapes, distributed_setup.device)
    main_weight = DBuffer.empty(
        mesh, [BlockAtomic(32)], shapes, torch.float32, distributed_setup.device, block_size=32
    )
    torch.manual_seed(1234 + distributed_setup.rank)
    main_weight.local_buffer.normal_()
    grouped.quantize_(main_weight)

    for index in range(len(shapes)):
        data = grouped.rowwise_data.get_local_tensor(index)
        assert grouped.rowwise_scale.get_local_tensor(index).shape == (
            data.shape[0],
            data.shape[1] // 32,
        )
        assert grouped.columnwise_scale.get_local_tensor(index).shape == (
            data.shape[0] // 32,
            data.shape[1],
        )
        if data.numel() == 0:
            continue
        reference = MXFP8Quantizer(tex.DType.kFloat8E4M3)(main_weight.get_local_tensor(index))
        for plane, expected in zip(
            grouped.planes,
            (
                reference._rowwise_data,
                reference._columnwise_data,
                reference._rowwise_scale_inv,
                reference._columnwise_scale_inv,
            ),
        ):
            actual = plane.get_local_tensor(index)
            torch.testing.assert_close(
                actual, expected[: actual.shape[0], : actual.shape[1]], rtol=0, atol=0
            )


def test_quantized_dbuffer_redistributes_every_plane(distributed_setup):
    """Redistribution shards into a destination and gathers every plane back."""
    if distributed_setup.world_size < 2:
        pytest.skip("QuantizedDBuffer redistribution requires at least two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    source = QuantizedDBuffer(mesh, [Replicate()], [(64, 64)], distributed_setup.device)
    for plane in source.planes:
        shard_size = plane.local_buffer.numel() // mesh.size()
        plane.local_buffer.copy_(
            torch.arange(plane.local_buffer.numel(), device=plane.device) // shard_size
        )
    destination = QuantizedDBuffer(mesh, [BlockAtomic(32)], [(64, 64)], distributed_setup.device)
    result = source.redistribute([BlockAtomic(32)], out=destination)
    assert result is destination

    gathered = result.redistribute([Replicate()])
    assert gathered.placements == (Replicate(),)
    for actual, expected in zip(gathered.planes, source.planes):
        torch.testing.assert_close(actual.local_buffer, expected.local_buffer)
