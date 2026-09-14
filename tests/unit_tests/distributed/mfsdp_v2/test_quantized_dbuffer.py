# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MFSDP v2 MXFP8 buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    BlockAtomic,
    Flat,
)

QuantizedDBuffer = pytest.importorskip(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer",
    reason="Transformer Engine MXFP8 support is required.",
).QuantizedDBuffer


def _grouped(mesh, placements, device: torch.device) -> QuantizedDBuffer:
    return QuantizedDBuffer(mesh, placements, [(64, 64)], device)


@pytest.mark.parametrize(
    "placement",
    [Replicate(), BlockAtomic(32), Flat(), Shard(0)],
    ids=["replicated", "block_atomic", "flat", "shard"],
)
@pytest.mark.parametrize("shape", [(64, 64), (128, 128)], ids=["padded_scales", "unpadded_scales"])
def test_quantized_dbuffer_quantization_matches_te(distributed_setup, placement, shape):
    """Quantized data and compact scales match independently quantized TE tensors."""
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

    if torch.cuda.get_device_capability(distributed_setup.device)[0] < 10:
        pytest.skip("MXFP8 quantization coverage requires Blackwell-or-newer GPUs.")
    if isinstance(placement, Shard):
        shape = (shape[0] * distributed_setup.world_size, shape[1])
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    grouped = QuantizedDBuffer(mesh, [placement], [shape], distributed_setup.device)
    main_weight = DBuffer.empty(
        mesh, [placement], [shape], torch.float32, distributed_setup.device, block_size=32
    )
    torch.manual_seed(1234)
    main_weight.local_buffer.normal_()
    reference = MXFP8Quantizer(tex.DType.kFloat8E4M3)(main_weight.get_local_tensor(0))
    grouped.quantize_(main_weight)

    for plane, expected in zip(
        grouped.planes,
        (
            reference._rowwise_data,
            reference._columnwise_data,
            reference._rowwise_scale_inv,
            reference._columnwise_scale_inv,
        ),
    ):
        actual = plane.get_local_tensor(0)
        torch.testing.assert_close(
            actual, expected[: actual.shape[0], : actual.shape[1]], rtol=0, atol=0
        )


def test_quantized_dbuffer_allgathers_every_plane(distributed_setup):
    """Every materialized MXFP8-like plane follows the same collective transition."""
    if distributed_setup.world_size < 2:
        pytest.skip("QuantizedDBuffer all-gather requires at least two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    grouped = _grouped(mesh, [Flat()], distributed_setup.device)
    for plane in grouped.planes:
        plane.local_buffer.fill_(mesh.get_local_rank())

    result = grouped.allgather(0)

    assert result.rowwise_data.placements == (Replicate(),)
    for plane in result.planes:
        assert plane.local_buffer.view(mesh.size(), -1)[0].eq(0).all()
        assert plane.local_buffer.view(mesh.size(), -1)[1].eq(1).all()


def test_quantized_dbuffer_redistributes_into_matching_destinations(distributed_setup):
    """A preallocated grouped destination receives every plane."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    source = _grouped(mesh, [Replicate()], distributed_setup.device)
    for plane in source.planes:
        plane.local_buffer.copy_(torch.arange(plane.local_buffer.numel(), device=plane.device))
    destination = _grouped(mesh, [Flat()], distributed_setup.device)
    result = source.redistribute([Flat()], out=destination)

    assert result is destination
    for result_plane, source_plane in zip(result.planes, source.planes):
        torch.testing.assert_close(
            result_plane.allgather(0).local_buffer, source_plane.local_buffer
        )


def test_quantized_dbuffer_derives_scale_shards_from_data(distributed_setup):
    """Scale planes own exactly the blocks represented by their local data views."""
    if distributed_setup.world_size != 2:
        pytest.skip("QuantizedDBuffer layout coverage requires exactly two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    grouped = QuantizedDBuffer(
        mesh, [BlockAtomic(32)], [(128, 64), (32, 64)], distributed_setup.device
    )

    for index in range(2):
        data = grouped.rowwise_data.get_local_tensor(index)
        assert grouped.rowwise_scale.get_local_tensor(index).shape == (
            data.shape[0],
            data.shape[1] // 32,
        )
        assert grouped.columnwise_scale.get_local_tensor(index).shape == (
            data.shape[0] // 32,
            data.shape[1],
        )
