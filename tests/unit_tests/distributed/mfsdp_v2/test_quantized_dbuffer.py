# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Tests for MFSDP v2 MXFP8 buffers."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.placement import (
    BlockAtomic,
    Flat,
)

QuantizedDBuffer = pytest.importorskip(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer",
    reason="Transformer Engine MXFP8 support is required.",
).QuantizedDBuffer

import transformer_engine_torch as tex
from transformer_engine.pytorch import is_mxfp8_available
from transformer_engine.pytorch.cpp_extensions import general_gemm
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer

pytestmark = pytest.mark.launch_on_gb200

if not is_mxfp8_available():
    pytest.skip("MXFP8 quantization is not available.", allow_module_level=True)


def test_quantized_dbuffer_quantization_matches_te(distributed_setup):
    """Quantized tensor views match TE's data and scales."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    shapes = [(128, 128), (128, 128), (64, 64), (32, 64)]
    quantized = QuantizedDBuffer.empty(mesh, [BlockAtomic(32)], shapes, device)
    main_weight = DBuffer(
        mesh, [BlockAtomic(32)], quantized.rowwise_data.layout, torch.float32, device
    )
    torch.manual_seed(1234 + distributed_setup.rank)
    main_weight.local_buffer.normal_()
    quantized.quantize_(main_weight)

    for index in range(len(shapes)):
        view = quantized.get_tensor_view(index)
        assert view._rowwise_scale_inv.shape == (view.shape[0], view.shape[1] // 32)
        assert view._columnwise_scale_inv.shape == (view.shape[0] // 32, view.shape[1])
        reference = MXFP8Quantizer(tex.DType.kFloat8E4M3)(main_weight.get_tensor_view(index))
        assert view.shape == reference.shape
        for attribute in (
            "_rowwise_data",
            "_columnwise_data",
            "_rowwise_scale_inv",
            "_columnwise_scale_inv",
        ):
            actual = getattr(view, attribute)
            expected = getattr(reference, attribute)
            # TE's allocating quantizer pads scales; our views keep them compact.
            # Ignore the padding. GEMM support for compact scales is tracked in:
            # https://github.com/NVIDIA/TransformerEngine/issues/3518
            torch.testing.assert_close(
                actual, expected[: actual.shape[0], : actual.shape[1]], rtol=0, atol=0
            )


def test_quantized_dbuffer_get_tensor_supports_gemm(distributed_setup):
    """Compute tensors prepare gathered scales for rowwise and columnwise GEMMs."""
    device = distributed_setup.device
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    shapes = [(128, 128), (128, 128), (64, 64), (32, 64)]
    quantized = QuantizedDBuffer.empty(mesh, [BlockAtomic(32)], shapes, device)
    main_weight = DBuffer(
        mesh, [BlockAtomic(32)], quantized.rowwise_data.layout, torch.float32, device
    )
    torch.manual_seed(1234 + distributed_setup.rank)
    main_weight.local_buffer.normal_()
    quantized.quantize_(main_weight)
    gathered = quantized.redistribute([Replicate()])
    gathered_main = main_weight.redistribute([Replicate()])
    quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
    for index, shape in enumerate(shapes):
        compute_tensor = gathered.get_tensor(index)
        reference = quantizer(gathered_main.get_tensor_view(index))
        for layout, inner_dim in (("TN", shape[1]), ("NN", shape[0])):
            activation = quantizer(torch.randn((64, inner_dim), device=device))
            actual = general_gemm(
                compute_tensor, activation, out_dtype=torch.bfloat16, layout=layout
            )[0]
            expected = general_gemm(reference, activation, out_dtype=torch.bfloat16, layout=layout)[
                0
            ]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_quantized_dbuffer_redistributes_every_plane(distributed_setup):
    """Redistribution shards into a destination and gathers every plane back."""
    if distributed_setup.world_size < 2:
        pytest.skip("QuantizedDBuffer redistribution requires at least two ranks.")

    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    shapes = [(64, 64)]
    source = QuantizedDBuffer.empty(mesh, [Replicate()], shapes, distributed_setup.device)
    for plane in source.planes:
        shard_size = plane.local_buffer.numel() // mesh.size()
        plane.local_buffer.copy_(
            torch.arange(plane.local_buffer.numel(), device=plane.device) // shard_size
        )
    destination = QuantizedDBuffer.empty(mesh, [BlockAtomic(32)], shapes, distributed_setup.device)
    result = source.redistribute([BlockAtomic(32)], out=destination)
    assert result is destination

    gathered = result.redistribute([Replicate()])
    assert gathered.placements == (Replicate(),)
    for actual, expected in zip(gathered.planes, source.planes):
        torch.testing.assert_close(actual.local_buffer, expected.local_buffer)


def test_quantized_dbuffer_view_shares_every_plane(distributed_setup):
    """A sharded view aliases exactly this rank's slice of each replicated plane."""
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    source = QuantizedDBuffer.empty(
        mesh, [Replicate()], [(128, 64), (32, 128)], distributed_setup.device
    )
    for plane in source.planes:
        plane.local_buffer.zero_()
    assert source.view([Replicate()]) is source

    view = source.view([BlockAtomic(32)])
    assert view.view([BlockAtomic(32)]) is view
    for index, (actual, original) in enumerate(zip(view.planes, source.planes)):
        assert actual.placements == ((Flat(),) if index == 3 else (BlockAtomic(32),))
        chunks = original.local_buffer.view(mesh.size(), -1)
        expected = chunks[mesh.get_local_rank()]
        assert actual.local_buffer.shape == expected.shape
        assert actual.local_buffer.data_ptr() == expected.data_ptr()

        actual.local_buffer.fill_(index + 1)
        assert expected.eq(index + 1).all()
        for rank in range(mesh.size()):
            if rank != mesh.get_local_rank():
                assert chunks[rank].eq(0).all()


@pytest.mark.parametrize("use_out", [False, True])
def test_quantized_dbuffer_allgathers_every_plane(distributed_setup, use_out):
    """All-gather preserves rank order in every plane, with or without an output buffer."""
    if distributed_setup.world_size < 2:
        pytest.skip("QuantizedDBuffer all-gather requires at least two ranks.")
    mesh = init_device_mesh(distributed_setup.device.type, (distributed_setup.world_size,))
    shapes = [(128, 64), (32, 128)]
    source = QuantizedDBuffer.empty(mesh, [BlockAtomic(32)], shapes, distributed_setup.device)
    for index, plane in enumerate(source.planes):
        plane.local_buffer.fill_(index * mesh.size() + mesh.get_local_rank())
    if use_out:
        destination = QuantizedDBuffer.empty(mesh, [Replicate()], shapes, distributed_setup.device)
        result = source.allgather(0, out=destination)
        assert result is destination
    else:
        result = source.allgather(0)
    for index, plane in enumerate(result.planes):
        assert plane.placements == (Replicate(),)
        chunks = plane.local_buffer.view(mesh.size(), -1)
        for rank in range(mesh.size()):
            assert chunks[rank].eq(index * mesh.size() + rank).all()
