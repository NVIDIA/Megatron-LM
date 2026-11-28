# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of MFSDP v2 quantization into compact MXFP8 scale planes."""

import pytest
import torch
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.dbuffer import DBuffer
from tests.unit_tests.distributed.mfsdp_v2.conftest import distributed_setup

quantized_dbuffer = pytest.importorskip(
    "megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantized_dbuffer"
)


@pytest.mark.launch_on_gb200
@pytest.mark.parametrize("rows", [32, 96])
def test_mfsdp_compact_quantization_replays(distributed_setup, rows):
    """Requantization overwrites every logical data and scale byte identically."""
    device = distributed_setup.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device)[0] < 10:
        pytest.skip("MXFP8 requires Blackwell-or-newer CUDA hardware.")
    mesh = init_device_mesh(device.type, (distributed_setup.world_size,))
    quantized = quantized_dbuffer.QuantizedDBuffer.empty(mesh, [Replicate()], [(rows, 128)], device)
    main = DBuffer(mesh, [Replicate()], quantized.rowwise_data.layout, torch.float32, device)
    torch.manual_seed(1234)
    main.local_buffer.normal_()
    quantized.quantize_(main)
    expected = [plane.local_buffer.clone() for plane in quantized.planes]
    for plane in quantized.planes:
        plane.local_buffer.fill_(0xAA)
    quantized.quantize_(main)
    for plane, reference in zip(quantized.planes, expected):
        assert torch.equal(plane.local_buffer, reference)
