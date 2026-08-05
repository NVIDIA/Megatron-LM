# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the MFSDP v2 MXFP8 block quantization kernels."""

import pytest
import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.quantization import (
    E4M3_BLOCK_SIZE,
    clear_payloads,
    compute_colwise_amax,
    dequantize_block_e4m3,
    dequantize_colwise_chunk,
    dequantize_rowwise_2d,
    pad_scale_inv,
    quantize_block_e4m3,
    quantize_colwise_with_scales,
    quantize_rowwise_2d,
)


def _encode_reference(values: torch.Tensor) -> torch.Tensor:
    """Scalar reference E4M3 encode (round-half-even, saturation) for cross-checking.

    ``values`` must already be block-scaled (divided by the block scale), so the
    reference encodes at scale 1.
    """
    bits = []
    for value in values.tolist():
        if value != value or value in (float("inf"), float("-inf")):
            bits.append(0b01111110)  # 448
            continue
        sign = 1 if value < 0 else 0
        a = abs(value)
        if a >= 448.0:
            bits.append((sign << 7) | 0b01111110)
            continue
        if a == 0.0:
            bits.append(sign << 7)
            continue
        import math

        exponent = math.floor(math.log2(a))
        if a < 2**-6:
            mant = int(round(a / 2**-9))
            mant = min(max(mant, 0), 7)
            bits.append((sign << 7) | mant)
            continue
        mant = int(round((a / 2**exponent - 1) * 8))
        if mant == 8:
            exponent += 1
            mant = 0
        if exponent + 7 > 15:
            bits.append((sign << 7) | 0b01111110)
            continue
        bits.append((sign << 7) | ((exponent + 7) << 3) | mant)
    return torch.tensor(bits, dtype=torch.uint8)


def _scale_blocks(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Divide each 32-element block by its block scale (amax/448)."""
    padded = torch.zeros(
        (values.numel() + E4M3_BLOCK_SIZE - 1) // E4M3_BLOCK_SIZE * E4M3_BLOCK_SIZE,
        dtype=torch.float32,
    )
    padded[: values.numel()] = values.float()
    blocks_2d = padded.view(-1, E4M3_BLOCK_SIZE)
    scale = blocks_2d.abs().amax(dim=1) / 448.0
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    return (blocks_2d / scale.unsqueeze(1)).reshape(-1), scale


def _roundtrip(values: torch.Tensor) -> torch.Tensor:
    payload, scales = quantize_block_e4m3(values)
    out = torch.empty(values.numel(), dtype=torch.bfloat16)
    dequantize_block_e4m3(payload, scales, out=out)
    return out


class TestE4m3BitEncoding:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (0.0, 0b00000000),
            (2**-9, 0b00000001),  # subnormal
            (2**-6, 0b00001000),  # E=1, M=0
            (1.0, 0b00111000),  # E=7, M=0
            (1.5, 0b00111100),  # E=7, M=4
            (448.0, 0b01111110),  # max normal (E=15, M=6)
            (-448.0, 0b11111110),
            (1024.0, 0b01111110),  # saturate
        ],
    )
    def test_bit_patterns(self, value, expected):
        # Anchor the block with 448.0 so the block scale is exactly 1.
        block = torch.tensor([value, 448.0], dtype=torch.float32)
        bits, _ = quantize_block_e4m3(block)
        assert bits[0].item() == expected

    def test_random_values_match_reference_encode(self):
        torch.manual_seed(42)
        values = torch.randn(1000, dtype=torch.float32) * 100
        bits, _ = quantize_block_e4m3(values)
        scaled, _ = _scale_blocks(values)
        assert torch.equal(bits, _encode_reference(scaled))

    def test_decode_roundtrip_for_e4m3_representable_values(self):
        values = torch.tensor([0.0, 2**-9, 2**-6, 1.0, 1.5, 448.0, -448.0], dtype=torch.float32)
        out = _roundtrip(values)
        assert torch.equal(out.float(), values)


class TestBlockQuantization:
    def test_padding_roundtrip(self):
        torch.manual_seed(0)
        for numel in (1, 31, 32, 33, 64, 100):
            values = torch.randn(numel, dtype=torch.bfloat16)
            out = _roundtrip(values)
            assert out.shape == values.shape
            assert torch.allclose(out.float(), values.float(), rtol=0.1, atol=1e-2)

    def test_blocks_are_independent(self):
        # A large-amplitude block must not affect a tiny-amplitude block.
        values = torch.cat(
            [
                torch.full((32,), 100.0, dtype=torch.bfloat16),
                torch.full((32,), 1e-4, dtype=torch.bfloat16),
            ]
        )
        out = _roundtrip(values)
        assert torch.allclose(out[32:].float(), values[32:].float(), rtol=0.05)
        assert torch.allclose(out[:32].float(), values[:32].float(), rtol=0.05)

    def test_roundtrip_error_bounded_by_block_scale(self):
        torch.manual_seed(1)
        values = torch.randn(256, dtype=torch.bfloat16) * 10
        out = _roundtrip(values)
        error = (out.float() - values.float()).abs()
        # Worst-case E4M3 rounding error is half an ulp at the block scale:
        # amax / 448 * 16 <= amax / 28, about 3.6% of the block max. Keep a
        # 5% bound with margin.
        block_max = values.view(-1, E4M3_BLOCK_SIZE).abs().amax(dim=1)
        bound = block_max.repeat_interleave(E4M3_BLOCK_SIZE)[: values.numel()] * 0.05
        assert (error <= bound).all()

    def test_out_buffers_are_reused(self):
        values = torch.randn(50, dtype=torch.bfloat16)
        payload = torch.empty(64, dtype=torch.uint8)
        scales = torch.empty(2, dtype=torch.bfloat16)
        payload_out, scales_out = quantize_block_e4m3(
            values, out_payload=payload, out_scales=scales
        )
        assert payload_out is payload and scales_out is scales
        out = torch.empty(50, dtype=torch.bfloat16)
        dequantize_block_e4m3(payload, scales, out=out)
        assert torch.allclose(out.float(), values.float(), rtol=0.1, atol=1e-2)

    def test_nan_inf_saturate_to_block_max(self):
        # NaN/Inf are excluded from the block amax and saturate in q-space to
        # 448, so they dequantize to the block's max representable value.
        values = torch.tensor(
            [float("nan"), float("inf"), float("-inf"), -0.0, 1e30], dtype=torch.float32
        )
        out = _roundtrip(values)
        assert torch.allclose(out[0].float(), torch.tensor(1e30), rtol=1e-2)
        assert torch.allclose(out[1].float(), torch.tensor(1e30), rtol=1e-2)
        assert torch.allclose(out[2].float(), torch.tensor(-1e30), rtol=1e-2)
        assert out[3].item() == 0.0
        assert torch.allclose(out[4].float(), torch.tensor(1e30), rtol=1e-2)

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            quantize_block_e4m3(torch.randn(2, 4))
        with pytest.raises(ValueError):
            quantize_block_e4m3(torch.randn(32), out_payload=torch.empty(32, dtype=torch.uint8))
        payload = torch.zeros(64, dtype=torch.uint8)
        scales = torch.zeros(1, dtype=torch.bfloat16)
        with pytest.raises(ValueError):
            dequantize_block_e4m3(payload, scales, out=torch.empty(64, dtype=torch.bfloat16))
        with pytest.raises(ValueError):
            dequantize_block_e4m3(
                payload,
                torch.zeros(2, dtype=torch.bfloat16),
                out=torch.empty(100, dtype=torch.bfloat16),
            )


class TestRowwise2D:
    def test_roundtrip_and_row_independence(self):
        torch.manual_seed(3)
        rows, cols = 16, 128
        tensor = torch.randn(rows, cols, dtype=torch.bfloat16)
        tensor[:4] *= 100  # amplify a few rows: their scales must not leak
        payload = torch.empty(rows * cols, dtype=torch.uint8)
        scales = torch.empty(rows * cols // E4M3_BLOCK_SIZE, dtype=torch.bfloat16)
        quantize_rowwise_2d(tensor, out_payload=payload, out_scales=scales)
        out = torch.empty(rows, cols, dtype=torch.bfloat16)
        dequantize_rowwise_2d(
            payload.view(rows, cols), scales.view(rows, cols // E4M3_BLOCK_SIZE), out=out
        )
        error = (out.float() - tensor.float()).abs()
        block_max = tensor.abs().view(rows, -1, E4M3_BLOCK_SIZE).amax(dim=2)
        bound = block_max.repeat_interleave(E4M3_BLOCK_SIZE, dim=1) * 0.05
        assert (error <= bound).all()
        # The quiet rows must not be affected by the amplified rows.
        assert error[:4].max() > error[4:].max() * 4

    def test_validation_errors(self):
        with pytest.raises(ValueError):
            quantize_rowwise_2d(
                torch.randn(4, 33),
                out_payload=torch.empty(4 * 33, dtype=torch.uint8),
                out_scales=torch.empty(4, dtype=torch.bfloat16),
            )


class TestColumnwiseProtocol:
    """Simulates the two-rank column-wise quantization protocol without dist."""

    def _run(self, height, cols, rows0, rows1, seed=7):
        torch.manual_seed(seed)
        full = torch.randn(height, cols, dtype=torch.bfloat16) * 3
        shard0 = full[:rows0]
        shard1 = full[rows0 : rows0 + rows1]

        partial0 = compute_colwise_amax(shard0, 0, height)
        partial1 = compute_colwise_amax(shard1, rows0, height)
        merged = torch.maximum(partial0, partial1)
        scales = merged / 448.0
        scales = torch.where(scales == 0, torch.ones_like(scales), scales)

        payload0 = torch.empty(rows0 * cols, dtype=torch.uint8)
        payload1 = torch.empty(rows1 * cols, dtype=torch.uint8)
        quantize_colwise_with_scales(shard0, scales, 0, out_payload=payload0)
        quantize_colwise_with_scales(shard1, scales, rows0, out_payload=payload1)

        # The all-gather concatenates rank chunks in group order; each rank's
        # chunk is the (rows, cols) payload of its own rows (same layout as the
        # row-wise payload; only the block direction differs).
        full_payload = torch.cat([payload0, payload1])
        out0 = torch.empty(rows0, cols, dtype=torch.bfloat16)
        out1 = torch.empty(rows1, cols, dtype=torch.bfloat16)
        dequantize_colwise_chunk(
            full_payload[: rows0 * cols].view(rows0, cols), scales, 0, out=out0
        )
        dequantize_colwise_chunk(
            full_payload[rows0 * cols :].view(rows1, cols), scales, rows0, out=out1
        )
        return full, torch.cat([out0, out1])

    def test_roundtrip_with_shard_straddling_blocks(self):
        # Shard boundary cuts through a 32-row block: rows0 = 40 -> block 1
        # spans both ranks.
        full, out = self._run(96, 64, rows0=40, rows1=56)
        error = (out.float() - full.float()).abs()
        block_max = full.abs().view(-1, E4M3_BLOCK_SIZE, 64).amax(dim=1)
        bound = block_max.repeat_interleave(E4M3_BLOCK_SIZE, dim=0) * 0.05
        assert (error <= bound).all()

    def test_aligned_shards(self):
        full, out = self._run(64, 32, rows0=32, rows1=32)
        error = (out.float() - full.float()).abs()
        assert error.max() < 1.0

    def test_scales_are_global_across_shards(self):
        # A large value in rank 1's rows of block 1 must scale rank 0's rows
        # of the same block.
        height, cols = 96, 32
        full = torch.ones(height, cols, dtype=torch.bfloat16) * 0.01
        full[33, 3] = 50.0  # rank 1 rows, block 1
        shard0, shard1 = full[:40], full[40:]
        partial0 = compute_colwise_amax(shard0, 0, height)
        partial1 = compute_colwise_amax(shard1, 40, height)
        merged = torch.maximum(partial0, partial1)
        assert merged[1, 3].item() == pytest.approx(50.0)

        scales = merged / 448.0
        payload0 = torch.empty(40 * cols, dtype=torch.uint8)
        quantize_colwise_with_scales(shard0, scales, 0, out_payload=payload0)
        out0 = torch.empty(40, cols, dtype=torch.bfloat16)
        dequantize_colwise_chunk(payload0.view(40, cols), scales, 0, out=out0)
        # Rank 0's rows in block 1 are quantized with the global block scale.
        assert torch.allclose(out0[32:40, 3].float(), shard0[32:40, 3].float(), atol=0.05)


class TestPayloadHelpers:
    def test_clear_payloads(self):
        tensor = torch.zeros(4, 4)
        tensor._rowwise_data = torch.zeros(4, 4)
        tensor._columnwise_data = torch.zeros(4, 4)
        clear_payloads(tensor)
        assert tensor._rowwise_data is None
        assert tensor._columnwise_data is None


class TestPadScaleInv:
    def test_padding(self):
        grid = torch.ones(65, 9)
        padded = pad_scale_inv(grid, 128, 4)
        assert padded.shape == (128, 12)
        assert padded[65:].sum() == 0 and padded[:, 9:].sum() == 0
        assert torch.equal(padded[:65, :9], grid)

    def test_no_padding_when_already_multiple(self):
        grid = torch.ones(128, 8)
        assert pad_scale_inv(grid, 128, 4) is grid
