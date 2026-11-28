# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/b57951ad292e43209f1cc7d4b5500768365a20ac/tests/unit_tests/models/test_quantization_utils.py
# Source license: Apache-2.0.

import math
from types import SimpleNamespace

import pytest
import torch

from megatron.bridge.models.conversion.quantization_utils import (
    dequantize_fp8_scale_inv,
    dequantize_fp8_scale_inv_weights,
    fp8_weight_block_size,
)


def _quantize_fp8_blocks(weight, block_size):
    """Quantize a 2-D weight in the fine-grained FP8 format: one scale per block, ceil-shaped grid."""
    rows, cols = block_size
    m, n = weight.shape
    scale_inv = torch.empty(math.ceil(m / rows), math.ceil(n / cols))
    codes = torch.empty(m, n, dtype=torch.float8_e4m3fn)
    for i in range(scale_inv.shape[0]):
        for j in range(scale_inv.shape[1]):
            tile = weight[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols]
            scale_inv[i, j] = tile.abs().max() / 448.0
            codes[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] = (
                tile / scale_inv[i, j]
            ).to(torch.float8_e4m3fn)
    return codes, scale_inv


def test_bridge_import_applies_scales_without_rescaling_custom_hook_results(monkeypatch):
    from megatron.bridge.models.qwen.qwen35_bridge import Qwen35Bridge

    bridge = Qwen35Bridge()
    bridge.hf_config = SimpleNamespace(quantization_config={"weight_block_size": [2, 2]})
    codes = torch.ones(4, 4, dtype=torch.float8_e4m3fn)
    state = {"weight": codes, "weight_scale_inv": torch.tensor([[2.0, 3.0], [4.0, 5.0]])}
    expected = torch.tensor(
        [[2.0, 2.0, 3.0, 3.0], [2.0, 2.0, 3.0, 3.0], [4.0, 4.0, 5.0, 5.0], [4.0, 4.0, 5.0, 5.0]]
    )
    torch.testing.assert_close(bridge._load_hf_weights("weight", state), expected)
    already_dequantized = expected.to(torch.bfloat16)
    monkeypatch.setattr(bridge, "maybe_modify_loaded_hf_weight", lambda *_: already_dequantized)
    assert bridge._load_hf_weights("weight", state) is already_dequantized


def _dequantize_fp8_blocks_reference(codes, scale_inv, block_size):
    rows, cols = block_size
    out = codes.float()
    for i in range(scale_inv.shape[0]):
        for j in range(scale_inv.shape[1]):
            out[i * rows : (i + 1) * rows, j * cols : (j + 1) * cols] *= scale_inv[i, j]
    return out


@pytest.mark.parametrize(
    ("shape", "block_size"),
    [((256, 384), (128, 128)), ((200, 320), (128, 128)), ((96, 80), (64, 32))],
)
def test_dequantize_fp8_scale_inv_scales_every_block_including_partial_ones(shape, block_size):
    torch.manual_seed(0)
    codes, scale_inv = _quantize_fp8_blocks(torch.randn(shape), block_size)

    result = dequantize_fp8_scale_inv(codes, scale_inv, block_size=block_size)

    assert result.dtype == torch.float32
    torch.testing.assert_close(
        result, _dequantize_fp8_blocks_reference(codes, scale_inv, block_size), rtol=0, atol=0
    )


def test_dequantize_fp8_scale_inv_applies_one_scale_to_the_whole_tensor():
    codes = torch.tensor([[1.5, -2.0], [0.5, 4.0]], dtype=torch.float8_e4m3fn)

    for scale_inv in (torch.tensor([[0.25]]), torch.tensor(0.25)):
        result = dequantize_fp8_scale_inv(codes, scale_inv, block_size=(128, 128))
        torch.testing.assert_close(result, codes.float() * 0.25, rtol=0, atol=0)


def test_dequantize_fp8_scale_inv_handles_stacked_expert_grids():
    torch.manual_seed(0)
    pairs = [_quantize_fp8_blocks(torch.randn(160, 128), (128, 128)) for _ in range(3)]
    codes = torch.stack([c for c, _ in pairs])
    scale_inv = torch.stack([s for _, s in pairs])

    result = dequantize_fp8_scale_inv(codes, scale_inv)

    expected = torch.stack([_dequantize_fp8_blocks_reference(c, s, (128, 128)) for c, s in pairs])
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_dequantize_fp8_scale_inv_rejects_a_grid_that_does_not_match_the_block_size():
    codes = torch.ones(256, 256, dtype=torch.float8_e4m3fn)

    with pytest.raises(ValueError, match="layer.weight.*128x128 blocks need \\(2, 2\\)"):
        dequantize_fp8_scale_inv(
            codes, torch.ones(4, 4), block_size=(128, 128), name="layer.weight"
        )
    with pytest.raises(ValueError, match="layer.bias.*at least two weight dimensions"):
        dequantize_fp8_scale_inv(
            torch.ones(8, dtype=torch.float8_e4m3fn), torch.ones(2), name="layer.bias"
        )


def test_fp8_weight_block_size_reads_the_quantization_config():
    assert fp8_weight_block_size(
        SimpleNamespace(quantization_config={"weight_block_size": [64, 32]})
    ) == (64, 32)
    to_dict_config = SimpleNamespace(
        to_dict=lambda: {"quant_method": "fp8", "weight_block_size": (128, 128)}
    )
    assert fp8_weight_block_size(SimpleNamespace(quantization_config=to_dict_config)) == (128, 128)
    assert (
        fp8_weight_block_size(SimpleNamespace(quantization_config={"quant_method": "fp8"})) is None
    )
    assert fp8_weight_block_size(SimpleNamespace()) is None
    assert fp8_weight_block_size(None) is None


def test_dequantize_fp8_scale_inv_weights_only_touches_fp8_tensors_with_a_companion():
    torch.manual_seed(0)
    codes, scale_inv = _quantize_fp8_blocks(torch.randn(130, 70), (128, 128))
    norm = torch.ones(70, dtype=torch.bfloat16)
    unscaled = torch.ones(2, 2, dtype=torch.float8_e4m3fn)
    state = {
        "q.weight": codes,
        "q.weight_scale_inv": scale_inv,
        "norm.weight": norm,
        "other.weight": unscaled,
    }

    single = dequantize_fp8_scale_inv_weights("q.weight", codes, state)
    torch.testing.assert_close(
        single, _dequantize_fp8_blocks_reference(codes, scale_inv, (128, 128)), rtol=0, atol=0
    )

    names = {"q": "q.weight", "norm": "norm.weight", "other": "other.weight"}
    loaded = {"q": codes, "norm": norm, "other": unscaled}
    compound = dequantize_fp8_scale_inv_weights(names, loaded, state)
    torch.testing.assert_close(compound["q"], single, rtol=0, atol=0)
    assert compound["norm"] is norm
    assert compound["other"] is unscaled

    plain = {"norm": norm}
    assert dequantize_fp8_scale_inv_weights({"norm": "norm.weight"}, plain, state) is plain
    assert dequantize_fp8_scale_inv_weights("norm.weight", norm, state) is norm
