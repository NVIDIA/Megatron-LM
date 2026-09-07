from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from megatron.lite.model.deepseek_v4.vllm.primitive.attention.module import (
    VLLMAttention,
)


class NativeFP8ProjectionScaleTest(unittest.TestCase):
    def test_all_input_projection_views_rebind_checkpoint_scales(self) -> None:
        def projection(value: float):
            module = nn.Linear(8, 8, bias=False, dtype=torch.bfloat16)
            scale = torch.full((1, 1), value, dtype=torch.float32)
            module._fp8_source_scales_by_parameter = {"weight": scale}
            return module, scale

        attention = SimpleNamespace()
        attention.wq_a, q_scale = projection(1.0)
        attention.wkv, kv_scale = projection(2.0)
        compressor_wkv, compressor_kv_scale = projection(3.0)
        compressor_wgate, compressor_gate_scale = projection(4.0)
        attention.compressor = SimpleNamespace(
            wkv=compressor_wkv, wgate=compressor_wgate
        )
        indexer_wkv, indexer_kv_scale = projection(5.0)
        indexer_wgate, indexer_gate_scale = projection(6.0)
        attention.indexer = SimpleNamespace(
            compressor=SimpleNamespace(wkv=indexer_wkv, wgate=indexer_wgate)
        )

        VLLMAttention._bind_input_projection_source_scales(attention)

        expected = (
            (attention.wq_a, q_scale),
            (attention.wkv, kv_scale),
            (compressor_wkv, compressor_kv_scale),
            (compressor_wgate, compressor_gate_scale),
            (indexer_wkv, indexer_kv_scale),
            (indexer_wgate, indexer_gate_scale),
        )
        for module, scale in expected:
            self.assertIs(module.weight._fp8_source_scales, scale)
            self.assertEqual(
                module.weight._fp8_source_scale_version, module.weight._version
            )


if __name__ == "__main__":
    unittest.main()
