# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real FSDP2 coverage for native block-FP8 source-scale preservation."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch import nn

from megatron.lite.model.deepseek_v4.vllm.primitive.block_fp8 import (
    bind_source_scale_to_visible_weight,
)
from megatron.lite.primitive.optimizers.fsdp2 import (
    build_fsdp2_training_optimizer,
    fsdp2_available,
)
from megatron.lite.primitive.parallel.state import ParallelState


class _NativeFP8Block(nn.Module):
    def __init__(self, master: torch.Tensor, scale: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(master)
        self.weight._fp8_source_scales = scale
        self.weight._fp8_source_scale_version = self.weight._version
        self._fp8_source_scales_by_parameter = {"weight": scale}
        self.observed_qweight = None
        self.observed_scale = None
        self.observed_source_scale = None
        self.observed_source_version = None
        self.observed_weight_version = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bind_source_scale_to_visible_weight(self, "weight", self.weight)
        self.observed_source_scale = getattr(self.weight, "_fp8_source_scales", None)
        self.observed_source_version = getattr(
            self.weight, "_fp8_source_scale_version", None
        )
        self.observed_weight_version = self.weight._version
        if self.observed_source_scale is None or (
            self.observed_source_version != self.observed_weight_version
        ):
            return x + self.weight[0, 0] * 0
        expanded = self.observed_source_scale.repeat_interleave(128, 0)
        expanded = expanded.repeat_interleave(128, 1)
        self.observed_qweight = (
            self.weight.detach().float().div(expanded).to(torch.float8_e4m3fn)
        )
        self.observed_scale = self.observed_source_scale.detach().clone()
        return x + self.weight[0, 0] * 0


@unittest.skipUnless(
    torch.cuda.is_available() and fsdp2_available(), "CUDA FSDP2 required"
)
class NativeFP8FSDP2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("LOCAL_RANK", "0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29517")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        cls.created_process_group = not dist.is_initialized()
        if cls.created_process_group:
            dist.init_process_group("nccl", init_method="env://")

    @classmethod
    def tearDownClass(cls):
        if cls.created_process_group:
            dist.destroy_process_group()

    def test_native_fp8_bytes_survive_fp32_shard_promotion_and_visible_forward(self):
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        qweight = torch.arange(128 * 128, device=device).reshape(128, 128)
        qweight = ((qweight % 31) - 15).to(torch.float8_e4m3fn)
        scale = torch.tensor([[2.0**-12]], device=device, dtype=torch.float32)
        master = (qweight.float() * scale.item()).to(torch.bfloat16)
        model = _NativeFP8Block(master, scale).to(device)
        ps = ParallelState(
            dp_group=dist.group.WORLD,
            dp_cp_group=dist.group.WORLD,
            dp_size=dist.get_world_size(),
            dp_cp_size=dist.get_world_size(),
            dp_rank=dist.get_rank(),
            dp_cp_rank=dist.get_rank(),
        )
        build_fsdp2_training_optimizer(
            [model],
            SimpleNamespace(
                optimizer="adam",
                lr=1.0e-6,
                weight_decay=0.0,
                adam_beta1=0.9,
                adam_beta2=0.95,
                adam_eps=1.0e-8,
                clip_grad=1.0,
                offload_fraction=0.0,
            ),
            ps,
            unit_modules=(),
            use_fp32_shards=True,
            cast_forward_inputs=False,
        )

        model(torch.zeros(1, device=device, dtype=torch.bfloat16))

        self.assertIsNotNone(model.observed_source_scale)
        self.assertEqual(model.observed_source_version, model.observed_weight_version)
        self.assertTrue(torch.equal(model.observed_qweight, qweight))
        self.assertTrue(torch.equal(model.observed_scale, scale))


if __name__ == "__main__":
    unittest.main()
