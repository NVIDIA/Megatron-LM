# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.lite.primitive.train_step import compute_and_clip_grad_norm, run_microbatch_loop


def test_train_step_microbatch_loop_and_grad_clip_cpu_contract():
    model = nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.fill_(0.5)
    data = iter(
        [
            {"x": torch.tensor([[1.0, 2.0]]), "y": torch.tensor([[1.0]])},
            {"x": torch.tensor([[3.0, 4.0]]), "y": torch.tensor([[2.0]])},
        ]
    )

    def forward_fn(module, batch):
        return {"loss": F.mse_loss(module(batch["x"]), batch["y"])}

    output = run_microbatch_loop(model, data, 2, forward_fn)

    assert output is not None
    assert output["loss"].ndim == 0
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all()

    grad_norm = compute_and_clip_grad_norm(model, optimizer=None, max_norm=0.25, use_dist_opt=False)

    assert torch.isfinite(grad_norm)
    assert model.weight.grad.norm() <= 0.25 + 1.0e-6
