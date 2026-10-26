# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Pipelined fused CPU-offload AdamW (`_step_param_groups_pipelined`): numeric parity
vs the serial path, D2H readiness, mixed dtype, big-param splitting, bounded pinned
slots, and env-var validation.

The pipeline is opt-in via ``overlap_cpu_optimizer_d2h_h2d=True`` (CPU-offload only);
it groups params into equal-size slots, overlaps grad D2H / fused CPU kernel / param
H2D on dedicated streams, and casts the fp32 master back to param dtype on the GPU.
"""
from __future__ import annotations

import copy
import os
from unittest.mock import patch

import pytest
import torch

from megatron.lite.primitive.optimizers.fsdp2.adamw import FP32AdamW

pytestmark = pytest.mark.gpus(1)


def _make_opt(params, pipeline=False):
    return FP32AdamW(
        [
            {"params": params[:2], "lr": 0.01, "weight_decay": 0.1},
            {"params": params[2:], "lr": 0.003, "weight_decay": 0.0},
        ],
        lr=0.01,
        weight_decay=0.1,
        betas=(0.9, 0.999),
        eps=1e-8,
        cpu_update=True,
        pipeline=pipeline,
    )


def _run_parity(params, depth, slot_numel):
    """Run pipeline vs serial for 8 steps (incl. a checkpoint round-trip and some
    grad-less params) and assert master/moments/step/param all match."""
    refs = [torch.nn.Parameter(p.detach().clone().cpu()) for p in params]
    pipe, serial = _make_opt(params, pipeline=True), _make_opt(refs)
    env = {
        "MLITE_FSDP2_ADAMW_SLOT_NUMEL": str(slot_numel),
        "MLITE_FSDP2_ADAMW_PIPELINE_DEPTH": str(depth),
    }
    with patch.dict(os.environ, env):
        for step in range(8):
            for i, (p, ref) in enumerate(zip(params, refs)):
                g = None if (step + i) % 4 == 0 else torch.randn_like(p)
                p.grad = g
                ref.grad = None if g is None else g.detach().clone().cpu()
            pipe._step_param_groups_pipelined()
            serial._step_param_groups_serial()
            if step == 3:  # checkpoint save/load must not perturb the run
                saved = copy.deepcopy(pipe.state_dict())
                pipe = _make_opt(params, pipeline=True)
                pipe.load_state_dict(saved)
            for p, ref in zip(params, refs):
                for key in ("master_param", "exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        pipe.state[p][key].cpu(),
                        serial.state[ref][key],
                        atol=2e-6,
                        rtol=2e-5,
                        msg=f"{key} step{step}",
                    )
                assert pipe.state[p]["step"] == serial.state[ref]["step"]
                torch.testing.assert_close(
                    p.detach().cpu(), ref.detach(), atol=0.02, rtol=0
                )


def _bf16_params(dev, n):
    torch.manual_seed(7)
    return [
        torch.nn.Parameter(torch.randn(3 + i, 5, dtype=torch.bfloat16, device=dev))
        for i in range(n)
    ]


def test_dispatch_follows_pipeline_flag():
    # The pipeline flag (from overlap_cpu_optimizer_d2h_h2d) selects the step path.
    for pipeline in (False, True):
        opt = _make_opt([], pipeline=pipeline)
        with patch.object(opt, "_step_param_groups_serial") as serial:
            with patch.object(opt, "_step_param_groups_pipelined") as pipe:
                opt.step()
                assert pipe.call_count == int(pipeline)
                assert serial.call_count == int(not pipeline)


@pytest.mark.parametrize(
    "name", ["MLITE_FSDP2_ADAMW_SLOT_NUMEL", "MLITE_FSDP2_ADAMW_PIPELINE_DEPTH"]
)
def test_non_integer_env_var_raises(name):
    # A non-integer knob surfaces int()'s ValueError rather than silently misbehaving;
    # non-positive values are floored to 1 by max(1, ...), so only non-ints raise.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    params = _bf16_params(dev, 3)
    opt = _make_opt(params, pipeline=True)
    for p in params:
        p.grad = torch.randn_like(p)
    with patch.dict(os.environ, {name: "abc"}):
        with pytest.raises(ValueError):
            opt._step_param_groups_pipelined()


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("slot", [7, 40, 10**9])  # split most / pack several / one slot
def test_depth_slot_variants(depth, slot):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    _run_parity(_bf16_params(dev, 7), depth, slot)


@pytest.mark.parametrize("depth", [1, 2, 3])
@pytest.mark.parametrize("slot", [5, 33, 10**9])
def test_mixed_dtype(depth, slot):
    # fp32 + bf16 params: per-fragment GPU cast handles both; must land correctly.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(1)
    params = [
        torch.nn.Parameter(torch.randn(6, 5, dtype=torch.bfloat16, device=dev)),
        torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32, device=dev)),
        torch.nn.Parameter(torch.randn(9, 3, dtype=torch.bfloat16, device=dev)),
        torch.nn.Parameter(torch.randn(7, 2, dtype=torch.float32, device=dev)),
    ]
    _run_parity(
        [torch.nn.Parameter(p.detach().clone()) for p in params], depth, slot
    )


def test_big_param_split():
    # A single param far larger than the slot must split across many slots.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    torch.manual_seed(2)
    params = [
        torch.nn.Parameter(torch.randn(200, 50, dtype=torch.bfloat16, device=dev)),
        torch.nn.Parameter(torch.randn(30, 10, dtype=torch.bfloat16, device=dev)),
        torch.nn.Parameter(torch.randn(64, 64, dtype=torch.bfloat16, device=dev)),
    ]
    _run_parity(
        [torch.nn.Parameter(p.detach().clone()) for p in params],
        depth=3,
        slot_numel=128,
    )


def test_pinned_pool_bounded():
    # Equal-size slots: resident pinned grad buffers are bounded by the in-flight
    # window (depth+1), not the parameter count.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    params = [
        torch.nn.Parameter(torch.randn(16, 4, dtype=torch.bfloat16, device=dev))
        for _ in range(20)
    ]
    opt = _make_opt(params, pipeline=True)
    depth = 3
    with patch.dict(
        os.environ,
        {
            "MLITE_FSDP2_ADAMW_SLOT_NUMEL": "8",
            "MLITE_FSDP2_ADAMW_PIPELINE_DEPTH": str(depth),
        },
    ):
        for _ in range(5):
            for p in params:
                p.grad = torch.randn_like(p)
            opt._step_param_groups_pipelined()
    total = sum(len(v) for v in opt._pin_pool.values())
    assert total <= depth + 1, f"pinned pool grew to {total} buffers"


def test_free_pin_pool():
    # MLITE_FSDP2_ADAMW_FREE_PIN_AFTER_OPT=1 must empty the pool after each step.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    params = [
        torch.nn.Parameter(torch.randn(16, 4, dtype=torch.bfloat16, device=dev))
        for _ in range(6)
    ]
    opt = _make_opt(params, pipeline=True)
    with patch.dict(
        os.environ,
        {
            "MLITE_FSDP2_ADAMW_SLOT_NUMEL": "8",
            "MLITE_FSDP2_ADAMW_FREE_PIN_AFTER_OPT": "1",
        },
    ):
        for p in params:
            p.grad = torch.randn_like(p)
        opt._step_param_groups_pipelined()
    assert sum(len(v) for v in opt._pin_pool.values()) == 0


def test_delayed_d2h_before_cpu_kernel():
    # Negative control: the fused CPU kernel must not read a grad slot before its async
    # D2H lands. Delay the grad fill on a side stream, then assert the staged grad holds
    # the expected value at kernel entry and the final state matches serial.
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    dev = torch.device("cuda", torch.cuda.current_device())
    for dtype in (torch.float32, torch.bfloat16):
        stream = torch.cuda.Stream(device=dev)
        p = torch.nn.Parameter(torch.ones(65536, device=dev, dtype=dtype))
        p.grad = torch.empty_like(p)
        opt = _make_opt([p], pipeline=True)
        ref = torch.nn.Parameter(p.detach().cpu())
        ref.grad = torch.full_like(ref, 3)
        reference = _make_opt([ref])
        reference._step_param_groups_serial()
        torch.cuda.synchronize(dev)

        original_fused = torch._fused_adamw_
        captured = {}

        def checked_fused(masters, grads, *args, **kwargs):
            captured["ok"] = torch.allclose(grads[0], torch.full_like(grads[0], 3.0))
            return original_fused(masters, grads, *args, **kwargs)

        with patch.dict(os.environ, {"MLITE_FSDP2_ADAMW_PIPELINE_DEPTH": "2"}):
            with torch.cuda.stream(stream):
                torch.cuda._sleep(200_000_000)  # delay the grad fill + D2H
                p.grad.fill_(3)
                with patch("torch._fused_adamw_", side_effect=checked_fused):
                    opt._step_param_groups_pipelined()
            stream.synchronize()
        assert captured.get("ok"), f"grad slot read before D2H ({dtype})"
        for key in ("master_param", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                opt.state[p][key], reference.state[ref][key], atol=1e-6, rtol=1e-5
            )
        torch.testing.assert_close(p.cpu(), ref, atol=0.016, rtol=0)
