# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import argparse
import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

try:
    import flashoptim  # noqa: F401

    HAVE_FLASHOPTIM = True
except ImportError:
    HAVE_FLASHOPTIM = False

from megatron.core.optimizer.optimizer_config import OptimizerConfig

requires_flashoptim = pytest.mark.skipif(
    not HAVE_FLASHOPTIM, reason="flashoptim>=0.1.3 not installed"
)
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bf16_model(in_features=64, out_features=32):
    return (
        nn.Sequential(
            nn.Linear(in_features, out_features, bias=False),
            nn.Linear(out_features, out_features, bias=False),
        )
        .to(torch.bfloat16)
        .cuda()
    )


def _train_steps(model, optimizer, steps=3, in_features=64, seed=0):
    torch.manual_seed(seed)
    losses = []
    for _ in range(steps):
        x = torch.randn(4, in_features, dtype=torch.bfloat16, device='cuda')
        loss = model(x).sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    return losses


def test_config_defaults():
    """Default flashadamw fields match documented values."""
    cfg = OptimizerConfig(optimizer='flashadamw', bf16=True, lr=1e-3)
    assert cfg.flashadamw_master_weight_bits == 24
    assert cfg.flashadamw_quantize is True
    assert cfg.flashadamw_compress_state_dict is False


def test_config_custom_32bit():
    """OptimizerConfig accepts master_weight_bits=32."""
    cfg = OptimizerConfig(
        optimizer='flashadamw', bf16=True, lr=1e-3, flashadamw_master_weight_bits=32
    )
    assert cfg.flashadamw_master_weight_bits == 32


def test_config_none_master_weight():
    """OptimizerConfig accepts master_weight_bits=None (pure BF16, no ECC)."""
    cfg = OptimizerConfig(
        optimizer='flashadamw', bf16=True, lr=1e-3, flashadamw_master_weight_bits=None
    )
    assert cfg.flashadamw_master_weight_bits is None


def test_config_compress_state_dict():
    """OptimizerConfig accepts compress_state_dict=True."""
    cfg = OptimizerConfig(
        optimizer='flashadamw', bf16=True, lr=1e-3, flashadamw_compress_state_dict=True
    )
    assert cfg.flashadamw_compress_state_dict is True


def test_config_no_quantize():
    """OptimizerConfig accepts quantize=False."""
    cfg = OptimizerConfig(optimizer='flashadamw', bf16=True, lr=1e-3, flashadamw_quantize=False)
    assert cfg.flashadamw_quantize is False


def _make_args(**kwargs):
    """Return a minimal Namespace that passes the flashadamw validation block."""
    defaults = dict(
        optimizer='flashadamw',
        flashadamw_master_weight_bits=24,
        flashadamw_no_master_weight_bits=False,
        flashadamw_quantize=True,
        flashadamw_compress_state_dict=False,
        use_distributed_optimizer=False,
        use_precision_aware_optimizer=False,
        optimizer_cpu_offload=False,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _run_flashadamw_validation(args):
    """Reproduce the validate_args block for flashadamw."""
    if args.optimizer == 'flashadamw':
        if args.flashadamw_no_master_weight_bits:
            args.flashadamw_master_weight_bits = None
        assert (
            not args.use_distributed_optimizer
        ), '--optimizer flashadamw is not compatible with --use-distributed-optimizer.'
        assert (
            not args.use_precision_aware_optimizer
        ), '--optimizer flashadamw is not compatible with --use-precision-aware-optimizer.'
        assert (
            not args.optimizer_cpu_offload
        ), '--optimizer flashadamw is not compatible with --optimizer-cpu-offload.'


def test_validate_args_ok():
    """Valid flashadamw args should pass without error."""
    _run_flashadamw_validation(_make_args())


def test_validate_args_no_master_weight_sets_none():
    """--flashadamw-no-master-weight-bits should set master_weight_bits=None."""
    args = _make_args(flashadamw_no_master_weight_bits=True)
    _run_flashadamw_validation(args)
    assert args.flashadamw_master_weight_bits is None


def test_validate_args_rejects_dist_optimizer():
    """flashadamw + use_distributed_optimizer should raise AssertionError."""
    args = _make_args(use_distributed_optimizer=True)
    with pytest.raises(AssertionError, match='not compatible with --use-distributed-optimizer'):
        _run_flashadamw_validation(args)


def test_validate_args_rejects_precision_aware():
    """flashadamw + use_precision_aware_optimizer should raise AssertionError."""
    args = _make_args(use_precision_aware_optimizer=True)
    with pytest.raises(AssertionError, match='not compatible with --use-precision-aware-optimizer'):
        _run_flashadamw_validation(args)


def test_validate_args_rejects_cpu_offload():
    """flashadamw + optimizer_cpu_offload should raise AssertionError."""
    args = _make_args(optimizer_cpu_offload=True)
    with pytest.raises(AssertionError, match='not compatible with --optimizer-cpu-offload'):
        _run_flashadamw_validation(args)


@requires_flashoptim
@requires_cuda
def test_instantiation_24bit():
    """FlashAdamW(master_weight_bits=24) instantiates and runs one step."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)
    assert isinstance(opt, FlashAdamW)
    _train_steps(model, opt, steps=1)


@requires_flashoptim
@requires_cuda
def test_instantiation_32bit():
    """FlashAdamW(master_weight_bits=32) instantiates and runs one step."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=32)
    _train_steps(model, opt, steps=1)


@requires_flashoptim
@requires_cuda
def test_instantiation_no_master_weight():
    """FlashAdamW(master_weight_bits=None) instantiates and runs one step."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=None)
    _train_steps(model, opt, steps=1)


@requires_flashoptim
@requires_cuda
def test_state_empty_before_step():
    """Optimizer state should be empty before any step or init call."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3)
    for p in model.parameters():
        assert len(opt.state[p]) == 0


@requires_flashoptim
@requires_cuda
def test_init_state_fn_populates_all_params():
    """Calling step_param for each parameter should populate all states."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)
    for group in opt.param_groups:
        for p in group['params']:
            if len(opt.state[p]) == 0:
                opt.step_param(p, group)
    for p in model.parameters():
        assert len(opt.state[p]) > 0, f"State missing for param shape={p.shape}"


@requires_flashoptim
@requires_cuda
def test_state_contains_expected_keys():
    """After a step, optimizer state should contain exp_avg and exp_avg_sq."""
    from flashoptim import FlashAdamW

    model = nn.Linear(32, 16, bias=False).to(torch.bfloat16).cuda()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24)
    _train_steps(model, opt, steps=1, in_features=32)
    for p in model.parameters():
        state_keys = set(opt.state[p].keys())
        assert 'exp_avg' in state_keys
        assert 'exp_avg_sq' in state_keys


@requires_flashoptim
@requires_cuda
def test_checkpoint_roundtrip_loss_consistency():
    """Resumed training after checkpoint load should reproduce the same loss."""
    from flashoptim import FlashAdamW

    torch.manual_seed(42)
    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24, quantize=True)

    # Train a few steps to build up optimizer state.
    _train_steps(model, opt, steps=3)

    # Save model + optimizer state.
    model_sd = copy.deepcopy(model.state_dict())
    opt_sd = copy.deepcopy(opt.state_dict())

    # Advance one more step and record loss.
    torch.manual_seed(99)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss_original = model(x).sum()
    loss_original.backward()
    opt.step()
    opt.zero_grad()

    # Restore model + optimizer from the saved checkpoint.
    model2 = _make_bf16_model()
    model2.load_state_dict(model_sd)
    opt2 = FlashAdamW(model2.parameters(), lr=1e-3, master_weight_bits=24, quantize=True)
    opt2.load_state_dict(opt_sd)

    # Same step should reproduce the same loss (within BF16 rounding tolerance).
    torch.manual_seed(99)
    x2 = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss_resumed = model2(x2).sum()
    loss_resumed.backward()
    opt2.step()
    opt2.zero_grad()

    assert abs(loss_original.item() - loss_resumed.item()) < 0.05, (
        f"Loss mismatch after checkpoint roundtrip: "
        f"{loss_original.item():.6f} vs {loss_resumed.item():.6f}"
    )


@requires_flashoptim
@requires_cuda
def test_checkpoint_roundtrip_no_master_weight():
    """Checkpoint roundtrip should work with master_weight_bits=None."""
    from flashoptim import FlashAdamW

    torch.manual_seed(1)
    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=None, quantize=True)
    _train_steps(model, opt, steps=2)

    model_sd = copy.deepcopy(model.state_dict())
    opt_sd = copy.deepcopy(opt.state_dict())

    # One more step.
    torch.manual_seed(7)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss1 = model(x).sum()
    loss1.backward()
    opt.step()
    opt.zero_grad()

    # Resume.
    model2 = _make_bf16_model()
    model2.load_state_dict(model_sd)
    opt2 = FlashAdamW(model2.parameters(), lr=1e-3, master_weight_bits=None, quantize=True)
    opt2.load_state_dict(opt_sd)

    torch.manual_seed(7)
    x2 = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss2 = model2(x2).sum()
    loss2.backward()
    opt2.step()
    opt2.zero_grad()

    assert (
        abs(loss1.item() - loss2.item()) < 0.05
    ), f"Roundtrip loss mismatch (no master weight): {loss1.item():.6f} vs {loss2.item():.6f}"


@requires_flashoptim
@requires_cuda
def test_compressed_checkpoint_saves_smaller_file():
    """compress_state_dict=True checkpoint file should be smaller than uncompressed."""
    from flashoptim import FlashAdamW

    model = nn.Linear(512, 512, bias=False).to(torch.bfloat16).cuda()
    _train_steps(
        model,
        FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24),
        steps=1,
        in_features=512,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        p = Path(tmpdir)

        opt_normal = FlashAdamW(
            model.parameters(), lr=1e-3, master_weight_bits=24, compress_state_dict=False
        )
        _train_steps(model, opt_normal, steps=1, in_features=512)
        torch.save(opt_normal.state_dict(), str(p / 'normal.pt'))

        opt_compressed = FlashAdamW(
            model.parameters(), lr=1e-3, master_weight_bits=24, compress_state_dict=True
        )
        _train_steps(model, opt_compressed, steps=1, in_features=512)
        torch.save(opt_compressed.state_dict(), str(p / 'compressed.pt'))

        normal_size = (p / 'normal.pt').stat().st_size
        compressed_size = (p / 'compressed.pt').stat().st_size

    assert compressed_size < normal_size, (
        f"Compressed checkpoint ({compressed_size} B) should be smaller than "
        f"uncompressed ({normal_size} B)"
    )


@requires_flashoptim
@requires_cuda
def test_compressed_checkpoint_roundtrip():
    """Training resumed from a compressed checkpoint should reproduce the same loss."""
    from flashoptim import FlashAdamW

    torch.manual_seed(5)
    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24, compress_state_dict=True)
    _train_steps(model, opt, steps=2)

    model_sd = copy.deepcopy(model.state_dict())
    opt_sd = copy.deepcopy(opt.state_dict())

    torch.manual_seed(11)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss1 = model(x).sum()
    loss1.backward()
    opt.step()
    opt.zero_grad()

    model2 = _make_bf16_model()
    model2.load_state_dict(model_sd)
    opt2 = FlashAdamW(model2.parameters(), lr=1e-3, master_weight_bits=24, compress_state_dict=True)
    opt2.load_state_dict(opt_sd)

    torch.manual_seed(11)
    x2 = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss2 = model2(x2).sum()
    loss2.backward()
    opt2.step()
    opt2.zero_grad()

    assert abs(loss1.item() - loss2.item()) < 0.05, (
        f"Compressed checkpoint roundtrip loss mismatch: "
        f"{loss1.item():.6f} vs {loss2.item():.6f}"
    )


@requires_flashoptim
@requires_cuda
def test_loss_parity_with_adamw():
    """FlashAdamW and AdamW should produce similar losses from the same init.

    Both optimizers run on BF16 models with identical initial weights.
    FlashAdamW uses quantised states; AdamW runs in full fp32 optimizer state.
    Losses should stay within a loose tolerance over 5 steps.
    """
    from flashoptim import FlashAdamW

    torch.manual_seed(0)
    model_flash = _make_bf16_model()
    torch.manual_seed(0)
    model_adam = _make_bf16_model()
    # Ensure identical initial weights.
    model_adam.load_state_dict(model_flash.state_dict())

    opt_flash = FlashAdamW(
        model_flash.parameters(), lr=1e-3, weight_decay=0.01, master_weight_bits=24
    )
    # AdamW on BF16 parameters directly (standard PyTorch behaviour).
    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=1e-3, weight_decay=0.01)

    losses_flash, losses_adam = [], []
    torch.manual_seed(42)
    for _ in range(5):
        x = torch.randn(8, 64, dtype=torch.bfloat16, device='cuda')

        loss_f = model_flash(x).sum()
        loss_f.backward()
        opt_flash.step()
        opt_flash.zero_grad()
        losses_flash.append(loss_f.item())

        loss_a = model_adam(x).sum()
        loss_a.backward()
        opt_adam.step()
        opt_adam.zero_grad()
        losses_adam.append(loss_a.item())

    max_delta = max(abs(lf - la) for lf, la in zip(losses_flash, losses_adam))
    assert max_delta < 2.0, (
        f"FlashAdamW and AdamW losses diverged too much (max delta={max_delta:.4f}).\n"
        f"FlashAdamW: {losses_flash}\nAdamW:      {losses_adam}"
    )


@requires_flashoptim
@requires_cuda
def test_no_master_weight_training_converges():
    """Training with master_weight_bits=None should not produce NaN or diverge."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=None)
    losses = _train_steps(model, opt, steps=5)
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN loss detected"
    # Loss should not blow up (random init ~ln(vocab_size) range is fine).
    assert all(l < 1e6 for l in losses), f"Loss diverged: {losses}"


@requires_flashoptim
@requires_cuda
def test_32bit_master_weight_training_converges():
    """Training with master_weight_bits=32 should not produce NaN or diverge."""
    from flashoptim import FlashAdamW

    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=32)
    losses = _train_steps(model, opt, steps=5)
    assert all(not torch.isnan(torch.tensor(l)) for l in losses), "NaN loss detected"
    assert all(l < 1e6 for l in losses), f"Loss diverged: {losses}"


@requires_flashoptim
@requires_cuda
def test_32bit_checkpoint_roundtrip():
    """Checkpoint roundtrip with master_weight_bits=32 should preserve loss."""
    from flashoptim import FlashAdamW

    torch.manual_seed(3)
    model = _make_bf16_model()
    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=32)
    _train_steps(model, opt, steps=2)

    model_sd = copy.deepcopy(model.state_dict())
    opt_sd = copy.deepcopy(opt.state_dict())

    torch.manual_seed(13)
    x = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss1 = model(x).sum()
    loss1.backward()
    opt.step()
    opt.zero_grad()

    model2 = _make_bf16_model()
    model2.load_state_dict(model_sd)
    opt2 = FlashAdamW(model2.parameters(), lr=1e-3, master_weight_bits=32)
    opt2.load_state_dict(opt_sd)

    torch.manual_seed(13)
    x2 = torch.randn(4, 64, dtype=torch.bfloat16, device='cuda')
    loss2 = model2(x2).sum()
    loss2.backward()
    opt2.step()
    opt2.zero_grad()

    assert abs(loss1.item() - loss2.item()) < 0.05, (
        f"32-bit checkpoint roundtrip loss mismatch: " f"{loss1.item():.6f} vs {loss2.item():.6f}"
    )


@requires_flashoptim
@requires_cuda
def test_optimizer_state_memory_vs_mixed_precision_adamw():
    """FlashAdamW state should be smaller than mixed-precision AdamW state.

    Mixed-precision AdamW stores fp32 exp_avg + fp32 exp_avg_sq (8 bytes/param).
    FlashAdamW stores INT8 exp_avg + INT8 exp_avg_sq (~2 bytes/param) plus ECC bits.
    On a 1024×1024 parameter matrix (~1M params) the difference is clearly measurable.
    """
    from flashoptim import FlashAdamW

    dim = 1024
    # FlashAdamW: BF16 model with quantised optimizer states.
    model_flash = nn.Linear(dim, dim, bias=False).to(torch.bfloat16).cuda()
    opt_flash = FlashAdamW(model_flash.parameters(), lr=1e-3, master_weight_bits=24, quantize=True)

    # Mixed-precision AdamW baseline: fp32 model → fp32 exp_avg + fp32 exp_avg_sq.
    model_adam_fp32 = nn.Linear(dim, dim, bias=False).to(torch.float32).cuda()
    opt_adam_fp32 = torch.optim.AdamW(model_adam_fp32.parameters(), lr=1e-3)

    # One step to initialise states.
    torch.manual_seed(0)
    x_bf16 = torch.randn(4, dim, dtype=torch.bfloat16, device='cuda')
    model_flash(x_bf16).sum().backward()
    opt_flash.step()
    opt_flash.zero_grad()

    x_fp32 = x_bf16.float()
    model_adam_fp32(x_fp32).sum().backward()
    opt_adam_fp32.step()
    opt_adam_fp32.zero_grad()

    flash_bytes = sum(
        v.element_size() * v.numel()
        for s in opt_flash.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )
    adam_bytes = sum(
        v.element_size() * v.numel()
        for s in opt_adam_fp32.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )

    assert flash_bytes < adam_bytes, (
        f"FlashAdamW state ({flash_bytes / 1024:.1f} KB) should be smaller than "
        f"mixed-precision AdamW state ({adam_bytes / 1024:.1f} KB).\n"
        f"FlashAdamW uses INT8 quantised states (~2 bytes/param) vs fp32 states (8 bytes/param)."
    )


@requires_flashoptim
@requires_cuda
def test_optimizer_state_bytes_per_param():
    """FlashAdamW optimizer state should be well under 4 bytes/param (INT8 quantised).

    Mixed-precision AdamW: 8 bytes/param (fp32 exp_avg + fp32 exp_avg_sq).
    FlashAdamW target: ~2 bytes/param (INT8 exp_avg + INT8 exp_avg_sq).
    We allow up to 4 bytes/param to account for ECC bits and step counters.
    """
    from flashoptim import FlashAdamW

    dim = 1024
    model = nn.Linear(dim, dim, bias=False).to(torch.bfloat16).cuda()
    num_params = sum(p.numel() for p in model.parameters())

    opt = FlashAdamW(model.parameters(), lr=1e-3, master_weight_bits=24, quantize=True)
    torch.manual_seed(0)
    x = torch.randn(4, dim, dtype=torch.bfloat16, device='cuda')
    model(x).sum().backward()
    opt.step()
    opt.zero_grad()

    state_bytes = sum(
        v.element_size() * v.numel()
        for s in opt.state.values()
        for v in s.values()
        if isinstance(v, torch.Tensor)
    )
    bytes_per_param = state_bytes / num_params
    assert bytes_per_param < 4.0, (
        f"FlashAdamW optimizer state uses {bytes_per_param:.2f} bytes/param, "
        f"expected < 4.0 bytes/param (INT8 quantised states). "
        f"Total state: {state_bytes / 1024:.1f} KB for {num_params} params."
    )
