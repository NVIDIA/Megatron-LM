# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""BSHD vs THD correctness test for multimodal_dev packed sequence support.

Validates that packing a [B, S] batch into [1, T] THD format produces
numerically equivalent loss values and gradient norms through a GPTModel.

The test uses equal-length sequences (no padding) so that BSHD causal
attention and THD cu_seqlens-based causal attention are mathematically
identical.  This makes any numerical deviation a real bug rather than an
expected consequence of different padding/masking strategies.

Usage::

    # Single GPU (flash attention):
    torchrun --nproc_per_node=1 \\
        examples/multimodal_dev/tests/test_thd_correctness.py

    # Override model size:
    torchrun --nproc_per_node=1 \\
        examples/multimodal_dev/tests/test_thd_correctness.py \\
        --num-layers 4 --hidden-size 512 --num-heads 8 --num-kv-heads 4
"""

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from megatron.core import parallel_state
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import (
    model_parallel_cuda_manual_seed,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

from examples.multimodal_dev.forward_step import (
    _build_packed_seq_params,
    _pack_batch,
)


# ===================================================================
# Helpers
# ===================================================================


def _grad_norm(model):
    """L2 norm of all parameter gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.float().norm(2).item() ** 2
    return total ** 0.5


def _mean_loss(per_token_loss, loss_mask):
    """Mean loss over valid tokens."""
    flat = per_token_loss.float().view(-1)
    mask = loss_mask.float().view(-1)
    return (flat * mask).sum() / mask.sum().clamp(min=1)


def _build_model(cfg, vocab_size, max_seq_len):
    """Build a small GPTModel for testing."""
    spec = get_gpt_layer_with_transformer_engine_spec()
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=spec,
        vocab_size=vocab_size,
        max_sequence_length=max_seq_len,
        pre_process=True,
        post_process=True,
        parallel_output=False,
        position_embedding_type="rope",
    )
    model.cuda()
    return model


# ===================================================================
# Core test logic
# ===================================================================


def run_equal_length_test(
    model,
    batch_size,
    seq_len,
    vocab_size,
    seed,
    atol_loss,
    rtol_grad,
):
    """Compare BSHD and THD with equal-length sequences (no padding).

    Returns a dict with test metrics for logging.
    """
    B, S = batch_size, seq_len

    # Deterministic data generation.
    torch.manual_seed(seed + 1)
    input_ids = torch.randint(0, vocab_size, (B, S), device="cuda")
    labels = torch.randint(0, vocab_size, (B, S), device="cuda")
    loss_mask = torch.ones(B, S, device="cuda")
    position_ids = (
        torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    )

    # ---- BSHD forward / backward ----
    output_bshd = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
    )
    bshd_loss = _mean_loss(output_bshd, loss_mask)
    bshd_loss.backward()
    bshd_gn = _grad_norm(model)
    bshd_lv = bshd_loss.item()
    bshd_per_token = output_bshd.detach().float().view(-1).clone()

    model.zero_grad()

    # ---- THD forward / backward ----
    batch = {
        "input_ids": input_ids.clone(),
        "labels": labels.clone(),
        "loss_mask": loss_mask.clone(),
        "position_ids": position_ids.clone(),
    }
    packed = _pack_batch(batch)
    psp = packed.pop("packed_seq_params")

    output_thd = model(
        input_ids=packed["input_ids"],
        position_ids=packed["position_ids"],
        attention_mask=None,
        labels=packed["labels"],
        loss_mask=packed["loss_mask"],
        packed_seq_params=psp,
    )
    thd_loss = _mean_loss(output_thd, packed["loss_mask"])
    thd_loss.backward()
    thd_gn = _grad_norm(model)
    thd_lv = thd_loss.item()
    thd_per_token = output_thd.detach().float().view(-1).clone()

    model.zero_grad()

    # ---- Comparison ----
    loss_diff = abs(bshd_lv - thd_lv)
    grad_diff = abs(bshd_gn - thd_gn)
    grad_rel = grad_diff / max(bshd_gn, 1e-8)
    token_max_diff = (bshd_per_token - thd_per_token).abs().max().item()
    token_mean_diff = (bshd_per_token - thd_per_token).abs().mean().item()

    loss_ok = loss_diff < atol_loss
    grad_ok = grad_rel < rtol_grad

    metrics = dict(
        bshd_loss=bshd_lv,
        thd_loss=thd_lv,
        loss_diff=loss_diff,
        bshd_grad_norm=bshd_gn,
        thd_grad_norm=thd_gn,
        grad_diff=grad_diff,
        grad_rel=grad_rel,
        token_max_diff=token_max_diff,
        token_mean_diff=token_mean_diff,
        loss_ok=loss_ok,
        grad_ok=grad_ok,
    )
    return metrics


def run_variable_length_smoke_test(model, vocab_size, seed):
    """Smoke test: variable-length sequences packed to THD.

    Does NOT compare against BSHD (padding in BSHD changes attention
    context).  Validates that:
    - Packing produces correct shapes
    - Forward + backward complete without error
    - Loss is finite
    - Gradients are finite and non-zero

    Returns a dict with test metrics.
    """
    seq_lengths = [128, 96, 112, 80]
    S = max(seq_lengths)
    B = len(seq_lengths)

    torch.manual_seed(seed + 2)
    input_ids = torch.randint(0, vocab_size, (B, S), device="cuda")
    labels = torch.randint(0, vocab_size, (B, S), device="cuda")
    loss_mask = torch.ones(B, S, device="cuda")
    position_ids = (
        torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    )

    # Build attention_mask to indicate valid tokens per sample.
    attention_mask = torch.zeros(B, S, device="cuda")
    for i, sl in enumerate(seq_lengths):
        attention_mask[i, :sl] = 1.0
        loss_mask[i, sl:] = 0.0

    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "loss_mask": loss_mask,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
    }
    packed = _pack_batch(batch)
    psp = packed.pop("packed_seq_params")

    T = sum(seq_lengths)
    assert packed["input_ids"].shape == (1, T), (
        f"Expected [1, {T}], got {packed['input_ids'].shape}"
    )
    assert packed["labels"].shape == (1, T)
    assert packed["loss_mask"].shape == (1, T)
    assert psp.cu_seqlens_q.tolist() == [
        0,
        seq_lengths[0],
        seq_lengths[0] + seq_lengths[1],
        seq_lengths[0] + seq_lengths[1] + seq_lengths[2],
        T,
    ]

    output = model(
        input_ids=packed["input_ids"],
        position_ids=packed["position_ids"],
        attention_mask=None,
        labels=packed["labels"],
        loss_mask=packed["loss_mask"],
        packed_seq_params=psp,
    )
    loss = _mean_loss(output, packed["loss_mask"])
    loss.backward()
    gn = _grad_norm(model)
    loss_val = loss.item()

    model.zero_grad()

    loss_finite = torch.isfinite(torch.tensor(loss_val)).item()
    grad_finite = torch.isfinite(torch.tensor(gn)).item()
    grad_nonzero = gn > 0

    return dict(
        loss=loss_val,
        grad_norm=gn,
        total_tokens=T,
        loss_finite=loss_finite,
        grad_finite=grad_finite,
        grad_nonzero=grad_nonzero,
        passed=loss_finite and grad_finite and grad_nonzero,
    )


# ===================================================================
# Main
# ===================================================================


def _print_banner(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="BSHD vs THD correctness test",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol-loss", type=float, default=1e-5,
                        help="Absolute tolerance for loss comparison")
    parser.add_argument("--rtol-grad", type=float, default=1e-3,
                        help="Relative tolerance for grad norm comparison")
    args = parser.parse_args()

    Utils.initialize_model_parallel(tensor_model_parallel_size=1)
    model_parallel_cuda_manual_seed(args.seed)

    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_heads,
        num_query_groups=args.num_kv_heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
    )

    model = _build_model(config, args.vocab_size, args.seq_len)

    all_passed = True

    # ----------------------------------------------------------------
    # Test 1: equal-length correctness (BSHD vs THD)
    # ----------------------------------------------------------------
    _print_banner("Test 1: Equal-length BSHD vs THD correctness")
    m = run_equal_length_test(
        model=model,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed,
        atol_loss=args.atol_loss,
        rtol_grad=args.rtol_grad,
    )
    print(f"  Config: B={args.batch_size}, S={args.seq_len}, "
          f"H={args.hidden_size}, L={args.num_layers}, "
          f"heads={args.num_heads}/{args.num_kv_heads}")
    print(f"  BSHD loss:             {m['bshd_loss']:.8f}")
    print(f"  THD  loss:             {m['thd_loss']:.8f}")
    print(f"  Loss abs diff:         {m['loss_diff']:.2e}")
    print(f"  BSHD grad norm:        {m['bshd_grad_norm']:.8f}")
    print(f"  THD  grad norm:        {m['thd_grad_norm']:.8f}")
    print(f"  Grad norm rel diff:    {m['grad_rel']:.2e}")
    print(f"  Per-token max diff:    {m['token_max_diff']:.2e}")
    print(f"  Per-token mean diff:   {m['token_mean_diff']:.2e}")
    print(f"  Loss match:            {'PASS' if m['loss_ok'] else 'FAIL'} "
          f"(atol={args.atol_loss})")
    print(f"  Grad norm match:       {'PASS' if m['grad_ok'] else 'FAIL'} "
          f"(rtol={args.rtol_grad})")
    if not (m["loss_ok"] and m["grad_ok"]):
        all_passed = False

    # ----------------------------------------------------------------
    # Test 2: variable-length smoke test (THD only)
    # ----------------------------------------------------------------
    _print_banner("Test 2: Variable-length THD smoke test")
    v = run_variable_length_smoke_test(model, args.vocab_size, args.seed)
    print(f"  Seq lengths:           [128, 96, 112, 80]")
    print(f"  Total packed tokens:   {v['total_tokens']}")
    print(f"  Loss:                  {v['loss']:.8f}")
    print(f"  Grad norm:             {v['grad_norm']:.8f}")
    print(f"  Loss finite:           {'PASS' if v['loss_finite'] else 'FAIL'}")
    print(f"  Grad finite:           {'PASS' if v['grad_finite'] else 'FAIL'}")
    print(f"  Grad nonzero:          {'PASS' if v['grad_nonzero'] else 'FAIL'}")
    if not v["passed"]:
        all_passed = False

    # ----------------------------------------------------------------
    # Summary
    # ----------------------------------------------------------------
    _print_banner("Summary")
    if all_passed:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED")
    print(f"{'='*60}\n")

    Utils.destroy_model_parallel()

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
