# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# This is a stdout-reporting standalone script; `print` is intentional.
# pylint: disable=bad-builtin

"""CP=1 vs CP>1 correctness test for THD and BSHD packing.

Runs the production forward path (:class:`MultimodalModel`) twice in a
single ``torchrun`` invocation:

  Phase 1 — CP=1 baseline. All ranks initialise with TP=1, CP=1
            (DP=world_size implicit). Each rank computes the full
            sequence, producing identical loss / grad_norm on every rank;
            rank 0's value is the baseline.

  Phase 2 — CP=cp_size. After ``destroy_model_parallel`` +
            ``initialize_model_parallel(CP=cp_size)`` the ranks form
            ``world_size // cp_size`` CP groups (one group when they are
            equal, which is the standalone 4-rank invocation below; two
            groups of 4 under the 8-GPU CI world size). The model's
            internal ``_cp_split_for_forward`` slices inputs per rank;
            per-rank loss / gradients are aggregated via AllReduce on the
            CP group.

We compare CP=1 and CP=cp_size results for both BSHD and THD packing modes,
asserting that loss and grad_norm match within tolerance.

Run with::

    # As a pytest module (this is what CI runs; skipped when the world
    # size is not a multiple of the CP size under test):
    PYTHONPATH=. torchrun --nproc-per-node 8 -m pytest -q \\
        examples/multimodal_dev/tests/test_cp_thd_correctness.py

    PYTHONPATH=. torchrun --nproc-per-node 4 \\
        examples/multimodal_dev/tests/test_cp_thd_correctness.py
"""

import argparse
import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.forward_step import pack_or_pad_batch
from examples.multimodal_dev.models.base import (
    MultimodalModel,
    _cp_split_tensor,
    _thd_cp_partition_index,
)
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_context_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# Run knobs -- exactly the keys ``run_cp_comparison`` consumes, so an override
# it would ignore is rejected rather than silently dropped. Shared with the CLI
# defaults in ``main`` so the two entry points cannot drift.
# ``seq_len`` must be divisible by 2*cp_size for zigzag CP splitting.
DEFAULTS = dict(
    cp_size=4,
    batch_size=2,
    seq_len=64,
    vocab_size=1024,
    hidden_size=256,
    num_layers=2,
    num_heads=4,
    num_kv_heads=2,
    ffn_hidden_size=512,
    seed=42,
    data_seed=123,
)

# Comparison bands, applied to both loss and grad_norm, hence the metric-neutral
# names. The assertion is ``abs_diff < atol or rel_diff < rtol``, so the looser
# leg wins; with CP=1 magnitudes of ~7.03 (loss) and ~4.85 (grad_norm) both sit
# far above ``atol/rtol = 0.5``, so ``rtol`` is the binding leg for both metrics.
# Measured on 4xH100 (bf16, CP=4 vs CP=1): BSHD loss 4.6e-5 rel, BSHD grad_norm
# 3.4e-4 rel (worst case), THD loss 1.2e-5, THD grad_norm 3.6e-6. rtol=2e-3
# keeps ~6x headroom on the worst case while cutting the effective band on the
# ~7 nat loss from 3.5e-2 to 1.4e-2.
TOLERANCES = dict(atol=1e-3, rtol=2e-3)

# ===================================================================
# Stub vision encoder
# ===================================================================


class _StubVisionEncoder(MegatronModule):
    """Vision encoder placeholder. The vision branch is skipped in
    :meth:`MultimodalModel.forward` whenever ``pixel_values is None``, so
    this module is never called — it only satisfies the constructor's
    ``vision_encoder: MegatronModule`` requirement.
    """

    def __init__(self, config):
        """Initialise the stub with the given TransformerConfig."""
        super().__init__(config=config)

    def forward(self, pixel_values, image_grid_thw):
        """Never called when ``pixel_values=None``; raises if it ever is."""
        raise RuntimeError("vision branch should not run when pixel_values=None")


# ===================================================================
# Model builder
# ===================================================================


def _build_model(config, vocab_size, max_seq_len, image_token_id):
    spec = get_gpt_layer_with_transformer_engine_spec()
    vision = _StubVisionEncoder(config)
    model = MultimodalModel(
        language_config=config,
        language_spec=spec,
        vision_encoder=vision,
        vocab_size=vocab_size,
        max_sequence_length=max_seq_len,
        image_token_id=image_token_id,
        position_embedding_type="rope",
        parallel_output=False,
    )
    model.cuda()
    return model


def _make_config(
    num_layers, hidden_size, ffn_hidden_size, num_heads, num_kv_heads, context_parallel_size
):
    return TransformerConfig(
        num_layers=num_layers,
        hidden_size=hidden_size,
        ffn_hidden_size=ffn_hidden_size,
        num_attention_heads=num_heads,
        num_query_groups=num_kv_heads,
        bf16=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        tensor_model_parallel_size=1,
        context_parallel_size=context_parallel_size,
        sequence_parallel=False,
    )


# ===================================================================
# Loss / grad-norm aggregation
# ===================================================================


def _global_loss(output, rank_loss_mask, cp_size):
    """Mean per-token loss over all CP shards (matches CP=1 mean exactly)."""
    num = (output.float().view(-1) * rank_loss_mask.float().view(-1)).sum()
    den = rank_loss_mask.float().view(-1).sum().clamp(min=1)
    if cp_size > 1:
        group = get_context_parallel_group()
        torch.distributed.all_reduce(num, group=group)
        torch.distributed.all_reduce(den, group=group)
    return (num / den).item()


def _global_grad_norm(model, cp_size):
    """Global L2 grad norm. For CP>1, AllReduce(SUM) gradients across CP
    then divide by ``cp_size`` so each rank holds the CP-mean gradient
    (matching CP=1's behaviour, where backward on the per-batch mean loss
    yields exactly that gradient).
    """
    if cp_size > 1:
        group = get_context_parallel_group()
        for p in model.parameters():
            if p.grad is not None:
                torch.distributed.all_reduce(p.grad, group=group)
                p.grad /= cp_size

    sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            sq += p.grad.float().norm(2).item() ** 2
    return sq**0.5


# ===================================================================
# Data — identical across all ranks (deterministic generator)
# ===================================================================


def _make_data(B, S, vocab_size, image_token_id, seed):
    """Same input on every rank thanks to the seeded generator."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    input_ids = torch.randint(0, vocab_size, (B, S), generator=g, device="cuda")
    # Ensure no accidental image tokens (we never run the vision branch).
    input_ids = torch.where(input_ids == image_token_id, (input_ids + 1) % vocab_size, input_ids)
    labels = torch.randint(0, vocab_size, (B, S), generator=g, device="cuda")
    loss_mask = torch.ones(B, S, device="cuda")
    position_ids = torch.arange(S, device="cuda").unsqueeze(0).expand(B, -1).contiguous()
    return input_ids, labels, loss_mask, position_ids


# ===================================================================
# One BSHD or THD forward+backward, returning (loss, grad_norm)
# ===================================================================


def _run_bshd(model, B, S, vocab_size, image_token_id, cp_size, seed):
    input_ids, labels, loss_mask, position_ids = _make_data(B, S, vocab_size, image_token_id, seed)

    output = model(
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=None,
        labels=labels,
        loss_mask=loss_mask,
        pixel_values=None,
        image_grid_thw=None,
        packed_seq_params=None,
    )

    # Slice loss_mask the same way forward_step does for BSHD + CP.
    rank_loss_mask = loss_mask
    if cp_size > 1:
        rank_loss_mask = _cp_split_tensor(
            rank_loss_mask, seq_dim=1, cp_size=cp_size, cp_rank=get_context_parallel_rank()
        )

    loss_val = _global_loss(output, rank_loss_mask, cp_size)

    # Backward on the LOCAL mean loss (each rank's contribution
    # equal-weighted; SUM-then-divide across CP recovers CP=1's gradient).
    local = (
        output.float().view(-1) * rank_loss_mask.float().view(-1)
    ).sum() / rank_loss_mask.float().view(-1).sum().clamp(min=1)
    model.zero_grad()
    local.backward()

    gn = _global_grad_norm(model, cp_size)
    return loss_val, gn


def _run_thd(model, B, S, vocab_size, image_token_id, cp_size, seed):
    input_ids, labels, loss_mask, _ = _make_data(B, S, vocab_size, image_token_id, seed)

    # Build the per-sample dict list and pack to [1, T].
    samples = []
    for i in range(B):
        samples.append(
            {
                "input_ids": input_ids[i].clone(),
                "labels": labels[i].clone(),
                "loss_mask": loss_mask[i].clone(),
                # No vision; empty tensors satisfy pack_or_pad_batch.
                "pixel_values": torch.zeros(0, 1, device="cuda"),
                "image_grid_thw": torch.empty(0, 3, dtype=torch.long, device="cuda"),
            }
        )
    packed = pack_or_pad_batch(samples, use_packed_sequence=True, device="cuda")
    psp = packed.pop("packed_seq_params")

    # THD position_ids: per-sample restart at 0.  Each sample has length S
    # (equal-length data), so this is arange(S) repeated B times.
    thd_pos = (
        torch.cat([torch.arange(S, device="cuda") for _ in range(B)]).unsqueeze(0).contiguous()
    )

    output = model(
        input_ids=packed["input_ids"],
        position_ids=thd_pos,
        attention_mask=None,
        labels=packed["labels"],
        loss_mask=packed["loss_mask"],
        pixel_values=None,
        image_grid_thw=None,
        packed_seq_params=psp,
    )

    rank_loss_mask = packed["loss_mask"]
    if cp_size > 1:
        T = rank_loss_mask.shape[1]
        idx = _thd_cp_partition_index(
            psp.cu_seqlens_q_padded, T, cp_size, get_context_parallel_rank()
        )
        rank_loss_mask = rank_loss_mask.index_select(1, idx)

    loss_val = _global_loss(output, rank_loss_mask, cp_size)

    local = (
        output.float().view(-1) * rank_loss_mask.float().view(-1)
    ).sum() / rank_loss_mask.float().view(-1).sum().clamp(min=1)
    model.zero_grad()
    local.backward()

    gn = _global_grad_norm(model, cp_size)
    return loss_val, gn


# ===================================================================
# State-dict roundtrip — keep weights identical across phases
# ===================================================================


def _cpu_state_dict(model):
    """Snapshot of model.state_dict() detached to CPU (kept in memory).

    Some entries (TransformerEngine ``_extra_state``) are non-tensor or
    ``None``; pass them through untouched.
    """
    snap = {}
    for k, v in model.state_dict().items():
        if isinstance(v, torch.Tensor):
            snap[k] = v.detach().to("cpu").clone()
        else:
            snap[k] = v
    return snap


def _restore_state_dict(model, snapshot):
    """Load a saved snapshot back into a freshly built model."""
    payload = {k: (v.to("cuda") if isinstance(v, torch.Tensor) else v) for k, v in snapshot.items()}
    model.load_state_dict(payload)


# ===================================================================
# Main
# ===================================================================


def _is_rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _print_banner(title):
    if _is_rank0():
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def _print_compare(label, baseline, trial, atol, rtol):
    """Print a CP=1 vs CP=``DEFAULTS["cp_size"]`` comparison line; return whether it passed.

    ``main`` is the only caller and it always runs the default CP size, so the
    interpolation below is the width that was actually compared.  A caller that
    overrides ``run_cp_comparison(cp_size=...)`` would have to add a ``cp_size``
    parameter here as well, or the printed label will not match the run.
    """
    if not _is_rank0():
        return True

    abs_diff = abs(baseline - trial)
    rel_diff = abs_diff / max(abs(baseline), 1e-8)
    ok = abs_diff < atol or rel_diff < rtol
    flag = "PASS" if ok else "FAIL"
    print(
        f"  {label:<30s} CP=1: {baseline:.8f}  CP={DEFAULTS['cp_size']}: {trial:.8f}"
        f"  abs={abs_diff:.2e}  rel={rel_diff:.2e}  [{flag}]"
    )
    return ok


def run_cp_comparison(**overrides):
    """Run the CP=1 baseline and the CP=``cp_size`` trial on identical weights.

    Any key of :data:`DEFAULTS` may be overridden, including ``cp_size``.
    Tolerances live in :data:`TOLERANCES` and are *not* accepted here -- this
    function only runs the comparison, the caller decides how to judge it.
    Returns ``{"BSHD loss": (cp1, cpN), "BSHD grad_norm": (...), ...}``.
    Leaves the model-parallel groups destroyed.
    """
    unknown = set(overrides) - set(DEFAULTS)
    assert not unknown, f"unknown overrides: {sorted(unknown)}"
    cfg = {**DEFAULTS, **overrides}
    cp_size = cfg["cp_size"]
    image_token_id = 0  # never appears in input (data filters this id out)

    def _phase(context_parallel_size, snapshot):
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, context_parallel_size=context_parallel_size
        )
        model_parallel_cuda_manual_seed(cfg["seed"])
        config = _make_config(
            cfg["num_layers"],
            cfg["hidden_size"],
            cfg["ffn_hidden_size"],
            cfg["num_heads"],
            cfg["num_kv_heads"],
            context_parallel_size=context_parallel_size,
        )
        torch.manual_seed(cfg["seed"])
        model = _build_model(config, cfg["vocab_size"], cfg["seq_len"], image_token_id)
        if snapshot is not None:
            _restore_state_dict(model, snapshot)

        run_args = (
            model,
            cfg["batch_size"],
            cfg["seq_len"],
            cfg["vocab_size"],
            image_token_id,
        )
        bshd = _run_bshd(*run_args, cp_size=context_parallel_size, seed=cfg["data_seed"])
        thd = _run_thd(*run_args, cp_size=context_parallel_size, seed=cfg["data_seed"])

        # Snapshot weights *before* the optimizer would have touched them.
        # (We've zeroed grads but never stepped; weights at this point are
        # the just-initialised baseline.)
        out_snapshot = _cpu_state_dict(model) if snapshot is None else None
        del model
        torch.cuda.empty_cache()
        return bshd, thd, out_snapshot

    _print_banner("Phase 1 — building CP=1 baseline (TP=1, CP=1)")
    (bshd_cp1, thd_cp1, weights_snapshot) = _phase(1, None)

    _print_banner(f"Phase 2 — re-initialising for CP={cp_size} (TP=1, CP={cp_size})")
    Utils.destroy_model_parallel()
    (bshd_cpN, thd_cpN, _) = _phase(cp_size, weights_snapshot)

    Utils.destroy_model_parallel()
    return {
        "BSHD loss": (bshd_cp1[0], bshd_cpN[0]),
        "BSHD grad_norm": (bshd_cp1[1], bshd_cpN[1]),
        "THD  loss": (thd_cp1[0], thd_cpN[0]),
        "THD  grad_norm": (thd_cp1[1], thd_cpN[1]),
    }


# ===================================================================
# pytest entry point
# ===================================================================


def test_cp_matches_cp1_for_bshd_and_thd():
    """CP=4 must reproduce the CP=1 loss and grad norm in both packings."""
    Utils.initialize_distributed()
    cp_size = DEFAULTS["cp_size"]
    world_size = torch.distributed.get_world_size()
    if world_size % cp_size != 0:
        pytest.skip(f"world_size={world_size} is not divisible by cp_size={cp_size}")

    results = run_cp_comparison()

    atol, rtol = TOLERANCES["atol"], TOLERANCES["rtol"]
    for label, (baseline, trial) in results.items():
        abs_diff = abs(baseline - trial)
        rel_diff = abs_diff / max(abs(baseline), 1e-8)
        assert abs_diff < atol or rel_diff < rtol, (
            f"{label}: CP=1 {baseline:.8f} vs CP={cp_size} {trial:.8f} "
            f"(abs={abs_diff:.2e}, rel={rel_diff:.2e}, atol={atol}, rtol={rtol})"
        )


def main():
    """Run the CP=1 baseline + CP=``DEFAULTS["cp_size"]`` trial, comparing losses / grad_norms.

    The CLI exposes the model/data knobs but not ``cp_size``: ``_print_compare``
    labels its output from ``DEFAULTS``, so a different context-parallel width
    would have to be threaded through both.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    # Must be divisible by 2*cp_size (=8 for CP=4 zigzag).
    parser.add_argument("--seq-len", type=int, default=DEFAULTS["seq_len"])
    parser.add_argument("--vocab-size", type=int, default=DEFAULTS["vocab_size"])
    parser.add_argument("--hidden-size", type=int, default=DEFAULTS["hidden_size"])
    parser.add_argument("--num-layers", type=int, default=DEFAULTS["num_layers"])
    parser.add_argument("--num-heads", type=int, default=DEFAULTS["num_heads"])
    parser.add_argument("--num-kv-heads", type=int, default=DEFAULTS["num_kv_heads"])
    parser.add_argument("--ffn-hidden-size", type=int, default=DEFAULTS["ffn_hidden_size"])
    parser.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    parser.add_argument(
        "--atol",
        type=float,
        default=TOLERANCES["atol"],
        help="Absolute tolerance, applied to loss and grad_norm (default %(default)s)",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=TOLERANCES["rtol"],
        help="Relative tolerance, applied to loss and grad_norm (default %(default)s)",
    )
    parser.add_argument("--data-seed", type=int, default=DEFAULTS["data_seed"])
    args = parser.parse_args()

    results = run_cp_comparison(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        ffn_hidden_size=args.ffn_hidden_size,
        seed=args.seed,
        data_seed=args.data_seed,
    )

    _print_banner(f"Results — CP=1 vs CP={DEFAULTS['cp_size']}")
    all_ok = True
    for label, (baseline, trial) in results.items():
        all_ok &= _print_compare(label, baseline, trial, args.atol, args.rtol)

    _print_banner("Summary")
    if _is_rank0():
        print(f"  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
        print(f"{'=' * 60}\n")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
