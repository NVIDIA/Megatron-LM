# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CP=1 vs CP=4 correctness test for THD and BSHD packing.

Runs the production forward path (:class:`MultimodalModel`) twice in a
single ``torchrun`` invocation:

  Phase 1 — CP=1 baseline. All 4 ranks initialise with TP=1, CP=1
            (DP=4 implicit). Each rank computes the full sequence,
            producing identical loss / grad_norm on every rank; rank 0's
            value is the baseline.

  Phase 2 — CP=4. After ``destroy_model_parallel`` + ``initialize_model_parallel(CP=4)``
            the 4 ranks form a single CP group. The model's internal
            ``_cp_split_for_forward`` slices inputs per rank; per-rank
            loss / gradients are aggregated via AllReduce on the CP group.

We compare CP=1 and CP=4 results for both BSHD and THD packing modes,
asserting that loss and grad_norm match within tolerance.

Run with::

    PYTHONPATH=. torchrun --nproc-per-node 4 \\
        examples/multimodal_dev/tests/test_cp_thd_correctness.py
"""

import argparse
import os
import sys

import torch
import torch.nn as nn

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../.."),
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from examples.multimodal_dev.forward_step import pack_or_pad_batch
from examples.multimodal_dev.models.base import (
    MultimodalModel,
    _cp_split_tensor,
    _thd_cp_partition_index,
)
from megatron.core import parallel_state
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.parallel_state import get_context_parallel_group, get_context_parallel_rank
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

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
        super().__init__(config=config)
        # MegatronModule expects at least one parameter for state_dict
        # round-tripping; this Linear is otherwise unused.
        self._unused = nn.Linear(1, 1)

    def forward(self, pixel_values, image_grid_thw):
        raise RuntimeError(
            "vision branch should not run when pixel_values=None"
        )


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


def _make_config(num_layers, hidden_size, ffn_hidden_size, num_heads,
                 num_kv_heads, context_parallel_size):
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
    num = (output.float().view(-1)
           * rank_loss_mask.float().view(-1)).sum()
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
    return sq ** 0.5


# ===================================================================
# Data — identical across all ranks (deterministic generator)
# ===================================================================


def _make_data(B, S, vocab_size, image_token_id, seed):
    """Same input on every rank thanks to the seeded generator."""
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)
    input_ids = torch.randint(
        0, vocab_size, (B, S), generator=g, device="cuda",
    )
    # Ensure no accidental image tokens (we never run the vision branch).
    input_ids = torch.where(
        input_ids == image_token_id,
        (input_ids + 1) % vocab_size,
        input_ids,
    )
    labels = torch.randint(
        0, vocab_size, (B, S), generator=g, device="cuda",
    )
    loss_mask = torch.ones(B, S, device="cuda")
    position_ids = (
        torch.arange(S, device="cuda")
        .unsqueeze(0)
        .expand(B, -1)
        .contiguous()
    )
    return input_ids, labels, loss_mask, position_ids


# ===================================================================
# One BSHD or THD forward+backward, returning (loss, grad_norm)
# ===================================================================


def _run_bshd(model, B, S, vocab_size, image_token_id, cp_size, seed):
    input_ids, labels, loss_mask, position_ids = _make_data(
        B, S, vocab_size, image_token_id, seed,
    )

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
            rank_loss_mask,
            seq_dim=1,
            cp_size=cp_size,
            cp_rank=get_context_parallel_rank(),
        )

    loss_val = _global_loss(output, rank_loss_mask, cp_size)

    # Backward on the LOCAL mean loss (each rank's contribution
    # equal-weighted; SUM-then-divide across CP recovers CP=1's gradient).
    local = ((output.float().view(-1) * rank_loss_mask.float().view(-1)).sum()
             / rank_loss_mask.float().view(-1).sum().clamp(min=1))
    model.zero_grad()
    local.backward()

    gn = _global_grad_norm(model, cp_size)
    return loss_val, gn


def _run_thd(model, B, S, vocab_size, image_token_id, cp_size, seed):
    input_ids, labels, loss_mask, _ = _make_data(
        B, S, vocab_size, image_token_id, seed,
    )

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
                "image_grid_thw": torch.empty(
                    0, 3, dtype=torch.long, device="cuda",
                ),
            }
        )
    packed = pack_or_pad_batch(
        samples, use_packed_sequence=True, device="cuda",
    )
    psp = packed.pop("packed_seq_params")

    # THD position_ids: per-sample restart at 0.  Each sample has length S
    # (equal-length data), so this is arange(S) repeated B times.
    thd_pos = (
        torch.cat([torch.arange(S, device="cuda") for _ in range(B)])
        .unsqueeze(0)
        .contiguous()
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
            psp.cu_seqlens_q_padded, T, cp_size,
            get_context_parallel_rank(),
        )
        rank_loss_mask = rank_loss_mask.index_select(1, idx)

    loss_val = _global_loss(output, rank_loss_mask, cp_size)

    local = ((output.float().view(-1) * rank_loss_mask.float().view(-1)).sum()
             / rank_loss_mask.float().view(-1).sum().clamp(min=1))
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
    payload = {
        k: (v.to("cuda") if isinstance(v, torch.Tensor) else v)
        for k, v in snapshot.items()
    }
    model.load_state_dict(payload)


# ===================================================================
# Main
# ===================================================================


def _is_rank0():
    return (
        not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _print_banner(title):
    if _is_rank0():
        print(f"\n{'=' * 60}")
        print(f"  {title}")
        print(f"{'=' * 60}")


def _print_compare(label, baseline, trial, atol, rtol):
    """Print a CP=1 vs CP=4 comparison line and return whether it passed."""
    if not _is_rank0():
        return True

    abs_diff = abs(baseline - trial)
    rel_diff = abs_diff / max(abs(baseline), 1e-8)
    ok = abs_diff < atol or rel_diff < rtol
    flag = "PASS" if ok else "FAIL"
    print(
        f"  {label:<30s} CP=1: {baseline:.8f}  CP=4: {trial:.8f}"
        f"  abs={abs_diff:.2e}  rel={rel_diff:.2e}  [{flag}]"
    )
    return ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    # Must be divisible by 2*cp_size (=8 for CP=4 zigzag).
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--ffn-hidden-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--atol-loss", type=float, default=1e-3)
    parser.add_argument("--rtol-grad", type=float, default=5e-3)
    parser.add_argument("--data-seed", type=int, default=123)
    args = parser.parse_args()

    image_token_id = 0  # never appears in input (data filters this id out)

    # ----------------------------------------------------------------
    # Phase 1: CP=1 baseline
    # ----------------------------------------------------------------
    _print_banner("Phase 1 — building CP=1 baseline (TP=1, CP=1, DP=4)")
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(args.seed)

    config_cp1 = _make_config(
        args.num_layers, args.hidden_size, args.ffn_hidden_size,
        args.num_heads, args.num_kv_heads, context_parallel_size=1,
    )
    torch.manual_seed(args.seed)
    model_cp1 = _build_model(
        config_cp1, args.vocab_size, args.seq_len, image_token_id,
    )

    bshd_loss_cp1, bshd_gn_cp1 = _run_bshd(
        model_cp1, args.batch_size, args.seq_len, args.vocab_size,
        image_token_id, cp_size=1, seed=args.data_seed,
    )
    thd_loss_cp1, thd_gn_cp1 = _run_thd(
        model_cp1, args.batch_size, args.seq_len, args.vocab_size,
        image_token_id, cp_size=1, seed=args.data_seed,
    )

    # Snapshot weights *before* the optimizer would have touched them.
    # (We've zeroed grads but never stepped; weights at this point are
    # the just-initialised baseline.)
    weights_snapshot = _cpu_state_dict(model_cp1)
    del model_cp1
    torch.cuda.empty_cache()

    # ----------------------------------------------------------------
    # Phase 2: CP=4
    # ----------------------------------------------------------------
    _print_banner("Phase 2 — re-initialising for CP=4 (TP=1, CP=4)")
    Utils.destroy_model_parallel()
    Utils.initialize_model_parallel(
        tensor_model_parallel_size=1, context_parallel_size=4,
    )
    model_parallel_cuda_manual_seed(args.seed)

    config_cp4 = _make_config(
        args.num_layers, args.hidden_size, args.ffn_hidden_size,
        args.num_heads, args.num_kv_heads, context_parallel_size=4,
    )
    torch.manual_seed(args.seed)
    model_cp4 = _build_model(
        config_cp4, args.vocab_size, args.seq_len, image_token_id,
    )
    _restore_state_dict(model_cp4, weights_snapshot)

    bshd_loss_cp4, bshd_gn_cp4 = _run_bshd(
        model_cp4, args.batch_size, args.seq_len, args.vocab_size,
        image_token_id, cp_size=4, seed=args.data_seed,
    )
    thd_loss_cp4, thd_gn_cp4 = _run_thd(
        model_cp4, args.batch_size, args.seq_len, args.vocab_size,
        image_token_id, cp_size=4, seed=args.data_seed,
    )

    # ----------------------------------------------------------------
    # Compare
    # ----------------------------------------------------------------
    _print_banner("Results — CP=1 vs CP=4")
    all_ok = True
    all_ok &= _print_compare(
        "BSHD loss", bshd_loss_cp1, bshd_loss_cp4,
        args.atol_loss, args.rtol_grad,
    )
    all_ok &= _print_compare(
        "BSHD grad_norm", bshd_gn_cp1, bshd_gn_cp4,
        args.atol_loss, args.rtol_grad,
    )
    all_ok &= _print_compare(
        "THD  loss", thd_loss_cp1, thd_loss_cp4,
        args.atol_loss, args.rtol_grad,
    )
    all_ok &= _print_compare(
        "THD  grad_norm", thd_gn_cp1, thd_gn_cp4,
        args.atol_loss, args.rtol_grad,
    )

    _print_banner("Summary")
    if _is_rank0():
        print(f"  {'ALL TESTS PASSED' if all_ok else 'SOME TESTS FAILED'}")
        print(f"{'=' * 60}\n")

    Utils.destroy_model_parallel()
    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
