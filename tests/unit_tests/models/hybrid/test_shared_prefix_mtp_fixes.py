# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared-prefix HybridModel MTP contracts: input validation, gradient determinism,
grouped loss normalization and the forest-stable router scope."""

from types import SimpleNamespace

import pytest
import torch

from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.multi_token_prediction import MTPLossAutoScaler, process_mtp_loss


def _run_process_mtp_loss(weight, hidden_depths, input_ids, loss_mask, input_mask, lengths, groups):
    cu_seqlens = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32, device=input_ids.device
    )

    def output_layer(hidden, weight=None, runtime_gather_output=None):
        return hidden @ weight.t(), None

    def language_model_loss(labels, logits):
        return torch.nn.functional.cross_entropy(
            logits.transpose(0, 1).reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).view(labels.shape)

    return process_mtp_loss(
        hidden_states=torch.cat(hidden_depths, dim=0),
        labels=None,
        loss_mask=loss_mask,
        output_layer=output_layer,
        output_weight=weight,
        runtime_gather_output=None,
        is_training=False,
        compute_language_model_loss=language_model_loss,
        config=SimpleNamespace(
            mtp_num_layers=len(hidden_depths) - 1,
            mtp_loss_scaling_factor=0.3,
            calculate_per_token_loss=True,
            mtp_detach_heads=False,
        ),
        packed_seq_params=PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=max(lengths),
            max_seqlen_kv=max(lengths),
        ),
        input_ids=input_ids,
        mtp_input_mask=input_mask,
        loss_group_lengths=groups,
    )


@pytest.mark.parametrize("use_input_mask", [False, True])
def test_grouped_mtp_normalization_equals_independent_forwards(use_input_mask):
    """One grouped process_mtp_loss call injects the gradients of one call per group."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator().manual_seed(3)
    depth, hidden, vocab = 2, 8, 11
    groups = ((6, 5), (7, 4, 3))
    lengths = [length for group in groups for length in group]
    total = sum(lengths)
    weight = torch.randn(vocab, hidden, generator=generator).to(device, torch.float64)
    input_ids = torch.randint(0, vocab, (1, total), generator=generator).to(device)
    loss_mask = (torch.rand(1, total, generator=generator) > 0.25).to(device, torch.float64)
    input_mask = (torch.rand(1, total, generator=generator) > 0.3).to(device)
    input_mask = input_mask if use_input_mask else None
    hidden_depths = [
        torch.randn(total, 1, hidden, generator=generator).to(device, torch.float64)
        for _ in range(depth + 1)
    ]
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0, device=device, dtype=torch.float64))
    grouped_inputs = [value.clone().requires_grad_() for value in hidden_depths]
    output = _run_process_mtp_loss(
        weight,
        grouped_inputs,
        input_ids,
        loss_mask,
        input_mask,
        lengths,
        tuple(sum(group) for group in groups),
    )
    output.sum().backward()

    start = 0
    for group in groups:
        span = slice(start, start + sum(group))
        independent_inputs = [value[span].clone().requires_grad_() for value in hidden_depths]
        output = _run_process_mtp_loss(
            weight,
            independent_inputs,
            input_ids[:, span],
            loss_mask[:, span],
            None if input_mask is None else input_mask[:, span],
            list(group),
            None,
        )
        output.sum().backward()
        for grouped, independent in zip(grouped_inputs[1:], independent_inputs[1:]):
            torch.testing.assert_close(grouped.grad[span], independent.grad, rtol=0, atol=1e-12)
        start = span.stop
    MTPLossAutoScaler.set_loss_scale(torch.tensor(1.0))
