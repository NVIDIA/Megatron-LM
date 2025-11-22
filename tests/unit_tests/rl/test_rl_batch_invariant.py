import os
import pytest
import torch

from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
    set_batch_invariant_mode,
)
from megatron.rl.rl_utils import selective_log_softmax


def test_selective_log_softmax_batch_invariant_matches_f_log_softmax():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    # Use float32 logits to ensure we force the F.log_softmax path via batch-invariant mode
    B, S, V = 4, 7, 16
    device = torch.device("cuda")
    logits = torch.randn(B, S, V, dtype=torch.float32, device=device)
    labels = torch.randint(low=0, high=V, size=(B, S), device=device)

    with set_batch_invariant_mode(True):
        bik_logps = selective_log_softmax(logits, labels)

    # Reference via F.log_softmax then gather
    ref = torch.nn.functional.log_softmax(logits, dim=-1).gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    assert torch.equal(bik_logps, ref)

