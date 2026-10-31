# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Paged-stash schedule hook for a Transformer Engine GroupedTensor expert input, which MXFP8
token dispatch produces."""

import pytest
import torch

from megatron.core.transformer.moe import experts, paged_stash

try:
    import transformer_engine_torch as tex
    from transformer_engine.pytorch.tensor.grouped_tensor import GroupedTensor
    from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
except ImportError:
    GroupedTensor = None


def dispatched_tokens(token_counts=(128, 256, 0, 128), hidden=256):
    """Expert-major MXFP8 tokens wrapped as a per-expert GroupedTensor, as Transformer Engine's
    NCCL-EP dispatch returns them."""
    counts = torch.tensor(token_counts, dtype=torch.int64, device="cuda")
    rows = sum(token_counts)
    offsets = torch.zeros(len(token_counts) + 1, dtype=torch.int64, device="cuda")
    offsets[1:] = torch.cumsum(counts * hidden, dim=0)
    return GroupedTensor(
        shape=(rows, hidden),
        dtype=torch.bfloat16,
        num_tensors=len(token_counts),
        quantizer=MXFP8Quantizer(tex.DType.kFloat8E4M3, rowwise=True, columnwise=False),
        data=torch.randint(0, 255, (rows * hidden,), dtype=torch.uint8, device="cuda"),
        scale_inv=torch.randint(100, 140, (rows * hidden // 32,), dtype=torch.uint8, device="cuda"),
        first_dims=counts,
        tensor_offsets=offsets,
    )


class StubStashManager:
    """An enabled manager whose schedule reloads layer 3 when the hook runs in backward."""

    enabled = True
    status = "captured"
    current_schedule_index = 0
    _pp_schedule = [-3]

    def __init__(self) -> None:
        self.waits = 0
        self.reloaded = []

    def wait_for_stash_to_complete(self):
        self.waits += 1

    def reload_paged_tensors(self, layer):
        self.reloaded.append(layer)


@pytest.fixture
def stash_manager(monkeypatch):
    manager = StubStashManager()
    monkeypatch.setattr(paged_stash.PagedStashManager, "STASH_MGR", manager)
    return manager


def test_hook_wraps_plain_hidden_states(stash_manager):
    hidden = torch.randn(512, 256, device="cuda", requires_grad=True)
    probs = torch.rand(512, device="cuda", requires_grad=True)
    hooked_hidden, hooked_probs = experts._paged_stash_group_start(hidden, probs)
    assert hooked_probs is probs
    assert hooked_hidden.grad_fn is not None and stash_manager.waits == 1


@pytest.mark.skipif(GroupedTensor is None, reason="Requires Transformer Engine's GroupedTensor")
def test_hook_wraps_probs_for_grouped_tensor_input(stash_manager):
    tokens = dispatched_tokens()
    probs = torch.rand(tokens.shape[0], device="cuda", requires_grad=True)
    hooked_tokens, hooked_probs = experts._paged_stash_group_start(tokens, probs)

    # The grouped tensor passes through; the hook runs in forward on the probs...
    assert hooked_tokens is tokens
    assert stash_manager.waits == 1
    # ...and in backward once the gradient of the probs arrives, as it would from the expert op.
    (hooked_probs * 2).sum().backward()
    assert stash_manager.reloaded == [3]
