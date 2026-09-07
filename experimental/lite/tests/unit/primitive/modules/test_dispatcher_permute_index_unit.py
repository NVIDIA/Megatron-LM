# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The all-to-all output reshuffle, as an index instead of a 64-operand ``cat``."""

from __future__ import annotations

import pytest
import torch

from megatron.lite.primitive.modules.dispatcher import TokenDispatcher

pytestmark = [pytest.mark.mlite]


def _legacy_reshuffle(recv_flat: torch.Tensor, recv_tpe_2d: torch.Tensor, sort_idxs: list[int]):
    """The split/cat form this replaced, kept verbatim as the contract."""
    chunk_sizes = recv_tpe_2d.ravel().tolist()
    chunks = torch.split(recv_flat, chunk_sizes, dim=0)
    return torch.cat([chunks[i] for i in sort_idxs], dim=0)


def _make_dispatcher(ep_size: int, num_local_experts: int) -> TokenDispatcher:
    """A dispatcher carrying only the fields the index builder reads."""
    d = TokenDispatcher.__new__(TokenDispatcher)
    d.ep_size = ep_size
    d.num_local_experts = num_local_experts
    chunk_idxs = torch.arange(ep_size * num_local_experts)
    d._sort_by_experts = chunk_idxs.reshape(ep_size, num_local_experts).T.ravel().tolist()
    d._sort_by_experts_dev = None
    return d


@pytest.mark.parametrize(("ep_size", "num_local_experts"), [(8, 8), (2, 4), (4, 2), (8, 2), (2, 2)])
def test_index_reshuffle_matches_split_cat(ep_size: int, num_local_experts: int) -> None:
    """Bitwise agreement with the split/cat form, over uneven expert loads."""
    generator = torch.Generator().manual_seed(0)
    d = _make_dispatcher(ep_size, num_local_experts)
    # Uneven counts including empty chunks: a routing that sends no token to some
    # (rank, expert) pair is normal and is where an off-by-one shows up.
    recv_tpe_2d = torch.randint(
        0, 5, (ep_size, num_local_experts), generator=generator, dtype=torch.long
    )
    total = int(recv_tpe_2d.sum())
    recv_flat = torch.randn(total, 6, generator=generator)

    index = d._expert_major_row_index(recv_tpe_2d, total)
    assert index.shape == (total,)
    assert torch.equal(index.sort().values, torch.arange(total)), "not a permutation"
    torch.testing.assert_close(
        recv_flat.index_select(0, index),
        _legacy_reshuffle(recv_flat, recv_tpe_2d, d._sort_by_experts),
        rtol=0,
        atol=0,
    )


def test_combine_index_inverts_dispatch_index() -> None:
    """Combine must undo dispatch exactly, or tokens come back to the wrong rank."""
    generator = torch.Generator().manual_seed(0)
    d = _make_dispatcher(8, 8)
    recv_tpe_2d = torch.randint(0, 5, (8, 8), generator=generator, dtype=torch.long)
    total = int(recv_tpe_2d.sum())
    index = d._expert_major_row_index(recv_tpe_2d, total)

    inverse = torch.empty_like(index)
    inverse.scatter_(0, index, torch.arange(total))

    x = torch.randn(total, 6, generator=generator)
    round_tripped = x.index_select(0, index).index_select(0, inverse)
    torch.testing.assert_close(round_tripped, x, rtol=0, atol=0)


def test_index_builder_reads_nothing_back_to_the_host() -> None:
    """No ``.item()``/``.tolist()`` on the sizes: that would sync every layer."""
    d = _make_dispatcher(8, 8)
    recv_tpe_2d = torch.full((8, 8), 3, dtype=torch.long)

    class _NoSync(torch.Tensor):
        @classmethod
        def __torch_function__(cls, func, types, args=(), kwargs=None):
            if func.__name__ in ("tolist", "item"):
                raise AssertionError(f"index builder synced to host via {func.__name__}")
            return super().__torch_function__(func, types, args, kwargs or {})

    guarded = recv_tpe_2d.as_subclass(_NoSync)
    d._expert_major_row_index(guarded, int(recv_tpe_2d.sum()))
