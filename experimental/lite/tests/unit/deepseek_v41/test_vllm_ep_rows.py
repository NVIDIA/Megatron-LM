# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Real Core/Gloo variable-row transport vs dense slot routing, without CUDA kernels."""
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.functional import pad

from megatron.core.tensor_parallel.mappings import all_to_all, gather_from_sequence_parallel_region
from megatron.lite.model.deepseek_v41.vllm.moe import dispatch_rows, return_rows


def _dense(x, scores, ids, weight, group):
    capacity = max(
        1, int(gather_from_sequence_parallel_region(torch.tensor([len(x)]), group=group).max())
    )

    def gather(t, fill=0):
        return gather_from_sequence_parallel_region(
            pad(t, (0, 0, 0, capacity - len(t)), value=fill), group=group
        )

    live, weights, routes = gather(x), gather(scores), gather(ids, -1)
    parts = [
        (routes.flatten() == e).nonzero().flatten()
        for e in range(group.rank() * 3, (group.rank() + 1) * 3)
    ]
    position = torch.cat(parts)
    multiplier = torch.cat([torch.full((len(p),), i + 1) for i, p in enumerate(parts)])
    rows = (
        (live[position // ids.shape[1]] @ weight)
        * weights.flatten()[position, None]
        * multiplier[:, None]
    )
    slots = rows.new_zeros((routes.numel(), x.shape[1])).index_copy(0, position, rows)
    received = all_to_all(group, slots.reshape(group.size(), capacity, ids.shape[1], x.shape[1]))
    owner = pad(ids, (0, 0, 0, capacity - len(x)), value=-1).clamp_min(0) // 3
    return (
        received.permute(1, 2, 0, 3)
        .gather(2, owner[:, :, None, None].expand(-1, -1, 1, x.shape[1]))
        .squeeze(2)[: len(x)]
    )


def _routed(x, scores, ids, weight, group):
    encoded = x.detach().to(torch.float8_e4m3fn), x.new_ones((len(x), 1))
    live, weights, received, counts, state = dispatch_rows(x, scores, ids, encoded, 3, group)
    assert torch.equal(
        received[0].view(torch.uint8), live.detach().to(torch.float8_e4m3fn).view(torch.uint8)
    )
    assert torch.equal(received[1], live.new_ones((len(live), 1)))
    multiplier = torch.repeat_interleave(torch.arange(1, 4), torch.tensor(counts))
    rows = (live @ weight) * weights[:, None] * multiplier[:, None]
    # Copy-only return preserves both zero signs before slot arithmetic.
    signs = torch.zeros_like(rows)
    if signs.numel():
        signs[:, 0] = -0.0
    returned = return_rows(signs, ids, state, group)
    valid = ids >= 0
    assert torch.all(returned[:, :, 0][valid].view(torch.int32) == -2147483648)
    return return_rows(rows, ids, state, group)


def _worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    original = torch.cuda.current_device
    try:
        # Core all_to_all allocates using current_device even for Gloo; only
        # adapt the device query, not collectives or autograd implementations.
        with patch('torch.cuda.current_device', return_value=torch.device('cpu')):
            for sizes in ((3, 5), (0, 7), (5, 0), (0, 0)):
                torch.manual_seed(230 + rank)
                count = sizes[rank]
                x, scores, ids = (
                    torch.randn(count, 8),
                    torch.rand(count, 3),
                    torch.randint(6, (count, 3)),
                )
                if count:
                    x[0, 0], ids[0, -1] = -0.0, -1
                weight, upstream = torch.randn(8, 8), torch.randn(count, 3, 8)
                results = []
                for implementation in (_dense, _routed):
                    leaves = [t.clone().requires_grad_() for t in (x, scores, weight)]
                    result = implementation(*leaves[:2], ids, leaves[2], dist.group.WORLD)
                    result.backward(upstream)
                    results.append((result.detach(), [t.grad for t in leaves]))
                assert torch.equal(results[0][0].view(torch.uint8), results[1][0].view(torch.uint8))
                for a, b in zip(results[0][1], results[1][1]):
                    assert (
                        a is not None
                        and b is not None
                        and torch.isfinite(a).all()
                        and torch.isfinite(b).all()
                    )
                    torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
        assert torch.cuda.current_device is original and not torch.cuda.is_initialized()
    finally:
        dist.destroy_process_group()


def test_actual_row_dispatch_preserves_dense_bits_and_vjp_with_empty_ranks(tmp_path):
    mp.spawn(_worker, args=('file://' + str(tmp_path / 'rendezvous'),), nprocs=2, join=True)
