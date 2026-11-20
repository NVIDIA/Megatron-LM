# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Ownership and differentiability checks for GDN's internal CP conversion."""

import pytest
import torch
from tests.unit_tests.test_utilities import Utils

from megatron.core.ssm.context_parallel.gdn_layout import _Redistributor, build_layout_plan


@pytest.mark.parametrize("boundaries", [(0, 64), (0, 8, 32, 64), (0, 16, 48, 128)])
@pytest.mark.parametrize("size", [2, 4])
def test_layout_ownership(boundaries, size):
    """Compare the complete collective plan with explicit per-document ownership."""
    local = boundaries[-1] // size
    sources = []
    for rank in range(size):
        source = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            chunks = list(range(start, stop))
            width = len(chunks) // (2 * size)
            source.extend(chunks[rank * width : (rank + 1) * width])
            reverse = 2 * size - 1 - rank
            source.extend(chunks[reverse * width : (reverse + 1) * width])
        sources.append(source)
    plans = [build_layout_plan(boundaries, size, rank) for rank in range(size)]
    received = [[] for _ in range(size)]
    for source, plan in zip(sources, plans):
        packed = [source[index] for index in plan.send_order]
        offset = 0
        for rank, count in enumerate(plan.send_counts):
            received[rank].extend(packed[offset : offset + count])
            offset += count
    for rank, (plan, values) in enumerate(zip(plans, received)):
        assert [values[index] for index in plan.receive_order] == list(
            range(rank * local, (rank + 1) * local)
        )


@pytest.mark.parametrize("boundaries", [(0, 63), (1, 64), (0, 0, 64), (0, 32, 24)])
def test_reject_invalid_physical_packing(boundaries):
    with pytest.raises(ValueError):
        build_layout_plan(boundaries, 4, 0)


@pytest.mark.parametrize("size", [2, 4])
def test_transport_values_and_adjoint(size):
    """A weighted loss checks both ownership and the reverse collective in backward."""
    Utils.initialize_model_parallel(context_parallel_size=size)
    try:
        from megatron.core.process_groups_config import ProcessGroupCollection

        group = ProcessGroupCollection.use_mpu_process_groups().cp
        rank = group.rank()
        boundaries = (0, 8, 32, 64)
        positions = []
        for start, stop in zip(boundaries[:-1], boundaries[1:]):
            width = (stop - start) // (2 * size)
            for chunk in (rank, 2 * size - 1 - rank):
                positions.extend(range(start + chunk * width, start + (chunk + 1) * width))
        x = (
            torch.tensor(positions, device="cuda", dtype=torch.float32)
            .unsqueeze(-1)
            .requires_grad_()
        )
        transport = _Redistributor(build_layout_plan(boundaries, size, rank), group, x.device)
        y = transport.forward(x)
        expected = torch.arange(rank * len(x), (rank + 1) * len(x), device=x.device).unsqueeze(-1)
        torch.testing.assert_close(y, expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(transport.inverse(y), x, rtol=0, atol=0)
        (y * (expected + 1)).sum().backward()
        torch.testing.assert_close(x.grad, x.detach() + 1, rtol=0, atol=0)
    finally:
        Utils.destroy_model_parallel()
