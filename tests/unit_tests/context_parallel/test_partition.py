# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from types import SimpleNamespace

import pytest
import torch

from megatron.core.context_parallel.partition import get_cp_partition_indices, partition_batch
from megatron.core.context_parallel.utils import _get_batch_on_this_cp_rank_contiguous
from megatron.core.datasets.data_schedule_utils import get_cp_slice_for_thd
from megatron.core.utils import _get_batch_on_this_cp_rank_per_document_balancing


@pytest.mark.parametrize("cp_rank", [0, 1])
@pytest.mark.parametrize("layout", ["zigzag", "contiguous"])
def test_scheduler_and_ordinary_batches_select_the_same_rows(monkeypatch, cp_rank, layout):
    """Rank ownership must not depend on a scheduler's flattened tensor representation."""
    cu = torch.tensor([0, 8, 24], device="cuda", dtype=torch.int32)
    tokens = torch.arange(24, device="cuda")
    ordinary = {"tokens": tokens[None], "labels": None, "cu_seqlens_padded": cu[None]}
    scheduler = {"tokens": tokens, "labels": None, "cu_seqlens_padded": cu}
    group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group: cp_rank)
    get_cp_slice_for_thd(scheduler, group, cp_partition_mode=layout)
    if layout == "zigzag":
        _get_batch_on_this_cp_rank_per_document_balancing(ordinary, group)
        expected = []
        for start, length in ((0, 8), (8, 16)):
            chunk = length // 4
            for offset in (cp_rank, 3 - cp_rank):
                expected.extend(range(start + offset * chunk, start + (offset + 1) * chunk))
        expected = torch.tensor(expected, device="cuda")
    else:
        _get_batch_on_this_cp_rank_contiguous(ordinary, group)
        expected = tokens[cp_rank * 12 : (cp_rank + 1) * 12]
    torch.testing.assert_close(scheduler["tokens"], expected)
    torch.testing.assert_close(ordinary["tokens"][0], expected)
    assert ordinary["labels"] is None and scheduler["labels"] is None


def test_contiguous_scheduler_tail_padding_preserves_metadata():
    """Tail padding and masks remain scheduler policy around shared row selection."""
    cu = torch.tensor([0, 6], dtype=torch.int32)
    batch = {
        "tokens": torch.arange(6),
        "padding_mask": torch.zeros(6, dtype=torch.bool),
        "cu_seqlens_padded": cu,
    }
    group = SimpleNamespace(size=lambda: 2, rank=lambda: 1)
    get_cp_slice_for_thd(
        batch,
        group,
        keys=("tokens", "padding_mask", "missing"),
        cp_partition_mode="contiguous",
        partition_total_tokens=8,
    )
    torch.testing.assert_close(batch["tokens"], torch.tensor([4, 5, 0, 0]))
    torch.testing.assert_close(batch["padding_mask"], torch.tensor([False, False, True, True]))
    assert batch["cu_seqlens_padded"] is cu
    assert "missing" not in batch


def test_shared_partition_preserves_sequence_dimension_and_gradients():
    tensor = torch.arange(24.0).reshape(2, 12).requires_grad_()
    batch = {"tokens": tensor, "optional": None}
    partition = get_cp_partition_indices(None, 12, 2, 1, "contiguous")
    partition_batch(batch, ("tokens", "optional", "missing"), partition, seq_dim=1)
    batch["tokens"].sum().backward()
    expected_grad = torch.zeros_like(tensor)
    expected_grad[:, 6:] = 1
    torch.testing.assert_close(tensor.grad, expected_grad)
    assert batch["optional"] is None
    assert "missing" not in batch


@pytest.mark.parametrize("cu_values", [[0, 5, 16], [0, 8, 16]])
@pytest.mark.parametrize("cp_rank", [0, 1])
def test_sample_level_masking_does_not_require_document_routes(monkeypatch, cu_values, cp_rank):
    from megatron.core.context_parallel import finalize_packed_seq_params
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.utils import get_batch_on_this_cp_rank

    cu = torch.tensor(cu_values, device="cuda", dtype=torch.int32)
    group = SimpleNamespace(size=lambda: 2, rank=lambda: cp_rank)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda _group: 2)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda _group: cp_rank)
    full = torch.arange(16, device="cuda")
    batch = {"tokens": full[None].clone(), "cu_seqlens": cu[None]}
    result = get_batch_on_this_cp_rank(
        batch,
        is_hybrid_cp=False,
        cp_group=group,
        use_per_sequence_balancing=True,
        cp_partition_mode="zigzag",
    )
    params = PackedSeqParams(qkv_format="thd", cu_seqlens_q=cu, tokens_per_sample=16)
    finalize_packed_seq_params(params, group, needs_layout_conversion=False)
    indices = [*range(cp_rank * 4, (cp_rank + 1) * 4), *range((3 - cp_rank) * 4, (4 - cp_rank) * 4)]
    torch.testing.assert_close(result["tokens"][0], full[indices])
    assert params.cp_partition_route is None
    with pytest.raises(NotImplementedError, match="sample-level inter-document masking"):
        get_batch_on_this_cp_rank(
            {"tokens": full[None], "cu_seqlens": cu[None]},
            is_hybrid_cp=False,
            cp_group=group,
            use_per_sequence_balancing=True,
            cp_partition_mode="contiguous",
        )
