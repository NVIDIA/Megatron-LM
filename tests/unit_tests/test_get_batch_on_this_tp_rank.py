# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Unit tests for get_batch_on_this_tp_rank.

Verifies that packed-sequence metadata (cu_seqlens, max_seqlen) is correctly
broadcast to all tensor-parallel ranks on first and intermediate pipeline
stages when SFT packing is enabled.

Regression for issue #4092: intermediate pipeline stages (neither first nor
last) previously never broadcast cu_seqlens/max_seqlen to TP ranks > 0,
causing RoPE and FlashAttention to fail with a shape mismatch.

Note: last stage intentionally receives cu_seqlens=None on TP ranks > 0;
it only computes the loss and does not run packed-sequence attention.
"""

from types import SimpleNamespace

import pytest
import torch

from megatron.training.global_vars import set_args
from megatron.training.utils import get_batch_on_this_tp_rank
from tests.unit_tests.test_utilities import Utils

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SEQ_LEN = 16
_MICRO_BATCH_SIZE = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_args(tp: int, pp: int) -> SimpleNamespace:
    """Minimal args namespace expected by get_batch_on_this_tp_rank."""
    return SimpleNamespace(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
        seq_length=_SEQ_LEN,
        micro_batch_size=_MICRO_BATCH_SIZE,
        sft=True,
        hybrid_context_parallel=False,
        create_attention_mask_in_dataloader=False,
    )


def _make_data_iterator():
    """One-shot iterator yielding a fake SFT-packed batch (consumed by TP rank-0 only)."""
    data = {
        "tokens": torch.randint(0, 1000, (_MICRO_BATCH_SIZE, _SEQ_LEN), dtype=torch.int64),
        "labels": torch.randint(0, 1000, (_MICRO_BATCH_SIZE, _SEQ_LEN), dtype=torch.int64),
        "loss_mask": torch.ones(_MICRO_BATCH_SIZE, _SEQ_LEN, dtype=torch.float32),
        "position_ids": (
            torch.arange(_SEQ_LEN, dtype=torch.int64)
            .unsqueeze(0)
            .expand(_MICRO_BATCH_SIZE, -1)
        ),
        "cu_seqlens": torch.tensor([[0, _SEQ_LEN]], dtype=torch.int32),
        "max_seqlen": torch.tensor([_SEQ_LEN], dtype=torch.int32),
    }
    return iter([data])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "tp,pp",
    [
        (2, 1),  # no pipeline parallel
        (2, 2),  # PP=2, no intermediate stage
        (2, 4),  # PP=4, 2 intermediate stages — primary regression target for #4092
    ],
)
def test_sft_packing_cu_seqlens_broadcast(tp: int, pp: int):
    """cu_seqlens and max_seqlen must be non-None on first and intermediate PP stages.

    On the unfixed branch, intermediate stages (pp_rank > 0 and pp_rank < pp - 1)
    never broadcast cu_seqlens/max_seqlen to TP ranks > 0, leaving them as None
    and causing RoPE / FlashAttention to fail with:

        RuntimeError: Tensors must have same number of dimensions: got 4 and 3

    The last stage only computes loss; pretrain_gpt.py intentionally uses
    cu_seqlens=None there to skip packed-sequence attention.
    """
    world_size = tp * pp
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs (tp={tp}, pp={pp})")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )

    import megatron.core.parallel_state as mpu

    set_args(_make_args(tp=tp, pp=pp))

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()
    is_first = mpu.is_pipeline_first_stage()
    is_last = mpu.is_pipeline_last_stage()
    is_intermediate = not is_first and not is_last

    # Only TP rank-0 feeds data; other ranks pass None.
    data_iterator = _make_data_iterator() if tp_rank == 0 else None

    batch = get_batch_on_this_tp_rank(data_iterator)

    # First and intermediate stages need cu_seqlens/max_seqlen for RoPE and FlashAttention.
    if is_first or is_intermediate:
        assert batch["cu_seqlens"] is not None, (
            f"cu_seqlens must not be None on pp_rank={pp_rank} tp_rank={tp_rank} "
            f"(first={is_first}, intermediate={is_intermediate})"
        )
        assert batch["max_seqlen"] is not None, (
            f"max_seqlen must not be None on pp_rank={pp_rank} tp_rank={tp_rank}"
        )
        assert batch["cu_seqlens"].dtype == torch.int32
        assert batch["max_seqlen"].dtype == torch.int32

    # Last stage: TP rank-0 keeps its loaded data; TP rank > 0 gets None (by design).
    if is_last and not is_first and tp_rank > 0:
        assert batch["cu_seqlens"] is None, (
            "cu_seqlens should be None on TP rank > 0 of the last pipeline stage"
        )

    Utils.destroy_model_parallel()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("tp,pp", [(2, 4)])
def test_sft_packing_cu_seqlens_consistent_across_tp_ranks(tp: int, pp: int):
    """All TP ranks on the same PP stage must receive identical cu_seqlens values.

    After the broadcast from TP rank-0, every other TP rank on the same stage
    must hold the exact same tensor contents.
    """
    world_size = tp * pp
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Need {world_size} GPUs (tp={tp}, pp={pp})")

    Utils.initialize_model_parallel(
        tensor_model_parallel_size=tp,
        pipeline_model_parallel_size=pp,
    )

    import megatron.core.parallel_state as mpu

    set_args(_make_args(tp=tp, pp=pp))

    tp_rank = mpu.get_tensor_model_parallel_rank()
    pp_rank = mpu.get_pipeline_model_parallel_rank()

    data_iterator = _make_data_iterator() if tp_rank == 0 else None
    batch = get_batch_on_this_tp_rank(data_iterator)

    # Last stage uses cu_seqlens=None by design; skip consistency check.
    if mpu.is_pipeline_last_stage() and not mpu.is_pipeline_first_stage():
        Utils.destroy_model_parallel()
        return

    assert batch["cu_seqlens"] is not None
    assert batch["max_seqlen"] is not None

    # Broadcast the TP-0 copy to all TP ranks and verify every rank agrees.
    cu_ref = batch["cu_seqlens"].clone()
    mx_ref = batch["max_seqlen"].clone()
    torch.distributed.broadcast(
        cu_ref,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )
    torch.distributed.broadcast(
        mx_ref,
        src=mpu.get_tensor_model_parallel_src_rank(),
        group=mpu.get_tensor_model_parallel_group(),
    )

    assert torch.equal(batch["cu_seqlens"], cu_ref), (
        f"cu_seqlens mismatch on pp_rank={pp_rank} tp_rank={tp_rank}"
    )
    assert torch.equal(batch["max_seqlen"], mx_ref), (
        f"max_seqlen mismatch on pp_rank={pp_rank} tp_rank={tp_rank}"
    )

    Utils.destroy_model_parallel()
