# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch

from megatron.core.inference.async_txn import AsyncDecodeSlot, AsyncDecodeSlotRing


class FakeEvent:
    def __init__(self, done: bool):
        self.done = done

    def query(self) -> bool:
        return self.done


class FakeGPUView:
    def __init__(self):
        self._buf = torch.zeros(16, dtype=torch.uint8)
        self.token_to_input_ids = torch.zeros(4, dtype=torch.int64)
        self.token_to_pos_ids = torch.zeros(4, dtype=torch.int64)
        self.token_to_block_idx = torch.zeros(4, dtype=torch.int64)
        self.mha_block_table = torch.zeros((2, 2), dtype=torch.int32)


def test_slot_cannot_reuse_before_h2d_event_retires():
    h2d = FakeEvent(False)
    slot = AsyncDecodeSlot(slot_id=1, gpu_view=FakeGPUView(), h2d_done_event=h2d)

    assert not slot.can_reuse()
    h2d.done = True
    assert slot.can_reuse()


def test_slot_cannot_reuse_before_forward_event_retires():
    forward = FakeEvent(False)
    slot = AsyncDecodeSlot(slot_id=1, gpu_view=FakeGPUView(), forward_done_event=forward)

    assert not slot.can_reuse()
    forward.done = True
    assert slot.can_reuse()


def test_graph_key_changes_when_slot_pointers_differ():
    slot_a = AsyncDecodeSlot(slot_id=0, gpu_view=FakeGPUView())
    slot_b = AsyncDecodeSlot(slot_id=1, gpu_view=FakeGPUView())

    assert slot_a.pointer_signature() != slot_b.pointer_signature()


def test_ring_promotes_only_reusable_child_slot():
    current = AsyncDecodeSlot(slot_id=0, gpu_view=FakeGPUView())
    child = AsyncDecodeSlot(
        slot_id=1,
        gpu_view=FakeGPUView(),
        h2d_done_event=FakeEvent(True),
        forward_done_event=FakeEvent(False),
    )
    ring = AsyncDecodeSlotRing((current, child))

    with pytest.raises(RuntimeError):
        ring.promote_child()

    child.forward_done_event.done = True
    assert ring.promote_child().slot_id == 1
