# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Trace-driven retention shares the normal double-buffered weight window."""

import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import patch

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.execution_order import (
    ExecutionAction,
    ExecutionEvent,
    ExecutionOrder,
)
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.phase import Phase
from megatron.core.distributed.fsdp.src.megatron_fsdp.experimental.weight_retention import (
    build_retain_mask,
)


class TestWeightRetention(unittest.TestCase):
    def _event(self, owner, action, size, *, rowwise=False):
        return ExecutionEvent(
            owner, Phase.FORWARD, action, rowwise_only=rowwise, unsharded_bytes=size
        )

    def _build(self, source, group):
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.distributed, "all_reduce") as agree,
        ):
            mask = build_retain_mask(source, group, device="cpu")
        self.assertIs(agree.call_args.kwargs["group"], group)
        return mask

    def _residency(self, source, mask):
        # Replay physical demand, successor prefetch and release, independently of planning.
        events = tuple(source)
        sizes = {event.module: event.unsharded_bytes for event in events}
        resident, profile = set(), []
        for index, event in enumerate(events):
            if event.action is ExecutionAction.UNSHARD:
                resident.add(event.module)
                successor = next(
                    (
                        item
                        for item in events[index + 1 :]
                        if item.action is ExecutionAction.UNSHARD
                    ),
                    None,
                )
                if successor is not None:
                    resident.add(successor.module)
            elif not event.rowwise_only and event.event_id not in mask:
                resident.discard(event.module)
            profile.append(sum(sizes[owner] != 0 for owner in resident))
        self.assertFalse(resident)
        return profile

    def test_reuse_shares_prefetch_slots_in_flat_nested_and_rowwise_schedules(self):
        u, r = ExecutionAction.UNSHARD, ExecutionAction.RESHARD
        cases = []
        for word, sizes, expected in (
            ("ABAB", {"A": 10, "B": 20}, {1, 3}),
            ("ABAB", {"A": 1000000, "B": 1}, {1, 3}),
            ("ABAB", {"A": None, "B": None}, {1, 3}),
            # Keeping A while B prefetches C would require a third buffer.
            ("ABCABC", {"A": 10, "B": 20, "C": 30}, set()),
            ("ABCABC", {"A": 0, "B": 20, "C": 30}, {1, 3, 5}),
        ):
            owners = {letter: torch.nn.Identity() for letter in sizes}
            source = ExecutionOrder(
                self._event(owners[letter], action, sizes[letter])
                for letter in word
                for action in (u, r)
            )
            cases.append((source, tuple(owners.values()), expected, 2))
        for depth in (3, 4):
            owners = tuple(torch.nn.Identity() for _ in range(depth))
            events = [self._event(owner, u, 10) for owner in owners]
            events += [self._event(owner, r, 10) for owner in reversed(owners)]
            # Root/layer/expert nesting can necessarily exceed two resident owners.
            cases.append(
                (ExecutionOrder(events * 2), owners, {2 * depth - 2, 2 * depth - 1}, depth)
            )
        first, second = torch.nn.Identity(), torch.nn.Identity()
        rowwise = ExecutionOrder(
            (
                self._event(first, u, 10),
                self._event(first, r, 10, rowwise=True),
                self._event(first, r, 10),
                self._event(second, u, 20),
                self._event(second, r, 20),
                self._event(first, u, 10),
                self._event(first, r, 10),
            )
        )
        cases.append((rowwise, (first, second), {1, 2}, 2))
        for source, owners, expected, peak_limit in cases:
            with self.subTest(expected=expected, peak_limit=peak_limit):
                before = tuple(source)
                mask = self._build(source, object())
                self.assertEqual(dict(mask), dict.fromkeys(expected, True))
                self.assertEqual(tuple(source), before)
                with self.assertRaises(TypeError):
                    mask[0] = True
                baseline, retained = self._residency(source, {}), self._residency(source, mask)
                self.assertLessEqual(max(retained), peak_limit)
                self.assertTrue(
                    all(actual <= max(2, normal) for actual, normal in zip(retained, baseline))
                )
                # The last release of every owner must remain physical.
                for owner in owners:
                    last = next(event for event in reversed(before) if event.module is owner)
                    self.assertNotIn(last.event_id, mask)

    def test_trace_agreement_errors_and_masks_keep_weak_ownership(self):
        u, r = ExecutionAction.UNSHARD, ExecutionAction.RESHARD
        owner, group = torch.nn.Identity(), object()
        source = ExecutionOrder(self._event(owner, action, 10) for action in (u, r, u, r))
        with (
            patch.object(torch.cuda, "is_current_stream_capturing", return_value=False),
            patch.object(torch.distributed, "all_reduce") as agree,
        ):
            with self.assertRaisesRegex(ValueError, "explicit agreement group"):
                build_retain_mask(source, None, device="cpu")
            agree.assert_not_called()
            invalid = ExecutionOrder((self._event(owner, u, 10), self._event(owner, r, 11)))
            with self.assertRaisesRegex(RuntimeError, "recorded unsharded_bytes changed"):
                build_retain_mask(invalid, group, device="cpu")
            for direction in ("forward_prefetch_size", "backward_prefetch_size"):
                owner._schedule_policy = SimpleNamespace(
                    forward_prefetch_size=None, backward_prefetch_size=None
                )
                setattr(owner._schedule_policy, direction, 1)
                with self.subTest(direction=direction):
                    with self.assertRaisesRegex(ValueError, "single-successor"):
                        build_retain_mask(source, group, device="cpu")
            owner._schedule_policy.forward_prefetch_size = 0
            owner._schedule_policy.backward_prefetch_size = None
            self.assertEqual(dict(build_retain_mask(source, group, device="cpu")), {1: True})

            def mismatch(header, **kwargs):
                self.assertIs(kwargs["group"], group)
                header[0] -= 1

            agree.side_effect = mismatch
            with self.assertRaisesRegex(RuntimeError, "differs across ranks"):
                build_retain_mask(source, group, device="cpu")
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                build_retain_mask(source, group, device="cpu")
        del invalid
        mask = self._build(source, group)
        reference = weakref.ref(owner)
        del owner
        gc.collect()
        self.assertIsNone(reference())
        self.assertEqual(dict(mask), {1: True})
        with self.assertRaisesRegex(RuntimeError, "expired"):
            self._build(source, group)


if __name__ == "__main__":
    unittest.main()
