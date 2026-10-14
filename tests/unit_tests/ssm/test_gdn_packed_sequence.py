# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Mutation and boundary validation coverage for the GDN metadata cache."""

import gc
import unittest
import weakref
from unittest.mock import patch

import torch

from megatron.core.ssm import gdn_packed_sequence as module


class PackedSequenceTest(unittest.TestCase):
    def setUp(self):
        module._VALIDATED.clear()

    def test_reuses_validation_without_host_copy(self):
        q = torch.tensor([0, 4, 12])
        self.assertIs(module.resolve_packed_sequences(q, q, 12)[0], q)
        with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected copy")):
            self.assertIs(module.resolve_packed_sequences(q, q, 12)[1], q)

    def test_inplace_mutation_invalidates(self):
        q = torch.tensor([0, 4, 12])
        module.resolve_packed_sequences(q, q, 12)
        q[-1] = 13
        with self.assertRaises(ValueError):
            module.resolve_packed_sequences(q, q, 12)

    def test_alias_mutation_invalidates(self):
        q = torch.tensor([0, 4, 12])
        kv = q.clone()
        module.resolve_packed_sequences(q, kv, 12)
        kv.view(-1)[1] = 5
        with self.assertRaises(AssertionError):
            module.resolve_packed_sequences(q, kv, 12)

    def test_changed_total_invalidates(self):
        q = torch.tensor([0, 4, 12])
        module.resolve_packed_sequences(q, q, 12)
        with self.assertRaises(ValueError):
            module.resolve_packed_sequences(q, q, 16)

    def test_equal_distinct_tensors(self):
        q, kv = torch.tensor([0, 4, 12]), torch.tensor([0, 4, 12], dtype=torch.int32)
        got = module.resolve_packed_sequences(q, kv, 12)
        self.assertIs(got[0], q)
        self.assertIs(got[1], kv)

    def test_context_parallel_divisibility(self):
        q = torch.tensor([0, 3, 12])
        module.resolve_packed_sequences(q, q, 12)
        with self.assertRaises(ValueError):
            module.resolve_packed_sequences(q, q, 12, cp_size=4)

    def test_inference_tensors_are_not_cached(self):
        with torch.inference_mode():
            q = torch.tensor([0, 4, 12])
            module.resolve_packed_sequences(q, q, 12)
            self.assertFalse(module._VALIDATED)
            q[-1] = 13
            with self.assertRaises(ValueError):
                module.resolve_packed_sequences(q, q, 12)

    def test_cache_does_not_keep_tensors_alive(self):
        q = torch.tensor([0, 4, 12])
        ref = weakref.ref(q)
        module.resolve_packed_sequences(q, q, 12)
        del q
        gc.collect()
        self.assertIsNone(ref())

    def test_cache_is_bounded(self):
        tensors = [torch.tensor([0, 4, 12]) for _ in range(32)]
        for q in tensors:
            module.resolve_packed_sequences(q, q, 12)
        self.assertEqual(len(module._VALIDATED), module._CACHE_SIZE)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_gpu_mutation_is_revalidated(self):
        q = torch.tensor([0, 4, 12], device="cuda")
        module.resolve_packed_sequences(q, q, 12)
        q[-1] = 13
        with self.assertRaises(ValueError):
            module.resolve_packed_sequences(q, q, 12)


if __name__ == "__main__":
    unittest.main()
