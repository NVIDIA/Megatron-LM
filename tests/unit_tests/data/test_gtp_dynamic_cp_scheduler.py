# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""CPU contracts for the ordinary schedulers over the full GTP data domain.

Run directly with Python without importing the CUDA-backed Megatron package.
The AST loader executes the production scheduler definitions unchanged.
"""

import ast
import random
import unittest
from collections import Counter
from functools import lru_cache
from math import ceil, log2
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple


def _load_schedulers():
    root = Path(__file__).resolve().parents[3]
    selections = {
        'megatron/core/context_parallel/groups.py': (
            'get_valid_dynamic_context_parallel_group_sizes',
        ),
        'megatron/core/datasets/data_schedule_utils.py': (
            'get_packed_sequence_length',
            'next_hdp_group_packing_aware',
            'align_sample_id_groups',
        ),
        'megatron/core/datasets/data_schedule.py': (
            'BasePackingScheduler',
            'DpBalancedScheduler',
            'DefaultDynamicCPScheduler',
        ),
    }
    namespace = dict(globals(), _DYNAMIC_CP_WORKLOAD_CAP_DELTA=0.05)
    for filename, names in selections.items():
        path = root / filename
        source = path.read_text()
        nodes = [node for node in ast.parse(source).body if getattr(node, 'name', None) in names]
        assert {node.name for node in nodes} == set(names)
        selected = 'from __future__ import annotations\n' + '\n\n'.join(
            ast.get_source_segment(source, node) for node in nodes
        )
        exec(compile(selected, str(path), 'exec'), namespace)
        namespace['parallel_state'] = SimpleNamespace(
            get_valid_dynamic_context_parallel_group_sizes=namespace[
                'get_valid_dynamic_context_parallel_group_sizes'
            ]
        )
    return namespace


_SCHEDULERS = _load_schedulers()


class TestFullDomainGTPScheduling(unittest.TestCase):
    def schedule(self, lengths, ranks, dynamic=True, multiple=None, capacity=1024):
        cp_size = 2
        cls = _SCHEDULERS['DefaultDynamicCPScheduler' if dynamic else 'DpBalancedScheduler']
        scheduler = cls(capacity, cp_size, ranks // cp_size, multiple)
        result = scheduler.get_groups_and_subsamples(list(enumerate(lengths)))
        seen = Counter()
        used_groups = set()
        for row in result:
            self.assertEqual(len(row), ranks)
            self.assertTrue(all(row))
            groups = {}
            for rank, ids in enumerate(row):
                groups.setdefault(tuple(ids), []).append(rank)
            for ids, members in groups.items():
                size = len(members)
                allowed_sizes = (
                    _SCHEDULERS['get_valid_dynamic_context_parallel_group_sizes'](ranks)
                    if dynamic
                    else [cp_size]
                )
                self.assertIn(size, allowed_sizes)
                self.assertEqual(
                    members, list(range(members[0] // size * size, (members[0] // size + 1) * size))
                )
                used_groups.add(tuple(members))
                for sid in ids:
                    seen[sid] += 1
                    self.assertEqual(
                        [rank for rank, pack in enumerate(row) if sid in pack], members
                    )
                # Inputs are physical padded lengths; zigzag's per-sequence
                # rounding and CP-local alignment must still fit capacity.
                seq_alignment = 2 * size if size > 1 else 1
                physical = sum(ceil(lengths[sid] / seq_alignment) * seq_alignment for sid in ids)
                self.assertLessEqual(ceil(physical / size), capacity)
        self.assertEqual(seen, Counter({sid: 1 for sid in range(len(lengths))}))
        if multiple:
            self.assertEqual(len(result) % multiple, 0)
        return result, used_groups

    def test_four_gpu_asymmetry_uses_original_scheduler_in_five_rounds(self):
        lengths = [128] * 8 + [2048] * 8
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                result, _ = self.schedule(lengths, 4, dynamic=dynamic)
                self.assertEqual(len(result), 5)

    def test_eight_gpu_g2_replica2_asymmetry(self):
        lengths = [128] * 16 + [2048] * 16
        for dynamic in (False, True):
            with self.subTest(dynamic=dynamic):
                result, _ = self.schedule(lengths, 8, dynamic=dynamic)
                self.assertEqual(len(result), 5)

    def test_runtime_cp_crosses_gtp_coordinates(self):
        for ranks in (4, 8):
            with self.subTest(ranks=ranks):
                _, groups = self.schedule([ranks * 1024] + [128] * 16, ranks)
                self.assertIn(tuple(range(ranks)), groups)

    def test_eight_gpu_cp4_groups_are_full_domain_contiguous(self):
        _, groups = self.schedule([4096, 4096] + [128] * 16, 8)
        self.assertIn((0, 1, 2, 3), groups)
        self.assertIn((4, 5, 6, 7), groups)

    def test_vpp_alignment_uses_public_scheduler(self):
        for ranks in (4, 8):
            for dynamic in (False, True):
                with self.subTest(ranks=ranks, dynamic=dynamic):
                    self.schedule([128] * 16 + [1024] * 16, ranks, dynamic, multiple=2)

    def test_static_alignment_can_take_a_sample_from_first_pack(self):
        # CP2, G2: one sample per data owner is a complete fixed-size batch.
        result, _ = self.schedule([8, 8], 4, dynamic=False, capacity=8)
        self.assertEqual(len(result), 1)

    def test_varied_lengths_conserve_samples_and_match_runtime_groups(self):
        rng = random.Random(20261026)
        for ranks in (4, 8):
            for case in range(40):
                lengths = [rng.randrange(1, 65) * 16 for _ in range(rng.randrange(8, 65))]
                with self.subTest(ranks=ranks, case=case):
                    self.schedule(lengths, ranks)


if __name__ == '__main__':
    unittest.main(verbosity=2)
