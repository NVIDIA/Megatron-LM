# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
import unittest

from megatron.rl.shared_prefix_alignment import align_rows_to_count


class AlignmentTests(unittest.TestCase):
    def test_equal_count_preserves_every_shared_layout_reference(self):
        units = ((2, 0), (1, 3))
        result = align_rows_to_count(
            units, padded_row_lengths=(8, 16, 24, 32), bin_capacity=64, target_count=2
        )
        self.assertEqual(tuple(unit.row_indices for unit in result), units)
        self.assertEqual(tuple(unit.source_unit for unit in result), (0, 1))

    def test_only_split_units_become_dense_without_dummy_rows(self):
        result = align_rows_to_count(
            ((0, 1, 2, 3), (4,)),
            padded_row_lengths=(40, 8, 8, 8, 8),
            bin_capacity=64,
            target_count=3,
        )
        self.assertEqual(tuple(u.row_indices for u in result), ((0,), (1, 2, 3), (4,)))
        self.assertEqual(tuple(u.source_unit for u in result), (None, None, 1))

    def test_can_align_to_singletons_without_reordering_or_copying_rows(self):
        result = align_rows_to_count(
            ((3, 1, 0), (2, 4)), padded_row_lengths=(1, 1, 1, 1, 1), bin_capacity=8, target_count=5
        )
        self.assertEqual(tuple(u.row_indices for u in result), ((3,), (1,), (0,), (2,), (4,)))
        self.assertTrue(all(u.source_unit is None for u in result))

    def test_rejects_bad_coverage_empty_units_and_unattainable_counts(self):
        for units, target in (
            (((0, 0), (1,)), 2),
            (((0,),), 1),
            (((0,), (1,), (2,)), 3),
            (((True,), (1,)), 2),
            (((), (0, 1)), 2),
            (((0, 1),), 0),
            (((0, 1),), 3),
            (((0, 1),), True),
        ):
            with self.subTest(units=units, target=target), self.assertRaises(ValueError):
                align_rows_to_count(
                    units, padded_row_lengths=(8, 8), bin_capacity=16, target_count=target
                )

    def test_rejects_physical_only_capacity_and_invalid_costs(self):
        for costs, capacity in (((16, 16), 16), ((0, 8), 16), ((1.5, 8), 16), ((8, 8), 0)):
            with self.subTest(costs=costs, capacity=capacity), self.assertRaises(ValueError):
                align_rows_to_count(
                    ((0, 1),), padded_row_lengths=costs, bin_capacity=capacity, target_count=2
                )

    def test_eight_rank_schedules_have_equal_counts_and_exact_coverage(self):
        rng = random.Random(185)
        for _ in range(100):
            shards = []
            for _rank in range(8):
                costs = tuple(8 * rng.randint(1, 16) for _ in range(32))
                order = list(range(32))
                rng.shuffle(order)
                units = [[]]
                used = 0
                for row in order:
                    if used + costs[row] > 512:
                        units.append([])
                        used = 0
                    units[-1].append(row)
                    used += costs[row]
                shards.append((costs, units))
            target = max(len(units) for _, units in shards)
            for costs, units in shards:
                result = align_rows_to_count(
                    units, padded_row_lengths=costs, bin_capacity=512, target_count=target
                )
                self.assertEqual(len(result), target)
                self.assertEqual(
                    [row for unit in result for row in unit.row_indices],
                    [row for unit in units for row in unit],
                )
                self.assertEqual(
                    sorted(row for unit in result for row in unit.row_indices), list(range(32))
                )
                for unit in result:
                    self.assertLessEqual(sum(costs[row] for row in unit.row_indices), 512)
                    if unit.source_unit is not None:
                        self.assertEqual(unit.row_indices, tuple(units[unit.source_unit]))


if __name__ == "__main__":
    unittest.main()
