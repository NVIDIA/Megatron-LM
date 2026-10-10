# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest

import community_request_assignee as assignment


class TestConfidence(unittest.TestCase):
    def test_nonfinite_values_cannot_authorize_automatic_assignment(self):
        for value in [float('inf'), float('-inf'), float('nan'), 'Infinity', '-Infinity', 'NaN']:
            with self.subTest(value=value):
                self.assertEqual(assignment.confidence_value(value), 0.0)

    def test_valid_confidence_is_preserved(self):
        self.assertEqual(assignment.confidence_value(0.9), 0.9)


if __name__ == '__main__':
    unittest.main()
