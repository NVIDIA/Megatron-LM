# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import tempfile
import unittest
from pathlib import Path

from tools import check_golden_values


class TestGoldenInvalidEncoding(unittest.TestCase):
    def test_reports_unreadable_utf8_and_continues_checking_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / 'invalid.json'
            bad.write_bytes(b'\xff')
            nonfinite = Path(directory) / 'nonfinite.json'
            nonfinite.write_text('{"metric": NaN}')
            with self.assertLogs(check_golden_values.logger, level='ERROR') as logs:
                result = check_golden_values.main([str(bad), str(nonfinite)])
            self.assertEqual(result, 1)
            self.assertTrue(any('Could not read' in line for line in logs.output))
            self.assertTrue(any('Found non-finite values' in line for line in logs.output))


if __name__ == '__main__':
    unittest.main()
