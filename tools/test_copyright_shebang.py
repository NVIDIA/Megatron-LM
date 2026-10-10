# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import check_copyright


class TestCopyright(unittest.TestCase):
    def test_header_can_follow_a_shebang(self):
        header = check_copyright.EXPECTED_HEADER.format(datetime.now().year)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'script.py'
            for prefix, expected in [
                ('#!/usr/bin/env python3\n', True),
                ('', True),
                ('print(1)\n', False),
            ]:
                with self.subTest(prefix=prefix):
                    path.write_text(prefix + header + '\n', encoding='utf-8')
                    self.assertEqual(check_copyright.has_correct_header(path), expected)


if __name__ == '__main__':
    unittest.main()
