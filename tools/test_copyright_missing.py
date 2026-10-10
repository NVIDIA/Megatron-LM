# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import check_copyright


class TestCopyright(unittest.TestCase):
    def test_invalid_input_paths_fail_the_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            for path in [tmp, str(Path(tmp) / 'missing.py')]:
                with self.subTest(path=path), mock.patch('sys.argv', ['check_copyright.py', path]):
                    with self.assertRaises(SystemExit) as result:
                        check_copyright.main()
                    self.assertEqual(result.exception.code, 1)


if __name__ == '__main__':
    unittest.main()
