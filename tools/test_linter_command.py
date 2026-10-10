# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest
from unittest import mock

import linter


class TestLinter(unittest.TestCase):
    def test_formatter_receives_a_literal_filename_argument(self):
        with (
            mock.patch.dict('sys.modules', {'autopep8': mock.Mock()}),
            mock.patch.object(linter.os, 'listdir', return_value=['model file.py']),
            mock.patch.object(linter.os, 'walk', return_value=[]),
            mock.patch.object(linter.subprocess, 'check_call') as call,
        ):
            linter.recursively_lint_files()
            args = call.call_args.args[0]
            self.assertIsInstance(args, list)
            self.assertEqual(
                args[:-1], ['autopep8', '--max-line-length', '100', '--aggressive', '--in-place']
            )
            self.assertTrue(args[-1].endswith('/model file.py'))


if __name__ == '__main__':
    unittest.main()
