# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import os
import subprocess
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from click.testing import CliRunner

from tests.test_utils.python_scripts import generate_local_jobs


class TestLocalJobOutputPaths(unittest.TestCase):
    def test_generated_script_preserves_spaces_and_shell_metacharacters_in_output_path(self):
        workload = SimpleNamespace(
            type='basic',
            spec={
                'model': 'gpt',
                'test_case': 'sample',
                'name': 'sample',
                'script': 'printf "%s\\n" "$OUTPUT_PATH"',
            },
        )
        for output_path in ('/tmp/experiment results', '/tmp/$OUTPUT_PARENT', "/tmp/owner's run"):
            with (
                self.subTest(output_path=output_path),
                CliRunner().isolated_filesystem(),
                patch.object(
                    generate_local_jobs.recipe_parser, 'load_workloads', return_value=[workload]
                ),
            ):
                generate_local_jobs.main.callback('gpt', 'mr', 'sample', 'dev', output_path)
                result = subprocess.run(
                    ['/bin/sh', str(Path('test_cases/gpt/sample.sh'))],
                    capture_output=True,
                    text=True,
                    env={**os.environ, 'OUTPUT_PARENT': 'expanded'},
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.splitlines()[0].startswith(output_path + '/runs/'))


if __name__ == '__main__':
    unittest.main()
