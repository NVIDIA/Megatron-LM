# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest

from tests.test_utils.python_scripts import download_golden_values


class TestGoldenArtifactPrefix(unittest.TestCase):
    def test_preserves_logs_inside_the_test_case_name(self):
        for test_name in ('logs-resume', 'restore-logs-state', 'ordinary-test'):
            with self.subTest(test_name=test_name):
                job = {'id': 1, 'name': 'gpt/' + test_name}
                artifact = {'name': f'logs-{test_name}-123-uuid'}
                self.assertIs(
                    download_golden_values._match_artifact_to_job(artifact, 123, {test_name: job}),
                    job,
                )


if __name__ == '__main__':
    unittest.main()
