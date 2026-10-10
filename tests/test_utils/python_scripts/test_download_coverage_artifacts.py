# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import io
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from tests.test_utils.python_scripts import download_coverage_results


def archive_bytes(files):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        for name, value in files.items():
            archive.writestr(name, value)
    return stream.getvalue()


class TestMissingCoverageArtifacts(unittest.TestCase):
    def test_skips_unrelated_artifacts_and_continues_to_the_next_job(self):
        self._assert_skips_missing_report({'README': 'setup'})

    def test_skips_iterations_without_attempt_directories(self):
        self._assert_skips_missing_report({'results/iteration=1/': ''})

    def test_skips_attempts_without_a_coverage_report(self):
        self._assert_skips_missing_report({'results/iteration=1/attempt_0/README': 'setup'})

    def _assert_skips_missing_report(self, missing_files):
        project = MagicMock()
        parent = MagicMock()
        downstream = MagicMock()
        parent.bridges.list.return_value = [
            SimpleNamespace(name='test:unit_tests_dev', downstream_pipeline={'id': 2})
        ]
        downstream.jobs.list.return_value = [SimpleNamespace(id=3), SimpleNamespace(id=4)]
        project.pipelines.get.side_effect = lambda key: parent if key == 1 else downstream
        missing = SimpleNamespace(
            name='setup', artifacts=lambda **kwargs: kwargs['action'](archive_bytes(missing_files))
        )
        coverage = SimpleNamespace(
            name='unit-tests',
            artifacts=lambda **kwargs: kwargs['action'](
                archive_bytes(
                    {
                        'results/iteration=1/attempt_0/assets/basic/test/coverage_report/index.html': 'coverage'
                    }
                )
            ),
        )
        project.jobs.get.side_effect = lambda key: missing if key == 3 else coverage
        with (
            CliRunner().isolated_filesystem(),
            patch.object(download_coverage_results.gitlab, 'Gitlab') as gitlab_handle,
        ):
            gitlab_handle.return_value.projects.get.return_value = project
            download_coverage_results.main.callback(1)
            self.assertEqual(
                Path('coverage_results/unit-tests/coverage_report/index.html').read_text(),
                'coverage',
            )


if __name__ == '__main__':
    unittest.main()
