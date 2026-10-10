# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.test_utils.python_scripts import download_coverage_results


class TestCoverageBridgePagination(unittest.TestCase):
    def test_downloads_unit_test_bridges_beyond_the_first_page(self):
        project = MagicMock()
        parent_pipeline = MagicMock()
        downstream = MagicMock()
        downstream.jobs.list.return_value = []
        first_page = [SimpleNamespace(name="build", downstream_pipeline=None)]
        later_bridge = SimpleNamespace(name="test:unit_tests_dev", downstream_pipeline={"id": 456})
        parent_pipeline.bridges.list.side_effect = lambda **kwargs: (
            first_page + [later_bridge] if kwargs.get("get_all") else first_page
        )
        project.pipelines.get.side_effect = lambda pipeline_id: (
            parent_pipeline if pipeline_id == 123 else downstream
        )
        with patch.object(download_coverage_results.gitlab, "Gitlab") as gitlab_handle:
            gitlab_handle.return_value.projects.get.return_value = project
            download_coverage_results.main.callback(123)

        project.pipelines.get.assert_any_call(456)
        parent_pipeline.bridges.list.assert_called_once_with(get_all=True)


if __name__ == "__main__":
    unittest.main()
