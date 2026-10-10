# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.test_utils.python_scripts import wait_for_resources


class TestResourceQueueCapacity(unittest.TestCase):
    def test_waits_when_older_jobs_fill_the_available_slots(self):
        current = SimpleNamespace(attributes={'created_at': '2026-10-09T12:00:00Z'})
        for waiting, expected_busy in [(0, False), (1, False), (2, True), (3, True)]:
            with self.subTest(waiting=waiting):
                project = MagicMock()
                project.pipelines.list.return_value = [
                    SimpleNamespace(
                        attributes={
                            'created_at': '2026-10-09T11:00:00Z',
                            'status': 'running',
                            'ref': f'refs/merge-requests/{index + 1}/head',
                        }
                    )
                    for index in range(waiting)
                ]
                project.mergerequests.get.return_value.target_branch = 'main'
                with (
                    patch.object(wait_for_resources, 'NUM_CONCURRENT_JOBS', 2),
                    patch.object(wait_for_resources, 'get_gitlab_handle') as handle,
                ):
                    handle.return_value.projects.get.return_value = project
                    self.assertEqual(wait_for_resources.ci_is_busy(current, 'main'), expected_busy)


if __name__ == '__main__':
    unittest.main()
