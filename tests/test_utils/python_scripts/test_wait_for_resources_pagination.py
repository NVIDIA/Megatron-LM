# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tests.test_utils.python_scripts import wait_for_resources


class TestResourceQueuePagination(unittest.TestCase):
    def test_counts_busy_pipelines_beyond_the_first_hundred_results(self):
        current = SimpleNamespace(attributes={'created_at': '2026-10-09T12:00:00Z'})
        first_page = [
            SimpleNamespace(attributes={'created_at': '2026-10-09T13:00:00Z'}) for _ in range(100)
        ]
        older_busy = [
            SimpleNamespace(
                attributes={
                    'created_at': '2026-10-09T11:00:00Z',
                    'status': 'running',
                    'ref': f'refs/merge-requests/{number}/head',
                }
            )
            for number in (101, 102)
        ]
        project = MagicMock()
        project.pipelines.list.side_effect = lambda **kwargs: (
            first_page + older_busy if kwargs.get('get_all') else first_page
        )
        project.mergerequests.get.return_value.target_branch = 'main'
        with (
            patch.object(wait_for_resources, 'NUM_CONCURRENT_JOBS', 1),
            patch.object(wait_for_resources, 'get_gitlab_handle') as handle,
        ):
            handle.return_value.projects.get.return_value = project
            self.assertTrue(wait_for_resources.ci_is_busy(current, 'main'))
        self.assertTrue(project.pipelines.list.call_args.kwargs['get_all'])


if __name__ == '__main__':
    unittest.main()
