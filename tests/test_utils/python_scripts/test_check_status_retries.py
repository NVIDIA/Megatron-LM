# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

from tests.test_utils.python_scripts import check_status_of_main


class TestMainPipelineRetryFailure(unittest.TestCase):
    def test_preserves_the_api_failure_when_all_attempts_fail(self):
        handle = MagicMock()
        listing = handle.projects.get.return_value.pipelines.list
        listing.side_effect = OSError('network unavailable')
        with (
            patch.object(check_status_of_main, 'get_gitlab_handle', return_value=handle),
            patch.object(check_status_of_main.time, 'sleep') as sleep,
        ):
            with self.assertRaisesRegex(OSError, 'network unavailable'):
                check_status_of_main.most_recent_pipeline('main')
        self.assertEqual(listing.call_count, 3)
        self.assertEqual(sleep.call_args_list, [call(10), call(20)])

    def test_recovers_when_the_third_attempt_succeeds(self):
        handle = MagicMock()
        pipeline = SimpleNamespace(id=123)
        handle.projects.get.return_value.pipelines.list.side_effect = [
            OSError('first failure'),
            OSError('second failure'),
            [pipeline],
        ]
        with (
            patch.object(check_status_of_main, 'get_gitlab_handle', return_value=handle),
            patch.object(check_status_of_main.time, 'sleep') as sleep,
        ):
            self.assertIs(check_status_of_main.most_recent_pipeline('main'), pipeline)
        self.assertEqual(sleep.call_args_list, [call(10), call(20)])


if __name__ == '__main__':
    unittest.main()
