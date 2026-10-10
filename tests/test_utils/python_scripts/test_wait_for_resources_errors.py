# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from unittest.mock import MagicMock, patch

import requests
from click.testing import CliRunner

from tests.test_utils.python_scripts import wait_for_resources


class TestResourceQueueErrors(unittest.TestCase):
    def test_an_unexpected_queue_error_does_not_report_resources_as_available(self):
        handle = MagicMock()
        with (
            patch.object(wait_for_resources, 'get_gitlab_handle', return_value=handle),
            patch.object(
                wait_for_resources, 'ci_is_busy', side_effect=ValueError('cannot read queue')
            ),
        ):
            result = CliRunner().invoke(
                wait_for_resources.main, ['--pipeline-id', '1', '--target-branch', 'main']
            )
        self.assertEqual(result.exit_code, 1)
        self.assertIn('cannot read queue', result.output)

    def test_a_temporary_network_error_is_still_retried(self):
        handle = MagicMock()
        with (
            patch.object(wait_for_resources, 'get_gitlab_handle', return_value=handle),
            patch.object(
                wait_for_resources,
                'ci_is_busy',
                side_effect=[requests.ConnectionError('temporary'), False],
            ) as busy,
            patch.object(wait_for_resources.time, 'sleep') as sleep,
        ):
            result = CliRunner().invoke(
                wait_for_resources.main, ['--pipeline-id', '1', '--target-branch', 'main']
            )
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(busy.call_count, 2)
        sleep.assert_called_once_with(15)


if __name__ == '__main__':
    unittest.main()
