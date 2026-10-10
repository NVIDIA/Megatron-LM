# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tests.test_utils.python_scripts import check_status_of_main


class TestMainPipelineTerminalStatus(unittest.TestCase):
    def test_terminal_statuses_do_not_keep_the_continuous_poll_running(self):
        for status in ('success', 'failed', 'canceled', 'skipped'):
            with self.subTest(status=status):
                pipeline = SimpleNamespace(id=1, attributes={'status': status})
                with patch.object(
                    check_status_of_main, 'most_recent_pipeline', return_value=pipeline
                ):
                    self.assertFalse(check_status_of_main.is_pending('main'))

    def test_active_statuses_still_wait(self):
        for status in ('created', 'waiting_for_resource', 'preparing', 'pending', 'running'):
            with self.subTest(status=status):
                pipeline = SimpleNamespace(id=1, attributes={'status': status})
                with patch.object(
                    check_status_of_main, 'most_recent_pipeline', return_value=pipeline
                ):
                    self.assertTrue(check_status_of_main.is_pending('main'))


if __name__ == '__main__':
    unittest.main()
