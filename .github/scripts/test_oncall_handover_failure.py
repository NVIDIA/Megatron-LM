# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest
from unittest import mock

import oncall_manager as oncall


class TestHandover(unittest.TestCase):
    def test_failed_add_keeps_existing_members_and_slack_unchanged(self):
        with (
            mock.patch.object(oncall, 'get_team_members', return_value={'old'}),
            mock.patch.object(oncall, 'get_headers', return_value={}),
            mock.patch.object(
                oncall.requests, 'put', return_value=mock.Mock(status_code=403, text='denied')
            ),
            mock.patch.object(oncall.requests, 'delete') as delete,
            mock.patch.object(oncall, 'update_slack_usergroup') as slack,
        ):
            with self.assertRaises(SystemExit):
                oncall.update_active_oncall_team('NVIDIA', 'new')
            delete.assert_not_called()
            slack.assert_not_called()

    def test_successful_add_still_retires_previous_oncall(self):
        with (
            mock.patch.object(oncall, 'get_team_members', return_value={'old'}),
            mock.patch.object(oncall, 'get_headers', return_value={}),
            mock.patch.object(oncall.requests, 'put', return_value=mock.Mock(status_code=200)),
            mock.patch.object(
                oncall.requests, 'delete', return_value=mock.Mock(status_code=204)
            ) as delete,
            mock.patch.object(oncall, 'update_slack_usergroup') as slack,
        ):
            oncall.update_active_oncall_team('NVIDIA', 'new')
            delete.assert_called_once()
            slack.assert_called_once_with('new', ['old'])


if __name__ == '__main__':
    unittest.main()
