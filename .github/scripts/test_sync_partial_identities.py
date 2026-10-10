# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest
from unittest import mock

import sync_team_usergroups as sync


class TestSync(unittest.TestCase):
    def test_incomplete_identity_resolution_keeps_current_members(self):
        slack = mock.Mock()
        with (
            mock.patch.object(sync, 'get_org', return_value='NVIDIA'),
            mock.patch.object(sync, 'get_slack_client', return_value=slack),
            mock.patch.object(sync, 'get_team_members', return_value={'alice', 'bob'}),
            mock.patch.object(sync, 'get_user_email', side_effect=lambda u: u + '@example.com'),
            mock.patch.object(sync, 'get_slack_user_id', side_effect=['U1', None]),
            mock.patch.object(sync, 'get_slack_usergroup_id', return_value=('S1', ['U1', 'U2'])),
        ):
            self.assertFalse(sync.sync_team_to_usergroup('team', 'mcore-team'))
            slack.usergroups_users_update.assert_not_called()
            slack.usergroups_create.assert_not_called()


if __name__ == '__main__':
    unittest.main()
