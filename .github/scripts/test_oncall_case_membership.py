# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest
from unittest import mock

import oncall_manager as oncall


class TestMembership(unittest.TestCase):
    def test_login_case_changes_remain_members(self):
        oncall.validate_schedule_users_in_rotation_team([{'user': 'ALICE'}], ['Alice'])

    def test_absent_user_still_rejected(self):
        with self.assertRaises(SystemExit):
            oncall.validate_schedule_users_in_rotation_team([{'user': 'Bob'}], ['Alice'])


if __name__ == '__main__':
    unittest.main()
