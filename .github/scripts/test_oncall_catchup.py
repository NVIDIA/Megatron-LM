# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
import unittest
from datetime import datetime, timezone
from unittest import mock

import oncall_manager as oncall


class TestCatchup(unittest.TestCase):
    def test_remove_all_expired_shifts_before_assigning_active_user(self):
        schedule = [
            {'user': 'Alice', 'date': '2026-09-16'},
            {'user': 'Bob', 'date': '2026-09-23'},
            {'user': 'Carol', 'date': '2026-09-30'},
            {'user': 'Dave', 'date': '2026-10-07'},
        ]
        with (
            mock.patch.object(oncall, 'load_schedule', return_value=schedule),
            mock.patch.object(
                oncall, 'get_rotation_order', return_value=['Alice', 'Bob', 'Carol', 'Dave']
            ),
            mock.patch.object(oncall, 'datetime', wraps=datetime) as clock,
            mock.patch.object(oncall, 'ensure_schedule_filled'),
            mock.patch.object(oncall, 'update_active_oncall_team') as update,
            mock.patch.object(oncall, 'save_schedule'),
        ):
            clock.now.return_value = datetime(2026, 10, 9, tzinfo=timezone.utc)
            oncall.rotate_schedule('NVIDIA')
            update.assert_called_once_with('NVIDIA', 'Dave')
            self.assertEqual(schedule, [{'user': 'Dave', 'date': '2026-10-07'}])


if __name__ == '__main__':
    unittest.main()
