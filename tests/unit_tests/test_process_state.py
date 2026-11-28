# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Regression coverage for restoring process-global test state."""

import pytest
import torch

from tests.unit_tests.test_utilities import _restore_torch_settings, _snapshot_torch_settings


@pytest.mark.parametrize("freeze_flags", [False, True])
def test_restore_torch_settings_preserves_backend_flag_lock(freeze_flags):
    """Restore changed cuDNN values without changing the backend mutation policy."""
    original = _snapshot_torch_settings()
    with torch.backends.__allow_nonbracketed_mutation():
        try:
            torch.backends.cudnn.deterministic = not original["cudnn"][0]
            torch.backends.cudnn.benchmark = not original["cudnn"][1]
            torch.backends.cudnn.allow_tf32 = not original["cudnn"][2]
            if freeze_flags:
                torch.backends.disable_global_flags()

            _restore_torch_settings(original)

            assert _snapshot_torch_settings() == original
            assert torch.backends.flags_frozen() is freeze_flags
        finally:
            _restore_torch_settings(original)
