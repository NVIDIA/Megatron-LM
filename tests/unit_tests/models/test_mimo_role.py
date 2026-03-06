# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""Tests for MIMO role data classes."""

import pytest

from megatron.core.models.mimo.config.role import ModuleStageInfo, RankRole


class TestMimoRole:
    """Tests for ModuleStageInfo and RankRole dataclasses."""

    def test_module_stage_info(self):
        """Test ModuleStageInfo creation and attributes."""
        first = ModuleStageInfo(is_first_stage=True, is_last_stage=False)
        last = ModuleStageInfo(is_first_stage=False, is_last_stage=True)
        only = ModuleStageInfo(is_first_stage=True, is_last_stage=True)

        assert (first.is_first_stage, first.is_last_stage) == (True, False)
        assert (last.is_first_stage, last.is_last_stage) == (False, True)
        assert (only.is_first_stage, only.is_last_stage) == (True, True)

    def test_rank_role(self):
        """Test RankRole properties and methods."""
        # Encoder-only role
        encoder_role = RankRole(
            modules={"vision": ModuleStageInfo(True, False)},
            language_module_name="language",
        )
        assert encoder_role.has_modality_modules is True
        assert encoder_role.has_language_module is False
        assert encoder_role.modality_module_names == ["vision"]

        # Language-only role
        lang_role = RankRole(
            modules={"language": ModuleStageInfo(True, True)},
            language_module_name="language",
        )
        assert lang_role.has_modality_modules is False
        assert lang_role.has_language_module is True

        # Mixed role with stage checks
        mixed = RankRole(
            modules={
                "vision": ModuleStageInfo(is_first_stage=True, is_last_stage=False),
                "language": ModuleStageInfo(is_first_stage=False, is_last_stage=True),
            },
            language_module_name="language",
        )
        assert mixed.is_first_stage("vision") is True
        assert mixed.is_last_stage("vision") is False
        assert mixed.is_first_stage("language") is False
        assert mixed.is_last_stage("language") is True
        assert mixed.is_first_stage("nonexistent") is False
