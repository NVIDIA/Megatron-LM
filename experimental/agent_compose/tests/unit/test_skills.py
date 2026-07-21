# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Structural checks for the initial Agent Compose skill registry."""

from __future__ import annotations

import re
from pathlib import Path

AGENT_COMPOSE_ROOT = Path(__file__).resolve().parents[2]
SKILL_ROOT = AGENT_COMPOSE_ROOT / "skills"
EXPECTED = {
    "basic.constitution": "basic/constitution.md",
    "basic.lint_skill": "basic/lint-skill.md",
    "primitive.contract": "primitive/contract.md",
    "model.compose": "model/compose.md",
    "runtime.validate": "runtime/validate.md",
}
SCHEMA_NAME = re.compile(r'schema\s*=\s*Skill\(\s*"([^"]+)"', re.DOTALL)
LIST_FIELD = re.compile(r"\b(imports|calls)\s*=\s*\[([^]]*)\]", re.DOTALL)
QUOTED = re.compile(r'"([^"]+)"')


def test_skill_registry_is_complete_and_resolved() -> None:
    for expected_name, relative_path in EXPECTED.items():
        path = SKILL_ROOT / relative_path
        text = path.read_text(encoding="utf-8")
        assert text.count("MLITE_SKILL_SCHEMA_BEGIN") == 1
        assert text.count("MLITE_SKILL_SCHEMA_END") == 1
        match = SCHEMA_NAME.search(text)
        assert match is not None
        assert match.group(1) == expected_name
        assert f"def {expected_name.rsplit('.', 1)[-1]}(" in text

        fields = {
            name: QUOTED.findall(values) for name, values in LIST_FIELD.findall(text)
        }
        for dependency in fields.get("imports", []) + fields.get("calls", []):
            assert dependency in EXPECTED
