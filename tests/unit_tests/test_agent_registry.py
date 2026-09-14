# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.agent.registry import AGENT_REGISTRY, get_agent_class

BUILTIN_AGENTS = {
    "RemoteAgent",
    "CountdownAgent",
    "OpenMathInstructAgent",
    "BigMathAgent",
    "DAPOAgent",
    "GSM8KAgent",
    "AIMEAgent",
}


def test_agent_registry_includes_builtin_agents():
    assert BUILTIN_AGENTS <= AGENT_REGISTRY.keys()


def test_agent_registry_targets_are_wellformed():
    # Every entry must be a "module.path:ClassName" import target so that
    # get_agent_class can resolve it without config-supplied import paths.
    for name, target in AGENT_REGISTRY.items():
        module_path, _, class_name = target.partition(":")
        assert module_path and class_name, f"malformed target for {name!r}: {target!r}"


def test_get_agent_class_lazily_imports_target(monkeypatch):
    # Uses a stdlib target to exercise the lazy import path without pulling in
    # optional agent dependencies.
    from collections import OrderedDict

    monkeypatch.setitem(AGENT_REGISTRY, "DummyAgent", "collections:OrderedDict")
    assert get_agent_class("DummyAgent") is OrderedDict


def test_get_agent_class_rejects_unknown_agent():
    with pytest.raises(ValueError, match="Unknown agent_type"):
        get_agent_class("examples.evil.MaliciousAgent")


def test_get_agent_class_rejects_module_paths():
    with pytest.raises(ValueError, match="Unknown agent_type"):
        get_agent_class("examples.rl.environments.countdown.countdown_agent.CountdownAgent")
