# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""A Ray-serialized cap must share the worker constructor's agent registry."""
import asyncio
import sys
from pathlib import Path
from types import ModuleType

import cloudpickle
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'examples/verl'))
from verl_mlite import compat

pytestmark = pytest.mark.optional


def test_serialized_cap_resolves_worker_registry(monkeypatch):
    name = 'verl.experimental.agent_loop.agent_loop'
    module = ModuleType(name)
    exec(
        "registry = {}\n"
        "class AgentLoopWorker:\n"
        "    async def _run_agent_loop(self, params, trajectory):\n"
        "        return registry[trajectory['agent']], params\n",
        module.__dict__,
    )
    monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setenv('VERL_MLITE_VAL_MAX_TOKENS', '7')
    assert compat._patch_verl_val_max_tokens()
    assert compat._patch_verl_val_max_tokens()  # idempotent
    # Export before constructor registration, as Ray does with the worker class.
    method = cloudpickle.loads(cloudpickle.dumps(module.AgentLoopWorker._run_agent_loop))
    module.registry['seeded'] = 'registered-in-worker'
    original = {'max_tokens': 16, 'seed': 123}
    for validate, expected in ((False, 16), (True, 7)):
        result, params = asyncio.run(
            method(None, original, dict(agent='seeded', validate=validate))
        )
        assert result == 'registered-in-worker'
        assert params == {'max_tokens': expected, 'seed': 123}
    assert original['max_tokens'] == 16
