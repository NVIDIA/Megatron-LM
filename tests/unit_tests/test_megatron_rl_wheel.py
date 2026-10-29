# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""pyproject's py-modules ship exactly megatron.rl's packing modules plus generation_api.

NeMo RL imports the packing modules from an installed Megatron-LM. The rest of megatron.rl needs
dependencies that Megatron-LM does not declare, so it must stay out of the wheel. The install CI
job imports the shipped packing modules (every py-module except generation_api) from a built
wheel; this test checks the list itself.
"""

import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _shipped_modules() -> set[str]:
    with open(REPO / "pyproject.toml", "rb") as handle:
        return set(tomllib.load(handle)["tool"]["setuptools"]["py-modules"])


def test_wheel_ships_exactly_the_packing_modules_and_generation_api():
    names = [path.stem for path in (REPO / "megatron" / "rl").glob("shared_prefix_*.py")]
    packing = {f"megatron.rl.{name}" for name in names + ["tree_layout"]}
    # Every shipped module except generation_api goes through the packing-module import test.
    assert _shipped_modules() == packing | {"megatron.rl.generation_api"}


def test_wheel_ships_no_rl_runtime():
    runtime = (
        "megatron.rl.rl_",
        "megatron.rl.agent",
        "megatron.rl.inference",
        "megatron.rl.server",
    )
    assert not any(name.startswith(runtime) for name in _shipped_modules())
