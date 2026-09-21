# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from importlib import import_module
from typing import Type

# Allowlist mapping stable config names to their fully-qualified import targets
# in "module.path:ClassName" form. Targets are imported lazily in
# get_agent_class() so that importing this module does not pull in optional
# agent dependencies (e.g. math_verify, nemogym2mrl).
AGENT_REGISTRY: dict[str, str] = {
    "RemoteAgent": "megatron.rl.agent.remote_agent:RemoteAgent",
    "CountdownAgent": "examples.rl.environments.countdown.countdown_agent:CountdownAgent",
    "OpenMathInstructAgent": "examples.rl.environments.math.openmath_agent:OpenMathInstructAgent",
    "BigMathAgent": "examples.rl.environments.math.bigmath_agent:BigMathAgent",
    "DAPOAgent": "examples.rl.environments.math.dapo_agent:DAPOAgent",
    "GSM8KAgent": "examples.rl.environments.math.gsm8k_agent:GSM8KAgent",
    "AIMEAgent": "examples.rl.environments.math.aime_agent:AIMEAgent",
    "NemoGymAgent": "nemogym2mrl.nemo_gym_agent:NemoGymAgent",
    "AceMathAgent": "environments.acemath_agent:AceMathAgent",
}


def get_agent_class(agent_name: str) -> Type:
    """Resolve a config agent_type string to a registered agent class.

    Only explicitly registered agent names are allowed, and each maps to a fixed
    import target defined in this module. This prevents arbitrary code execution
    from untrusted environment configuration files. The target module is imported
    lazily so importing this module does not require optional agent dependencies.
    """
    try:
        import_path = AGENT_REGISTRY[agent_name]
    except KeyError as exc:
        known = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(
            f"Unknown agent_type {agent_name!r}. "
            f"Registered agent types: {known or '(none)'}"
        ) from exc
    module_path, class_name = import_path.split(":")
    module = import_module(module_path)
    return getattr(module, class_name)
