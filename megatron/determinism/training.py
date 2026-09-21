# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Explicit early opt-in for Megatron's CLI and experimental YAML entrypoints."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence

from ._policy import configure_determinism


def bootstrap_training_determinism(argv: Sequence[str] | None = None) -> dict | None:
    """Apply requested policy before a training script imports the GPU stack.

    Only the mode and YAML path are parsed here. The real argument/config parser
    still validates every model option later, without changing the original argv.
    As in parse_args, YAML replaces CLI settings. Model options must live in
    model_parallel or language_model, which the model config consumes.
    """
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--deterministic-mode", action="store_true")
    parser.add_argument("--yaml-cfg")
    args, _ = parser.parse_known_args(argv)
    options = {"deterministic_mode": args.deterministic_mode}
    if args.yaml_cfg is not None:
        import yaml

        with open(args.yaml_cfg, encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, Mapping):
            raise ValueError("Training YAML must contain a mapping")
        options = {}
        for name in ("model_parallel", "language_model"):
            section = config.get(name, {})
            if not isinstance(section, Mapping):
                raise ValueError(f"Training YAML {name} must contain a mapping")
            options.update(section)
        if config.get("deterministic_mode") and "deterministic_mode" not in options:
            raise ValueError("YAML deterministic_mode must be in model_parallel or language_model")
    return configure_determinism(options) if options.get("deterministic_mode", False) else None
