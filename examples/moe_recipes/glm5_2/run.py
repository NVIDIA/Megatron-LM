# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render or execute a GLM-5.2 example inside the qualified runtime environment."""

import argparse
import os
import shlex
import sys
from pathlib import Path
from string import Template

import yaml


def _arguments(values: dict) -> list[str]:
    result = []
    for name, value in values.items():
        if value is None or value is False:
            continue
        result.append('--' + name.replace('_', '-'))
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def main() -> None:
    """Translate ENV_VARS and ARGS without importing the GPU training stack."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recipe', type=Path)
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the environment and command; do not run training.',
    )
    options = parser.parse_args()
    with options.recipe.open() as stream:
        recipe = yaml.safe_load(stream)

    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, MEGATRON_ROOT=str(root))
    overrides = {}
    try:
        for name, value in recipe['ENV_VARS'].items():
            overrides[name] = Template(str(value)).substitute(environment)
    except KeyError as error:
        parser.error(f'Missing environment variable {error.args[0]}; see the recipe README.')
    environment.update(overrides)
    command = [sys.executable, str(root / 'pretrain_gpt.py'), *_arguments(recipe['ARGS'])]

    if options.dry_run:
        exports = [f'{name}={value}' for name, value in overrides.items()]
        print(f'cd {shlex.quote(str(root))} && {shlex.join(["env", *exports, *command])}')
        return

    expected_world_size = recipe['DEPENDENCIES']['hardware']['num_gpus']
    if environment.get('WORLD_SIZE') != str(expected_world_size):
        parser.error(
            f'This recipe requires {expected_world_size} distributed ranks; launch it with torchrun.'
        )
    os.chdir(root)
    os.execvpe(sys.executable, command, environment)


if __name__ == '__main__':
    main()
