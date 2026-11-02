# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Render or execute a GLM-5.2 example inside the qualified runtime environment."""

import argparse
import os
import shlex
import sys
from pathlib import Path
from string import Template

import yaml

# Run by --prewarm-deepgemm in the recipe environment on one GPU: prints where the runtime
# artifacts are imported from, then compiles DeepGEMM's indexer kernels into DG_JIT_CACHE_DIR (or
# loads them), so the training ranks only load cached kernels.
_PREWARM_DEEPGEMM = '''
import importlib, os, sys
import torch
for name in ("cudnn", "cutlass", "mok", "emerging_optimizers", "ds_tune", "deep_gemm"):
    module = importlib.import_module(name)
    print(f"{name} {getattr(module, '__version__', '-')} {os.path.dirname(module.__file__)}")
from megatron.core.transformer.experimental_attention_variant import dsa_deepgemm_scorer
dsa_deepgemm_scorer.warmup(num_heads=int(sys.argv[1]))
torch.cuda.synchronize()
print(f"DeepGEMM JIT cache ready: {os.environ['DG_JIT_CACHE_DIR']} ({sys.argv[1]} indexer heads)")
'''


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
        '--variant', help="Apply one entry of the recipe's VARIANTS (e.g. frozen_indexer) to ARGS."
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--dry-run',
        action='store_true',
        help='Print the environment and command; do not run training.',
    )
    mode.add_argument(
        '--prewarm-deepgemm',
        action='store_true',
        help="Fill DG_JIT_CACHE_DIR with DeepGEMM's indexer kernels on one GPU; do not train.",
    )
    options = parser.parse_args()
    with options.recipe.open() as stream:
        recipe = yaml.safe_load(stream)

    arguments = dict(recipe['ARGS'])
    if options.variant is not None:
        variants = recipe.get('VARIANTS', {})
        if options.variant not in variants:
            parser.error(
                f'Unknown variant {options.variant!r}; the recipe defines {sorted(variants)}.'
            )
        unknown = sorted(set(variants[options.variant]) - set(arguments))
        if unknown:
            parser.error(f'Variant {options.variant!r} sets arguments absent from ARGS: {unknown}.')
        arguments.update(variants[options.variant])

    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ, MEGATRON_ROOT=str(root))
    overrides = {}
    try:
        for name, value in recipe['ENV_VARS'].items():
            overrides[name] = Template(str(value)).substitute(environment)
    except KeyError as error:
        parser.error(f'Missing environment variable {error.args[0]}; see the recipe README.')
    environment.update(overrides)
    if options.prewarm_deepgemm:
        if arguments.get('dsa_indexer_scorer_backend') != 'deepgemm':
            parser.error('This recipe does not score the DSA indexer with DeepGEMM.')
        command = [sys.executable, '-c', _PREWARM_DEEPGEMM, str(arguments['dsa_indexer_n_heads'])]
    else:
        command = [sys.executable, str(root / 'pretrain_gpt.py'), *_arguments(arguments)]

    if options.dry_run:
        exports = [f'{name}={value}' for name, value in overrides.items()]
        print(f'cd {shlex.quote(str(root))} && {shlex.join(["env", *exports, *command])}')
        return

    if not options.prewarm_deepgemm:
        expected_world_size = recipe['DEPENDENCIES']['hardware']['num_gpus']
        if environment.get('WORLD_SIZE') != str(expected_world_size):
            parser.error(
                f'This recipe requires {expected_world_size} distributed ranks; '
                'launch it as the recipe README describes.'
            )
    os.chdir(root)
    os.execvpe(sys.executable, command, environment)


if __name__ == '__main__':
    main()
