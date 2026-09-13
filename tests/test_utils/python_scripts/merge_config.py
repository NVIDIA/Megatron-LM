"""
Merges base_config, runtime_config and model_config into one final config that the CI can launch.

Starting Dec 19th 2025 MCore CI supports a new format of defining tests. We are decoupling the test
config into a modular system of base_config, model_config and runtime_config. This allows us to
re-use and parametrize a given model easily with multiple runtime configs, like parallelism settings.

With this DRY principle, we simplify test maintenance and reduce the amount of code duplication.

This refactoring is fully compliant with the original CI system as we merge the three configs into one
final config that the CI can launch.

Precendence: Base config > Model config > Runtime config.

Usage:

python merge_config.py \
    --model_config model_config.yaml \
    --base_config base_config.yaml \
    --runtime_config runtime_config.yaml \
    --output_config output_config.yaml
"""

import logging

import click
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option("--model_config", type=str, help="Model config to merge")
@click.option("--base_config", type=str, help="Base config to merge")
@click.option("--runtime_config", type=str, help="Run time config to merge")
@click.option("--output_config", type=str, help="Output config to merge")
def main(model_config, base_config, runtime_config, output_config):

    with open(model_config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(base_config, "r") as f:
        base_config = yaml.safe_load(f)
    with open(runtime_config, "r") as f:
        runtime_config = yaml.safe_load(f)

    config = {}

    # Collect all top-level keys (ENV_VARS, MODEL_ARGS, etc.)
    all_keys = set(base_config.keys()) | set(model_config.keys()) | set(runtime_config.keys())

    for key in all_keys:
        base_val = base_config.get(key)
        model_val = model_config.get(key)
        runtime_val = runtime_config.get(key)

        # Get first non-None value to check type
        first_val = base_val or model_val or runtime_val

        if isinstance(first_val, dict):
            # Merge dicts
            config[key] = {}
            for val in [base_val, model_val, runtime_val]:
                if val:
                    config[key].update(val)
        elif isinstance(first_val, list):
            # Concatenate lists (deduplicate while preserving order)
            config[key] = []
            seen = set()
            for val in [base_val, model_val, runtime_val]:
                if val:
                    for item in val:
                        if item not in seen:
                            config[key].append(item)
                            seen.add(item)
        else:
            # Scalar value (string, int, bool, etc.) - use last defined
            if runtime_val is not None:
                config[key] = runtime_val
            elif model_val is not None:
                config[key] = model_val
            else:
                config[key] = base_val

    with open(output_config, "w") as f:
        yaml.dump(config, f)

    logger.info(f"Config merged and saved to {output_config}")


if __name__ == "__main__":
    main()
