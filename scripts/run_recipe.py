# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Launch a native MLM recipe under torch.distributed.run.

Select a zero-argument Python config factory with --recipe module:function.
Trailing dotted KEY=VALUE overrides update existing config fields. Dataset
selection belongs to the recipe's cfg.data; this runner does not synchronize
sequence lengths or finalize configs. Python recipes must be trusted code.
"""

import argparse
import importlib
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from pprint import pformat

logger = logging.getLogger(__name__)


def _load_callable(path: str) -> Callable:
    """Load trusted Python selected explicitly on the command line, not a YAML target."""
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected module:function, got {path!r}")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError(f"{path!r} is not callable")
    return function


def main(argv: list[str] | None = None) -> None:
    """Load an MLM recipe, apply overrides, and validate or launch training."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--recipe", required=True, help="Config factory as module:function")
    parser.add_argument(
        "--forward-step", help="MLM forward step as module:function; defaults based on model config"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and display config without runtime setup"
    )
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # Direct script execution must also find the source-checkout package.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch

    from megatron.core._rank_utils import safe_get_rank
    from megatron.training.api import pretrain
    from megatron.training.config import PretrainConfigContainer
    from megatron.training.config.utils import apply_config_overrides
    from megatron.training.models.gpt import GPTModelConfig
    from megatron.training.models.hybrid import HybridModelConfig

    config = _load_callable(options.recipe)()
    if not isinstance(config, PretrainConfigContainer):
        raise TypeError("Recipe must return an MLM PretrainConfigContainer")
    apply_config_overrides(config, options.overrides)
    step = options.forward_step
    if step is None:
        # Temporary default: exact GPT/Hybrid configs use the text step; all others
        # use the VLM step. VLM configs inherit from HybridModelConfig, so isinstance
        # would select the wrong step. Expand this logic as recipes and steps are added.
        if type(config.model) in (GPTModelConfig, HybridModelConfig):
            step = "megatron.training.forward_steps.llm_step:forward_step"
        else:
            step = "megatron.training.forward_steps.vlm_step:forward_step"
    forward_step = _load_callable(step)
    if options.dry_run:
        config.validate()
        if safe_get_rank() == 0:
            logger.info("Recipe config (not finalized):\n%s", pformat(config.to_dict()))
        return
    pretrain(config, forward_step)
    # On failure, let torchrun report the exception and terminate peer ranks.
    # Collective teardown in a finally block can hang before the traceback is printed.
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
