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
from dataclasses import fields, is_dataclass
from pathlib import Path
from pprint import pformat
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from megatron.training.config import PretrainConfigContainer

logger = logging.getLogger(__name__)


def _load_callable(path: str) -> Callable:
    module, separator, name = path.partition(":")
    if not separator or not module or not name:
        raise ValueError(f"Expected module:function, got {path!r}")
    function = getattr(importlib.import_module(module), name)
    if not callable(function):
        raise TypeError(f"{path!r} is not callable")
    return function


def _apply_overrides(config: "PretrainConfigContainer", overrides: list[str]) -> None:
    """Update existing public fields without reconstructing configs or losing callbacks."""
    from omegaconf import OmegaConf

    for override in overrides:
        path, separator, literal = override.partition("=")
        if not separator:
            raise ValueError(f"Expected KEY=VALUE, got {override!r}")
        names = path.split(".")
        owner = config
        for index, name in enumerate(names):
            if (
                not name
                or name.startswith("_")
                or not is_dataclass(owner)
                or isinstance(owner, type)
                or name not in {field.name for field in fields(owner) if field.init}
            ):
                raise ValueError(f"Not an editable config field: {path!r}")
            if index < len(names) - 1:
                owner = getattr(owner, name)
        current = getattr(owner, names[-1])
        if is_dataclass(current):
            raise ValueError(f"Override a field within {path!r}, not the entire config section")
        if callable(current):
            raise ValueError(f"Define the Python callable or class {path!r} in the recipe")
        value = OmegaConf.to_container(OmegaConf.from_dotlist([f"value={literal}"]), resolve=True)[
            "value"
        ]
        setattr(owner, names[-1], value)


def main(argv: list[str] | None = None) -> None:
    """Load an MLM recipe, apply overrides, and validate or launch training."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--recipe", required=True, help="Config factory as module:function")
    parser.add_argument(
        "--forward-step", help="MLM forward step as module:function; defaults to GPT/Hybrid's step"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate and display config without runtime setup"
    )
    parser.add_argument("overrides", nargs="*", metavar="KEY=VALUE")
    options = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    # Direct script execution must also find repository-root forward steps and recipes.
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)

    import torch

    from megatron.bridge.models.nemotron_3.super_vl_config import Nemotron35SuperVLModelConfig
    from megatron.bridge.models.qwen_vl.qwen35_vl_config import Qwen35VLModelConfig
    from megatron.core._rank_utils import safe_get_rank
    from megatron.training.api import pretrain
    from megatron.training.config import PretrainConfigContainer
    from megatron.training.models.gpt import GPTModelConfig
    from megatron.training.models.hybrid import HybridModelConfig

    config = _load_callable(options.recipe)()
    if not isinstance(config, PretrainConfigContainer):
        raise TypeError("Recipe must return an MLM PretrainConfigContainer")
    _apply_overrides(config, options.overrides)
    is_vlm = isinstance(config.model, (Nemotron35SuperVLModelConfig, Qwen35VLModelConfig))
    if is_vlm and options.forward_step is None:
        raise ValueError("VLM recipes require an explicit --forward-step")
    step = options.forward_step
    if step is None:
        if isinstance(config.model, HybridModelConfig):
            step = "pretrain_hybrid:forward_step"
        elif isinstance(config.model, GPTModelConfig):
            step = "pretrain_gpt:forward_step"
        else:
            raise TypeError("Default forward steps require a GPTModelConfig or HybridModelConfig")
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
