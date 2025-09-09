# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import importlib
from typing import Type

from pydantic import BaseModel, ConfigDict


def import_class(class_path: str) -> Type:
    """Import a class from a string path.

    Args:
        class_path: String path to the class (e.g. 'examples.rl.environments.countdown.countdown_agent.CountdownAgent')

    Returns:
        The class object
    """
    module_path, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class GenericGenerationArgs(BaseModel):
    """Generic generation arguments."""

    model_config = ConfigDict(frozen=True)
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    max_tokens: int | None = None

    def add(self, generation_args: 'GenericGenerationArgs') -> 'GenericGenerationArgs':
        return GenericGenerationArgs.model_validate(
            {**self.model_dump(), **generation_args.model_dump(exclude_unset=True)}
        )


class Request(BaseModel):
    """Generation Request."""

    generation_args: GenericGenerationArgs = GenericGenerationArgs()
