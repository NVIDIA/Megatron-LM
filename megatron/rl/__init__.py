# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import functools
import importlib
import os
import sys
import time
import traceback
from typing import Callable, Coroutine, Type

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, Type

def import_class(class_path: str) -> Type:
    """Import a class from a string path.

    Args:
        class_path: String path to the class (e.g. 'examples.rl.environments.countdown.countdown_agent.CountdownAgent' or '../environments.countdown.py:CountdownAgent')

    Returns:
        The class object
    """
    if '.py:' in class_path:
        # filepath.py:Classname branch.
        module_path, class_name = class_path.split(':')
        abs_path = os.path.abspath(module_path)
        spec = importlib.util.spec_from_file_location('acemath_agent', abs_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    else:
        module_path, class_name = class_path.rsplit('.', 1)
        module = importlib.import_module(module_path, package=__package__)
    return getattr(module, class_name)


class TypeLookupable(BaseModel, extra='allow'):
    """Supports 'unwrapping' of base class into subclasses."""

    type_name: str = Field('Null', frozen=True)

    def unwrap(self) -> Self:
        """Turn instance of base class into registered subclass."""
        return type(self).Library.type_names[self.type_name](**self.model_dump())

    @classmethod
    def register_subclass(cls, register_type: Type[Self]) -> Type[Self]:
        """Register subclass for unwrapping."""
        if 'Library' not in cls.__dict__:

            class Library:
                type_names = {}

            cls.Library = Library
        cls.Library.type_names[register_type.__fields__['type_name'].default] = register_type
        return register_type


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
