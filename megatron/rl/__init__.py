# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import functools
import time
import traceback
from typing import Callable, Coroutine

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Self, Type


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
