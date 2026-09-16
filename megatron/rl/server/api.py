# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Self, Type

from .. import TypeLookupable

from ..__init__ import Request
from ..agent.api import EvaluationRequest, RolloutRequest
from ..inference import InferenceInterface, LLMChatMessage


class Server(TypeLookupable):
    """Server interface class. Implements launch and kill control methods."""

    @classmethod
    async def launch(cls) -> Self:
        raise NotImplementedError

    async def suspend(self):
        pass

    async def resume(self):
        pass

    async def kill(self):
        raise NotImplementedError


class InferenceServer(Server, InferenceInterface):
    """Base Inference Server."""

    ...


class EnvironmentServer(Server):
    """Base Environment Server."""

    ...


class RemotePromptRequest(BaseModel):
    """Trainer -> env server: sample the prompt one group of rollouts will share."""

    validation: bool = False


class RemotePrompt(BaseModel):
    """Env server -> trainer: a sampled prompt and the golden data that scores it."""

    prompt: str | list[LLMChatMessage]
    golden: Any = None


### Intentionally force `inference_interface` to be `None` in these subclasses.


class RemoteEpisodeRequest(Request):
    """Trainer -> env server: run one episode on a sampled prompt and score it."""

    prompt: str | list[LLMChatMessage]
    golden: Any = None
    validation: bool = False
    inference_interface: None = None


class RemoteRolloutRequest(RolloutRequest):
    inference_interface: None = None


class RemoteEvaluationRequest(EvaluationRequest):
    inference_interface: None = None
