# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from pydantic import BaseModel, Field
from typing_extensions import Self, Type

from .. import TypeLookupable
from ..agent.api import EvaluationRequest, RolloutRequest
from ..inference import InferenceInterface


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


### Intentionally force `inference_interface` to be `None` in these subclasses.

class RemoteRolloutRequest(RolloutRequest):
    inference_interface: None = None


class RemoteEvaluationRequest(EvaluationRequest):
    inference_interface: None = None
