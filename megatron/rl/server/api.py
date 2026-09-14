# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter
from typing_extensions import Self

from .. import TypeLookupable
from ..agent.api import EvaluationRequest, GroupedRolloutRequest, RolloutRequest
from ..inference import InferenceInterface, InferenceRequest


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


class RemoteRolloutRequest(RolloutRequest):
    inference_interface: None = None


class RemoteGroupedRolloutRequest(GroupedRolloutRequest):
    inference_interface: None = None


class RemoteEvaluationRequest(EvaluationRequest):
    inference_interface: None = None


class InferenceRequestMessage(BaseModel):
    """Env server -> trainer: one generation the remote agent is waiting on.

    Answered with an InferenceResponse body at POST /jobs/{job_id}/responses/{request_id}.
    """

    kind: Literal['inference'] = 'inference'
    job_id: str
    request_id: int
    request: InferenceRequest


class ResultMessage(BaseModel):
    """Env server -> trainer: the job's result."""

    kind: Literal['result'] = 'result'
    result: Any


class ErrorMessage(BaseModel):
    """Env server -> trainer: the job failed."""

    kind: Literal['error'] = 'error'
    detail: str


JobMessage = InferenceRequestMessage | ResultMessage | ErrorMessage
_job_message_adapter = TypeAdapter(Annotated[JobMessage, Field(discriminator='kind')])


def parse_job_message(line: str) -> JobMessage:
    """Parse one NDJSON line of a job's response stream."""
    return _job_message_adapter.validate_json(line)
