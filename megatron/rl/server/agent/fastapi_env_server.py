# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import itertools
import socket
import uuid
from typing import Any, AsyncGenerator, AsyncIterator, Awaitable, Callable, Iterator

import httpx
import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, PrivateAttr
from typing_extensions import Self
from uvicorn import Config, Server
from uvicorn.config import LOGGING_CONFIG

LOGGING_CONFIG['root'] = {"handlers": ["default"], "level": "INFO"}

from ...agent.api import (
    Agent,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    GroupRolloutParams,
    Rollout,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from ...agent.registry import get_agent_class
from ...inference import (
    InferenceInterface,
    InferenceRequest,
    InferenceResponse,
    ReturnsRaw,
    ReturnsTokens,
)
from ..api import (
    EnvironmentServer,
    ErrorMessage,
    InferenceRequestMessage,
    RemoteEvaluationRequest,
    RemoteGroupedRolloutRequest,
    RemoteRolloutRequest,
    ResultMessage,
    parse_job_message,
)


class Job(ReturnsRaw, ReturnsTokens):
    """Server half of the job.

    Generation requests open a stream to the trainer that is answered by request id;
    the stream ends when the job has a result or error."""

    job_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    _lines: asyncio.Queue[str | None] = PrivateAttr(default_factory=asyncio.Queue)
    _pending: dict[int, asyncio.Future[InferenceResponse]] = PrivateAttr(default_factory=dict)
    _request_ids: Iterator[int] = PrivateAttr(default_factory=itertools.count)
    _closed: bool = PrivateAttr(False)

    @classmethod
    def start(
        cls, jobs: dict[str, "Job"], work: Callable[[InferenceInterface], Awaitable[Any]]
    ) -> StreamingResponse:
        """Stream a new job; `work(job)` runs with the job as the agent's inference interface."""
        job = cls()
        return StreamingResponse(job._stream(jobs, work), media_type="application/x-ndjson")

    async def _stream(
        self, jobs: dict[str, "Job"], work: Callable[[InferenceInterface], Awaitable[Any]]
    ) -> AsyncIterator[str]:
        jobs[self.job_id] = self
        task = asyncio.create_task(self._run(work))
        try:
            while (line := await self._lines.get()) is not None:
                yield line
        finally:
            task.cancel()
            self._closed = True
            for future in list(self._pending.values()):
                future.cancel()
            del jobs[self.job_id]

    async def _run(self, work: Callable[[InferenceInterface], Awaitable[Any]]):
        try:
            self._emit(ResultMessage(result=await work(self)))
        except Exception as exc:
            self._emit(ErrorMessage(detail=repr(exc)))
        finally:
            self._lines.put_nowait(None)

    def _emit(self, message: BaseModel):
        self._lines.put_nowait(message.model_dump_json() + "\n")

    async def base_generate(self, request: InferenceRequest) -> InferenceResponse:
        if self._closed:
            raise RuntimeError("Job is closed")
        request_id = next(self._request_ids)
        future: asyncio.Future[InferenceResponse] = asyncio.get_running_loop().create_future()
        self._pending[request_id] = future
        try:
            self._emit(
                InferenceRequestMessage(job_id=self.job_id, request_id=request_id, request=request)
            )
            return await future
        finally:
            del self._pending[request_id]

    def resolve(self, request_id: int, response: InferenceResponse) -> bool:
        """Deliver the trainer's answer; False when the request id is invalid."""
        future = self._pending.get(request_id)
        if future is None or future.done():
            return False
        future.set_result(response)
        return True


def _bind(wire: BaseModel, request_cls: type, interface: InferenceInterface):
    """Rebuild the agent request from its wire form, with the job as its inference interface."""
    return request_cls.model_validate({**wire.model_dump(), 'inference_interface': interface})


async def _grouped_rollouts(
    env: GroupedRolloutGenerator, request: GroupedRolloutRequest
) -> list[list[Rollout]]:
    """num_groups groups of rollouts_per_group episodes; same-reward groups redrawn if filtered."""

    async def one_group() -> list[Rollout]:
        while True:
            params = await env.prepare_group_rollout(request)
            episodes = await asyncio.gather(
                *[params.run_episode() for _ in range(request.rollouts_per_group)]
            )
            group = [await params.build_rollout(episode) for episode in episodes]
            if (
                not request.filter_groups_with_same_reward
                or np.std([r.reward for r in group]) > 1e-6
            ):
                return group

    return list(await asyncio.gather(*[one_group() for _ in range(request.num_groups)]))


async def _answer(
    client: httpx.AsyncClient,
    base_url: str,
    message: InferenceRequestMessage,
    interface: InferenceInterface,
):
    """Answer one generation request with the trainer's inference interface."""
    response = await interface.agenerate(message.request)
    posted = await client.post(
        f"{base_url}/jobs/{message.job_id}/responses/{message.request_id}",
        content=response.model_dump_json(),
        headers={"content-type": "application/json"},
    )
    if posted.status_code != 404:
        posted.raise_for_status()


@EnvironmentServer.register_subclass
class FastAPIEnvServer(EnvironmentServer):
    server_type: str = Field('FastAPIEnvServer', frozen=True, Literal=True)
    env_server_host_port: str
    _server_task: asyncio.Task = PrivateAttr(None)

    @classmethod
    def build_app(cls, env_cls: type[Agent], cls_args: dict) -> FastAPI:
        """Env server app. Each job streams generation requests, then a result or error."""
        app = FastAPI()
        jobs: dict[str, Job] = {}
        app.state.jobs = jobs

        @app.post("/jobs/{job_id}/responses/{request_id}", status_code=204)
        async def answer(job_id: str, request_id: int, response: InferenceResponse) -> Response:
            job = jobs.get(job_id)
            if job is None or not job.resolve(request_id, response):
                raise HTTPException(status_code=404, detail="Unknown job or request id")
            return Response(status_code=204)

        if issubclass(env_cls, GroupedRolloutGenerator):

            @app.post("/grouped_rollouts/")
            async def grouped_rollouts(request: RemoteGroupedRolloutRequest) -> StreamingResponse:
                env = env_cls(**cls_args)
                return Job.start(
                    jobs,
                    lambda job: _grouped_rollouts(env, _bind(request, GroupedRolloutRequest, job)),
                )

        if issubclass(env_cls, ContrastiveRolloutGenerator):

            @app.post("/contrastive_rollouts/")
            async def contrastive_rollouts(request: RemoteRolloutRequest) -> StreamingResponse:
                env = env_cls(**cls_args)
                return Job.start(
                    jobs,
                    lambda job: env.get_contrastive_rollouts(_bind(request, RolloutRequest, job)),
                )

        if issubclass(env_cls, RolloutGenerator):

            @app.post("/rollouts/")
            async def rollouts(request: RemoteRolloutRequest) -> StreamingResponse:
                env = env_cls(**cls_args)
                return Job.start(
                    jobs, lambda job: env.get_reward_rollouts(_bind(request, RolloutRequest, job))
                )

        if issubclass(env_cls, EvaluationAgent):

            @app.post("/evaluation/")
            async def run_evaluation(request: RemoteEvaluationRequest) -> StreamingResponse:
                env = env_cls(**cls_args)
                return Job.start(
                    jobs, lambda job: env.run_evaluation(_bind(request, EvaluationRequest, job))
                )

        return app

    @classmethod
    async def launch(cls, env_cls: type[Agent], cls_args: dict, port: int, **kwargs) -> Self:
        app = cls.build_app(env_cls, cls_args)

        loop = asyncio.get_event_loop()
        config = Config(app=app, loop=loop, host='0.0.0.0', port=port)
        server = Server(config)
        server_task = loop.create_task(server.serve())

        ip = socket.gethostbyname(socket.gethostname())

        launched_server = cls(env_server_host_port=f"{ip}:{config.port}", **kwargs)
        launched_server._server_task = server_task

        return launched_server

    def kill(self):
        return self._server_task.cancel()

    async def _run_job(
        self, path: str, request: RolloutRequest | GroupedRolloutRequest | EvaluationRequest
    ) -> Any:
        """Run one remote job, answering its generation requests with the request's interface."""
        interface = request.inference_interface
        assert isinstance(interface, ReturnsRaw) and isinstance(
            interface, ReturnsTokens
        ), "FastAPIEnvServer forwards raw text and token ids; the interface must return both"
        payload = request.model_dump(mode='json', exclude={'inference_interface'})
        base_url = f"http://{self.env_server_host_port}"
        outcome: ResultMessage | ErrorMessage | None = None
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", f"{base_url}{path}", json=payload) as response:
                if response.status_code != 200:
                    detail = (await response.aread()).decode(errors='replace')
                    raise RuntimeError(
                        f"Env server {path} returned HTTP {response.status_code}: {detail}"
                    )
                # A failing answer cancels this reader;
                # leaving the stream closes the connection, which ends the job on the env server.
                async with asyncio.TaskGroup() as answers:
                    in_flight: set[asyncio.Task] = set()
                    async for line in response.aiter_lines():
                        if not line.strip():
                            continue
                        message = parse_job_message(line)
                        if not isinstance(message, InferenceRequestMessage):
                            outcome = message
                            break
                        task = answers.create_task(_answer(client, base_url, message, interface))
                        in_flight.add(task)
                        task.add_done_callback(in_flight.discard)
                    # The job is over; an answer still in flight is for a generation nobody awaits.
                    for task in list(in_flight):
                        task.cancel()
        if outcome is None:
            raise RuntimeError(f"Env server {path} closed the job stream without a result")
        if isinstance(outcome, ErrorMessage):
            raise RuntimeError(f"Env server {path} failed: {outcome.detail}")
        return outcome.result

    async def get_contrastive_rollouts(self, request: RolloutRequest) -> list[ContrastiveRollout]:
        result = await self._run_job("/contrastive_rollouts/", request)
        return [ContrastiveRollout.model_validate(r) for r in result]

    async def prepare_group_rollout(self, request: GroupedRolloutRequest) -> GroupRolloutParams:
        raise NotImplementedError(
            "FastAPIEnvServer overrides get_grouped_rollouts; prepare_group_rollout is not used."
        )

    async def get_rollout_response(self, request, inference_request):
        raise NotImplementedError(
            "FastAPIEnvServer overrides get_grouped_rollouts/get_reward_rollouts/run_evaluation; "
            "get_rollout_response is not used."
        )

    async def get_grouped_rollouts(
        self, request: GroupedRolloutRequest
    ) -> AsyncGenerator[list[TokenRollout], None]:
        assert (
            request.submission_granularity != "R"
        ), "FastAPIEnvServer does not support rollout submission granularity"
        for group in await self._run_job("/grouped_rollouts/", request):
            yield [TokenRollout.model_validate(r) for r in group]

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        result = await self._run_job("/rollouts/", request)
        return [TokenRollout.model_validate(r) for r in result]

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        result = await self._run_job("/evaluation/", request)
        return EvaluationResponse.validate_registered(result)


def run(agent_cls: type[Agent], cls_args: dict, port: int):
    loop = asyncio.new_event_loop()

    async def run_server():
        server: FastAPIEnvServer = await FastAPIEnvServer.launch(
            env_cls=agent_cls, cls_args=cls_args, port=port
        )
        print(server.model_dump(exclude={'_server_task'}))
        await server._server_task

    loop.run_until_complete(run_server())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with open(args.env_config, 'r') as f:
        config = yaml.safe_load(f)[0]
    agent_cls = get_agent_class(config['agent_type'])
    cls_args = config['agent_args']
    run(agent_cls, cls_args, port=args.port)
