# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
from typing import AsyncGenerator

import httpx
import yaml
from fastapi import FastAPI
from pydantic import Field, PrivateAttr
from typing_extensions import Self
from uvicorn import Config, Server
from uvicorn.config import LOGGING_CONFIG

LOGGING_CONFIG['root'] = {"handlers": ["default"], "level": "INFO"}

from ... import import_class, inference
from ...agent.api import (
    Agent,
    ContrastiveRollout,
    ContrastiveRolloutGenerator,
    EvaluationAgent,
    EvaluationRequest,
    EvaluationResponse,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from ...server.api import (
    EnvironmentServer,
    InferenceServer,
    RemoteEvaluationRequest,
    RemoteGroupedRolloutRequest,
    RemoteRolloutRequest,
)
from .. import agent
from ..api import EnvironmentServer, InferenceServer, RemoteEvaluationRequest, RemoteRolloutRequest


@EnvironmentServer.register_subclass
class FastAPIEnvServer(EnvironmentServer):
    server_type: str = Field('FastAPIEnvServer', frozen=True, Literal=True)
    env_server_host_port: str
    _server_task: asyncio.Task = PrivateAttr(None)

    @classmethod
    async def launch(cls, env_cls: type[Agent], cls_args: dict, port: int, **kwargs) -> Self:

        app = FastAPI()

        if issubclass(env_cls, GroupedRolloutGenerator):

            @app.post("/grouped_rollouts/")
            async def grouped_rollouts(
                request: RemoteGroupedRolloutRequest,
            ) -> list[list[TokenRollout]]:
                env = env_cls(**cls_args)
                request.inference_interface = request.inference_interface.unwrap()
                return await env.get_grouped_rollouts(request)

        if issubclass(env_cls, ContrastiveRolloutGenerator):

            @app.post("/contrastive_rollouts/")
            async def contrastive_rollouts(
                request: RemoteRolloutRequest,
            ) -> list[ContrastiveRollout]:
                env = env_cls(**cls_args)
                request.inference_interface = request.inference_interface.unwrap()
                return await env.get_contrastive_rollouts(request)

        if issubclass(env_cls, RolloutGenerator):

            @app.post("/rollouts/")
            async def rollouts(request: RemoteRolloutRequest) -> list[TokenRollout]:
                env = env_cls(**cls_args)
                request.inference_interface = request.inference_interface.unwrap()
                return await env.get_reward_rollouts(request)

        if issubclass(env_cls, EvaluationAgent):

            @app.post("/evaluation/")
            async def run_evaluation(request: RemoteEvaluationRequest):
                env = env_cls(**cls_args)
                request.inference_interface = request.inference_interface.unwrap()
                return await env.run_evaluation(request)

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

    async def get_contrastive_rollouts(self, request: RolloutRequest) -> list[ContrastiveRollout]:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Rollout requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/contrastive_rollouts/",
                json=payload,
                timeout=None,
            )
        rollouts = [ContrastiveRollout.model_validate(r) for r in response.json()]
        return rollouts

    async def group_rollout(self, request: GroupedRolloutRequest):
        assert (
            False
        ), "Calling group_rollout on FastAPIEnvServer is not supported, use get_grouped_rollouts"

    async def get_grouped_rollouts(
        self, request: GroupedRolloutRequest
    ) -> AsyncGenerator[list[TokenRollout], None]:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Rollout requests to remote server must contain an InferenceServer object"
        assert request.num_groups != -1, "FastAPIEnvServer does not support group rollout streaming"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/grouped_rollouts/", json=payload, timeout=None
            )
        rollouts = [[TokenRollout.model_validate(r) for r in group] for group in response.json()]
        for rollout in rollouts:
            yield rollout

    async def rollout(self, request: RolloutRequest) -> TokenRollout:
        assert (
            False
        ), "Calling rollout on FastAPIEnvServer is not supported, use get_reward_rollouts"

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Rollout requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/rollouts/", json=payload, timeout=None
            )
        rollouts = [TokenRollout.model_validate(r) for r in response.json()]
        return rollouts

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        assert isinstance(
            request.inference_interface, InferenceServer
        ), "Evaluation requests to remote server must contain an InferenceServer object"
        payload = request.model_dump()
        payload["inference_interface"] = request.inference_interface.model_dump()
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/evaluation/", json=payload, timeout=None
            )
        response = EvaluationResponse.model_validate(response.json()).unwrap()
        return response


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
    agent_cls = import_class(config['agent_type'])
    cls_args = config['agent_args']
    run(agent_cls, cls_args, port=args.port)
