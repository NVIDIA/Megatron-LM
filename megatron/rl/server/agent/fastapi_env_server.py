# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket

import httpx
import yaml
from fastapi import FastAPI
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
    GroupedRolloutRequest,
    GroupRolloutParams,
    RolloutGenerator,
    RolloutRequest,
    TokenRollout,
)
from ...agent.registry import get_agent_class
from ...inference import InferenceInterface
from ...inference.chat_interface import MegatronChatInterface
from ..api import EnvironmentServer, RemoteEvaluationRequest, RemoteRolloutRequest


def _bind(wire: BaseModel, request_cls: type, interface: InferenceInterface):
    """Rebuild an agent request from its wire form, generating through `interface`."""
    return request_cls.model_validate({**wire.model_dump(), "inference_interface": interface})


@EnvironmentServer.register_subclass
class FastAPIEnvServer(EnvironmentServer):
    server_type: str = Field('FastAPIEnvServer', frozen=True, Literal=True)
    env_server_host_port: str
    _server_task: asyncio.Task = PrivateAttr(None)

    @classmethod
    def build_app(cls, env_cls: type[Agent], cls_args: dict, policy: InferenceInterface) -> FastAPI:
        """Env server app; every route generates through `policy`."""
        app = FastAPI()

        if issubclass(env_cls, ContrastiveRolloutGenerator):

            @app.post("/contrastive_rollouts/")
            async def contrastive_rollouts(
                request: RemoteRolloutRequest,
            ) -> list[ContrastiveRollout]:
                env = env_cls(**cls_args)
                return await env.get_contrastive_rollouts(_bind(request, RolloutRequest, policy))

        if issubclass(env_cls, RolloutGenerator):

            @app.post("/rollouts/")
            async def rollouts(request: RemoteRolloutRequest) -> list[TokenRollout]:
                env = env_cls(**cls_args)
                return await env.get_reward_rollouts(_bind(request, RolloutRequest, policy))

        if issubclass(env_cls, EvaluationAgent):

            @app.post("/evaluation/")
            async def run_evaluation(request: RemoteEvaluationRequest):
                env = env_cls(**cls_args)
                return await env.run_evaluation(_bind(request, EvaluationRequest, policy))

        return app

    @classmethod
    async def launch(
        cls, env_cls: type[Agent], cls_args: dict, port: int, policy: InferenceInterface, **kwargs
    ) -> Self:
        app = cls.build_app(env_cls, cls_args, policy)

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

    async def _post(self, path: str, body: BaseModel):
        """POST a wire request; a failure on the env server is raised with its detail."""
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://{self.env_server_host_port}{path}",
                content=body.model_dump_json(),
                headers={"content-type": "application/json"},
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Env server {path} returned HTTP {response.status_code}: {response.text}"
            )
        return response.json()

    async def get_contrastive_rollouts(self, request: RolloutRequest) -> list[ContrastiveRollout]:
        wire = RemoteRolloutRequest.model_validate(
            request.model_dump(exclude={"inference_interface"})
        )
        result = await self._post("/contrastive_rollouts/", wire)
        return [ContrastiveRollout.model_validate(r) for r in result]

    async def prepare_group_rollout(self, request: GroupedRolloutRequest) -> GroupRolloutParams:
        raise NotImplementedError(
            "FastAPIEnvServer overrides get_reward_rollouts; prepare_group_rollout is not used."
        )

    async def get_rollout_response(self, request, inference_request):
        raise NotImplementedError(
            "FastAPIEnvServer overrides get_reward_rollouts/run_evaluation; "
            "get_rollout_response is not used."
        )

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[TokenRollout]:
        wire = RemoteRolloutRequest.model_validate(
            request.model_dump(exclude={"inference_interface"})
        )
        result = await self._post("/rollouts/", wire)
        return [TokenRollout.model_validate(r) for r in result]

    async def run_evaluation(self, request: EvaluationRequest) -> EvaluationResponse:
        wire = RemoteEvaluationRequest.model_validate(
            request.model_dump(exclude={"inference_interface"})
        )
        return EvaluationResponse.validate_registered(await self._post("/evaluation/", wire))


def run(agent_cls: type[Agent], cls_args: dict, port: int, policy: InferenceInterface):
    loop = asyncio.new_event_loop()

    async def run_server():
        server: FastAPIEnvServer = await FastAPIEnvServer.launch(
            env_cls=agent_cls, cls_args=cls_args, port=port, policy=policy
        )
        print(server.model_dump(exclude={'_server_task'}))
        await server._server_task

    loop.run_until_complete(run_server())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env-config", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--policy-url",
        type=str,
        required=True,
        help="Megatron text-generation server to generate through, e.g. http://<rank 0>:8294.",
    )
    parser.add_argument(
        "--add-bos",
        action="store_true",
        help="Prepend BOS to prompts; set it when the trainer runs without --rl-skip-bos-token.",
    )
    args = parser.parse_args()
    with open(args.env_config, 'r') as f:
        config = yaml.safe_load(f)[0]
    agent_cls = get_agent_class(config['agent_type'])
    cls_args = config['agent_args']
    run(
        agent_cls,
        cls_args,
        port=args.port,
        policy=MegatronChatInterface(base_url=args.policy_url, add_bos=args.add_bos),
    )
