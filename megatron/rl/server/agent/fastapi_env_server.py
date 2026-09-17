# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import socket
from typing import Annotated, Any

import httpx
import yaml
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, PrivateAttr, TypeAdapter
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
    Rollout,
    RolloutRequest,
    TokenRollout,
)
from ...agent.registry import get_agent_class
from ...agent.reward_only_agent import RewardOnlyAgent
from ...inference import InferenceInterface
from ...inference.chat_interface import MegatronChatInterface
from ..api import (
    EnvironmentServer,
    RemoteEpisodeRequest,
    RemoteEvaluationRequest,
    RemotePrompt,
    RemotePromptRequest,
    RemoteRolloutRequest,
)

# A finished rollout on the wire; an empty placeholder validates as the first member.
_ROLLOUT = TypeAdapter(Annotated[TokenRollout | Rollout, Field(union_mode="left_to_right")])


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
        """Env server app: one environment instance generating through `policy`."""
        app = FastAPI()
        env = env_cls(**cls_args)

        @app.exception_handler(Exception)
        async def env_error(_, exc: Exception) -> JSONResponse:
            # The trainer sees the environment's failure the way a local agent would raise it.
            return JSONResponse(status_code=500, content={"detail": repr(exc)})

        if isinstance(env, RewardOnlyAgent):

            @app.post("/prompts/")
            async def sample_prompt(request: RemotePromptRequest) -> RemotePrompt:
                prompt, golden = await env.get_prompt(validation=request.validation)
                return RemotePrompt(prompt=prompt, golden=golden)

            @app.post("/episodes/")
            async def run_episode(request: RemoteEpisodeRequest) -> dict[str, Any]:
                group_request = GroupedRolloutRequest(
                    num_groups=1,
                    rollouts_per_group=1,
                    inference_interface=policy,
                    generation_args=request.generation_args,
                    validation=request.validation,
                )
                params = env.group_rollout_params(
                    group_request, prompt=request.prompt, golden=request.golden
                )
                rollout = await params.build_rollout(await params.run_episode())
                return rollout.model_dump()

        if isinstance(env, ContrastiveRolloutGenerator):

            @app.post("/contrastive_rollouts/")
            async def contrastive_rollouts(
                request: RemoteRolloutRequest,
            ) -> list[ContrastiveRollout]:
                return await env.get_contrastive_rollouts(_bind(request, RolloutRequest, policy))

        if isinstance(env, EvaluationAgent):

            @app.post("/evaluation/")
            async def run_evaluation(request: RemoteEvaluationRequest):
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

    async def prepare_group_rollout(
        self, request: GroupedRolloutRequest | RolloutRequest
    ) -> GroupRolloutParams:
        sampled = RemotePrompt.model_validate(
            await self._post("/prompts/", RemotePromptRequest(validation=request.validation))
        )
        episode_request = RemoteEpisodeRequest(
            prompt=sampled.prompt,
            golden=sampled.golden,
            generation_args=request.generation_args,
            validation=request.validation,
        )

        async def run_episode() -> Rollout | TokenRollout:
            # The env server scores the episode itself, so the episode is already the rollout.
            return _ROLLOUT.validate_python(await self._post("/episodes/", episode_request))

        async def build_rollout(rollout: Rollout | TokenRollout) -> Rollout | TokenRollout:
            return rollout

        return GroupRolloutParams(run_episode=run_episode, build_rollout=build_rollout)

    async def get_rollout_response(self, request, inference_request):
        raise NotImplementedError(
            "Episodes run on the env server; FastAPIEnvServer never generates itself."
        )

    async def get_reward_rollouts(self, request: RolloutRequest) -> list[Rollout | TokenRollout]:
        async def single_rollout() -> Rollout | TokenRollout:
            params = await self.prepare_group_rollout(request)
            return await params.build_rollout(await params.run_episode())

        return list(await asyncio.gather(*[single_rollout() for _ in range(request.num_rollouts)]))

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
