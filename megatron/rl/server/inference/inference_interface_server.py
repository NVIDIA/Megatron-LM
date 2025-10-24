# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import asyncio
import os
import socket
import weakref

import httpx
from fastapi import FastAPI
from pydantic import Field, PrivateAttr
from typing_extensions import Self
from uvicorn import Config, Server

from ...inference.api import ChatInferenceRequest, ChatInferenceResponse
from ...inference.inference_interface import (
    ChatInferenceInterface,
    InferenceInterface,
    ReturnsRaw,
    ReturnsTokens,
)
from ...server.api import InferenceServer


@InferenceServer.register_subclass
class InferenceInterfaceClient(ChatInferenceInterface, InferenceServer):
    type_name: str = Field(default='InferenceInterfaceClient', frozen=True)
    env_server_host_port: str
    conversation_template: None = None

    async def base_generate(self, request: ChatInferenceRequest) -> list[ChatInferenceResponse]:
        async with httpx.AsyncClient(timeout=None) as client:
            response = await client.post(
                f"http://{self.env_server_host_port}/base_generate/", json=request.model_dump()
            )
            return [
                ChatInferenceResponse.model_validate(inference_response)
                for inference_response in response.json()
            ]


@InferenceServer.register_subclass
class InferenceInterfaceServer(InferenceInterfaceClient, ReturnsRaw, ReturnsTokens):
    type_name: str = Field(default='InferenceInterfaceServer', frozen=True)
    _server: Server
    _server_task: asyncio.Task
    _inference_interface: InferenceInterface
    _interface_launched: bool = PrivateAttr(False)

    @classmethod
    async def launch(cls, interface_cls: type[InferenceInterface], **kwargs) -> Self:
        app = FastAPI()
        loop = asyncio.get_event_loop()
        config = Config(
            app=app,
            loop=loop,
            host='0.0.0.0',
            port=os.getenv('MEGATRON_RL_INFERENCE_SERVER_PORT', 8294),
        )
        ip = socket.gethostbyname(socket.gethostname())
        launched_server = cls(env_server_host_port=f"{ip}:{config.port}")

        if issubclass(interface_cls, InferenceServer):
            launched_server._inference_interface = await interface_cls.launch(**kwargs)
            launched_server._interface_launched = True
        else:
            launched_server._inference_interface = interface_cls(**kwargs)

        # Use a weak reference to avoid circular reference
        server_ref = weakref.ref(launched_server)

        @app.post("/base_generate/")
        async def base_generate(request: ChatInferenceRequest):
            server = server_ref()
            if server is None:
                raise RuntimeError("Server has been garbage collected")
            return await server._inference_interface.base_generate(request)

        server = Server(config)
        launched_server._server = server
        launched_server._server_task = loop.create_task(server.serve())

        print(f"Launched server on {ip}:{config.port}")
        return launched_server

    async def kill(self):
        self._server.should_exit = True
        if isinstance(self._inference_interface, InferenceServer) and self._interface_launched:
            self._interface_launched = False
            await self._inference_interface.kill()
        await self._server_task

    async def suspend(self):
        if isinstance(self._inference_interface, InferenceServer):
            await self._inference_interface.suspend()

    def resume(self):
        if isinstance(self._inference_interface, InferenceServer):
            self._inference_interface.resume()
