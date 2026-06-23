# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from ..server.agent.fastapi_env_server import FastAPIEnvServer
from .api import EvaluationAgent, GroupedRolloutGenerator, RolloutGenerator


class RemoteAgent(FastAPIEnvServer, RolloutGenerator, GroupedRolloutGenerator, EvaluationAgent):
    env_id: str = "remote"
    env_server_host_port: str
