# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.rl.agent.remote_agent import RemoteAgent


def test_remote_agent_instantiates():
    """RemoteAgent must satisfy all abstract methods inherited from Agent."""
    agent = RemoteAgent(env_server_host_port="example:1")
    assert agent.env_server_host_port == "example:1"
