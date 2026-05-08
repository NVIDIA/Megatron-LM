# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from megatron.inference.serve_config import ServeConfig


class TestServeConfig:
    def test_defaults(self):
        cfg = ServeConfig()
        assert cfg.host == "0.0.0.0"
        assert cfg.port == 5000
        assert cfg.parsers == []
        assert cfg.verbose is False
        assert cfg.frontend_replicas == 4

    def test_overrides_preserved(self):
        cfg = ServeConfig(
            host="127.0.0.1",
            port=8080,
            parsers=["json", "tool_use"],
            verbose=True,
            frontend_replicas=1,
        )
        assert cfg.host == "127.0.0.1"
        assert cfg.port == 8080
        assert cfg.parsers == ["json", "tool_use"]
        assert cfg.verbose is True
        assert cfg.frontend_replicas == 1
