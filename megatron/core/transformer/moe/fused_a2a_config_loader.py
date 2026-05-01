# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""
FusedA2A config loader and resolver for Megatron-LM MoE fused all-to-all.
"""
import os
import json
from typing import Optional

try:
    import yaml
    HAVE_YAML = True
except ImportError:
    HAVE_YAML = False

from .fused_a2a_config import FusedA2AConfig


def load_a2a_config_from_file(path: str) -> dict:
    """
    Load config from JSON or YAML file.
    Raises ValueError on error.
    """
    if path.endswith('.json'):
        with open(path, 'r') as f:
            return json.load(f)
    elif path.endswith(('.yaml', '.yml')):
        if not HAVE_YAML:
            raise ImportError('pyyaml is required for YAML config files')
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {path}")

def resolve_fused_a2a_config_from_sources(cli_args=None, env=os.environ, config_file_path=None) -> FusedA2AConfig:
    """
    Resolve FusedA2AConfig from CLI args, environment, and config file.
    Precedence: CLI > ENV > CONFIG FILE > DEFAULTS.
    Raises ValueError on any invalid or unknown keys.
    """
    file_cfg = {}
    if config_file_path:
        file_cfg = load_a2a_config_from_file(config_file_path)
    env_cfg = {}
    if env.get('MOE_A2A_CHUNK_SIZE'):
        env_cfg['chunk_size'] = int(env['MOE_A2A_CHUNK_SIZE'])
    if env.get('MOE_A2A_NUM_SMS'):
        env_cfg['num_sms'] = int(env['MOE_A2A_NUM_SMS'])
    cli_cfg = {}
    if cli_args is not None:
        if getattr(cli_args, 'moe_a2a_chunk_size', None) is not None:
            cli_cfg['chunk_size'] = cli_args.moe_a2a_chunk_size
        if getattr(cli_args, 'moe_a2a_num_sms', None) is not None:
            cli_cfg['num_sms'] = cli_args.moe_a2a_num_sms
    merged = {**file_cfg, **env_cfg, **cli_cfg}
    config = FusedA2AConfig.from_dict(merged)
    config.validate()
    return config
