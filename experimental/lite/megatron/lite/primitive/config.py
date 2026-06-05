"""Configuration utilities.

Model-specific configs live under `megatron.lite/models/*/config.py`
as standalone dataclasses.
"""

from __future__ import annotations

import json
from dataclasses import fields as dc_fields
from pathlib import Path
from typing import Any


def to_dataclass(oc_cfg, cls):
    """Convert an OmegaConf DictConfig back to a typed dataclass instance."""
    from omegaconf import OmegaConf

    init_names = {f.name for f in dc_fields(cls) if f.init}
    d = OmegaConf.to_container(oc_cfg, resolve=True)
    return cls(**{k: v for k, v in d.items() if k in init_names})


def load_hf_config_dict(path_or_name: str) -> dict[str, Any]:
    """Load HF config dict from local path or Hub."""
    p = Path(path_or_name)

    if p.is_file() and p.name == "config.json":
        with open(p) as f:
            return json.load(f)

    if p.is_dir():
        cfg_file = p / "config.json"
        if cfg_file.exists():
            with open(cfg_file) as f:
                return json.load(f)
        raise FileNotFoundError(f"No config.json in {p}")

    try:
        from transformers import AutoConfig

        hf_config = AutoConfig.from_pretrained(path_or_name, trust_remote_code=True)
        return hf_config.to_dict()
    except ImportError as err:
        raise ImportError(
            f"'{path_or_name}' is not a local path. "
            "Install transformers to load from HuggingFace Hub: pip install transformers"
        ) from err
    except Exception as e:
        raise ValueError(f"Failed to load config from '{path_or_name}': {e}") from e


__all__ = [
    "load_hf_config_dict",
    "to_dataclass",
]
