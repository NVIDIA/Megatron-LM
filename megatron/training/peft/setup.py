# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Native model-build integration for LoRA."""

from megatron.core.utils import unwrap_model

from .checkpointing import load_base_weights
from .transform import apply_lora


def prepare_lora_model(model, *, config):
    """Restore the base, then attach adapters before DDP/FSDP and optimizer creation."""
    chunks = unwrap_model(model)
    if not isinstance(chunks, list):
        chunks = [chunks]
    pg_collection = chunks[0].pg_collection
    load_base_weights(chunks, config, pg_collection)
    apply_lora(chunks, config.peft, pg_collection=pg_collection)
    return model
