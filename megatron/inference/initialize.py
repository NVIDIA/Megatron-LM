# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Runtime-service initialization for inference entry points."""

from argparse import Namespace

from megatron.core.config import set_experimental_flag
from megatron.core.jit import disable_jit_fuser
from megatron.training import global_vars


def initialize_runtime_services_for_inference(
    args: Namespace, *, build_tokenizer: bool = True
) -> None:
    """Initialize inference services without training batch or progress state.

    Tokenizer, W&B and telemetry use the shared service implementations. Request
    batching belongs to the inference engine, not the training microbatch calculator.

    Args:
        args: Normalized command-line arguments for inference.
        build_tokenizer: Whether to construct and register the tokenizer.
    """
    if build_tokenizer:
        global_vars._build_tokenizer(args)
    global_vars._set_wandb_writer(args)
    global_vars._set_telemetry(args, include_training=False)

    if args.enable_experimental:
        set_experimental_flag(True)

    if args.disable_jit_fuser:
        disable_jit_fuser()
