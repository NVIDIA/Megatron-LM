# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Eval-oriented OpenAI-compatible server for Megatron checkpoints.

This is a thin wrapper around ``tools.run_dynamic_text_generation_server`` for
external evaluation runners. It keeps the same Megatron/inference CLI surface,
but defaults checkpoint loading and log-prob behavior to what
loglikelihood-style evals need.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from tools.run_dynamic_text_generation_server import main

if __name__ == "__main__":
    main(
        args_defaults={'exit_on_missing_checkpoint': True},
        force_return_log_probs=True,
        force_prompt_log_probs=True,
    )
