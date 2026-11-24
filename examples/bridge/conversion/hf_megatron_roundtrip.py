# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/11d8b349d2daff3523bb9655b04f4ad33a9e9ea9/examples/conversion/hf_megatron_roundtrip.py
# Source license: Apache-2.0.

"""Import HF weights into a Megatron model and export them back to HF on one GPU.

Use a supported Nemotron or Qwen checkpoint that fits on one GPU. This example
prints a per-weight comparison and saves HF weights/config/tokenizer artifacts;
it does not train a model or write an intermediate native Megatron checkpoint.
"""

import argparse
import os
from pathlib import Path

import torch
from rich.console import Console

from megatron.bridge import AutoBridge
from megatron.bridge.models.conversion.utils import weights_verification_table
from megatron.core import parallel_state

console = Console()


def main(
    hf_model_id: str,
    output_dir: str | None = None,
    trust_remote_code: bool = False,
    revision: str | None = None,
) -> None:
    """Display weight comparisons and save the converted model in a new directory."""
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This example supports one GPU; use run_conversion.py for distributed I/O."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA GPU.")
    save_path = Path(output_dir or ".") / Path(hf_model_id).name
    if save_path.exists():
        raise FileExistsError(f"Output already exists: {save_path}")

    bridge = AutoBridge.from_hf_pretrained(
        hf_model_id, trust_remote_code=trust_remote_code, revision=revision
    )
    model_config = bridge.get_model_config()
    megatron_model = bridge.get_model(model_config, wrap_with_ddp=False)
    console.print(weights_verification_table(bridge, megatron_model))

    console.print(f"Saving HF checkpoint in {save_path}...")
    bridge.save_hf_pretrained(megatron_model, save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-model-id", required=True, help="Supported HF model ID or local path")
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow custom code from a trusted checkpoint",
    )
    parser.add_argument("--revision", help="Pinned HF revision for remote inputs")
    parser.add_argument(
        "--output-dir",
        help="Parent of the new model output directory (default: current working directory)",
    )
    args = parser.parse_args()
    try:
        main(args.hf_model_id, args.output_dir, args.trust_remote_code, args.revision)
    finally:
        parallel_state.destroy_model_parallel()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
