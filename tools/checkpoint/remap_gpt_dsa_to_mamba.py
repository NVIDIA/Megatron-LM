#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Convert a GPTModel DSA checkpoint to a MambaModel-compatible checkpoint.

A GPTModel with ``--experimental-attention-variant dsa`` uses one combined
TransformerLayer per model layer (attention + MLP).  The equivalent MambaModel
with pattern ``S-S-...`` stores them as two separate layers:

* Layer 2N   – DSA attention (TransformerLayer: input_layernorm + MLASelfAttention)
* Layer 2N+1 – MLP          (MLPLayer: fused-norm MLP)

This script loads a GPTModel Distributed Checkpoint (DCP), remaps the state-dict
keys, and saves a new DCP that can be loaded by MambaModel.

Usage
-----
::

    python tools/checkpoint/remap_gpt_dsa_to_mamba.py \\
        --input  /path/to/gpt_dsa_dcp_checkpoint \\
        --output /path/to/mamba_dsa_dcp_checkpoint \\
        --num-gpt-layers 4

Key remapping rules
-------------------
* ``decoder.layers.{N}.input_layernorm.*``  →  ``decoder.layers.{2N}.input_layernorm.*``
* ``decoder.layers.{N}.self_attention.*``   →  ``decoder.layers.{2N}.self_attention.*``
* ``decoder.layers.{N}.mlp.*``             →  ``decoder.layers.{2N+1}.mlp.*``
* ``decoder.final_layernorm.*``            →  ``decoder.final_norm.*``
* All other keys                           →  unchanged
"""

import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Dict


def _remap_key(key: str, num_gpt_layers: int) -> str:
    """Return the MambaModel state-dict key corresponding to *key* from GPTModel.

    Args:
        key: A key from the GPTModel state dict.
        num_gpt_layers: Total number of GPT decoder layers (across all PP stages).

    Returns:
        The remapped key for MambaModel.

    Raises:
        ValueError: If an unexpected sub-key is encountered in a decoder layer.
    """
    layer_prefix = "decoder.layers."
    final_ln_prefix = "decoder.final_layernorm."

    # Final layernorm name differs between TransformerBlock and MambaStack
    if key.startswith(final_ln_prefix):
        return "decoder.final_norm." + key[len(final_ln_prefix):]

    if not key.startswith(layer_prefix):
        return key  # embedding, output_layer, rotary_pos_emb, etc.

    # Parse "decoder.layers.{N}.{rest}"
    remainder = key[len(layer_prefix):]
    dot_idx = remainder.index('.')
    layer_n = int(remainder[:dot_idx])
    rest = remainder[dot_idx + 1:]

    if rest.startswith("input_layernorm.") or rest.startswith("self_attention."):
        return f"{layer_prefix}{2 * layer_n}.{rest}"
    elif rest.startswith("mlp."):
        return f"{layer_prefix}{2 * layer_n + 1}.{rest}"
    else:
        raise ValueError(
            f"Unexpected sub-key '{rest}' in GPT layer {layer_n} (full key='{key}'). "
            "Expected: input_layernorm.*, self_attention.*, mlp.*"
        )


def _remap_state_dict(
    gpt_sd: Dict, num_gpt_layers: int
) -> Dict:
    """Apply key remapping to the full GPTModel state dict."""
    return {_remap_key(k, num_gpt_layers): v for k, v in gpt_sd.items()}


def convert(input_path: Path, output_path: Path, num_gpt_layers: int) -> None:
    """Load a GPTModel DCP checkpoint, remap keys, and save as MambaModel DCP.

    Args:
        input_path: Path to the GPTModel DCP checkpoint directory.
        output_path: Destination directory for the MambaModel DCP checkpoint.
        num_gpt_layers: Number of GPT decoder layers in the original model.
    """
    try:
        import torch
        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint.format_utils import (
            dcp_to_torch_save,
            torch_save_to_dcp,
        )
    except ImportError as exc:
        raise SystemExit(
            "PyTorch distributed checkpoint (torch.distributed.checkpoint) is required. "
            "Please upgrade to PyTorch >= 2.0."
        ) from exc

    import torch

    print(f"Loading GPTModel checkpoint from: {input_path}")

    # --- Load the flat state dict from DCP ---
    # We use dcp_to_torch_save to materialize the DCP into a regular .pt file,
    # then remap keys, then convert back to DCP.
    tmp_flat = output_path.parent / "_tmp_gpt_flat.pt"
    try:
        dcp_to_torch_save(str(input_path), str(tmp_flat))
        gpt_sd = torch.load(tmp_flat, map_location="cpu")
        print(f"Loaded {len(gpt_sd)} keys from GPTModel checkpoint.")

        # --- Remap keys ---
        mamba_sd = _remap_state_dict(gpt_sd, num_gpt_layers)
        print(
            f"Remapped state dict: {len(gpt_sd)} GPT keys → {len(mamba_sd)} Mamba keys."
        )

        # --- Save remapped state dict as a new flat .pt then convert to DCP ---
        tmp_mamba = output_path.parent / "_tmp_mamba_flat.pt"
        torch.save(mamba_sd, tmp_mamba)

        output_path.mkdir(parents=True, exist_ok=True)
        torch_save_to_dcp(str(tmp_mamba), str(output_path))
        print(f"MambaModel DCP checkpoint saved to: {output_path}")

    finally:
        for tmp in (tmp_flat, output_path.parent / "_tmp_mamba_flat.pt"):
            if tmp.exists():
                tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert GPTModel DSA checkpoint to MambaModel-compatible format."
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to the source GPTModel DCP checkpoint directory.",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Destination path for the MambaModel DCP checkpoint.",
    )
    parser.add_argument(
        "--num-gpt-layers", required=True, type=int,
        help="Number of decoder layers in the GPTModel (e.g. 4).",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input checkpoint not found: {args.input}")
    if args.output.exists():
        print(f"Warning: output path already exists and will be overwritten: {args.output}")
        shutil.rmtree(args.output)

    convert(args.input, args.output, args.num_gpt_layers)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
