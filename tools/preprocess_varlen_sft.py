# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Pre-tokenize SFT conversations for ``--varlen-pretokenized-dataset``.

Reads a JSONL file whose rows are either ``{"messages": [...]}`` or a bare
messages list, tokenizes each conversation with
:meth:`SFTTokenizer.tokenize_conversation`, and writes two IndexedDataset
streams:

  * ``<output-prefix>.bin/.idx`` -- token ids.
  * ``<output-prefix>.mask.bin/.idx`` -- assistant-only loss mask (0/1).

The prompt format must return assistant-only targets, i.e. ``IGNORE_INDEX`` on
prompt tokens. ``--prompt-format default`` returns all-1 targets by design, so
it cannot produce an assistant-only mask and is rejected.

Example:
    python tools/preprocess_varlen_sft.py \\
        --input conversations.jsonl \\
        --output-prefix /data/sft/train \\
        --tokenizer-model /models/Qwen2.5-0.5B-Instruct \\
        --prompt-format nemotron-h-aligned
"""

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

# Allow running this file directly: python tools/preprocess_varlen_sft.py.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import torch

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from megatron.core.tokenizers.text.libraries.sft_tokenizer import IGNORE_INDEX, SFTTokenizer


def assistant_only_mask(targets: Sequence[int], prompt_format: str) -> np.ndarray:
    """Build a 0/1 assistant-only loss mask from SFTTokenizer targets."""
    if prompt_format == "default":
        raise ValueError(
            "The 'default' prompt format returns all-1 targets and cannot produce "
            "an assistant-only loss mask. Use a format such as "
            "'nemotron-h-aligned' or 'nemotron-nano-v2'."
        )
    mask = np.asarray(targets) != IGNORE_INDEX
    if not mask.any():
        raise ValueError("Conversation contains no assistant target tokens.")
    if mask.all():
        raise ValueError(
            "SFTTokenizer returned all-1 targets; assistant-only masking needs at "
            "least one non-assistant token."
        )
    return mask.astype(np.uint8)


def _load_messages(line: str) -> List[Dict[str, str]]:
    row = json.loads(line)
    if isinstance(row, dict):
        row = row.get("messages")
    if not isinstance(row, list):
        raise ValueError(
            "Each input line must be a messages list or an object with a 'messages' list."
        )
    return row


def write_pretokenized(
    input_path: str,
    output_prefix: str,
    tokenizer: SFTTokenizer,
    prompt_format: str,
    append_eod: bool = True,
) -> int:
    """Tokenize ``input_path`` into tokens and mask IndexedDataset streams."""
    output_dir = os.path.dirname(os.path.abspath(output_prefix))
    os.makedirs(output_dir, exist_ok=True)

    token_builder = IndexedDatasetBuilder(output_prefix + ".bin", dtype=np.int32)
    mask_builder = IndexedDatasetBuilder(output_prefix + ".mask.bin", dtype=np.uint8)
    count = 0
    with open(input_path) as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                messages = _load_messages(line)
                tokens, targets = tokenizer.tokenize_conversation(
                    messages, return_target=True, add_generation_prompt=False
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"{input_path}:{line_number}: {exc}") from exc

            tokens_list = tokens.tolist()
            targets_list = targets.tolist()
            if len(tokens_list) != len(targets_list):
                raise ValueError(
                    f"{input_path}:{line_number}: token/target length mismatch "
                    f"({len(tokens_list)} vs {len(targets_list)})."
                )
            mask = assistant_only_mask(targets_list, prompt_format)
            if append_eod and tokens_list[-1] != tokenizer.eod:
                tokens_list.append(tokenizer.eod)
                mask = np.append(mask, 1)

            token_builder.add_item(torch.tensor(tokens_list, dtype=torch.int32))
            mask_builder.add_item(torch.tensor(mask, dtype=torch.uint8))
            token_builder.end_document()
            mask_builder.end_document()
            count += 1

    if count == 0:
        raise ValueError(f"{input_path} contains no samples.")
    token_builder.finalize(output_prefix + ".idx")
    mask_builder.finalize(output_prefix + ".mask.idx")
    return count


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL file; each line is a messages list or an object with a "
        "'messages' list.",
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix; tokens go to <prefix>.bin/.idx and the mask to "
        "<prefix>.mask.bin/.idx.",
    )
    parser.add_argument("--tokenizer-model", required=True, help="HF tokenizer path.")
    parser.add_argument(
        "--prompt-format",
        default="nemotron-h-aligned",
        choices=["nemotron-nano-v2", "nemotron-h-aligned", "identity", "default"],
        help="SFT prompt format; it must return assistant-only targets "
        "(IGNORE_INDEX on prompt tokens). 'default' returns all-1 targets and "
        "is rejected.",
    )
    parser.add_argument("--use-gigatoken", action="store_true", help="Tokenize with Gigatoken.")
    parser.add_argument(
        "--no-append-eod",
        action="store_true",
        help="Do not append an EOD token when a conversation does not end with one.",
    )
    return parser.parse_args()


def main() -> None:
    """Tokenize the input JSONL and write both IndexedDataset streams."""
    args = get_args()
    tokenizer = SFTTokenizer(
        args.tokenizer_model, args.prompt_format, use_gigatoken=args.use_gigatoken
    )
    count = write_pretokenized(
        args.input,
        args.output_prefix,
        tokenizer,
        args.prompt_format,
        append_eod=not args.no_append_eod,
    )
    sys.stdout.write(f"Wrote {count} samples to {args.output_prefix}\n")


if __name__ == "__main__":
    main()
