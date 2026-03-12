#!/usr/bin/env python3
"""Inspect pretokenized SFT samples from Megatron-LM .bin/.idx file pairs.

Usage:
    python inspect_sft_file_prefix.py <file_prefix>

Example:
    python inspect_sft_file_prefix.py /path/to/dataset-materialized_text_document
"""

import argparse
import struct
import numpy
from functools import lru_cache
from typing import Optional, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Hardcoded config for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8 chat template
# ──────────────────────────────────────────────────────────────────────────────
TOKENIZER_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8"

# These are pre-computed from:
#   tokenizer.encode("<|im_start|>system\n", add_special_tokens=False)  etc.
ROLE_START_TOKENS = {
    "system":    [10, 25708, 1010],       # <|im_start|>system\n
    "user":      [10, 3263, 1010],        # <|im_start|>user\n
    "assistant": [10, 1503, 19464, 1010], # <|im_start|>assistant\n
}
END_TOKENS = [11, 1010]                   # <|im_end|>\n
THINK_START_ID = 12                       # <think>
THINK_END_ID = 13                         # </think>
TOOL_CALL_START = [14, 1010]              # <tool_call>\n
TOOL_CALL_END = [15, 1010]               # </tool_call>\n
TOOL_RESPONSE_START = [16, 1010]         # <tool_response>\n
TOOL_RESPONSE_END = [17, 1010]           # </tool_response>\n

# ──────────────────────────────────────────────────────────────────────────────
# Index / bin readers (from Megatron-LM)
# ──────────────────────────────────────────────────────────────────────────────
_INDEX_HEADER = b"MMIDIDX\x00\x00"


class _MMapBinReader:
    def __init__(self, bin_path: str) -> None:
        self._bin_file_reader = open(bin_path, mode="rb")
        self._bin_buffer_mmap = numpy.memmap(self._bin_file_reader, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap.data)

    def read(self, dtype, count: int, offset: int) -> numpy.ndarray:
        return numpy.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        if self._bin_file_reader is not None:
            self._bin_file_reader.close()
        del self._bin_buffer_mmap
        del self._bin_file_reader


class _IndexReader:
    def __init__(self, idx_path: str) -> None:
        with open(idx_path, "rb") as stream:
            header = stream.read(9)
            assert header == _INDEX_HEADER, f"bad header, cannot read: {idx_path}"

            version = struct.unpack("<Q", stream.read(8))[0]
            assert version == 1, f"bad version, cannot read: {idx_path}"

            _code = struct.unpack("<B", stream.read(1))[0]
            self.sequence_count = struct.unpack("<Q", stream.read(8))[0]
            self.document_count = struct.unpack("<Q", stream.read(8))[0]
            offset = stream.tell()

        self.bin_buffer_mmap = numpy.memmap(idx_path, mode="r", order="C")
        self.bin_buffer = memoryview(self.bin_buffer_mmap)

        self.sequence_lengths = numpy.frombuffer(
            self.bin_buffer, dtype=numpy.int32, count=self.sequence_count, offset=offset
        )
        self.sequence_pointers = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.sequence_count,
            offset=offset + self.sequence_lengths.nbytes,
        )
        self.document_indices = numpy.frombuffer(
            self.bin_buffer,
            dtype=numpy.int64,
            count=self.document_count,
            offset=offset + self.sequence_lengths.nbytes + self.sequence_pointers.nbytes,
        )

    def __len__(self) -> int:
        return self.sequence_count

    @lru_cache(maxsize=8)
    def __getitem__(self, idx: int) -> Tuple[numpy.int64, numpy.int32]:
        return (self.sequence_pointers[idx], self.sequence_lengths[idx])

    def __del__(self) -> None:
        if hasattr(self, "bin_buffer_mmap"):
            self.bin_buffer_mmap._mmap.close()
            del self.bin_buffer_mmap


# ──────────────────────────────────────────────────────────────────────────────
# Segment extraction logic
# ──────────────────────────────────────────────────────────────────────────────
def find_subsequence(sequence, subsequence, start=0):
    sub_len = len(subsequence)
    for i in range(start, len(sequence) - sub_len + 1):
        if sequence[i : i + sub_len] == subsequence:
            return i
    return -1


def split_tool_calls(tokens, offset):
    """Split a token sequence into assistant text and tool_call sub-segments."""
    tc_start_len = len(TOOL_CALL_START)
    tc_end_len = len(TOOL_CALL_END)
    result = []
    pos = 0
    while pos < len(tokens):
        # Find next <tool_call>\n
        tc_start = find_subsequence(tokens, TOOL_CALL_START, pos)

        if tc_start == -1:
            # No more tool calls, rest is regular assistant content
            if pos < len(tokens):
                result.append({"role": "assistant", "tokens": tokens[pos:], "start": offset + pos, "end": offset + len(tokens)})
            break

        # Assistant content before tool_call
        if tc_start > pos:
            result.append({"role": "assistant", "tokens": tokens[pos:tc_start], "start": offset + pos, "end": offset + tc_start})

        # Find matching </tool_call>\n
        content_start = tc_start + tc_start_len
        tc_end = find_subsequence(tokens, TOOL_CALL_END, content_start)

        if tc_end == -1:
            # No closing tag, treat rest as tool_call
            result.append({"role": "tool_call", "tokens": tokens[content_start:], "start": offset + content_start, "end": offset + len(tokens)})
            break

        # Tool call content (excluding markers)
        result.append({"role": "tool_call", "tokens": tokens[content_start:tc_end], "start": offset + content_start, "end": offset + tc_end})
        pos = tc_end + tc_end_len

    return result


def extract_segments(tokenized_conversation, role_start_tokens, end_tokens, think_start_id, think_end_id):
    markers = []
    for role, start_tokens in role_start_tokens.items():
        pos = 0
        while True:
            idx = find_subsequence(tokenized_conversation, start_tokens, pos)
            if idx == -1:
                break
            markers.append((idx, role, len(start_tokens)))
            pos = idx + len(start_tokens)
    markers.sort(key=lambda x: x[0])

    segments = []
    for start_pos, role, marker_len in markers:
        content_start = start_pos + marker_len
        end_pos = find_subsequence(tokenized_conversation, end_tokens, content_start)
        if end_pos == -1:
            content_end = len(tokenized_conversation)
        else:
            content_end = end_pos
        content_tokens = tokenized_conversation[content_start:content_end]

        # Check if this user turn is actually a tool response
        if role == "user" and len(content_tokens) >= len(TOOL_RESPONSE_START) and content_tokens[:len(TOOL_RESPONSE_START)] == TOOL_RESPONSE_START:
            segments.append({"role": "tool_response", "tokens": content_tokens, "start": content_start, "end": content_end})
            continue

        if role == "assistant":
            think_start_idx = None
            think_end_idx = None
            for i, tok in enumerate(content_tokens):
                if tok == think_start_id and think_start_idx is None:
                    think_start_idx = i
                elif tok == think_end_id:
                    think_end_idx = i
                    break

            if think_start_idx is not None and think_end_idx is not None:
                reasoning_tokens = content_tokens[think_start_idx + 1 : think_end_idx]
                response_tokens = content_tokens[think_end_idx + 1 :]
                if reasoning_tokens:
                    abs_start = content_start + think_start_idx + 1
                    abs_end = content_start + think_end_idx
                    segments.append({"role": "reasoning", "tokens": reasoning_tokens, "start": abs_start, "end": abs_end})
                # Split the response part by tool calls
                if response_tokens:
                    abs_start = content_start + think_end_idx + 1
                    segments.extend(split_tool_calls(response_tokens, abs_start))
                continue

            # No think tags — split entire content by tool calls
            if find_subsequence(content_tokens, TOOL_CALL_START) != -1:
                segments.extend(split_tool_calls(content_tokens, content_start))
                continue

        segments.append({"role": role, "tokens": content_tokens, "start": content_start, "end": content_end})

    return segments


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inspect pretokenized SFT samples.")
    parser.add_argument("file_prefix", help="Path prefix for .bin/.idx files (without extension)")
    args = parser.parse_args()

    print(f"Loading tokenizer: {TOKENIZER_NAME}")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)

    # Verify hardcoded token ids match this tokenizer
    assert tokenizer.encode("<|im_start|>system\n", add_special_tokens=False) == ROLE_START_TOKENS["system"]
    assert tokenizer.encode("<|im_start|>user\n", add_special_tokens=False) == ROLE_START_TOKENS["user"]
    assert tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False) == ROLE_START_TOKENS["assistant"]
    assert tokenizer.encode("<|im_end|>\n", add_special_tokens=False) == END_TOKENS
    assert tokenizer.convert_tokens_to_ids("<think>") == THINK_START_ID
    assert tokenizer.convert_tokens_to_ids("</think>") == THINK_END_ID
    assert tokenizer.encode("<tool_call>\n", add_special_tokens=False) == TOOL_CALL_START
    assert tokenizer.encode("</tool_call>\n", add_special_tokens=False) == TOOL_CALL_END
    assert tokenizer.encode("<tool_response>\n", add_special_tokens=False) == TOOL_RESPONSE_START
    assert tokenizer.encode("</tool_response>\n", add_special_tokens=False) == TOOL_RESPONSE_END

    print(f"Loading index: {args.file_prefix}.idx")
    index = _IndexReader(args.file_prefix + ".idx")
    print(f"Loading bin:   {args.file_prefix}.bin")
    reader = _MMapBinReader(args.file_prefix + ".bin")
    print(f"Total sequences: {len(index)}\n")

    while True:
        try:
            raw = input(f"Enter sample index [0-{len(index) - 1}] (q to quit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if raw.lower() == "q":
            break

        # Parse optional -d suffix for raw detokenized output
        detokenize_only = raw.endswith("-d")
        if detokenize_only:
            raw = raw[:-2].strip()

        try:
            sample_idx = int(raw)
        except ValueError:
            print(f"Invalid input: {raw!r}")
            continue

        if sample_idx < 0 or sample_idx >= len(index):
            print(f"Out of range. Must be 0-{len(index) - 1}")
            continue

        pointer, length = index[sample_idx]
        sample = reader.read(numpy.int32, int(length), int(pointer))

        if detokenize_only:
            print(f"\n{'=' * 80}")
            print(f"Sample {sample_idx} | {len(sample)} total tokens (raw detokenized)")
            print(f"{'=' * 80}\n")
            print(tokenizer.decode(sample.tolist()))
            print(f"\n{'=' * 80}\n")
            continue

        segments = extract_segments(
            sample.tolist(), ROLE_START_TOKENS, END_TOKENS, THINK_START_ID, THINK_END_ID
        )

        print(f"\n{'=' * 80}")
        print(f"Sample {sample_idx} | {len(sample)} total tokens")
        print(f"{'=' * 80}")

        assistant_tokens = 0
        reasoning_tokens = 0
        tool_call_tokens = 0

        for seg in segments:
            decoded = tokenizer.decode(seg["tokens"])
            role_label = seg["role"].upper()
            n_tokens = len(seg["tokens"])
            print(f"\n[{role_label:>13}] ({n_tokens:>5} tokens | {seg['start']}:{seg['end']})")
            print(decoded)

            if seg["role"] == "assistant":
                assistant_tokens += n_tokens
            elif seg["role"] == "reasoning":
                reasoning_tokens += n_tokens
            elif seg["role"] == "tool_call":
                tool_call_tokens += n_tokens

        has_tools = tool_call_tokens > 0
        has_reasoning = reasoning_tokens > 0

        print(f"\n{'-' * 80}")
        print(f"Training tokens (assistant only):                    {assistant_tokens}")
        if has_tools:
            print(f"Training tokens (assistant + tool calls):            {assistant_tokens + tool_call_tokens}")
        if has_reasoning:
            print(f"Training tokens (assistant + reasoning):             {assistant_tokens + reasoning_tokens}")
        if has_tools or has_reasoning:
            print(f"Training tokens (assistant + tool calls + reasoning): {assistant_tokens + tool_call_tokens + reasoning_tokens}")
        print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
