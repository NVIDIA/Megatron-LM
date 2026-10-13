#!/usr/bin/env python3
"""
Compare tokenization throughput of Megatron-Core's MegatronTokenizer vs.
`tokenizers` v1 (the new Rust-rewritten `Tokenizer` class) on a .jsonl file.

Each line of the input file is a JSON object with a "text" field.

Usage example:
    python compare_tokenizers.py \
        --jsonl-file data.jsonl \
        --megatron-tokenizer-path /path/to/megatron/tokenizer \
        --tokenizers-v1-json /path/to/tokenizer.json \
        --limit 100000
"""

import argparse
import json
import sys
import time


def load_texts(jsonl_path: str, limit: int | None = None) -> list[str]:
    texts = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["text"])
    return texts


def build_megatron_tokenizer(args):
    from megatron.core.tokenizers import MegatronTokenizer

    return MegatronTokenizer.from_pretrained(
        tokenizer_path=args.megatron_tokenizer_path,
        metadata_path={"library": "huggingface"},
        use_gigatoken=True,
    )


def build_tokenizers_v1(args):
    from tokenizers import Tokenizer

    if args.tokenizers_v1_pretrained:
        return Tokenizer.from_pretrained(args.tokenizers_v1_pretrained)
    return Tokenizer.from_pretrained(args.tokenizers_v1_json)


def time_megatron(tokenizer, texts: list[str]) -> tuple[float, int]:
    total_tokens = 0
    start = time.perf_counter()
    batch = True
    if batch:
        ids = tokenizer._tokenizer.tokenizer.tokenizer.encode_batch(texts)
        total_tokens += len(ids)
    for text in texts:
        ids = tokenizer.tokenize(text)
        total_tokens += len(ids)
    elapsed = time.perf_counter() - start
    return elapsed, total_tokens


def time_tokenizers_v1(tokenizer, texts: list[str], batch: bool) -> tuple[float, int]:
    total_tokens = 0
    import os
    os.environ["RAYON_NUM_THREADS"] = "128"
    start = time.perf_counter()
    if batch:
        encodings = tokenizer.encode_batch(texts)
        total_tokens = sum(len(e.ids) for e in encodings)
    else:
        for text in texts:
            enc = tokenizer.encode(text)
            total_tokens += len(enc.ids)
    elapsed = time.perf_counter() - start
    return elapsed, total_tokens


def report(name: str, elapsed: float, n_docs: int, n_tokens: int) -> None:
    print(f"\n== {name} ==")
    print(f"  docs:          {n_docs}")
    print(f"  tokens:        {n_tokens}")
    print(f"  elapsed:       {elapsed:.4f}s")
    print(f"  docs/sec:      {n_docs / elapsed:,.1f}")
    print(f"  tokens/sec:    {n_tokens / elapsed:,.1f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--jsonl-file", required=True, help="Path to input .jsonl file with a 'text' field per line")
    parser.add_argument("--limit", type=int, default=None, help="Only read the first N lines")

    # megatron.core.tokenizers.MegatronTokenizer config
    parser.add_argument(
        "--megatron-tokenizer-path",
        required=False,
        help="Path passed to MegatronTokenizer.from_pretrained(tokenizer_path=...). "
        "Directory (or file) must contain/sit next to tokenizer_metadata.json, "
        "unless --megatron-metadata-path is given.",
    )
    parser.add_argument(
        "--megatron-metadata-path",
        default=None,
        help="Optional explicit path to tokenizer_metadata.json (or a dict-like JSON string). "
        "If omitted, it's auto-derived from --megatron-tokenizer-path.",
    )

    # tokenizers v1 config
    parser.add_argument("--tokenizers-v1-json", default=None, help="Path to tokenizer.json")
    parser.add_argument("--tokenizers-v1-pretrained", default=None, help="Hub id, alternative to --tokenizers-v1-json")
    parser.add_argument("--tokenizers-v1-batch", action="store_true", help="Use encode_batch instead of per-line encode")

    parser.add_argument("--skip-megatron", action="store_true", help="Only benchmark tokenizers v1")
    parser.add_argument("--skip-tokenizers-v1", action="store_true", help="Only benchmark Megatron")

    args = parser.parse_args()

    print(f"Loading texts from {args.jsonl_file} ...")
    texts = load_texts(args.jsonl_file, args.limit)
    print(f"Loaded {len(texts)} documents.")

    if not args.skip_megatron:
        if not args.megatron_tokenizer_path:
            print("Skipping Megatron: pass --megatron-tokenizer-path", file=sys.stderr)
        else:
            print("\nBuilding MegatronTokenizer ...")
            megatron_tok = build_megatron_tokenizer(args)
            elapsed, n_tokens = time_megatron(megatron_tok, texts)
            report("MegatronTokenizer", elapsed, len(texts), n_tokens)

    if not args.skip_tokenizers_v1:
        if not args.tokenizers_v1_json and not args.tokenizers_v1_pretrained:
            print("Skipping tokenizers v1: pass --tokenizers-v1-json or --tokenizers-v1-pretrained", file=sys.stderr)
        else:
            print("\nBuilding tokenizers v1 Tokenizer ...")
            tok_v1 = build_tokenizers_v1(args)
            elapsed, n_tokens = time_tokenizers_v1(tok_v1, texts, batch=args.tokenizers_v1_batch)
            report("tokenizers v1", elapsed, len(texts), n_tokens)


if __name__ == "__main__":
    main()
