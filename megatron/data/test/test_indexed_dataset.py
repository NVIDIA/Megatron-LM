# This file isn't really a formal automated test, it's just a place to
# put some code used during development and manual testing of
# indexed_dataset.

from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer
import argparse
import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))


def test_indexed_dataset(args):
    ds = indexed_dataset.MMapIndexedDataset(args.data)
    tokenizer = build_tokenizer(args)
    print(len(ds.doc_idx))
    print(len(ds))
    print(ds.doc_idx[-1])
    if ds.supports_prefetch:
        # just prefetch the whole thing in test (so assume it is small)
        ds.prefetch(range(len(ds)))
    if args.count > len(ds.doc_idx) - 1:
        args.count = len(ds.doc_idx) - 1

    for i in range(args.count):
        start = ds.doc_idx[i]
        end = ds.doc_idx[i + 1]
        ids = ds[start:end]
        print(f"Document {i}:")
        print("--------------")
        for s in ids:
            assert len(s) > 0
            l = s.data.tolist()
            text = tokenizer.detokenize(l)
            print(text)
            print("---")


def test_indexed_dataset_get(args):
    ds = indexed_dataset.MMapIndexedDataset(args.data)
    tokenizer = build_tokenizer(args)
    size = ds.sizes[0]
    print(f"size: {size}")
    full = ds.get(0)
    print(full)
    # print(tokenizer.detokenize(full.data.tolist()))
    print("---")
    end = ds.get(0, offset=size - 10)
    print(end)
    # print(tokenizer.detokenize(end.data.tolist()))

    start = ds.get(0, length=10)
    print(start)
    # print(tokenizer.detokenize(start.data.tolist()))

    part = ds.get(0, offset=2, length=8)
    print(part)
    # print(tokenizer.detokenize(part.data.tolist()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--count', type=int, default=10,
                        help='Number of samples/documents to print')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase',
                                'GPT2BPETokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')

    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs to plan for')
    parser.add_argument('--max-num-samples', type=int, default=None,
                        help='Maximum number of samples to plan for')
    parser.add_argument('--masked-lm-prob', type=float, default=0.15,
                        help='probability of masking tokens')
    parser.add_argument('--seq-length', type=int, default=512,
                        help='maximum sequence length')
    parser.add_argument('--short-seq-prob', type=float, default=0.1,
                        help='probability of creating a short sequence')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    args = parser.parse_args()
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1

    test_indexed_dataset_get(args)


if __name__ == "__main__":
    main()
