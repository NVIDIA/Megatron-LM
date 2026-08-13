# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining using gigatoken."""

import argparse
import awkward as ak
import gigatoken as gt
import time

import multiprocessing
from multiprocessing import Pool

import orjson
from itertools import islice
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron.core.datasets import indexed_dataset
from megatron.core.tokenizers.utils.build_tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args


def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON line file')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--log-interval', type=int, required=False, default=10000,
                       help='Interval to log the progress')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.use_gigatoken = True
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def process_key(args, key, level):
    tokenizer = build_tokenizer(args)  # each process needs its own tokenizer

    encoded_docs = tokenizer.tokenize_files(args.input, key)

    bin_file = "{}_{}_{}.bin".format(args.output_prefix, key, level)
    idx_file = "{}_{}_{}.idx".format(args.output_prefix, key, level)

    builder = indexed_dataset.IndexedDatasetBuilder(
        bin_file,
        dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
    )
    for doc in encoded_docs:
        if args.append_eod:
            doc = ak.concatenate([doc, [tokenizer.eod]])
        builder.add_document(doc, [len(doc)])

    builder.finalize(idx_file)


def main():
    args = get_args()
    level = "document"

    ctx = multiprocessing.get_context('spawn')
    with ctx.Pool(processes=len(args.json_keys)) as pool:
        pool.starmap(process_key, [(args, key, level) for key in args.json_keys])


if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
