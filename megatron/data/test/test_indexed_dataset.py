import argparse
import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))

from megatron.data import indexed_dataset, FullBertTokenizer

def test_indexed_dataset(args):
    ds_impl = indexed_dataset.infer_dataset_impl(args.data)
    ds = indexed_dataset.make_dataset(args.data, ds_impl)
    tokenizer = FullBertTokenizer(args.vocab, do_lower_case=True)
    for sample in ds:
        print(sample)
        print(sample.data.tolist())
        print(tokenizer.convert_ids_to_tokens(sample.data.tolist()))
        print("---")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--vocab', type=str, help='Path to vocab.txt')
    args = parser.parse_args()

    test_indexed_dataset(args)

if __name__ == "__main__":
    main()
