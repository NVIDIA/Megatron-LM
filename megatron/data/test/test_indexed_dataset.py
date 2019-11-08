import argparse
import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))

from megatron.data import indexed_dataset, FullBertTokenizer

def test_indexed_dataset(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = FullBertTokenizer(args.vocab, do_lower_case=True)
    print(len(ds.doc_idx))
    print(len(ds))
    print(ds.doc_idx[-1])
    if ds.supports_prefetch:
        # just prefetch the whole thing in test (so assume it is small)
        ds.prefetch(range(len(ds)))
    for i in range(1):
        start = ds.doc_idx[i]
        end = ds.doc_idx[i+1]
        print(start, end)
        for j in range(start, end):
            ids = ds[j].data.tolist()
            print(ids)
            tokens = tokenizer.convert_ids_to_tokens(ids)
            print(tokens)
        print("******** END DOCUMENT **********")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--vocab', type=str, help='Path to vocab.txt')
    parser.add_argument('--dataset-impl', type=str, default='infer',
                        choices=['lazy', 'cached', 'mmap', 'infer'])
    args = parser.parse_args()

    if args.dataset_impl == "infer":
        args.dataset_impl = indexed_dataset.infer_dataset_impl(args.data)

    test_indexed_dataset(args)

if __name__ == "__main__":
    main()
