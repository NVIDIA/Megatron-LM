import argparse
import os
import sys

import torch

script_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(script_dir, "../../../"))

from megatron.data import indexed_dataset, FullBertTokenizer, AlbertDataset

def test_indexed_dataset(args):
    ds = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    tokenizer = FullBertTokenizer(args.vocab, do_lower_case=True)
    print(len(ds.doc_idx))
    print(len(ds))
    print(ds.doc_idx[-1])
    if ds.supports_prefetch:
        # just prefetch the whole thing in test (so assume it is small)
        ds.prefetch(range(len(ds)))
    for i in range(len(ds.doc_idx)-1):
        start = ds.doc_idx[i]
        end = ds.doc_idx[i+1]
        ids = ds[start:end]
        for s in ids:
            assert len(s) > 0
            l = s.data.tolist()
            tokens = tokenizer.convert_ids_to_tokens(l)
            for t in tokens:
                if '\n' in t:
                    print("Newline in string!")
        print(i)

def test_albert_dataset(args):
    # tokenizer = FullBertTokenizer(args.vocab, do_lower_case=True)
    # idataset = indexed_dataset.make_dataset(args.data, args.dataset_impl)
    # ds = AlbertDataset(idataset, tokenizer)
    ds = AlbertDataset.from_paths(args.vocab, args.data, args.dataset_impl,
                                  args.epochs, args.max_num_samples,
                                  args.masked_lm_prob, args.seq_length,
                                  args.short_seq_prob, args.seed)
    truncated = 0
    total = 0
    for s in ds:
        ids = s['text']
        tokens = ds.tokenizer.convert_ids_to_tokens(ids)
        print(tokens)
        exit()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='prefix to data files')
    parser.add_argument('--vocab', type=str, help='Path to vocab.txt')
    parser.add_argument('--dataset-impl', type=str, default='infer',
                        choices=['lazy', 'cached', 'mmap', 'infer'])
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

    if args.dataset_impl == "infer":
        args.dataset_impl = indexed_dataset.infer_dataset_impl(args.data)

    test_albert_dataset(args)
#    test_indexed_dataset(args)

if __name__ == "__main__":
    main()
