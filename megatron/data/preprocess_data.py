import argparse
import json
import multiprocessing
import nltk
import sys
import time

import torch

from bert_tokenization import FullTokenizer
import indexed_dataset

class CustomLanguageVars(nltk.tokenize.punkt.PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = FullTokenizer(self.args.vocab, do_lower_case=True)
        spliter = nltk.load("tokenizers/punkt/english.pickle")
        if self.args.keep_newlines:
            # this prevents punkt from eating newlines after sentences
            Encoder.spliter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                train_text = spliter._params,
                lang_vars = CustomLanguageVars())
        else:
            Encoder.splitter = spliter

    def encode(self, json_line):
        text = json.loads(json_line)[self.args.json_key]
        doc_ids = []
        for sentence in Encoder.splitter.tokenize(text):
            tokens = Encoder.tokenizer.tokenize(sentence)
            ids = Encoder.tokenizer.convert_tokens_to_ids(tokens)
            doc_ids.append(ids)
        return doc_ids, len(json_line)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to input JSON')
    parser.add_argument('--vocab', type=str, help='Path to vocab.txt')
    parser.add_argument('--json-key', type=str, default='text',
                        help='Key to extract from json')
    parser.add_argument('--output-prefix', type=str, help='Path to binary output file without suffix')
    parser.add_argument('--workers', type=int, default=20,
                        help='Number of worker processes to launch')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='Interval between progress updates')
    parser.add_argument('--keep-newlines', action='store_true',
                        help='Keep newlines between sentences.')
    parser.add_argument('--dataset-impl', type=str, default='mmap',
                        choices=['lazy', 'cached', 'mmap'])
    args = parser.parse_args()
    args.keep_empty = False

    startup_start = time.time()

    print("Opening", args.input)
    fin = open(args.input, 'r', encoding='utf-8')

    nltk.download("punkt", quiet=True)

    encoder = Encoder(args)
    tokenizer = FullTokenizer(args.vocab, do_lower_case=True)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)
    encoded_docs = pool.imap(encoder.encode, fin, 25)

    print(f"Vocab size: {tokenizer.vocab_size()}")

    output_bin_file = "{}.bin".format(args.output_prefix)
    output_idx_file = "{}.idx".format(args.output_prefix)
    builder = indexed_dataset.make_builder(output_bin_file,
                                      impl=args.dataset_impl,
                                      vocab_size=tokenizer.vocab_size())

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0
    print("Time to startup:", startup_end - startup_start)
    for i, (doc, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        for sentence in doc:
            #print(sentence)
            #print(tokenizer.convert_ids_to_tokens(sentence))
            builder.add_item(torch.IntTensor(sentence))
        builder.end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    builder.finalize(output_idx_file)

if __name__ == '__main__':
    main()
