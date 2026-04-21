# coding=utf-8
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Processing text modality data for MultiModal pretraining."""

import argparse
import json
import multiprocessing
import os
import sys
import numpy as np
from torchvision.transforms import ToTensor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    from nltk.tokenize.punkt import PunktLanguageVars
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class

from megatron.training.tokenizer import build_tokenizer
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder


# https://stackoverflow.com/questions/33139531/preserve-empty-lines-with-nltks-punkt-tokenizer
class CustomLanguageVars(PunktLanguageVars):

    _period_context_fmt = r"""
        \S*                          # some word material
        %(SentEndChars)s             # a potential sentence ending
        \s*                       #  <-- THIS is what I changed
        (?=(?P<after_tok>
            %(NonWord)s              # either other punctuation
            |
            (?P<next_tok>\S+)     #  <-- Normally you would have \s+ here
        ))"""

class IdentitySplitter(object):
    def tokenize(self, *text):
        return text

class Encoder(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        # Use Encoder class as a container for global data
        Encoder.tokenizer = build_tokenizer(self.args)

    def encode(self, input_pair):
        json_line, img_path = input_pair
        data = json.loads(json_line)
        key = "text"
        text = data[key]
        sentence_ids = Encoder.tokenizer.tokenize(text)
        pad_len = self.args.pad_length
        if len(sentence_ids) > 0 and self.args.append_eod:
            sentence_ids = sentence_ids[:pad_len]
            current_length = len(sentence_ids)
            sentence_ids.extend([Encoder.tokenizer.eod for _ in range(max(0,pad_len-current_length))])

        with open(img_path, "rb") as tf:
            xs = bytearray(tf.read())
            img_pad = (4 - len(xs) % 4) % 4
            xs.extend([0 for _ in range(img_pad)])
            img_raw = np.frombuffer(xs, dtype=np.int32)
            img_raw = np.insert(img_raw, 0, img_pad)
        
        return sentence_ids, img_raw, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--input-image', type=str, required=True,
                       help='Path to input image folder')

    group.add_argument('--pad-length', type=int, required=True,
                       help='Pad length of preprocessed text')

    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')

    group = parser.add_argument_group(title='tokenizer')
    group.add_argument('--tokenizer-type', type=str, required=True,
                       choices=['BertWordPieceLowerCase','BertWordPieceCase',
                                'GPT2BPETokenizer', 'SentencePieceTokenizer', 'GPTSentencePieceTokenizer'],
                       help='What type of tokenizer to use.')
    group.add_argument('--vocab-file', type=str, default=None,
                       help='Path to the vocab file')
    group.add_argument('--merge-file', type=str, default=None,
                       help='Path to the BPE merge file (if necessary).')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group.add_argument('--tokenizer-model', type=str, default=None,
                       help='sentencepeice tokenizer model.')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    # some default/dummy values for the tokenizer
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args

def main():
    args = get_args()
    startup_start = time.time()

    encoder = Encoder(args)
    tokenizer = build_tokenizer(args)
    pool = multiprocessing.Pool(args.workers, initializer=encoder.initializer)

    fin = open(args.input, 'r', encoding='utf-8')
    img_paths = [os.path.join(args.input_image, basename) for basename in os.listdir(args.input_image)]

    encoded_docs = pool.imap(encoder.encode, zip(fin, img_paths), 25)

    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Output prefix: {args.output_prefix}")
    
    output_bin_files = "{}.bin".format(args.output_prefix)
    output_idx_files = "{}.idx".format(args.output_prefix)

    builders = IndexedDatasetBuilder(output_bin_files, dtype=np.int32, multimodal=True)

    startup_end = time.time()
    proc_start = time.time()
    total_bytes_processed = 0

    print("Time to startup:", startup_end - startup_start)
    
    for i, (sentence, img_raw, bytes_processed) in enumerate(encoded_docs, start=1):
        total_bytes_processed += bytes_processed
        builders.add_item(torch.IntTensor(sentence))
        builders.add_item(torch.from_numpy(img_raw), 1)
        builders.end_document()
        if i % args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {i} documents",
                  f"({i/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)
    
    builders.finalize(output_idx_files)


if __name__ == '__main__':
    main()

