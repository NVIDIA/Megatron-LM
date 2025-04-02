# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing large data for pretraining."""
import argparse
import math
import json
import os
import sys
import glob
from pyspark.sql import SparkSession
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time
import gzip
import glob
import torch
import numpy as np
import multiprocessing

# Force 'spawn' method for all multiprocessing
try:
    multiprocessing.set_start_method('spawn')
    print(f"{time.strftime('%H:%M:%S', time.localtime())} Process - Using 'spawn' multiprocessing start method")
except RuntimeError:
    print(f"{time.strftime('%H:%M:%S', time.localtime())} Process - Multiprocessing start method already set to: {multiprocessing.get_start_method()}")

import functools
try:
    import nltk
    from nltk.tokenize.punkt import PunktLanguageVars
    nltk_available = True
except ImportError:
    PunktLanguageVars = object  # Fallback to the built-in object class
    nltk_available = False

from megatron.training.tokenizer import build_tokenizer
from megatron.training.arguments import _add_tokenizer_args
from megatron.core.datasets import indexed_dataset


def timing_decorator(func):
    """Decorator to measure and print the execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"{time.strftime('%H:%M:%S', time.localtime())} Process - {func.__name__} took {elapsed:.2f} seconds")
        return result
    return wrapper

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
        if self.args.split_sentences:
            if not nltk_available:
                print("NLTK is not available to split sentences.")
                exit()
            if os.environ.get("NLTK_DATA"):
                library = os.path.join(os.environ.get("NLTK_DATA"), "tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"file:{library}"
            else:
                library = os.path.join("tokenizers", "punkt", f"{self.args.lang}.pickle")
                url = f"nltk:{library}"
            splitter = nltk.load(url)
            if self.args.keep_newlines:
                # this prevents punkt from eating newlines after sentences
                Encoder.splitter = nltk.tokenize.punkt.PunktSentenceTokenizer(
                    train_text = splitter._params,
                    lang_vars = CustomLanguageVars())
            else:
                Encoder.splitter = splitter

        else:
            Encoder.splitter = IdentitySplitter()

    def split(self, json_line):
        data = json.loads(json_line)
        output = {}
        for key in self.args.json_keys:
            text = data[key]
            max_len = 1000000
            tokens_list = [Encoder.splitter.tokenize(text[i:i+max_len]) for i in range(0, len(text), max_len)]
            output[key] = [tokens for partial in tokens_list for tokens in partial]
        return json.dumps(output), len(json_line)

    def encode(self, json_line):
        data = json.loads(json_line)
        ids = {}
        lens = {}
        for key in self.args.json_keys:
            text = data[key]
            if isinstance(text, list):
                sentences = text
            else:
                sentences = [text]
            doc_ids = []
            sentence_lens = []
            for sentence in sentences:
                sentence_ids = Encoder.tokenizer.tokenize(sentence)
                if len(sentence_ids) > 0:
                    doc_ids.extend(sentence_ids)
                    sentence_lens.append(len(sentence_ids))
            if len(doc_ids) > 0 and self.args.append_eod:
                doc_ids.append(Encoder.tokenizer.eod)
                sentence_lens[-1] += 1
            ids[key] = doc_ids
            lens[key] = sentence_lens
        return ids, lens, len(json_line)


class Partition(object):
    @timing_decorator
    def __init__(self, args, workers):
        self.args = args
        self.workers = workers

    def print_processing_stats(self, count, proc_start, total_bytes_processed):
        if count % self.args.log_interval == 0:
            current = time.time()
            elapsed = current - proc_start
            mbs = total_bytes_processed/elapsed/1024/1024
            print(f"Processed {count} documents",
                  f"({count/elapsed} docs/s, {mbs} MB/s).",
                  file=sys.stderr)

    @timing_decorator
    def split_sentences(self, file_name):
        input_file_name, output_file_name = file_name
        print("Opening", input_file_name)
        fin = open(input_file_name, 'r', encoding='utf-8')
        fout = open(output_file_name, 'w')

        encoder = Encoder(self.args)
        pool = multiprocessing.Pool(self.workers, initializer=encoder.initializer)
        split_docs = pool.imap(encoder.split, fin, 32)

        proc_start = time.time()
        total_bytes_processed = 0
        for i, (doc, bytes_processed) in enumerate(split_docs, start=1):
            total_bytes_processed += bytes_processed
            fout.write(doc + "\n")
            self.print_processing_stats(i, proc_start, total_bytes_processed)

        fin.close()
        fout.close()

    def process_json_file(self, file_name):
        # Unpack input file name and output prefix.
        input_file_name, output_prefix = file_name
        print("Opening", input_file_name)
        
        file_open_start = time.time()
        n_workers = f"local[{self.args.workers}]" if hasattr(self.args, 'workers') and self.args.workers > 0 else "local[*]"
        
        # Create a SparkSession (or get the existing one).
        spark = SparkSession.builder \
            .master(n_workers) \
            .config("spark.driver.memory", "200g") \
            .config("spark.executor.memory", "200g") \
            .config("spark.executorEnv.PYTHONPATH", "/shared/all/nemo_workspace/Megatron_yuli/:$PYTHONPATH") \
            .config("spark.driverEnv.PYTHONPATH", "/shared/all/nemo_workspace/Megatron_yuli/:$PYTHONPATH") \
            .getOrCreate()
        sc = spark.sparkContext
        
        # Read the input file and repartition it to the desired number of partitions.
        # This ensures the output is split into exactly args.partitions parts.
        rdd = sc.textFile(input_file_name)
        
        file_open_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Opening file took {file_open_end - file_open_start:.2f} seconds")
        
        # Startup phase: initialize a dummy encoder (for driver) and obtain the tokenizer.
        startup_start = time.time()
        encoder_dummy = Encoder(self.args)  # for driver initialization only
        tokenizer = build_tokenizer(self.args)
        
        # Determine processing level.
        level = "document"
        if self.args.split_sentences:
            level = "sentence"
        
        # Define a function to process a partition and write out its results
        # in the same bin/idx format using the IndexedDatasetBuilder.
        def process_and_write_partition(index, iterator):
            # Each partition gets its own Encoder instance.
            local_encoder = Encoder(self.args)
            local_encoder.initializer()
            
            # Initialize an IndexedDatasetBuilder for each json key.
            local_builders = {}
            for key in self.args.json_keys:
                bin_file = "{}_{}_{}_{}.bin".format(output_prefix, key, level, index)
                # Note: We create a new builder for each key.
                local_builders[key] = indexed_dataset.IndexedDatasetBuilder(
                    bin_file,
                    dtype=indexed_dataset.DType.optimal_dtype(build_tokenizer(self.args).vocab_size)
                )
            
            # Process each line in the partition.
            for line in iterator:
                doc, sentence_lens, bytes_processed = local_encoder.encode(line)
                for key in doc.keys():
                    local_builders[key].add_document(doc[key], sentence_lens[key])
            
            # Finalize each builder to write out the corresponding idx file.
            for key in local_builders:
                idx_file = "{}_{}_{}_{}.idx".format(output_prefix, key, level, index)
                local_builders[key].finalize(idx_file)
            
            # Return an empty iterator
            return iter([])
        
        rdd.mapPartitionsWithIndex(process_and_write_partition).count()  # Just to trigger execution
        
        startup_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Partition processing and writing took {startup_end - file_open_end:.2f} seconds")
        
        # ---------------------------
        # Phase 2: Merge the per-partition output files into one final dataset.
        # ---------------------------
        
        # Now, for each JSON key, merge all partition outputs.
        for key in self.args.json_keys:
            # Final output names.
            output_full_prefix = "{}_{}_{}".format(output_prefix, key, level)
            output_bin_file = "{}.bin".format(output_full_prefix)
            output_idx_file = "{}.idx".format(output_full_prefix)
            # Initialize the final builder.
            builder = indexed_dataset.IndexedDatasetBuilder(
                output_bin_file,
                dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size)
            )
            # Find all partitions with the common output prefix and extract unique partition prefixes
            partitions = sorted(
                glob.glob(f"{output_prefix}_{key}_{level}_*.idx"),
                key=lambda x: int(x.split('_')[-1].split('.')[0])  # Sort by numeric partition index
            )
            print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Found {len(partitions)} partitions")
            for partition in partitions:
                # Extract the base name without extension
                partition_name = os.path.splitext(partition)[0]
                builder.add_index(partition_name)
            # Finalize the final builder to merge all indices and write the idx file.
            builder.finalize(output_idx_file)
        
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Merging partitions completed.")
        
        # ---------------------------
        # Phase 3: Clean up intermediate files
        # ---------------------------
        cleanup_start = time.time()
        deleted_count = 0
        
        # Delete all partition .idx files
        for idx_file in glob.glob(f"{output_prefix}_*_{level}_*.idx"):
            os.remove(idx_file)
            deleted_count += 1
            
        # Delete all partition .bin files 
        for bin_file in glob.glob(f"{output_prefix}_*_{level}_*.bin"):
            os.remove(bin_file)
            deleted_count += 1
        
        cleanup_end = time.time()
        print(f"{time.strftime('%H:%M:%S', time.localtime())} IN - Deleted {deleted_count} intermediate files in {cleanup_end - cleanup_start:.2f} seconds")

def get_args():
    parser = argparse.ArgumentParser()
    parser = _add_tokenizer_args(parser)
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')
    group.add_argument('--split-sentences', action='store_true',
                       help='Split documents into sentences.')
    group.add_argument('--keep-newlines', action='store_true',
                       help='Keep newlines between sentences when splitting.')
    group = parser.add_argument_group(title='tokenization process')
    group.add_argument('--append-eod', action='store_true',
                       help='Append an <eod> token to the end of a document.')
    group.add_argument('--lang', type=str, default='english',
                       help='Language to use for NLTK-powered sentence splitting.')
    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help=('Number of worker processes to launch.'
                             'A good default for fast pre-processing '
                             'is: (workers * partitions) = available CPU cores.'))
    group.add_argument('--partitions', type=int, default=1,
                        help='Number of file partitions')
    group.add_argument('--log-interval', type=int, default=1000,
                       help='Interval between progress updates')
    group.add_argument('--keep-sequential-samples', action='store_true',
                       help='Ensure ordering of samples in .jsonl files is '
                            'preserved when using partitions>1.')
    args = parser.parse_args()
    args.keep_empty = False

    if args.tokenizer_type.lower().startswith('bert') and not args.split_sentences:
        print("Are you sure you don't want to split sentences?")

    # some default/dummy values for the tokenizer
    args.rank = 1
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.vocab_extra_ids = 0

    return args


def get_file_name(args, file_id):
    """Constructs file names for input, sentence split, and output files based on file_id."""
    file_name, extension = os.path.splitext(args.input)
    input_file_name = file_name + "_" + str(file_id) + extension
    sentence_split_file = file_name + "_ss_" + str(file_id) + extension
    output_prefix = args.output_prefix + "_" + str(file_id)
    file_names = {
        'partition': input_file_name,
        'sentence_split': sentence_split_file,
        'output_prefix': output_prefix}
    return file_names


@timing_decorator
def check_files_exist(in_ss_out_names, key, num_partitions):
    """Checks if all partition files for a given output directory and key exist."""
    for i in range(num_partitions):
        if not os.path.exists(in_ss_out_names[i][key]):
            return False
    return True


def main():
    args = get_args()

    if args.split_sentences:
        if nltk_available:
            nltk.download("punkt", quiet=True, download_dir=os.environ.get("NLTK_DATA"))
        else:
            raise Exception(
                "nltk library required for sentence splitting is not available.")

    in_ss_out_names = []

    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Sentence splitting is {'enabled' if args.split_sentences else 'disabled'}")
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Number of partitions: {args.partitions}")

    file_name, extension = os.path.splitext(args.input)
    sentence_split_file = file_name + "_ss" + extension
    file_names = {
        'partition': args.input,
        'sentence_split': sentence_split_file,
        'output_prefix': args.output_prefix}
    in_ss_out_names.append(file_names)

    partition = Partition(args, args.workers)

    # check to see if paritions with split sentences already created
    split_sentences_present = check_files_exist(in_ss_out_names, 'sentence_split', args.partitions)

    # TODO: to resolve and test by fixing the error in splitter = nltk.load(url)
    # split sentences in partition files
    if args.split_sentences and not split_sentences_present:
        for name in in_ss_out_names:
            partition.split_sentences((name['partition'], name['sentence_split']))
        return

    # encode partition files in parallel
    input_key = 'sentence_split' if args.split_sentences else 'partition'
    
    process_json_start = time.time()
    # it applies to both single and multiple partitions
    for name in in_ss_out_names:
        partition.process_json_file((name[input_key], name['output_prefix']))

    process_json_end = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Process - Process json took {process_json_end - process_json_start:.2f} seconds")

    if args.partitions == 1:
        return

    # merge bin/idx partitions
    level = "document"
    if args.split_sentences:
        level = "sentence"

    output_bin_files = {}
    output_idx_files = {}
    builders = {}
    tokenizer_start = time.time()
    tokenizer = build_tokenizer(args)
    tokenizer_end = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())}  Process - Building tokenizer took {tokenizer_end - tokenizer_start:.2f} seconds")

    merge_start = time.time()
    # the final file is output_prefix-key specific
    for key in args.json_keys:
        output_bin_files[key] = "{}_{}_{}.bin".format(args.output_prefix,
                                                      key, level)
        output_idx_files[key] = "{}_{}_{}.idx".format(args.output_prefix,
                                                      key, level)
        # initialize a builder for each key
        builders[key] = indexed_dataset.IndexedDatasetBuilder(
            output_bin_files[key],
            dtype=indexed_dataset.DType.optimal_dtype(tokenizer.vocab_size),
        )

        # add all output rows from all partitions to the builder
        for name in in_ss_out_names:
            parition_output_prefix = name['output_prefix']
            full_partition_output_prefix = "{}_{}_{}".format(parition_output_prefix,
                                                             key, level)
            builders[key].add_index(full_partition_output_prefix)
        # close all output file handlers, write out the accumulated data to disk
        builders[key].finalize(output_idx_files[key])
    merge_end = time.time()
    print(f"{time.strftime('%H:%M:%S', time.localtime())} Process - Merging partitions took {merge_end - merge_start:.2f} seconds")


if __name__ == '__main__':

    main()

