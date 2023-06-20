# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Processing visual modality data for MultiModal pretraining."""

import gc
import argparse
import json
import multiprocessing
import os
import sys
import glob
from PIL import Image
from torchvision.transforms import ToTensor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input tensor files')

    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')
    group.add_argument('--dataset-impl', type=str, default='mmap',
                       choices=['lazy', 'cached', 'mmap'])

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, default=1,
                       help='Number of worker processes to launch')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()
    args.keep_empty = False

    return args

def main():
    args = get_args()
    startup_start = time.time()

    import numpy as np

    key="img"
    output_bin_files = "{}_{}.bin".format(args.output_prefix, key)
    output_idx_files = "{}_{}.idx".format(args.output_prefix, key)

    builders = MMapIndexedDatasetBuilder(output_bin_files, dtype=np.uint8)

    proc_start = time.time()
    total_bytes_processed = 0

    img_files = open(args.input)

    count = 0
    for img_file in img_files:
        count += 1
        with open(img_file[:-1], "rb") as tf:
            img_raw = np.frombuffer(tf.read(), dtype=np.uint8)
        startup_end = time.time()
        if count % 1000 == 0:
            print("Time to process %d samples:" % (count), startup_end - startup_start)
        builders.add_item(ToTensor(img_raw))
        builders.end_document()

    builders.finalize(output_idx_files)

if __name__ == '__main__':
    main()
