# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Processing data for multimodal pretraining."""
import gc
import argparse
import json
import multiprocessing
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

import torch
try:
    import nltk
    nltk_available = True
except ImportError:
    nltk_available = False

from megatron.tokenizer import build_tokenizer
from megatron.data import indexed_dataset
from megatron.data.indexed_dataset import MMapIndexedDatasetBuilder


def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input Tensor')
    group.add_argument('--input-bs', type=int, required=True,
                       help='Image tensor loading batch size')
    group.add_argument('--start', type=int, required=True,
                       help='Start of input tensor split index')
    group.add_argument('--end', type=int, required=True,
                       help='End of input tensor split index')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

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

    output_bin_files = "{}_img.bin".format(args.output_prefix,
                                                      key)
    output_idx_files = "{}_img.idx".format(args.output_prefix,
                                                      key)
    builders = MMapIndexedDatasetBuilder(output_bin_files, dtype=np.float32)

    proc_start = time.time()
    total_bytes_processed = 0
    
    for i in range(args.start, args.end):
        img_tensor = np.load(args.input + "_%d.npy" % (i))
        N = img_tensor.shape[0]    
        img_tensor = img_tensor.reshape(N, -1)
        startup_end = time.time()
        print("Time to Load image tensor:", startup_end - startup_start)
        
        bs = args.input_bs
        for j in range(ceil(N / bs)):
            builders.add_batched_item(img_tensor[j*bs:min((j+1)*bs, N)])
            current = time.time()
            elapsed = current - proc_start
            print(elapsed)

        del img_tensor
        gc.collect()

    builders.finalize(output_idx_files)

if __name__ == '__main__':
    main()
