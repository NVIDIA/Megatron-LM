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
#
# Author:  bugface (https://github.com/bugface)
"""A tool to Merge several .bin/.idx files separately generated using the same vocab and tokenizer"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
from megatron.data import indexed_dataset
from megatron.tokenizer import build_tokenizer
import argparse
from pathlib import Path


def main(args):
    args.rank = 0
    args.make_vocab_size_divisible_by = 128
    args.tensor_model_parallel_size = 1
    args.merge_file = None

    pin = Path(args.input)
    data_path_prefix = [str(each)[:-4] for each in pin.glob("*.bin")]

    pout = Path(args.output)
    pout.mkdir(parents=True, exist_ok=True)
    output_bin_files = pout / f"{args.output_prefix}.bin"
    output_idx_files = pout / f"{args.output_prefix}.idx"
    try:
        os.remove(output_bin_files)
        os.remove(output_idx_files)
    except:
        pass
    
    tokenizer = build_tokenizer(args)

    builders = indexed_dataset.make_builder(output_bin_files,  impl='mmap', vocab_size=tokenizer.vocab_size)
    for each in data_path_prefix:
        builders.merge_file_(each)

    builders.finalize(output_idx_files) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True,
                        help="input dir where multiple .bin and .idx files are located")
    parser.add_argument("--output", type=str, required=True,
                        help="output dir where the merged .bin and .idx files stored")
    parser.add_argument("--output_prefix", type=str, required=True,
                        help="the filename for the output bin and idx files")
    parser.add_argument("--vocab_file", type=str, required=True,
                        help="the original vocab file used to generate the bin files")
    parser.add_argument("--tokenizer_type", type=str, default="BertWordPieceCase",
                        help="the original tokenizer used to generate the bin files")
    global_args = parser.parse_args()
    main(global_args)