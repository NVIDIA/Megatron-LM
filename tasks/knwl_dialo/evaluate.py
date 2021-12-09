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

"""Model evaluation"""

from megatron import get_args
from megatron import print_rank_0
from tasks.knwl_dialo.metrics import F1Metric
from tqdm import tqdm


def evaluate_f1(guess_file, answer_file):
    """Evaluating F1 Score"""

    guess_list = []
    print_rank_0('reading %s' % guess_file)
    with open(guess_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if "<|endoftext|>" in line:
                line = line.replace("<|endoftext|>", "")
            guess_list.append(line)

    answer_list = []
    print_rank_0('reading %s' % answer_file)
    with open(answer_file, "r") as f:
        for i, line in enumerate(tqdm(f)):
            line = line.strip()
            if line == "no_passages_used":
                line = ""
            answer_list.append(line)

    assert len(guess_list) == len(answer_list), \
        "lengths of guess and answer are different!"

    precision, recall, f1 = F1Metric.compute_all_pairs(guess_list, answer_list)
    print_rank_0('Precision: %.4f; recall: %.4f; f1: %.4f' % (precision, recall, f1))

    print_rank_0('done :-)')


def main():
    args = get_args()
    
    evaluate_f1(args.guess_file, args.answer_file)

