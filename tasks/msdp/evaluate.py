# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Model evaluation"""

from megatron.training import get_args
from megatron.training import print_rank_0
from tasks.msdp.metrics import F1Metric
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

