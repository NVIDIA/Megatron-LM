import json
import time
from typing import Dict, Optional, List

import joblib
from googleapiclient import discovery
from googleapiclient.errors import HttpError

import argparse

from tqdm import tqdm

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--out-path', type=str, default='',
                    help='data path to load the jsonl')
parser.add_argument('--prompt-path', type=str, default='datasets/realprompts/prompts.jsonl',
                    help='data path to load the prompt jsonl')
parser.add_argument('--workers', type=int, default=10,
                   help='Number of worker processes to launch')


def get_corpus_scores(lines):
    scores = []
    corpus = []

    for line in tqdm(lines):
        data = json.loads(line)
        score = data['score']
        if score is not None:
            scores.append(score['toxicity'])
            corpus.append(data)
    return scores, corpus


def main():
    args = parser.parse_args()

    with open(args.data_path) as f:
        lines = f.readlines()

    print(f"total line number: {len(lines)}")

    scores, corpus = get_corpus_scores(lines)
    import numpy as np
    scores = np.array(scores)
    indices = np.argsort(scores)

    print(f"total valid samples: {len(scores)}")

    print(f"Avg toxicity (all): {np.mean(scores)} +- {np.std(scores)}")
    print(f"Avg toxicity (toxic): {np.mean(scores[scores > 0.5])} +- {np.std(scores[scores > 0.5])}")
    print(f"Toxic Percentage {sum(scores > 0.5) / len(scores)}")
    print(f"Avg toxicity (nontoxic): {np.mean(scores[scores <= 0.5])} +- {np.std(scores[scores <= 0.5])}")
    print(f"Nontoxic Percentage {sum(scores <= 0.5) / len(scores)}")

    samples_left = len(lines) // 2
    print(f"After filtering: {samples_left} of samples are left")
    nontoxic_indices = indices[:samples_left]
    print(f"Avg toxicity (filtered): {np.mean(scores[nontoxic_indices])} +- {np.std(scores[nontoxic_indices])}")
    print(f"Toxicity Range (filtered): {np.min(scores[nontoxic_indices])} ~ {np.max(scores[nontoxic_indices])}")
    nontoxic_data = [corpus[ind] for ind in nontoxic_indices]
    print(f"Total samples after filtering: {len(nontoxic_data)}")
    print(f"Examples: {nontoxic_data[:3]}")

    from sklearn.utils import shuffle
    nontoxic_data = shuffle(nontoxic_data)

    with open(args.out_path, 'w') as f:
        for x in nontoxic_data:
            f.write(json.dumps(x) + '\n')


main()