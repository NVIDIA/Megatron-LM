import argparse
import json
from typing import List

from .evaluate_mmmu import get_input_output_paths
from open_flamingo.eval.vqa_metric import VQAEval

# ANLS score calculation based on https://github.com/shunk031/ANLS/blob/6472e1d71e84d6cee28e3c6d2e18564bafaa312d/anls/metrics/dist.py#L1
# and https://github.com/shunk031/ANLS/blob/6472e1d71e84d6cee28e3c6d2e18564bafaa312d/anls/metrics/score.py#L6
# MIT License. Copyright (c) 2022 Shunsuke KITADA
def levenshtein_distance(s1: str, s2: str) -> int:

    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = list(range(len(s1) + 1))
    for i2, c2 in enumerate(s2):
        dists = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                dists.append(distances[i1])
            else:
                dists.append(1 + min((distances[i1], distances[i1 + 1], dists[-1])))
        distances = dists

    return distances[-1]


def normalized_levenshtein_distance(s1: str, s2: str) -> float:
    dist = levenshtein_distance(s1, s2)
    length = max(len(s1.upper()), len(s2.upper()))
    return 0.0 if length == 0 else dist / length

def similarity_function(prediction: str, gold_label: str, threshold: float) -> float:
    nl_score = normalized_levenshtein_distance(prediction, gold_label)
    return 1 - nl_score if nl_score < threshold else 0.0

def anls_score(
    prediction: str, gold_labels: List[str], threshold: float = 0.5
) -> float:

    # not case sensitive, but space sensitive
    y_pred = " ".join(prediction.strip().lower().split())

    anls_scores: List[float] = []
    for gold_label in gold_labels:

        # not case sensitive, but space sensitive
        y_true = " ".join(gold_label.strip().lower().split())

        anls_score = similarity_function(y_pred, y_true, threshold)
        anls_scores.append(anls_score)

    score = max(anls_scores)

    return score

def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="VQAv2")

    results = dict()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                sample_id = res["sample_id"]

                # Skip possible duplicates.
                if sample_id in results:
                    continue

                res["question_id"] = sample_id
                results[sample_id] = res

    results = list(results.values())

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4, sort_keys=True)

    return output_file_path


def is_number(n: str):
    """Check if input is a number."""
    try:
        float(n)
        return True
    except ValueError:
        return False


def compute_vqa_accuracy(result_file, task):
    """Compute VQA accuracy."""
    merged_results = json.load(open(result_file))

    vqa = VQAEval(vqa=None, vqaRes=None)
    all_acc = []
    for res in merged_results:
        pred = res["answer"]
        pred = vqa.processPunctuation(pred)
        pred = vqa.processDigitArticle(pred)

        gt = res["gt_answer"]
        gt = [vqa.processPunctuation(ans) for ans in gt]
        gt = [vqa.processDigitArticle(ans) for ans in gt]

        # ChartQA uses relaxed accuracy:
        # "We consider an answer to be correct if it is within 5% of the gold answer.
        #  For non-numeric answers, we still need an exact match to consider an answer to be correct."
        if task == "ChartQA":
            acc = 0.0
            assert len(gt) == 1, "expected exactly one groundtruth answer."
            gt = gt[0]

            pred = pred.rstrip("%")
            gt = gt.rstrip("%")

            if is_number(pred) and is_number(gt):
                pred = float(pred)
                gt = float(gt)
                if pred >= (gt * 0.95) and pred <= (gt * 1.05):
                    acc = 1.0
            elif pred == gt:
                acc = 1.0

            all_acc.append(acc)
        elif task in ("VQAv2", "TextVQA"):
            num_match = sum([pred == ans for ans in gt])
            acc = min(1.0, num_match / 3.0)
            all_acc.append(acc)
        elif task in ("SPDocVQA", "InfoVQA"):
            acc = anls_score(prediction=pred, gold_labels=gt, threshold=0.5)
            all_acc.append(acc)
        elif task in ("AI2D", "RealworldQA", "MotionBench"):
            assert len(gt) == 1, f"Expected exactly 1 GT, got {gt}"
            acc = pred == gt[0]
            all_acc.append(acc)
        else:
            raise NotImplementedError(f"unknown task {task}")

    acc_avg = sum(all_acc) / len(all_acc) * 100

    return acc_avg


def vqav2_eval(input_path):
    """Run VQAv2 evaluation."""
    result_file = merge_input_files(input_path)
    avg_acc = compute_vqa_accuracy(result_file, task="VQAv2")
    return avg_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    avg_acc = vqav2_eval(args.input_path)

    print(f"===== VQAv2 Accuracy {avg_acc:.2f}% =====")
