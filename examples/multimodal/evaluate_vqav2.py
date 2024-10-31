import argparse
import json

from evaluate_mmmu import get_input_output_paths
from open_flamingo.eval.vqa_metric import VQAEval


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="VQAv2")

    results = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                res["question_id"] = res["sample_id"]

                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

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
        elif task == "AI2D":
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
