import argparse
import json

from evaluate_mmmu import get_input_output_paths
from evaluate_vqav2 import compute_vqa_accuracy


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="ChartQA")

    results = dict()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                sample_id = res["sample_id"]

                # Ignore possible duplicates.
                if sample_id in results:
                    continue

                res["question_id"] = sample_id
                results[sample_id] = res

    results = list(results.values())

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

    return output_file_path


def chartqa_eval(input_path):
    """Run ChartQA evaluation."""
    result_file_path = merge_input_files(input_path)
    return compute_vqa_accuracy(result_file_path, task="ChartQA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    avg_acc = chartqa_eval(args.input_path)

    print(f"ChartQA accuracy: {avg_acc:.2f}")
