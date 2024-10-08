import argparse
import glob
import json

from evaluate_vqav2 import compute_vqa_accuracy


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    output_file_path = input_path + "-ChartQA-merged.json"

    pattern = input_path + "-ChartQA-[0-9].*jsonl"
    input_file_paths = glob.glob(pattern)

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


def chartqa_eval(input_path):
    """Run ChartQA evaluation."""
    result_file_path = merge_input_files(input_path)
    compute_vqa_accuracy(result_file_path, use_chartqa_metric=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    chartqa_eval(args.input_path)
