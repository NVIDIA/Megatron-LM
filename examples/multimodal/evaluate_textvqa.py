import argparse
import glob
import json
import re

from evaluate_vqav2 import compute_vqa_accuracy


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    output_file_path = input_path + "-TextVQA-merged.json"

    pattern = input_path + "-TextVQA-[0-9].*jsonl"
    input_file_paths = glob.glob(pattern)

    results = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                results.append(
                    {
                        "question_id": res["sample_id"],
                        "answer": res["answer"],
                        "gt_answer": res["gt_answer"],
                    }
                )

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file)

    return output_file_path


def textvqa_eval(input_path):
    """Run TextVQA evaluation."""
    result_file_path = merge_input_files(input_path)
    compute_vqa_accuracy(result_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    textvqa_eval(args.input_path)
