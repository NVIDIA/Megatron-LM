import argparse
import json

from .evaluate_vqav2 import compute_vqa_accuracy
from .evaluate_mmmu import get_input_output_paths


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, task="RealworldQA")

    results = []
    collected = set()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                res["question_id"] = res["sample_id"]
                if res['sample_id'] in collected:
                    continue
                collected.add(res['sample_id'])

                results.append(res)

    with open(output_file_path, "w") as output_file:
        json.dump(results, output_file, indent=4, sort_keys=True)

    return output_file_path


def realworldqa_eval(input_path):
    """Run RealWorldQA evaluation."""
    result_file_path = merge_input_files(input_path)
    return compute_vqa_accuracy(result_file_path, task="RealworldQA")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    args = parser.parse_args()

    avg_acc = realworldqa_eval(args.input_path)

    print(f"RealworldQA accuracy: {avg_acc:.2f}")
