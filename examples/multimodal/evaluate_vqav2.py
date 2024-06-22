import argparse
import glob
import json

from open_flamingo.eval.vqa_metric import compute_vqa_accuracy


def merge_input_files(input_path):
    """Merge input files to a format compatible with the evaluator."""
    output_file_path = input_path + "-VQAv2-merged.json"

    pattern = input_path + "-VQAv2-[0-9].*jsonl"
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, help="Path to input file(s)")
    parser.add_argument('--groundtruth-path', type=str, help="Path to groundtruth file")
    parser.add_argument('--question-path', type=str, help="Path to questions file")
    args = parser.parse_args()

    result_file = merge_input_files(args.input_path)

    accuracy = compute_vqa_accuracy(result_file, args.question_path, args.groundtruth_path)
    print(accuracy)
