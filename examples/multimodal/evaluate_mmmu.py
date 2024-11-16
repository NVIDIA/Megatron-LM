import argparse
import glob
import json
import os
import re
import subprocess

from run_text_generation import get_output_path
from config import EvaluationConfig


def get_input_output_paths(input_path, task):
    """Get all input files and an output path for a merged file."""
    # Single input file.
    if os.path.exists(input_path):
        input_file_paths = [input_path]
        output_file_path = input_path.replace(".jsonl", "-merged.json")
    # Select multiple partitions and dp ranks.
    else:
        cfg = EvaluationConfig(task=task, output_path=input_path, partition_id="*")
        pattern = get_output_path(cfg, dp_rank="*")
        input_file_paths = glob.glob(pattern)

        output_file_path = input_path + f"-{task}-merged.json"

    return input_file_paths, output_file_path


def convert_to_mmmu_format(input_path):
    """Convert input files to MMMU compatible format."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, "MMMU")

    output = dict()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)

                sample_id = res["sample_id"]
                prediction = res["prediction"]

                if res["question_type"] == "multiple-choice":
                    from MMMU.mmmu.utils.eval_utils import parse_multi_choice_response

                    prediction = parse_multi_choice_response(
                        prediction, res["all_choices"], res["index2ans"]
                    )

                # MMMU eval script expects just a sample_id to prediction mapping.
                output[sample_id] = prediction

    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file)

    return output_file_path


def mmmu_eval(input_path, groundtruth_path):
    """Run MMMU evaluation."""
    result_file = convert_to_mmmu_format(input_path)

    # The MMMU repo has a script for running the actual evaluation but no API. So launching the script here.
    output = subprocess.run(
        [
            "python",
            "examples/multimodal/MMMU/mmmu/main_eval_only.py",
            "--output_path",
            result_file,
            "--answer_path",
            groundtruth_path,
        ],
        capture_output=True,
        text=True,
    )

    print(output.stderr)
    print(output.stdout)

    m = re.search("'Overall': {'num': \d+, 'acc': (\d.\d+)}", output.stdout)

    return float(m.group(1)) * 100.0


def main():
    """Run MMMU evaluation."""
    # Using the validation groundtruth file from the MMMU repo by default. This assumes you have cloned the MMMU github repo here.
    default_groundtruth_path = "examples/multimodal/MMMU/mmmu/answer_dict_val.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    parser.add_argument(
        "--groundtruth-path",
        type=str,
        default=default_groundtruth_path,
        help="Path to groundtruth file. Defaults to the validation file in the MMMU repo.",
    )
    args = parser.parse_args()

    avg_acc = mmmu_eval(args.input_path, args.groundtruth_path)

    print(f"MMMU average accuracy: {avg_acc:.2f}")


if __name__ == "__main__":
    main()
