import argparse
import glob
import json
import subprocess


def convert_to_mmmu_format(input_path):
    """Convert input files to MMMU compatible format."""
    output_file_path = input_path + "-MMMU-merged.json"

    pattern = input_path + "-MMMU-[0-9].*jsonl"
    input_file_paths = glob.glob(pattern)

    output = dict()

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)

                sample_id = res["sample_id"]
                prediction = res["prediction"]

                output[sample_id] = prediction

    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file)

    return output_file_path


def main():
    # Using the validation groundtruth file from the MMMU repo by default. This assumes you have cloned the MMMU github repo here.
    default_groundtruth_path = "examples/multimodal/MMMU/eval/answer_dict_val.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    parser.add_argument(
        "--groundtruth-path",
        type=str,
        default=default_groundtruth_path,
        help="Path to groundtruth file. Defaults to the validation file in the MMMU repo.",
    )
    args = parser.parse_args()

    result_file = convert_to_mmmu_format(args.input_path)

    # The MMMU repo has a script for running the actual evaluation but no API. So launching the script here.
    output = subprocess.run(
        [
            "python",
            "examples/multimodal/MMMU/eval/main_eval_only.py",
            "--output_path",
            result_file,
            "--answer_path",
            default_groundtruth_path,
        ],
        capture_output=True,
        text=True,
    )

    print(output.stdout)


if __name__ == "__main__":
    main()
