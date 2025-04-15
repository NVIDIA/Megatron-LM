import argparse
import json
import subprocess
import nltk
nltk.download("wordnet")

from .evaluate_mmmu import get_input_output_paths


def convert_to_ocrbench_v2_format(input_path, groundtruth_path):
    """Convert input files to OCRBenchV2 compatible format."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, "OCRBench_v2")

    output = []

    with open(groundtruth_path) as f:
        gt = json.load(f)

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)

                out = gt[res["sample_id"]]
                out["predict"] = res["predict"]

                output.append(out)

    output = sorted(output, key=lambda x: x["id"])

    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file)

    return output_file_path


def ocrbench_v2_eval(input_path, groundtruth_path, output_path):
    """Run OCRBenchV2 evaluation."""
    result_file = convert_to_ocrbench_v2_format(input_path, groundtruth_path)

    # The OCRBenchV2 repo has scripts for running the actual evaluation
    output = subprocess.run(
        [
            "python",
            "examples/multimodal/MultimodalOCR/OCRBench_v2/eval_scripts/eval.py",
            "--output_path",
            output_path,
            "--input_path",
            result_file,
        ],
        capture_output=True,
        text=True,
    )
    print(output.stderr)
    print(output.stdout)

    output = subprocess.run(
        [
            "python",
            "examples/multimodal/MultimodalOCR/OCRBench_v2/eval_scripts/get_score.py",
            "--json_file",
            output_path,
        ],
        capture_output=True,
        text=True,
    )
    print(output.stderr)
    print(output.stdout)


def main():
    """Run OCRBenchV2 evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    parser.add_argument(
        "--groundtruth-path",
        type=str,
        required=True,
        help="Path to groundtruth file",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to dump outputs from the OCRBench V2 eval script",
    )
    args = parser.parse_args()

    ocrbench_v2_eval(args.input_path, args.groundtruth_path, args.output_path)


if __name__ == "__main__":
    main()
