import argparse
import glob
import json
import os
import re
import subprocess
import sys
import numpy as np

from .evaluate_mmmu import get_input_output_paths

# The rd-tablebench repo has functions for grading table predictions.
# Get the absolute path of the rd-tablebench repo
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'rd-tablebench'))
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

from grading import table_similarity
from convert import html_to_numpy


def convert_to_rdtablebench_format(input_path):
    """Convert input files to RDTableBench compatible format."""
    input_file_paths, output_file_path = get_input_output_paths(input_path, "RD_TableBench")

    output = []

    for input_file_path in input_file_paths:
        with open(input_file_path, "r") as input_file:
            for line in input_file:
                res = json.loads(line)
                output.append(res)

    output = sorted(output, key=lambda x: x["sample_id"])

    with open(output_file_path, "w") as output_file:
        json.dump(output, output_file)

    return output_file_path


def rdtablebench_eval(input_path):
    """Run RD-TableBench evaluation."""
    result_file = convert_to_rdtablebench_format(input_path)

    with open(result_file) as f:
        data = json.load(f)

    similarities = []
    num_failed = 0
    for sample in data:
        pred = sample["predict"]
        target = sample["ground_truth"]
        target_np = html_to_numpy(target)
        try:
            pred_np = html_to_numpy(pred)
            similarity = table_similarity(target_np, pred_np)
        except Exception as e:
            print("Failed to grade table: ", e)
            similarity = 0
            num_failed += 1
        similarities.append(similarity)

    print(f"Accuracy: {np.mean(similarities)}")
    print(f"Failed: {num_failed}")

def main():
    """Run RD-TableBench evaluation."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, required=True, help="Path to input file(s)")
    args = parser.parse_args()

    rdtablebench_eval(args.input_path)


if __name__ == "__main__":
    main()
