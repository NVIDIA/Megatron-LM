"""
python3 scripts/tools/create_data_mixture.py --folders /capstor/store/cscs/swissai/a06/datasets_tokenized/fineweb-2 /capstor/store/cscs/swissai/a06/datasets_tokenized/fineweb-edu --weights 0.1 0.9 --output datasets/mixtures/my-first-data-mixture
"""

import argparse
import os
import random

from pathlib import Path
from typing import List

import numpy as np

SEED = 1234

def compute_current_sizes(dataset_sample_index, dataset_files):
    current_files = []
    for samples, dataset in zip(dataset_sample_index, dataset_files):
        current_files.append(dataset[:samples])
    return [sum([os.path.getsize(file) for file in dataset]) for dataset in current_files]

def build_dataset_mixture(weights: List[float], dataset_files: List[int]) -> List:
    """
    Given multiple datasets and weights, build a data mixture such that it follows those weights
    """
    weights = np.array(weights)
    dataset_sizes = [len(dataset) for dataset in dataset_files]
    total_number_of_files = sum(dataset_sizes)

    # Initialize buffer for number of samples used for each dataset
    current_samples = [0 for _ in range(len(weights))]
    current_weights = weights
    
    # Iterate over all samples
    for _ in range(total_number_of_files):
        # Find the dataset with the highest error
        errors = weights - current_weights
        max_error_index = np.argmax(errors)
        # Update the total samples for the selected dataset
        current_samples[max_error_index] += 1
        
        # Exit when consuming entirely a dataset
        if current_samples[max_error_index] % dataset_sizes[max_error_index] == 0:
            print(f"Dataset {max_error_index} exhausted!")
            break
        
        # Compute weights of the ongoing data mixture
        current_sizes = compute_current_sizes(current_samples, dataset_files)
        current_weights = [float(i)/sum(current_sizes) for i in current_sizes]

    return current_samples

def get_bin_files(path_to_folder: str) -> List:
    files = [
        os.path.join(dp, f)
        for dp, _, fn in os.walk(os.path.expanduser(path_to_folder), followlinks=True)
        for f in fn
    ]

    if len(files) == 0:
        raise ValueError(f"No .bin files found in {path_to_folder}")

    filtered_files = [
        raw_file
        for raw_file in files
        if Path(raw_file).suffix.lower().endswith(".bin")
    ]

    return filtered_files

def create_symlink_mixture(folders, weights, output_folder):
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Collect bin files from each folder
    bin_files = []
    for folder in folders:
        folder_bin_files = sorted([f for f in get_bin_files(folder)])
        random.Random(SEED).shuffle(folder_bin_files)  # Shuffle to prevent bias from ordering
        bin_files.append(folder_bin_files)
    
    # Create the mixture without repeating files
    sample_list = build_dataset_mixture(weights, bin_files)

    used_files = set()
    for folder, count, folder_bin_files in zip(folders, sample_list, bin_files):
        dataset_name = os.path.basename(folder)
        output_dataset_mix_folder = os.path.join(output_folder, dataset_name)
        chosen_files = folder_bin_files[:count]
        for original_file in chosen_files:
            symlink_file = os.path.join(output_dataset_mix_folder, Path(original_file).relative_to(folder))
            Path(os.path.dirname(symlink_file)).mkdir(parents=True, exist_ok=True)
            used_files.add(original_file)
            if not os.path.exists(symlink_file):  # Avoid overwriting
                os.symlink(f"{original_file[:-4]}.bin", f"{symlink_file[:-4]}.bin")
                os.symlink(f"{original_file[:-4]}.idx", f"{symlink_file[:-4]}.idx")
    
    # Compute mixture statistics
    dataset_sizes = []
    for dataset in os.listdir(output_folder):
        dataset_output_folder = os.path.join(output_folder, dataset)
        if os.path.isdir(dataset_output_folder):
            dataset_files = get_bin_files(dataset_output_folder)
            sizes = [os.path.getsize(file) for file in dataset_files]
            dataset_sizes.append(sum(sizes))
    
    total_mix_size = sum(dataset_sizes) 
    produced_mix = [round(dataset_size / total_mix_size, 4) for dataset_size in dataset_sizes]
    produced_mix = {os.path.join(output_folder, dataset): weight for dataset, weight in zip(os.listdir(output_folder), produced_mix)}

    summary = f"Dataset mixture created in {output_folder} | {len(used_files)} files ({sample_list}) | {round(total_mix_size/(1e12), 2)} TB | {round(total_mix_size/(4e9), 2)} Billion Tokens (x2 if Vocab < 65536) | Resulting mixture is {dict(sorted(produced_mix.items()))}"
    print(summary)
    with open(os.path.join(output_folder, "dataset_mixture_summary.txt"), 'w') as outfile:
        outfile.write(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mixture of bin files using symlinks WITHOUT repetition.")
    parser.add_argument("--folders", nargs='+', required=True, help="List of folders containing bin files recursively")
    parser.add_argument("--weights", nargs='+', type=float, required=True, help="Weights for each folder")
    parser.add_argument("--output", required=True, help="Output folder for symlinks.")
    
    args = parser.parse_args()

    if len(args.folders) != len(args.weights):
        raise ValueError("Number of folders and weights must be the same.")
    
    # Normalize weights
    weights = [float(i)/sum(args.weights) for i in args.weights]
    # NOTE(tj.solergibert) For logging purposes
    mixture = {folder: round(weight, 4) for folder, weight in zip(args.folders, weights)}
    print(f"Creating data mixture from {dict(sorted(mixture.items()))}...")
    ##################################
    create_symlink_mixture(args.folders, weights, args.output)
