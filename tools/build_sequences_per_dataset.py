# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""
Script to build a json file with the sequences per dataset to use with the --per-dataset-sequences-path. Accepts the same arguments as the training script.

Usage:
python3 tools/build_sequences_per_dataset.py --per-split-data-args-path my-training-dataset-blend.json --per-dataset-sequences-path my-training-dataset-blend-sequences-per-dataset.json

"""

import argparse
import json
from typing import Optional, Tuple, List


from megatron.core.datasets.indexed_dataset import _IndexReader
from megatron.training.utils import get_blend_and_blend_per_split

def get_paths_from_blend(
    blend: Optional[Tuple[List[str], Optional[List[float]]]],
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]],
) -> List[str]:
    """Extract all dataset paths from blend and blend_per_split.

    Args:
        blend (Optional[Tuple[List[str], Optional[List[float]]]]): A blend tuple containing
            a list of dataset paths and optionally a list of weights, e.g.,
            (["path/to/dataset_1", "path/to/dataset_2"], [0.3, 0.7])
        blend_per_split (Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]): 
            A list of 3 blend tuples (for train, valid, test splits), where each element has 
            the same structure as blend

    Returns:
        List[str]: A list of all unique dataset paths found in blend and blend_per_split
    """
    paths = []
    
    # Extract paths from blend
    if blend is not None:
        paths_list, _ = blend
        paths.extend(paths_list)
    
    # Extract paths from blend_per_split
    if blend_per_split is not None:
        for split_blend in blend_per_split:
            if split_blend is not None:
                split_paths, _ = split_blend
                paths.extend(split_paths)
    
    # Remove duplicates while preserving order
    seen = set()
    unique_paths = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique_paths.append(path)
    
    return unique_paths

def build_sequences_per_dataset(args):
    print("Building sequences per dataset...")

    blend, blend_per_split = get_blend_and_blend_per_split(args)

    file_prefixes = get_paths_from_blend(blend, blend_per_split)

    print(f"Number of unique file prefixes: {len(file_prefixes)}")

    sequence_count_dict = {}
    for file_prefix in file_prefixes:
        # NOTE(asolergi-nv): For every file prefix, read index file and get the number of sequences and documents
        index_reader = _IndexReader(file_prefix + ".idx", False)
        count = (index_reader.sequence_count, index_reader.document_count)
        sequence_count_dict[file_prefix] = count

    return sequence_count_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', nargs='*', default=None,
                       help='The weight and prefix list for a set of train, validation, and test'
                       'datasets which split according to --split. The accepted formats are: '
                       '(1) a single prefix, '
                       '(2) a list of weight prefix pairs e.g. weight1 prefix1 weight2 prefix2, '
                       '(3) a list of prefixes e.g. prefix1 prefix2. '
                       'For (3), weights are inferred from the lengths of the contributing datasets. '
                       'This argument is exclusive to the other independent --*-data-path arguments.')
    parser.add_argument('--train-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent train dataset. '
                       'Follows the same pattern rules as --data-path.')
    parser.add_argument('--valid-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent validation dataset. '
                       'Follows the same pattern rules as --data-path.')
    parser.add_argument('--test-data-path', nargs='*', default=None,
                       help='The weight and prefix list for an independent test dataset. '
                       'Follows the same pattern rules as --data-path.')
    parser.add_argument('--data-args-path', type=str, default=None,
                       help='Path to data-args. Instead of feeding `--data-path` '
                       'with weighted dataset, we pass in a file path from which '
                       'we read that argument. This is useful when the list of data is '
                       'too big.')
    parser.add_argument('--per-split-data-args-path', type=str, default=None,
                       help='Path to per-split-data-args. Instead of feeding '
                       '`--(train|valid|test)-data-path` with weighted dataset, '
                       'we pass in a file path from which we read those arguments. '
                       'This is useful when the list of data is too big. Format is a '
                       'json file with `train`, `valid, `test` keys')
    parser.add_argument('--per-dataset-sequences-path', type=str, required=True,
                       help='Path to the output json file with the sequences per dataset.')
    args = parser.parse_args()

    sequence_count_dict = build_sequences_per_dataset(args)

    with open(args.path_to_sequences_per_dataset_json, "w") as f:
        json.dump(sequence_count_dict, f)

    print(f"Done! Saving --path-to-sequences-per-dataset file to {args.path_to_sequences_per_dataset_json}")