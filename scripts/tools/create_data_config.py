import argparse
import os
from pathlib import Path
from typing import List


def create_data_prefix(list_of_paths: List[str]):
    list_of_bin_files = []
    # Select all .bin files
    for path in list_of_paths:
        path_to_files = [
            os.path.join(dp, f)
            for dp, _, fn in os.walk(os.path.expanduser(path))
            for f in fn
        ]
        list_of_bin_files.extend(
            [
                raw_file
                for raw_file in path_to_files
                if Path(raw_file).suffix.lower().endswith(".bin")
            ]
        )

    list_of_bin_files = [
        bin_file[:-4] for bin_file in list_of_bin_files
    ]  # NOTE(tj.solergibert) Delete .bin extension to have file prefixes
    
    return list_of_bin_files

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--paths",
        type=str,
        required=True,
        help="Comma separated list of paths to generate the config from. e.g. -p /path/to/dataset/A,/path/to/dataset/B,/path/to/dataset/C",
    )
    args = parser.parse_args()

    paths = [x.strip() for x in args.paths.split(",")]
    data_prefix = create_data_prefix(paths)
    print(*data_prefix, sep=" ")
