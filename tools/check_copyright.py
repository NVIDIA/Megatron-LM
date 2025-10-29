#!/usr/bin/env python3
"""
Script to check and optionally add NVIDIA copyright headers to files.
"""

import sys
import argparse
import re  # Import the regular expression module
from pathlib import Path
from datetime import datetime

# Regex to check for in the first line
# Matches "Copyright.*NVIDIA CORPORATION.*All rights reserved."
HEADER_REGEX = re.compile(r"Copyright.*NVIDIA CORPORATION.*All rights reserved\.")

# Original format string, still used for the error message suggestion
EXPECTED_HEADER = """# Copyright (c) {} NVIDIA CORPORATION & AFFILIATES. All rights reserved."""


def has_correct_header(file_path):
    """Check if file's first line has the correct copyright header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Read only the first line
            first_line = f.readline()

        # Check if the regex pattern is found anywhere in the first line
        return HEADER_REGEX.search(first_line) is not None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Check and add NVIDIA copyright headers to files.'
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Files to check/modify'
    )

    args = parser.parse_args()

    missing_headers = []

    for file_path in args.files:
        path = Path(file_path)

        if not path.exists():
            print(f"File not found: {file_path}")
            continue

        if not path.is_file():
            print(f"Not a file: {file_path}")
            continue

        if not has_correct_header(path):
            print(f"✗ Header missing: {file_path}")
            missing_headers.append(path)

    # Exit with error code if headers are missing and not added
    if missing_headers:
        print(f"\n{len(missing_headers)} file(s) missing copyright header.")
        print("\n")
        # The error message still suggests a specific header to add
        print("Add the header in those files with the following content:")
        print(EXPECTED_HEADER.format(str(datetime.now().year)))
        print("\n")
        print(
            "Disclaimer: This must done irrespective of the magnitude of the change "
            "or whether your are the file/module author."
        )
        sys.exit(1)

    sys.exit(0)


if __name__ == '__main__':
    main()