#!/usr/bin/env python3
"""
Script to check and optionally add NVIDIA copyright headers to files.
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

EXPECTED_HEADER = """# Copyright (c) {} NVIDIA CORPORATION & AFFILIATES. All rights reserved."""


def has_correct_header(file_path):
    """Check if file has the correct copyright header."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Check if the expected header is at the start of the file
        return content.startswith(EXPECTED_HEADER.format(str(datetime.now().year)))
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

        if has_correct_header(path):
            print(f"✓ Header present: {file_path}")
        else:
            print(f"✗ Header missing: {file_path}")
            missing_headers.append(path)

    # Exit with error code if headers are missing and not added
    if missing_headers:
        print(f"\n{len(missing_headers)} file(s) missing copyright header.")
        print("\n")
        print("Add or replace the header in those files with the following content:")
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
