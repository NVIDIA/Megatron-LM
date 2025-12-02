# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def get_test_cases(yaml_file):
    result = subprocess.run(
        ['yq', 'eval', '.products[].test_case[]', yaml_file],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]


def get_base_path(pattern):
    if '**' in pattern:
        return pattern.split('/**')[0]
    elif '*' in pattern:
        return pattern.rsplit('/', 1)[0]
    return pattern.rstrip('/')


def is_child_of_bucket(test_case, bucket):
    test_base = get_base_path(test_case)
    bucket_base = get_base_path(bucket)
    return test_base.startswith(bucket_base + '/')


def expand_pattern(pattern):
    if '**' in pattern:
        parts = pattern.split('/**/')
        if len(parts) == 2:
            base_dir, file_pattern = parts
        else:
            # Handle case like 'dir/**'
            base_dir = pattern.split('/**')[0]
            file_pattern = '*.py'
        return [str(f) for f in Path(base_dir).rglob(file_pattern) if f.is_file()]
    elif '*' in pattern:
        base_dir, file_pattern = pattern.rsplit('/', 1)
        return [str(f) for f in Path(base_dir).glob(file_pattern) if f.is_file()]
    elif Path(pattern).is_file():
        return [pattern]
    return []


def main():
    BUCKET = sys.argv[1]
    YAML_FILE = 'tests/test_utils/recipes/unit-tests.yaml'

    all_test_cases = get_test_cases(YAML_FILE)
    bucket_files = set(expand_pattern(BUCKET))

    # Collect files from child test cases to ignore
    files_to_ignore = set()
    for test_case in all_test_cases:
        if test_case != BUCKET and is_child_of_bucket(test_case, BUCKET):
            files_to_ignore.update(expand_pattern(test_case))

    # Output files to ignore
    for file in sorted(files_to_ignore & bucket_files):
        print(f"--ignore={file}")


if __name__ == '__main__':
    main()
