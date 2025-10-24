# Copyright (c) 2019-2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


def get_test_cases(yaml_file):
    """Extract all test cases from YAML file using yq."""
    result = subprocess.run(
        ['yq', 'eval', '.products[].test_case[]', yaml_file],
        capture_output=True,
        text=True,
        check=True,
    )
    return [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]


def expand_test_case(pattern):
    """Expand a test case pattern to all matching files."""
    files = []

    if '**' in pattern:
        # Recursive glob: base_dir/**/*.py
        base_dir = pattern.split('/**')[0]
        file_pattern = pattern.split('**/')[-1]

        base_path = Path(base_dir)
        if base_path.is_dir():
            files = list(base_path.rglob(file_pattern))

    elif '*' in pattern:
        # Non-recursive glob: base_dir/*.py
        parts = pattern.rsplit('/', 1)
        base_dir = parts[0]
        file_pattern = parts[1]

        base_path = Path(base_dir)
        if base_path.is_dir():
            files = list(base_path.glob(file_pattern))

    elif Path(pattern).is_dir():
        # Directory: find all .py files recursively
        files = list(Path(pattern).rglob('*.py'))

    elif Path(pattern).is_file():
        # Specific file
        files = [Path(pattern)]

    # Return as strings and filter to only files
    return [str(f) for f in files if f.is_file()]


def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <BUCKET> [--verbose]", file=sys.stderr)
        sys.exit(1)

    BUCKET = sys.argv[1]
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    YAML_FILE = 'tests/test_utils/recipes/unit-tests.yaml'

    if verbose:
        print(f"BUCKET: {BUCKET}", file=sys.stderr)
        print(file=sys.stderr)

    # Get all test cases
    all_test_cases = get_test_cases(YAML_FILE)

    if verbose:
        print("All test cases:", file=sys.stderr)
        for tc in all_test_cases:
            print(f"  {tc}", file=sys.stderr)
        print(file=sys.stderr)

    # Verify BUCKET is in the list
    if BUCKET not in all_test_cases:
        print(f"ERROR: BUCKET '{BUCKET}' not found in test cases!", file=sys.stderr)
        sys.exit(1)

    # Sort by length (most specific first - longer paths are more specific)
    sorted_test_cases = sorted(all_test_cases, key=len, reverse=True)

    if verbose:
        print("Processing test cases in order of specificity...", file=sys.stderr)

    file_ownership = {}

    for test_case in sorted_test_cases:
        files = expand_test_case(test_case)
        if verbose:
            print(f"  Processing: {test_case} ({len(files)} files)", file=sys.stderr)

        for file in files:
            # Only assign file if not already owned by a more specific test case
            if file not in file_ownership:
                file_ownership[file] = test_case

    if verbose:
        print(file=sys.stderr)

    # Find files owned by BUCKET
    bucket_files = sorted([file for file, owner in file_ownership.items() if owner == BUCKET])

    if verbose:
        print(f"Files owned by BUCKET ({len(bucket_files)} total):", file=sys.stderr)
        for file in bucket_files[:5]:
            print(f"  {file}", file=sys.stderr)
        if len(bucket_files) > 5:
            print(f"  ... and {len(bucket_files) - 5} more", file=sys.stderr)
        print(file=sys.stderr)

    # Output each file on a separate line for bash array capture
    for file in bucket_files:
        print(f"--ignore={file}")


if __name__ == '__main__':
    main()
