import os
import sys
from pathlib import Path


def main(paths_file):
    with open(paths_file) as f:
        files = f.readlines()

    files = [file.rstrip("\n") for file in files]
    dataset_symlink = os.path.join(
        Path(paths_file).parent.parent.absolute(), "raw-dataset-link"
    )
    total_dump_size = sum(
        [os.path.getsize(os.path.join(dataset_symlink, file)) for file in files]
    )
    print(total_dump_size)


if __name__ == "__main__":
    assert len(sys.argv) == 2, (
        "Usage: python3 compute_dump_size.py /path/to/paths/file/dir"
    )
    main(sys.argv[1])
