from megatron.core.datasets.indexed_dataset import IndexedDataset

path = "/lustre/fsw/coreai_dlalgo_llm/dpykhtar/data_prep/gigatoken"

default = IndexedDataset(f"{path}/hf_default_text_document")
fast = IndexedDataset(f"{path}/hf_gigatoken_fast_usual_text_document")

import hashlib
import os

def get_file_hash(file_path, algorithm="sha256"):
    """Calculates the cryptographic hash of a file by reading it in chunks."""
    # Create the hash object dynamically (e.g., sha256, md5)
    hasher = hashlib.new(algorithm)

    # Open the file in binary mode ('rb')
    with open(file_path, 'rb') as f:
        # Read in 64KB chunks to efficiently handle large files
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)

    return hasher.hexdigest()

def are_files_identical(file1, file2):
    """Compares two files using their sizes and hashes."""
    # Step 1: Early-fail optimization (compare file sizes first)
    if os.path.getsize(file1) != os.path.getsize(file2):
        return False

    # Step 2: Calculate and compare hashes
    return get_file_hash(file1) == get_file_hash(file2)

# Example Usage
file_a = f"{path}/hf_default_text_document"
file_b = f"{path}/hf_gigatoken_fast_usual_text_document"

if are_files_identical(file_a, file_b):
    print("Success: Both files have the identical hash!")
else:
    print("Warning: Files have different hashes.")
    print(file_a)
    print(file_b)
