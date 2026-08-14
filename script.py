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
file_a = "/lustre/fsw/coreai_dlalgo_llm/dpykhtar/data_prep/gigatoken/hf_default_text_document"
file_b = "/lustre/fsw/coreai_dlalgo_llm/dpykhtar/data_prep/gigatoken/hf_giga_final_text_document"

#if are_files_identical(f"{file_a}.bin", f"{file_b}.bin"):
#    print("Success: Both bin files have the identical hash!")
#else:
#    print("Warning: Bin Files have different hashes.")

#if are_files_identical(f"{file_a}.idx", f"{file_b}.idx"):
#    print("Success: Both idx files have the identical hash!")
#else:
#    print("Warning: idx files have different hashes.")

from megatron.core.datasets.indexed_dataset import IndexedDataset

default = IndexedDataset(file_a)
fast = IndexedDataset(file_b)
import numpy as np

for index, (val1, val2) in enumerate(zip(default, fast), start=1):
    if index % 1000000 == 0:
        print(index)
    if len(val1) != len(val2):
        print(index)
        print(f"len {len(val1)} != {len(val2)}")
    if not np.array_equal(val1, val2):
        print(index)
        print(f"{val1} != {val2}")
