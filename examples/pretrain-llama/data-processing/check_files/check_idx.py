import mmap
import struct

def load_memory_mapped_file(filename):
    with open(filename, 'rb') as f:
        return mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

def read_from_idx_file(idx_map, offset):
    idx_map.seek(offset)
    try:
        position = struct.unpack('<Q', idx_map.read(8))[0]
        return position
    except struct.error:
        return None

def check_idx_file(idx_file, num_entries=20):
    idx_map = load_memory_mapped_file(idx_file)
    print(f"Diagnosing index file: {idx_file}")
    for i in range(num_entries):
        offset = i * 8  # Assuming 64-bit integers
        position = read_from_idx_file(idx_map, offset)
        if position is not None:
            print(f"Entry {i}: {position}")
        else:
            print(f"Entry {i}: Error reading position")
    idx_map.close()

if __name__ == "__main__":
    idx_path = "/eph/nvme0/azureml/cr/j/efa3fb01dbcf493fae769213df552ae0/exe/wd/ALLaM-Megatron-LM/examples/pretrain-llama/data-processing/detokenize/outputs/data/allam_data_2-1_splits-llama2-indexed_data/llama2_bin_idx/ar_encyclopedias_split_00_text_document.idx"
    check_idx_file(idx_path)