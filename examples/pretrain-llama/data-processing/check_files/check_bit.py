import struct
import mmap

def is_64bit_integers(filename):
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for offset in range(0, mm.size(), 8):
            try:
                struct.unpack('<Q', mm[offset:offset + 8])
            except struct.error:
                return False
        return True

def is_32bit_integers(filename):
    with open(filename, 'rb') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for offset in range(0, mm.size(), 4):
            try:
                struct.unpack('<I', mm[offset:offset + 4])
            except struct.error:
                return False
        return True

if __name__ == "__main__":
    index_file = 'outputs/data/allam_data_2-1_splits-llama2-VE-indexed_data/bin_idx/en_books_books_split_02_text_document.idx'
    bin_file = 'outputs/data/allam_data_2-1_splits-llama2-VE-indexed_data/bin_idx/en_books_books_split_02_text_document.bin'

    print(f"Index file 64-bit check: {is_64bit_integers(index_file)}")
    print(f"Bin file 32-bit check: {is_32bit_integers(bin_file)}")
