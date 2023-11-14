import mmap
import struct
import sentencepiece as spm
import json
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--tokenizer_model")
    parser.add_argument("--output")
    return parser.parse_args()

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

def read_tokens_from_bin(bin_map, start, end):
    if start >= bin_map.size() or end > bin_map.size() or start >= end:
        print(f"Invalid range: start={start}, end={end}, bin_map size={bin_map.size()}")
        return []
    bin_map.seek(start)
    token_ids = []
    while bin_map.tell() < end:
        try:
            token_id = struct.unpack('<I', bin_map.read(4))[0]
            token_ids.append(token_id)
        except struct.error:
            break
    return token_ids

def detokenize_tokens(tokenizer, token_ids):
    return tokenizer.DecodeIds(token_ids)

def main(bin_file, idx_file, sp_model, output_file):
    bin_map = load_memory_mapped_file(bin_file)
    idx_map = load_memory_mapped_file(idx_file)

    tokenizer = spm.SentencePieceProcessor(model_file=sp_model)

    with open(output_file, 'w') as out:
        offset = 0
        while offset < idx_map.size():
            start = read_from_idx_file(idx_map, offset)
            if start is None:
                break
            end = read_from_idx_file(idx_map, offset + 8)
            if end is None:
                break
            
            token_ids = read_tokens_from_bin(bin_map, start, end)
            text = detokenize_tokens(tokenizer, token_ids)
            json_line = json.dumps({"text": text})
            out.write(json_line + '\n')
            offset += 16

if __name__ == "__main__":
    args = get_args()
    bin_path = args.file_path+'.bin'
    idx_path = args.file_path+'.idx'
    main(bin_path, idx_path, args.tokenizer_model, args.output)
