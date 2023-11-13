import sentencepiece as sp
import argparse
import struct

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path")
    parser.add_argument("--tokenizer_model")
    parser.add_argument("--output")
    return parser.parse_args()

def read_bin_idx(bin_path, idx_path):
    with open(bin_path, 'rb') as bin_file, open(idx_path, 'rb') as idx_file:
        while True:
            idx_bytes = idx_file.read(8)
            if not idx_bytes:
                break
            start = struct.unpack('q', idx_bytes)[0]
            end = struct.unpack('q', idx_file.read(8))[0]
            bin_file.seek(start)
            yield bin_file.read(end - start)

def detokenize_data(tokenizer, bin_path, idx_path):
    sp = spm.SentencePieceProcessor(model_file=tokenizer)
    for token_ids in read_bin_idx(bin_path, idx_path):
        tokens = struct.unpack(f'{len(token_ids) // 4}i', token_ids)
        yield sp.decode_ids(tokens)

def write_jsonl(output_path, detokenized_data):
    with open(output_path, 'w') as out_file:
        for line in detokenized_data:
            out_file.write(f'{{"text": "{line}"}}\n')

if __name__ == "__main__":
    args = get_args()
    bin_path = args.file_path+'.bin'
    idx_path = args.file_path+'file.idx'

    detokenized_data = detokenize_data(args.tokenizer_model, bin_path, idx_path)
    write_jsonl(args.output, detokenized_data)