import os
import sys
import glob
import argparse
import subprocess
from tqdm import tqdm
from transformers import LlamaTokenizer


"""
example:
python count_token_and_rename_bin_idx.py --source_prefix_path "../DUMPED/*" 
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-prefix-paths', type=str, required=True,
                       help='Glob path to folder where all the bin, idx files are.')
    parser.add_argument('--megatron-path', type=str, default=None,
                       help='Base directory of deepspeed repository')
    args = parser.parse_args()
    # Search in directory above this
    sys.path.append(os.path.abspath(
        os.path.join(os.path.dirname(__file__),
                     os.path.pardir)))
    if args.megatron_path is not None:
        sys.path.insert(0, args.megatron_path)

    return args

def test_indexed_dataset(source_prefix_path: str):
    
    from megatron.data.dataset_utils import get_indexed_dataset_

    dataset = get_indexed_dataset_(source_prefix_path, 'gpt')
    doc_cnt = dataset.document_indices.shape[0] - 1
    sent_cnt = dataset.sequence_lengths.shape[0]
    token_cnt = 0
    for doc in tqdm(dataset):
        token_cnt += len(doc)
    return doc_cnt, sent_cnt, token_cnt

def main():
    args = get_args()
    source_prefix_paths = glob.glob(args.source_prefix_paths)
    source_prefix_paths = sorted([ source_prefix_path.replace(".bin", "") for source_prefix_path in source_prefix_paths if source_prefix_path.endswith(".bin")])
    for idx, source_prefix_path in enumerate(source_prefix_paths):
        doc_cnt, sent_cnt, token_cnt = test_indexed_dataset(source_prefix_path)
        print(f"[{idx+1}/{len(source_prefix_paths)}][OK][{doc_cnt=}][{sent_cnt=}][{token_cnt=}] {source_prefix_path}")
        old_bin_file_path = source_prefix_path + '.bin'
        old_idx_file_path = source_prefix_path + '.idx'
        if "dc=" in source_prefix_path:
            dc = int(source_prefix_path.split("dc=")[1].split("_")[0])
            assert dc == doc_cnt
            assert "sc=" in source_prefix_path
            assert "tc=" in source_prefix_path
        if "sc=" in source_prefix_path:
            sc = int(source_prefix_path.split("sc=")[1].split("_")[0])
            assert sc == sent_cnt
            assert "dc=" in source_prefix_path
            assert "tc=" in source_prefix_path
        if "tc=" in source_prefix_path:
            tc = int(source_prefix_path.split("tc=")[1].split("_")[0])
            assert tc == token_cnt
            assert "dc=" in source_prefix_path
            assert "tc=" in source_prefix_path
        if "sc=" not in source_prefix_path and "dc=" not in source_prefix_path and "tc=" not in source_prefix_path:
            new_bin_file_path = source_prefix_path + f"_dc={doc_cnt}_sc={sent_cnt}_tc={token_cnt}" + '.bin'
            new_idx_file_path = source_prefix_path + f"_dc={doc_cnt}_sc={sent_cnt}_tc={token_cnt}" + '.idx'
            cmd1 = f"mv {old_bin_file_path} {new_bin_file_path}"
            cmd2 = f"mv {old_idx_file_path} {new_idx_file_path}"
            print(f"Running {cmd1}")
            subprocess.check_output(cmd1, shell=True)
            print(f"Running {cmd2}")
            subprocess.check_output(cmd2, shell=True)

if __name__ == '__main__':
    main()