import argparse
import os
import tqdm
import concurrent.futures
import subprocess
from glob import glob
import time

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Multi-Process runner for data processing")
    parser.add_argument("--glob-input-path", type = str, help="Glob path to input jsonl files", required=True)
    parser.add_argument("--output-folder", type = str, help="Path to the output bin files", required=True)
    parser.add_argument("--tokenizer-type", type = str, help="Type of the tokenizer model", required=True)
    parser.add_argument("--tokenizer-model", type = str, help="Path to the tokenizer model", required=True)
    parser.add_argument("--per-file-workers", type = int, help="Number of workers per file", required=True)
    parser.add_argument("--num-file-workers", type = int, help="Number of file to be processed at the same time", required=True)
    args = parser.parse_args()

    def process(file):
        cmd = "python tools/preprocess_data.py"
        cmd = cmd + f' --input {file}'
        cmd = cmd + f' --output-prefix {os.path.join(args.output_folder, os.path.basename(file).replace(".jsonl", ""))}'
        cmd = cmd + f' --tokenizer-type {args.tokenizer_type}'
        cmd = cmd + f' --tokenizer-model {args.tokenizer_model}'
        cmd = cmd + f' --workers {args.per_file_workers}'
        cmd = cmd + f' --append-eod'
        subprocess.check_output(cmd, shell=True)
        return file

    input_paths = glob(args.glob_input_path, recursive=True)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.num_file_workers) as executor:
        for _out in tqdm(
                executor.map(
                        process, 
                        [ input_path for input_path in input_paths ],
                ),
                total=len(input_paths),
        ):
                try:
                    print(f"Finished processing {_out}")
                except Exception as emsg:
                    print("Exception msg: {}".format(emsg))