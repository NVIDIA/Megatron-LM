import os
import tqdm
import argparse
import concurrent.futures
import subprocess
from glob import glob


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Multi-Process runner for data processing")
    parser.add_argument("--glob-input-path", type = str, help="Glob path to input jsonl files", required=True)
    parser.add_argument("--output-folder", type = str, help="Path to the output bin files", required=True)
    parser.add_argument("--tokenizer-module", type = str, default='megatron', choices=['megatron', 'nemo'], help="Which version of script will be used to tokenize the data.")
    parser.add_argument("--tokenizer-type", type = str, help="Type of the tokenizer model", required=True)
    parser.add_argument("--tokenizer-model", type = str, help="Path to the tokenizer model", required=True)
    parser.add_argument("--vocab-file", type = str, default=None, help="Vocab file of the tokenizer.", required=False)
    parser.add_argument("--merge-file", type = str, default=None, help="Merge file of the tokenizer.", required=False)
    parser.add_argument("--per-file-workers", type = int, help="Number of workers per file", required=True)
    parser.add_argument("--num-file-workers", type = int, help="Number of file to be processed at the same time", required=True)
    parser.add_argument('--log-interval', type=int, default=1000, help='Interval between progress updates')
    parser.add_argument('--path-to-nemo', type=int, default='/workspace/', help='Path to the nemo directory')
    
    
    args = parser.parse_args()

    def process(file):
        if args.tokenizer_module == 'nemo':
            cmd = f"python {args.path_to_nemo}/scripts/nlp_language_modeling/preprocess_data_for_megatron.py"
            cmd = cmd + f' --tokenizer-library {args.tokenizer_type}'
        elif args.tokenizer_module == 'megatron':
            cmd = "python tools/preprocess_data.py"
            cmd = cmd + f' --tokenizer-type {args.tokenizer_type}'
        cmd = cmd + f' --input {file}'
        cmd = cmd + f' --output-prefix {os.path.join(args.output_folder, os.path.basename(file).replace(".jsonl", ""))}'
        cmd = cmd + f' --tokenizer-model {args.tokenizer_model}'
        if args.vocab_file is not None: cmd = cmd + f' --vocab-file {args.vocab_file}'
        if args.merge_file is not None: cmd = cmd + f' --merge-file {args.merge_file}'
        cmd = cmd + f' --workers {args.per_file_workers}'
        cmd = cmd + f' --append-eod'
        cmd = cmd + f' --log-interval {args.log_interval}'
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
                    pass
                except Exception as emsg:
                    print("Exception msg: {}".format(emsg))
                    raise