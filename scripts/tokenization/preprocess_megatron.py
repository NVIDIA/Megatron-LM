"""
python3 preprocess_megatron.py --tokenizer-name-or-path meta-llama/Meta-Llama-3-8B --output-folder tokenized_datasets/fineweb-edu --n-tasks 16 --dataset datasets/fineweb-edu/raw-dataset-link --paths-file datasets/fineweb-edu/dumps/paths_file_0.txt
"""

import argparse

from data_pipeline_pretrain.pipeline.tokens import MegatronDocumentTokenizer
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader


def get_args():
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="Tokenizer")
    group.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        required=True,
        help="A path to a directory containing vocabulary files required by the tokenizer or the model id of a predefined tokenizer hosted inside a model repo on the Hugging Face Hub.",
    )
    group.add_argument(
        "--eos-token",
        type=str,
        default=None,
        help="EOS token to add after each document. Default: None",
    )

    group = parser.add_argument_group(title="Output data")
    group.add_argument(
        "--output-folder",
        type=str,
        required=True,
        help="Path to the output folder to store the tokenized documents",
    )
    group = parser.add_argument_group(title="Miscellaneous configs")
    group.add_argument(
        "--logging-dir",
        type=str,
        default=None,
        help="Path to a folder for storing the logs of the preprocessing step. Default: None",
    )
    group.add_argument(
        "--n-tasks",
        type=int,
        default=8,
        help="Total number of tasks to run the preprocessing step. Default: 8",
    )
    group.add_argument(
        "--n-workers",
        type=int,
        default=-1,
        help="Number of workers executing concurrently --n-tasks tasks. Default: -1, which means --n-workers==--n-tasks",
    )
    group = parser.add_argument_group(title="Dataset configs")
    group.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to a folder recursively containing multiple .parquet files",
    )
    group.add_argument(
        "--paths-file",
        type=str,
        required=True,
        help="A file with one path per line (without the `dataset` prefix) to read",
    )
    group.add_argument(
        "--column",
        type=str,
        default="text",
        help="Column to preprocess from the Dataset. Default: text",
    )

    args = parser.parse_args()

    return args


def main(args):
    n_tasks = args.n_tasks
    # Check number of files > n tasks
    with open(args.paths_file, "rb") as f:
        number_of_files = sum(1 for _ in f)
    if n_tasks > number_of_files:
        n_tasks = number_of_files

    preprocess_executor = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                data_folder=args.dataset,
                paths_file=args.paths_file,
                text_key=args.column,
            ),
            MegatronDocumentTokenizer(
                output_folder=args.output_folder,
                tokenizer_name_or_path=args.tokenizer_name_or_path,
                eos_token=args.eos_token,
            ),
        ],
        tasks=n_tasks,
        workers=args.n_workers,
        logging_dir=args.logging_dir,
    )
    preprocess_executor.run()


if __name__ == "__main__":
    _args = get_args()
    main(_args)
