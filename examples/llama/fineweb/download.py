import os
import argparse

from datasets import load_dataset
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download and process FineWeb-EDU dataset")
    parser.add_argument("--out-dir", default="./fineweb-edu/", help="Target directory for dataset (default: ./fineweb-edu/)")
    args = parser.parse_args()

    target_dir = args.out_dir
    os.makedirs(target_dir, exist_ok=True)

    folder = snapshot_download(
        "HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        local_dir=target_dir,
        # limit download to just one parquet file
        allow_patterns="sample/10BT/000_*",
    )

    print("Loading dataset")
    dataset = load_dataset("parquet", data_dir=os.path.join(target_dir, "sample/10BT/"))
    dataset["train"].to_json(os.path.join(target_dir, "data.jsonl"), lines=True)


if __name__ == "__main__":
    main()
