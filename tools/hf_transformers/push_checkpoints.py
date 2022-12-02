import re
from huggingface_hub import Repository
from pathlib import Path
import subprocess
import argparse

import tools.hf_transformers.convert_checkpoint



"""
Script to upload Megatron checkpoints to a HF repo on the Hub.

The script clones/creates a repo on the Hub, checks out a branch `--branch_name`,
and converts each `iter_` checkpoint and saves it as a commit on that branch.
"""

def get_iter_number(iter_dir: str):
    m = re.match(r'iter_(\d+)', iter_dir)
    if m is not None:
        return int(m.group(1))
    else:
        raise ValueError(f"Invalid directory name: {iter_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default=None, help="Path where repository is cloned to locally. Will use {exp_dir}/hf_checkpoints if not provided")
    parser.add_argument("--exp_dir", type=str, help="Path to experiment folder.")
    parser.add_argument("--repo_name", type=str, help="Name of repository on the Hub in 'ORG/NAME' format.")
    parser.add_argument("--branch_name", type=str, help="Name of branch in repository to save experiments.")
    args = parser.parse_args()

    all_ckpt_dir = Path(args.exp_dir)
    save_dir = args.save_dir if args.save_dir is not None else all_ckpt_dir / "hf_checkpoints"

    hf_repo = Repository(save_dir, clone_from=args.repo_name)
    hf_repo.git_checkout(args.branch_name, create_branch_ok=True)
    # Find last checkpoint that was uploaded
    head_hash = hf_repo.git_head_hash()
    commit_msg = subprocess.check_output(["git", "show", "-s", "--format=%B", head_hash], cwd=save_dir).decode()
    try:
        last_uploaded_iter = get_iter_number(commit_msg.strip())
    except ValueError:
        last_uploaded_iter = -1
    
    # The checkpoint dirs should be in ascending iteration order, so that the last commit corresponds to the latest checkpoint
    ckpt_dirs = sorted([x for x in all_ckpt_dir.iterdir() if x.name.startswith("iter_") and x.is_dir()])

    for ckpt_dir in ckpt_dirs:
        iter_number = get_iter_number(ckpt_dir.name)
        if iter_number <= last_uploaded_iter:
            print(f"Will skip iter: {iter_number}")
            continue
        # TODO: this only works for 1-way tensor/pipeline parallelism
        file_path = next((ckpt_dir / "mp_rank_00").glob('*.pt'))
        tools.hf_transformers.convert_checkpoint.main(path_to_checkpoint=file_path, output_dir=save_dir, print_checkpoint_structure=False)
        hf_repo.push_to_hub(commit_message=f"{ckpt_dir.name}")

if __name__ == "__main__":
    main()