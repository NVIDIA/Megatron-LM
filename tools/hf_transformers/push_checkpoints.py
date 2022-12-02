from huggingface_hub import Repository
from pathlib import Path
import subprocess
import argparse


"""
Script to upload Megatron checkpoints to a HF repo on the Hub.

The script clones/creates a repo on the Hub, checks out a branch `--branch_name`,
and converts each `iter_` checkpoint and saves it as a commit on that branch.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, help="Path where repository is cloned to locally.")
    parser.add_argument("--exp_dir", type=str, help="Path to experiment folder.")
    parser.add_argument("--repo_name", type=str, help="Name of repository on the Hub in 'ORG/NAME' format.")
    parser.add_argument("--branch_name", type=str, help="Name of branch in repository to save experiments.")
    args = parser.parse_args()

    hf_repo = Repository(args.save_dir, clone_from=args.repo_name)
    hf_repo.git_checkout(args.branch_name, create_branch_ok=True)
    
    all_ckpt_dir = Path(args.exp_dir) / "hf_checkpoints/"
    ckpt_dirs = [x for x in all_ckpt_dir.iterdir() if x.name.startswith("iter_") and x.is_dir()]
    # TODO: some sorting necessary? they `ckpt_dirs` should be in ascending order

    for ckpt_dir in ckpt_dirs:
        file_path = next(ckpt_dir.glob('*.pt'))
        # TODO: if we format convert_checkpoint.py such that the main logic is in a function with args instead of argparse we can avoid using the `suprocess` and import the function instead
        subprocess.Popen(["python", "convert_checkpoint.py", "--path_to_checkpoint", file_path, "--output-dir", args.save_dir])
        hf_repo.push_to_hub(commit_message=f"{ckpt_dir.name}")

if __name__ == "__main__":
    main()