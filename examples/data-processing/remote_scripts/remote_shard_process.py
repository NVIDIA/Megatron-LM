import argparse 
import subprocess
from typing import Dict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure batch job submission for processing a shard")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-subscription", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-resource-group", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-workspace-name", default=None, type = str, help="Azure subscription id.")
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")
    group.add_argument("--az-sample-yaml-job-file", default=None, type = str, help="Path to a sample job file.")
    group.add_argument("--az-configs", default=None, type = str, help="Path to a sample job file.")




if __name__ == "__main__":
    args =  get_args()
    # input_shard_dict = get_shard_info(args, args.input_folder_path)
    # output_shards_dict = get_shard_info(args, args.output_folder_path)
    # if not args.overwrite:
    #     input_shard_dict = match_pre_existing_bin_idx(input_shard_dict, output_shards_dict)
    # submit_jobs(args, input_shard_dict)