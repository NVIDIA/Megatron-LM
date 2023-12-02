import argparse 
import json
import subprocess
from typing import Dict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Azure batch job submission for processing a shard")
    group = parser.add_argument_group(title='Azure login params.')
    group.add_argument("--az-sas-token", default=None, type = str, help="Azure blob SAS token")

    group = parser.add_argument_group(title='Process I/O params.')
    group.add_argument("--shard-names", nargs='+', type = str, help="Name of the shards to be processed.")
    group.add_argument("--input-folder-path", type = str, help="Compute target folder path to input jsonl files, each job process a single jsonl file.")
    group.add_argument("--output-folder-path", type = str, help="Compute target folder path to output folder path.")
    group.add_argument("--config-path-for-process-function", type = str, help="Config path for process function.")
    parser.add_argument("--compute-target", type = str, default='azure', choices=['local', 'azure'], help="Conpute targets. Both --input-folder-path, --output-folder-path"
                        " TODO: Enable cross compute.")
    args = parser.parse_args()

    if args.az_configs is not None:
        args.az_configs = json.load(open(args.az_configs))
        if args.az_subscription is not None:
            print("Overloading config args from  --az-subscription")
            args.az_configs['az-subscription'] = args.az_subscription
        if args.az_resource_group is not None:
            print("Overloading config args from  --az-resource-group")
            args.az_configs['az-resource-group'] = args.az_resource_group
        if args.az_workspace_name is not None:
            print("Overloading config args from  --az-workspace-name")
            args.az_configs['az-workspace-name'] = args.az_workspace_name
        if args.az_sas_token is not None:
            print("Overloading config args from  --az-sas-token")
            args.az_configs['az-sas-token'] = args.az_sas_token
        if args.az_sample_yaml_job_file is not None:
            print("Overloading config args from  --az-sample-yaml-job-file")
            args.az_configs['az-sample-yaml-job-file'] = args.az_sample_yaml_job_file

    return args

def download_azure_files():
    pass

if __name__ == "__main__":
    args =  get_args()

    