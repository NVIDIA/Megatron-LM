# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

# Downloads the specified taks in the evaluation harness
# This is particularly useful when running in environments where the GPU nodes 
# do not have internet access. This way we can pre-download them and use the cached data-set during evaluation.

from lm_eval import tasks
from lm_eval.tasks import ALL_TASKS
import argparse
import os


parser = argparse.ArgumentParser(description='Download evaluation harness', allow_abbrev=False)
parser.add_argument('--task_list', type=str, default = "all", help='Either "all" or comma separated list of tasks to download.')
args = parser.parse_args()

def main():
    task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
    tasks.get_task_dict(task_list)

if __name__ == '__main__':
    main()


    