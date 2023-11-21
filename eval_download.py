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

