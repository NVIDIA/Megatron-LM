# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Main tasks functionality."""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from arguments import get_args


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group('tasks', 'tasks configurations')
    parser.add_argument('--task', type=str, required=True,
                        help='task name.')
    group.add_argument('--epochs', type=int, required=True,
                       help='number of finetunning epochs. Zero results in '
                       'evaluation only.')
    parser.add_argument('--pretrained-checkpoint', type=str, default=None,
                        help='pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='keep the last batch (maybe incomplete) in'
                       'the data loader')
    return parser


if __name__ == '__main__':

    args = get_args(extra_args_provider=get_tasks_args)

    if args.task == 'RACE':
        from race.finetune import main
    elif args.task in ['MNLI', 'QQP']:
        from glue.finetune import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main(args)
