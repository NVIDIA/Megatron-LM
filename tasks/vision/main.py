# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.join(os.path.dirname(__file__), os.path.pardir),
            os.path.pardir,
        )
    )
)
from megatron import get_args
from megatron.initialize import initialize_megatron

def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title="tasks")

    group.add_argument('--task', type=str, default='segment',
                       choices=['classify', 'segment_setr', 'segment_segformer'],
                       help='task name.')
    group.add_argument("--epochs", type=int, default=None,
                       help="Number of finetunning epochs. Zero results in "
                       "evaluation only.")
    group.add_argument('--pretrained-checkpoint-type', type=str, default='default',
                       choices=['default', 'external', 'constrastive'],
                       help='Type of pretrained checkpoint')
    group.add_argument("--pretrained-checkpoint", type=str, default=None,
                       help="Pretrained checkpoint used for finetunning.")
    group.add_argument('--seg-stride', type=int, default=None,
                       help='sliding window stride during evaluation')
    return parser


if __name__ == "__main__":

    initialize_megatron(extra_args_provider=get_tasks_args)
    args = get_args()

    if args.task == 'classify':
        from tasks.vision.classification.classification import main
        main()
    elif args.task == 'segment_setr':
        from tasks.vision.segmentation.finetune_setr import main
        main()
    elif args.task == 'segment_segformer':
        from tasks.vision.segmentation.finetune_segformer import main
        main()

