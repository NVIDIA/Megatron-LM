# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

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
from megatron.training import get_args
from megatron.training.initialize import initialize_megatron

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

