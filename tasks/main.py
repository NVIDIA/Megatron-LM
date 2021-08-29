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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

from megatron import get_args
from megatron.initialize import initialize_megatron


def get_tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='tasks')

    group.add_argument('--task', type=str, required=True,
                       help='Task name.')
    group.add_argument('--epochs', type=int, default=None,
                       help='Number of finetunning epochs. Zero results in '
                       'evaluation only.')
    group.add_argument('--pretrained-checkpoint', type=str, default=None,
                       help='Pretrained checkpoint used for finetunning.')
    group.add_argument('--keep-last', action='store_true',
                       help='Keep the last batch (maybe incomplete) in'
                       'the data loader')
    group.add_argument('--train-data', nargs='+', default=None,
                       help='Whitespace separated paths or corpora names '
                       'for training.')
    group.add_argument('--valid-data', nargs='*', default=None,
                       help='path(s) to the validation data.')
    group.add_argument('--overlapping-eval', type=int, default=32,
                       help='Sliding window for overlapping evaluation.')
    group.add_argument('--strict-lambada', action='store_true',
                       help='Use more difficult formulation of lambada.')
    # Retriever args
    group.add_argument('--qa-data-dev', type=str, default=None,
                       help='Path to the QA dataset dev file.')
    group.add_argument('--qa-data-test', type=str, default=None,
                       help='Path to the QA dataset test file.')

    # Faiss arguments for retriever
    group.add_argument('--faiss-use-gpu', action='store_true',
                       help='Whether create the FaissMIPSIndex on GPU')
    group.add_argument('--faiss-match', type=str, default='string', \
                        choices=['regex', 'string'], help="Answer matching '\
                        'logic type")
    group.add_argument('--faiss-topk-retrievals', type=int, default=100,
                       help='Number of blocks to use as top-k during retrieval')

    # finetune for retriever
    group.add_argument('--eval-micro-batch-size', type=int, default=None,
                       help='Eval Batch size per model instance (local batch '
                            'size). Global batch size is local batch size '
                            'times data parallel size.')
    group.add_argument('--train-with-neg', action='store_true',
                       help='Whether to use negative examples during model '
                        'training')
    group.add_argument('--train-hard-neg', type=int, default=0,
                       help='Number of hard negative exmaples to use during '
                        'training')


    # parameters for Av.rank validation method
    # Following options/arguments have been taken directly from DPR codebase
    group.add_argument('--val-av-rank-hard-neg', type=int, default=30,
                        help='Av.rank validation: how many hard negatives to'
                        ' take from each question pool')
    group.add_argument('--val-av-rank-other-neg', type=int, default=30,
                        help='Av.rank validation: how many other negatives to'
                        ' take from each question pool')

    # finetune for controllable dialogue
    group.add_argument('--train-module', type=str, default="",
                       help='either control module or dialogue model (control or dialog)')
    group.add_argument('--train-data-path', type=str, default="",
                       help='datapath for training set')
    group.add_argument('--test-data-path', type=str, default="",
                       help='datapath for test set')
    group.add_argument('--guess-file', type=str, default="",
                       help='datapath for generated sentences')
    group.add_argument('--answer-file', type=str, default="",
                       help='datapath for golden sentences')
    group.add_argument('--max-seq-len', type=int, default=1024,
                       help='maximum sequence length')
    group.add_argument('--spec-toks', type=str, default=None,
                       help='additional special tokens')
    group.add_argument('--last-turn', action='store_true',
                       help='only use last turn for control model')
    group.add_argument('--no-control-code', action='store_true',
                       help='removing control code in the training for control model')
    group.add_argument('--remove-stopwords', action='store_true',
                       help='removing stopwords when evaluating F1-score')
    group.add_argument('--add-separator', action='store_true', 
                       help='add separator between turns and add colon before generation')
    group.add_argument('--add-ctrl-code-to-dialog', action='store_true', 
                       help='add control code in the dialog modeling')
    group.add_argument('--remove-ctrl-sent', action='store_true', 
                       help='dont use control sentence in dialog modeling')


    # finetune for controllable generation
    group.add_argument('--wiki-path', type=str, default="",
                       help='data path for the wikipedia corpus')
    group.add_argument('--tokenized-path', type=str, default="",
                       help='data path for the tokenized file')
    group.add_argument('--prop', type=float, default=1.0,
                       help='Proportion of data used for training')
    group.add_argument('--max-instance', type=int, default=10000000,
                       help='Proportion of data used for training')

    return parser


if __name__ == '__main__':

    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for downstream tasks.")
        exit()

    if args.task == 'RACE':
        from race.finetune import main
    elif args.task in ['MNLI', 'QQP']:
        from glue.finetune import main
    elif args.task in ['LAMBADA', 'WIKITEXT103']:
        from zeroshot_gpt.evaluate import main
    elif args.task in ['ICT-ZEROSHOT-NQ', 'RETRIEVER-EVAL']:
        from orqa.evaluate_orqa import main
    elif args.task in ['RET-FINETUNE-NQ']:
        from orqa.supervised.finetune import main
    elif args.task == 'control-gen':
        from control_gen.finetune import main
    elif args.task == 'dialctrl':
        from dialctrl.finetune import main
    elif args.task in ['dialctrl-eval-ppl', 'dialctrl-eval-f1']:
        from dialctrl.evaluate import main
    else:
        raise NotImplementedError('Task {} is not implemented.'.format(
            args.task))

    main()
