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

#sys.path.append(
#    os.path.abspath(
#        os.path.join(
#            os.path.join(os.path.dirname(__file__), os.path.pardir),
#            os.path.pardir,
#        )
#    )
#)

from megatron import get_args, print_rank_0
from megatron.indexer import IndexBuilder
from tasks.orqa.evaluate_utils import ORQAEvaluator

def main():
    """
    Main program
    """
    #initialize_megatron(extra_args_provider=None,
    #                    args_defaults={'tokenizer_type': 'BertWordPieceLowerCase'})

    args = get_args()

    """Create a BlockData data structure by running an IndexBuilder over an ICT Dataset
    - Include all args needed for initial model specification

    Other key args:
        --block-data-path: path to write to
        --ict-load or --realm-load: path to checkpoint with which to embed
        --data-path and --titles-data-path: paths for dataset
        --indexer-log-interval: reporting interval
        --indexer-batch-size: size specific for indexer jobs

    Check README.md for example script
    """

    #print_rank_0("Starting index builder!")

    index_builder = IndexBuilder()
    index_builder.build_and_save_index()
    print_rank_0("Build and save indices: done!")

    # Set up the model and evaluator
    evaluator = ORQAEvaluator()

    # Run evaluation
    if args.qa_data_dev is not None:
        evaluator.evaluate(args.qa_data_dev, "DEV")

    if args.qa_data_test is not None:
        evaluator.evaluate(args.qa_data_test, "TEST")
    
