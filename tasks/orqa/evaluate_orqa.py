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

from megatron import get_args, print_rank_0
from megatron.indexer import IndexBuilder
from tasks.orqa.evaluate_utils import ORQAEvaluator

def main():
    """
    Main program
    """

    args = get_args()

    """
    Create a BlockData data structure by running an IndexBuilder over an
    ICT Dataset and then evaluate on NQ task
    """

    print_rank_0("Starting index builder!")

    index_builder = IndexBuilder()
    index_builder.build_and_save_index()
    print_rank_0("Build and save indices: done!")


    print_rank_0("Starting evaluations!")

    # Set up the model and evaluator
    evaluator = ORQAEvaluator()

    # Run evaluation
    if args.qa_data_dev is not None:
        evaluator.evaluate(args.qa_data_dev, "DEV")

    if args.qa_data_test is not None:
        evaluator.evaluate(args.qa_data_test, "TEST")

