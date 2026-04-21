# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Main tasks functionality."""

from megatron.training import get_args, print_rank_0
from megatron.legacy.indexer import IndexBuilder
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

