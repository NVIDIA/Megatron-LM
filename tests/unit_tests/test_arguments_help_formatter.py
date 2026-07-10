# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse

from megatron.training.arguments import _MegatronHelpFormatter


def test_megatron_help_formatter_allows_literal_percent_characters():
    parser = argparse.ArgumentParser(formatter_class=_MegatronHelpFormatter)
    parser.add_argument(
        "--cache-ratio",
        default="1.0",
        help="Prepare 5% training cache shards; default %(default)s; escaped 10%%.",
    )

    help_text = parser.format_help()

    assert "Prepare 5% training cache shards" in help_text
    assert "default 1.0" in help_text
    assert "escaped 10%." in help_text
