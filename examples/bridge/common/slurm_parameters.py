# Adapted from https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/08035723e7a86f94fc54a7f4473bd0519f2ac941/scripts/common/slurm_parameters.py
# Source license: Apache-2.0.
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Shared allocation-level Slurm arguments for public launchers."""

import argparse


def _parse_additional_slurm_params(value: str) -> dict[str, str]:
    """Parse semicolon-separated Slurm executor parameters."""
    parameters: dict[str, str] = {}
    for item in value.split(";"):
        key, separator, parameter_value = item.partition("=")
        if not separator or not key or not parameter_value:
            raise argparse.ArgumentTypeError(
                "--additional-slurm-params expects semicolon-separated KEY=VALUE pairs."
            )
        parameters[key] = parameter_value
    return parameters


def add_slurm_parameter_args(parser: argparse._ArgumentGroup) -> None:
    """Add allocation parameters without forwarding them to the worker.

    Args:
        parser: Launcher execution argument group.
    """
    parser.add_argument(
        "--additional-slurm-params",
        "--additional_slurm_params",
        type=_parse_additional_slurm_params,
        default={},
        help="Additional sbatch parameters as semicolon-separated KEY=VALUE pairs.",
    )
