# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import megatron.training.arguments


def _add_dmc_args(parser):
    group = parser.add_argument_group(title='dmc')
    group.add_argument('--dmc-init-val', type=float, default=5)
    group.add_argument('--dmc-window-size', type=int, default=12)
    group.add_argument('--dmc-temp', type=float, default=0.1)
    group.add_argument('--dmc-cr', type=float, default=4.0)
    group.add_argument('--dmc-finetune', action='store_true')
    group.add_argument('--dmc-paged-cache-size', type=float, default=40.0)  # in GiB
    group.add_argument('--dmc-paged-block-size', type=int, default=256)
    group.add_argument('--dmc-is-stage-one', action='store_true')

    return parser


def add_dmc_args():
    megatron.training.arguments.__add_experimental_args = (
        megatron.training.arguments._add_experimental_args
    )

    def _add_experimental_args_with_dmc(parser):
        parser = megatron.training.arguments.__add_experimental_args(parser)
        parser = _add_dmc_args(parser)
        return parser

    megatron.training.arguments._add_experimental_args = _add_experimental_args_with_dmc
