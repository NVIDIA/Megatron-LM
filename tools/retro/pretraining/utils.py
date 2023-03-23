# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import os

from megatron import get_retro_args


def get_pretraining_workdir():
    args = get_retro_args()
    return os.path.join(args.retro_workdir, "pretraining")
