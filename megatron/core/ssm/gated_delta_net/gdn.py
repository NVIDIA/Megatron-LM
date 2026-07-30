# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Copyright (c) 2025, Songlin Yang, Jan Kautz, Ali Hatamizadeh.

# Some of this code was adopted from https://github.com/huggingface/transformers
# This source code is licensed under the Apache license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=unused-import

"""GatedDeltaNet layer.

The full ``GatedDeltaNet`` implementation lives in
:mod:`megatron.core.ssm.gated_delta_net.common`; this module re-exports it so the
``megatron.core.ssm.gated_delta_net.gdn`` import path keeps working.
"""

from megatron.core.ssm.gated_delta_net.common import GatedDeltaNet

__all__ = ["GatedDeltaNet"]
