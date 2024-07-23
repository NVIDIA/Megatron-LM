# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import logging
from enum import Enum
from typing import List, Optional, Tuple

import numpy
import torch

from ..utils import log_single_rank

logger = logging.getLogger(__name__)


class Split(Enum):
    train = 0
    valid = 1
    test = 2


def compile_helpers():
    """Compile C++ helper functions at runtime. Make sure this is invoked on a single process.
    """
    import os
    import subprocess

    command = ["make", "-C", os.path.abspath(os.path.dirname(__file__))]
    if subprocess.run(command).returncode != 0:
        import sys

        log_single_rank(logger, logging.ERROR, "Failed to compile the C++ dataset helper functions")
        sys.exit(1)


def normalize(weights: List[float]) -> List[float]:
    """Do non-exponentiated normalization

    Args:
        weights (List[float]): The weights

    Returns:
        List[float]: The normalized weights
    """
    w = numpy.array(weights, dtype=numpy.float64)
    w_sum = numpy.sum(w)
    w = (w / w_sum).tolist()
    return w


def get_cu_seqlens(token_ids: torch.Tensor, delim_bos: Optional[int], delim_eos: Optional[int]) -> torch.Tensor:
    """Get the cumulative sequence lengths required for 'thd' attention as specified by
    Transformer Engine DotProductAttention for a GPT-like causal auto-regressive masking scheme

    Args:
        token_ids (torch.Tensor): The token tensor in (batch, sequence) format

        delim_bos (Optional[int]): The initial prefix delimiting token id, required if eos is None

        delim_eos (Optional[int]): The terminal suffix delimiting token id, required if bos is None

    Returns:
        torch.Tensor: The cumulative sequence lengths for tokens according to bos/eos
    
    """
    dim_b = token_ids.shape[0]
    dim_s = token_ids.shape[0]

    # We deliminate documents with the bos token
    if delim_bos is not None:
        start_mask = torch.zeros_like(token_ids)
        start_mask[:, 0] = 1
        start_mask = torch.logical_or(start_mask, token_ids == delim_bos)
        cu_seqlens = torch.argwhere(start_mask)
        cu_seqlens = cu_seqlens[:, 1] + (cu_seqlens[:, 0] * dim_s)
        cu_seqlens = cu_seqlens[1:]
        cu_seqlens = torch.concat(
            [
                cu_seqlens, 
                torch.tensor([dim_b * dim_s], dtype=cu_seqlens.dtype, device=cu_seqlens.device)
            ]
        )

    # We deliminate documents with the eos token
    elif delim_eos is not None:
        end_mask = torch.zeros_like(token_ids)
        end_mask[:, -1] = 1
        end_mask = torch.logical_or(end_mask, token_ids == delim_eos)
        cu_seqlens = torch.argwhere(end_mask)
        cu_seqlens = cu_seqlens[:, 1] + (cu_seqlens[:, 0] * dim_s) + 1

    # We don't care about intra-sequence inter-document breaks
    else:
        cu_seqlens = (torch.arange(dim_b, device=token_ids.device) + 1) * dim_s

    return cu_seqlens


def get_blend_from_list(
    blend: Optional[List[str]],
) -> Optional[Tuple[List[str], Optional[List[float]]]]:
    """Get the megatron.core.datasets.blended_megatron_dataset_config.BlendedMegatronDatasetConfig blend from the blend list
    
    Args:
        blend (Optional[List[str]]): The blend list, which can be either (1) a list of prefixes, e.g. ["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], or (2) a flattened, zipped list of weights and prefixes, e.g. ["30", "path/to/dataset_1_prefix", "70", "path/to/dataset_2_prefix"]

    Returns:
        Optional[Tuple[List[str], Optional[List[float]]]]: The blend, consisting of a list of dataset prefixes and optionally a list of dataset weights, e.g. [["path/to/dataset_1_prefix", "path/to/dataset_2_prefix"], [30.0, 70.0]].
    """
    if blend is None:
        return None

    if len(blend) % 2 == 1:
        weight_per_dataset = None
        raw_prefix_per_dataset = blend
    else:
        raw_weight_per_dataset, raw_prefix_per_dataset = zip(
            *[(blend[i], blend[i + 1]) for i in range(0, len(blend), 2)]
        )

        weight_per_dataset = []
        for rwpd in raw_weight_per_dataset:
            try:
                weight = float(rwpd)
            except ValueError:
                weight = None
            weight_per_dataset.append(weight)

        is_none = map(lambda _: _ is None, weight_per_dataset)
        if any(is_none):
            assert all(is_none)
            weight_per_dataset = None
            raw_prefix_per_dataset = blend

    prefix_per_dataset = [rppd.strip() for rppd in raw_prefix_per_dataset]

    return prefix_per_dataset, weight_per_dataset
