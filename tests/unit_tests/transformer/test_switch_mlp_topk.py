# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

import pytest

import torch

from megatron.core.transformer.switch_mlp import SwitchMLP
from tests.unit_tests.test_utilities import Utils
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_layer_specs import gpt_layer_with_transformer_engine_spec_moe



import os
from torch import Tensor
from functools import partial
from typing import Union
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.blended_megatron_dataset_config import GPTDatasetConfig
from megatron.core.datasets.gpt_dataset import GPTDataset
import megatron.model
from megatron.core.models.gpt import GPTModel
from megatron.training import pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.utils import (
    get_ltor_masks_and_position_ids,
    get_batch_on_this_cp_rank,
    average_losses_across_data_parallel_group
)
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import (
    gpt_layer_with_transformer_engine_spec,
    gpt_layer_with_transformer_engine_spec_moe
)


import gc
from datetime import datetime
import math
import logging
import sys
import time
# The earliest we can measure the start time.
_TRAIN_START_TIME = time.time()
import torch

from megatron import get_args
from megatron import get_signal_handler
from megatron import get_timers
from megatron import get_tensorboard_writer
from megatron import get_wandb_writer
from megatron import get_current_global_batch_size
from megatron import get_num_microbatches
from megatron import is_last_rank
from megatron import update_num_microbatches
from megatron.core import mpu, tensor_parallel
from megatron.core.utils import get_model_config
from megatron import print_rank_0
from megatron import print_rank_last
from megatron.checkpointing import load_checkpoint
from megatron.checkpointing import save_checkpoint
from megatron.model import Float16Module
from megatron.model import GPTModel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import finalize_model_grads
from megatron.core.enums import ModelType
from megatron.optimizer import get_megatron_optimizer
from megatron.initialize import initialize_megatron
from megatron.initialize import write_args_to_tensorboard
from megatron.initialize import set_jit_fusion_options
from megatron.optimizer_param_scheduler import OptimizerParamScheduler
from megatron.utils import check_adlr_autoresume_termination
from megatron.utils import unwrap_model
from megatron.data.data_samplers import build_pretraining_data_loader
from megatron.utils import calc_params_l2_norm
from megatron.utils import throughput_calculator
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.utils import report_memory
from megatron.model.vision.knn_monitor import compute_feature_bank





Utils.initialize_model_parallel(1,1)
print('cucu1')
model_parallel_cuda_manual_seed(123)
print('cucu2')
print("done intializing")
transformer_config = TransformerConfig(num_layers=2, hidden_size=12, num_attention_heads=4, num_moe_experts= 2, use_cpu_initialization=True)
switch_mlp = SwitchMLP(transformer_config,
               gpt_layer_with_transformer_engine_spec_moe.submodules.mlp.submodules)

switch_mlp.cuda()
# [sequence length, batch size, hidden size]
hidden_states = torch.ones((32, 2, switch_mlp.config.hidden_size))
hidden_states = hidden_states.cuda()
output, output_bias = switch_mlp(hidden_states)
assert output.shape[0] == 32
assert output.shape[1] == 2
assert output.shape[2] == switch_mlp.config.hidden_size
assert output_bias.shape[2] == switch_mlp.config.hidden_size
assert output.dtype == torch.float32
assert output.device.type == 'cuda'
assert output_bias.device.type == 'cuda'

