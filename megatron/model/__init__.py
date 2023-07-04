# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

from deepspeed.accelerator.real_accelerator import get_accelerator
if get_accelerator().device_name() == 'cuda':
    from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
else:
    from torch.nn import LayerNorm

from .distributed import DistributedDataParallel
from .bert_model import BertModel
from .gpt_model import GPTModel, GPTModelPipe
from .t5_model import T5Model
from .language_model import get_language_model
from .module import Float16Module
