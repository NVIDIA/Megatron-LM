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

_LAYER_NORM = None


def import_layernorm(fp32_residual_connection):

    global _LAYER_NORM
    if not _LAYER_NORM:
        if fp32_residual_connection:
            from .fused_layer_norm import MixedFusedLayerNorm as LayerNorm
        else:
            from apex.normalization.fused_layer_norm import FusedLayerNorm as LayerNorm
        _LAYER_NORM = LayerNorm
            
    return _LAYER_NORM


from .distributed import *
from .bert_model import (BertModel,
                         BertModelFirstStage,
                         BertModelIntermediateStage,
                         BertModelLastStage)
from .realm_model import ICTBertModel
from .gpt_model import (GPTModel,
                        GPTModelFirstStage,
                        GPTModelIntermediateStage,
                        GPTModelLastStage)
from .language_model import get_language_model
from .module import FP16Module
from .realm_model import ICTBertModel


