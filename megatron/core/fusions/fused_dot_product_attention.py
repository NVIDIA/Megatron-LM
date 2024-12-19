# Copyright (C) 2024 Habana Labs, Ltd. an Intel Company.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from torch import Tensor

from megatron.core import parallel_state
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.utils import divide


class FusedDotProductAttention(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: float = None,
    ):
        super(FusedDotProductAttention, self).__init__(config=config)

        self.config: TransformerConfig = config

        assert (
            self.config.context_parallel_size == 1
        ), "Context parallelism is not supported by FusedDotProductAttention!"

        assert (
            self.config.window_size is None
        ), "Sliding Window Attention is not supported by FusedDotProductAttention!"

        from habana_frameworks.torch.hpex.kernels import FusedSDPA
        self.fused_sdpa = FusedSDPA
        self.attn_mask_type = attn_mask_type
        self.attention_type = attention_type  # unused for now
        if attention_dropout is not None:
            self.config.attention_dropout = attention_dropout
        self.use_fast_softmax = "fast" if config.use_fast_softmax is True else "None"

        projection_size = config.kv_channels * config.num_attention_heads

        # Per attention head and per partition values.
        world_size = parallel_state.get_tensor_model_parallel_world_size()
        self.hidden_size_per_partition = divide(projection_size, world_size)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor,
        attn_mask_type: AttnMaskType = None,
        packed_seq_params: PackedSeqParams = None,
        ):
        assert packed_seq_params is None, (
            "Packed sequence is not supported by FusedDotProductAttention."
        )

        # [sq, b, np, hn] -> [b, np, sq, hn]
        q, k, v = [x.transpose(0, 1).transpose(1, 2) for x in [query, key, value]]
        causal = True
        scale = None
        attn_mask = None
        context_layer = self.fused_sdpa.apply(
            q, k, v, attn_mask, self.config.attention_dropout, causal, scale,
            self.use_fast_softmax, self.config.use_fused_sdpa_with_recompute
        )

        # [b, np, sq, hn] --> [sq, b, np, hn]
        context_layer = context_layer.permute(2, 0, 1, 3).contiguous()

        # [sq, b, np, hn] --> [sq, b, hp]
        new_context_layer_shape = context_layer.size()[:-2] + \
            (self.hidden_size_per_partition,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer
