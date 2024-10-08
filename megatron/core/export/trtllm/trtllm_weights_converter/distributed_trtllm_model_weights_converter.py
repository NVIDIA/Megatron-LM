# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch
from tqdm import tqdm

from megatron.core import parallel_state
from megatron.core.export.data_type import DataType
from megatron.core.export.trtllm.trtllm_layers import NON_TRANSFORMER_LAYERS_NAMES, TRTLLMLayers
from megatron.core.export.trtllm.trtllm_layers import get_layer_name_without_prefix as suffix
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.core.transformer.transformer_config import TransformerConfig


def str_dtype_to_torch(dtype: DataType):
    """Get torch datatype from input datatype"""
    from tensorrt_llm._utils import str_dtype_to_torch

    return str_dtype_to_torch(dtype.name)


# pylint: disable=line-too-long
class DistributedTRTLLMModelWeightsConverter:
    """The TRTLLM Converter class used for GPU (on device) conversion

    This class is used to convert models sharded and on gpus. (It assumes that the model is already sharded appropriate to how you want to export it). (i.e) If you want to export to tp2pp2, then load the model in tp2pp2 setting and pass in their respective state dictionaries
    """

    def __init__(
        self,
        transformer_config: TransformerConfig,
        dtype: DataType,
        multi_query_mode: bool = False,
        activation: str = "gelu",
    ):
        """Constructor for the TRTLLMModelWeightsConverterGPU class

        This class is responsible to convert the model weights to TRTLLM equivalent weights.

        Args:
            transformer_config (TransformerConfig): The transformer config
            dtype (DataType): The data type or model precision
            multi_query_mode (bool, optional): Defaults to False.
            activation (str, optional): Defaults to "gelu".
        """
        self.transformer_config = transformer_config
        self.trtllm_model_weights = {}
        self.storage_type = str_dtype_to_torch(dtype)
        self.activation = activation
        num_kv_heads = self.transformer_config.num_query_groups
        if num_kv_heads == 0:
            if multi_query_mode:
                num_kv_heads = 1
            else:
                num_kv_heads = self.transformer_config.num_attention_heads
        self.num_kv_heads = num_kv_heads

        self.inference_pp_size = parallel_state.get_pipeline_model_parallel_world_size()
        self.inference_tp_size = parallel_state.get_tensor_model_parallel_world_size()
        self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
        self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        self.tp_group = parallel_state.get_tensor_model_parallel_group()
        vp_size = parallel_state.get_virtual_pipeline_model_parallel_world_size()

        assert (
            vp_size is None or vp_size == 1
        ), "Virtual parallelism is not supported in GPU Converter. Gather the VP chunks and use PP config."

    def _add_to_trtllm_model_weights(self, val: torch.Tensor, layer_name: str):
        assert torch.is_tensor(val), f"Expected a tensor for {layer_name} but got {type(val)}"
        val = val.to(self.storage_type)
        val = val.detach().contiguous()
        if val.ndim >= 2:
            val = torch.transpose(val.reshape(val.shape[0], -1), 0, 1)
        if layer_name not in self.trtllm_model_weights:
            self.trtllm_model_weights[layer_name] = torch.empty(
                val.size(), dtype=val.dtype, layout=val.layout, device="cpu", pin_memory=True
            )
        self.trtllm_model_weights[layer_name] = val

    def _convert_transformer_layer(self, layer_name: str, val: torch.Tensor):
        """Convert Transformer layers to TRTLLM weights

        Transformer layers referes to layers within the transformber block. They have a layer number associated with them. Depending on the layer we either directly save it to trtllm_model_weights, or split it across some dimension and save the splits

        Args:
            model_state_dict (dict): The input model state dictionary (All collected on CPU)
            layer (TRTLLMLayerNames): The TRTLLM Layer that we want to change
        """
        if val.ndim == 2:
            val = val.T

        if (
            layer_name.endswith(suffix(TRTLLMLayers.input_layernorm_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.input_layernorm_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.post_layernorm_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.post_layernorm_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.attention_dense_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.attention_dense_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.mlp_projection_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.mlp_router_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.attention_dense_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.mlp_projection_weight))
        ):
            # Same as layernorm1p in NeMo
            if (
                self.transformer_config.layernorm_zero_centered_gamma
                and self.transformer_config.normalization == "LayerNorm"
                and 'layernorm.weight' in layer_name
            ):
                val = val + 1.0

            self._add_to_trtllm_model_weights(val=val, layer_name=layer_name)

        elif layer_name.endswith(suffix(TRTLLMLayers.mlp_fc_weight)) or layer_name.endswith(
            suffix(TRTLLMLayers.mlp_fc_bias)
        ):

            split_gated_activation = self.activation in [
                "swiglu",
                "geglu",
                "fast-swiglu",
                "fast-geglu",
            ]
            if split_gated_activation:
                vals, gates = [[n] for n in torch.chunk(val, 2, axis=-1)]
                gate_layer_name = layer_name.replace("fc", "gate")
                self._add_to_trtllm_model_weights(val=gates[0], layer_name=gate_layer_name)
                val = vals[0]

            self._add_to_trtllm_model_weights(val=val, layer_name=layer_name)

        elif layer_name.endswith(suffix(TRTLLMLayers.attention_qkv_bias)):
            qkv_hidden_dim = val.shape[0]
            size_per_head = (
                qkv_hidden_dim
                // (self.transformer_config.num_attention_heads + 2 * self.num_kv_heads)
                * self.inference_tp_size
            )
            q_num = self.transformer_config.num_attention_heads // self.num_kv_heads

            # We first concat all sub weights per tp rank together.
            val = val.reshape(self.num_kv_heads // self.inference_tp_size, q_num + 2, size_per_head)
            qkv = torch.split(val, [q_num, 1, 1], dim=1)
            split_vals = torch.concatenate(
                [qkv[0].reshape(-1), qkv[1].reshape(-1), qkv[2].reshape(-1)], dim=0
            )
            self._add_to_trtllm_model_weights(val=split_vals, layer_name=layer_name)

        # TODO : Should add a atten layer dimension "qkvqkv, qqkkvv etc to see how to reshape here"
        elif layer_name.endswith(suffix(TRTLLMLayers.attention_qkv_weight)):
            hidden_dim = val.shape[0]
            size_per_head = self.transformer_config.kv_channels
            if size_per_head is None:
                size_per_head = hidden_dim // self.transformer_config.num_attention_heads
            q_num = self.transformer_config.num_attention_heads // self.num_kv_heads

            val = val.reshape(
                hidden_dim, self.num_kv_heads // self.inference_tp_size, q_num + 2, size_per_head
            )
            qkv = torch.split(val, [q_num, 1, 1], dim=2)
            split_vals = torch.concatenate(
                [
                    qkv[0].reshape(hidden_dim, -1),
                    qkv[1].reshape(hidden_dim, -1),
                    qkv[2].reshape(hidden_dim, -1),
                ],
                dim=1,
            )
            self._add_to_trtllm_model_weights(val=split_vals, layer_name=layer_name)

        else:
            raise ValueError(f"{layer_name} cannot be handled by GPU converter")

    def _convert_non_transformer_layer(self, model_state_dict: dict, layer_name: str):
        """Convert Non Transformer layers to TRTLLM weights

        Non transformer layers referes to layers that occur only once in the model (e.g Embedding , final output layer etc. ) They dont have any layer number associated with them. We remove this layer from the original state dict and cast it to storage type and convert to numpy and add it to trtllm_model_weights

        Args:
            model_state_dict (dict): The input model state dictionary (All collected on CPU)
            layer (TRTLLMLayerNames): The TRTLLM Layer that we want to change
        """
        if layer_name in model_state_dict:
            val = model_state_dict.pop(layer_name)
            self._add_to_trtllm_model_weights(val=val, layer_name=layer_name)

    # ----------------Convert Embeddings----------------
    def _get_remove_vocab_padding(self, layer_name, model_state_dict, tokenizer_vocab_size):
        val = model_state_dict.get(layer_name, None)
        if val is None:
            return None

        if self.inference_tp_size > 1:  # Gather padded tensor chunks
            vocab_size_padded = val.shape[0] * self.inference_tp_size
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                vocab_size_padded, self.tp_rank, self.inference_tp_size
            )
            dim_size = list(val.size())
            dim_size[0] = vocab_size_padded
            gathered_val = torch.zeros(
                dim_size, dtype=val.dtype, device=torch.cuda.current_device()
            )
            gathered_val[vocab_start_index:vocab_end_index] = val
            torch.distributed.all_reduce(gathered_val, group=self.tp_group)
            val = gathered_val
        unpadded = val[:tokenizer_vocab_size]
        if self.inference_tp_size > 1:  # Split gathered val for val parallel embedding
            vocab_start_index, vocab_end_index = VocabUtility.vocab_range_from_global_vocab_size(
                tokenizer_vocab_size, self.tp_rank, self.inference_tp_size
            )
            unpadded = unpadded[vocab_start_index:vocab_end_index]
        return unpadded.T  # TRTLLM expects (vocab_size, hidden_size) so need extra transpose

    @torch.no_grad()
    def convert(
        self, model_state_dict: dict, trtllm_conversion_dict: dict, tokenizer_vocab_size: int
    ):
        """Convert model weights to trtllm model weights

        This method goes through each layer in the model state dict and converts to equivalent trtllm model weights. It also handles splitting across TP dimension , expert split etc.

        Args:
            model_state_dict (dict): The full model state dict (all on CPU)
            trtllm_conversion_dict (dict): The conversion dictionary used to convert model layer names to trtllm layer names
            tokenizer_vocab_size (int): The vocab size of the tokenizer
        """

        # First step is to convert input model layer names to equivalent trtllm layer names
        model_state_dict = TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
            model_state_dict=model_state_dict, trtllm_conversion_dict=trtllm_conversion_dict
        )

        # Convert the non transformer layers
        for layer_name in NON_TRANSFORMER_LAYERS_NAMES:
            if (
                layer_name in TRTLLMLayers.vocab_embedding.value
                or layer_name in TRTLLMLayers.lm_head.value
            ):
                # For embedding layers alone we do some pre processing
                embed_val = self._get_remove_vocab_padding(
                    layer_name, model_state_dict, tokenizer_vocab_size
                )
                model_state_dict[layer_name] = embed_val
            # TODO : Check if this handling of position embedding is right.
            if layer_name == TRTLLMLayers.position_embedding.value:
                position_embedding = model_state_dict[layer_name]
                req_position_embedding = position_embedding.chunk(self.inference_tp_size)[
                    self.tp_rank
                ]
                model_state_dict[layer_name] = req_position_embedding.T
            self._convert_non_transformer_layer(
                model_state_dict=model_state_dict, layer_name=layer_name
            )

        for layer_name, value in tqdm(
            model_state_dict.items(), desc="Converting to TRTLLM Weights"
        ):
            self._convert_transformer_layer(layer_name, value)
