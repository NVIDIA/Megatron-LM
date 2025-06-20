# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import re
from typing import Optional

import torch
from tqdm import tqdm

from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.trtllm.trtllm_layers import NON_TRANSFORMER_LAYERS_NAMES, TRTLLMLayers
from megatron.core.export.trtllm.trtllm_layers import get_layer_name_without_prefix as suffix
from megatron.core.export.trtllm.trtllm_weights_converter.utils import is_gated_activation
from megatron.core.transformer.transformer_config import TransformerConfig


# pylint: disable=line-too-long
# TODO: Writing TRT imports this way so that it can be mocked in the test_trtllm_cpu_converter.py unit test
# TODO: Figure out how to patch it directly from the trtllm library
def pad_vocab_size(vocab_size: int, tp_size: int):
    """Pad vocab size based on inference size"""
    from tensorrt_llm._utils import pad_vocab_size

    return pad_vocab_size(vocab_size, tp_size)


def str_dtype_to_torch(dtype: DataType):
    """Get torch datatype from input datatype"""
    from tensorrt_llm._utils import str_dtype_to_torch

    return str_dtype_to_torch(dtype.name)


class SingleDeviceTRTLLMModelWeightsConverter:
    """Class to convert Model weights to TRTLLM weights on CPU"""

    def __init__(
        self,
        export_config: ExportConfig,
        transformer_config: TransformerConfig,
        dtype: DataType,
        multi_query_mode: bool = False,
        activation: str = "gelu",
        scales: Optional[dict] = None,
    ):
        """Constructor for the TRTLLMModelWeightsConverterCPU class

        This class is responsible to convert the model weights to TRTLLM equivalent weights and also split them for each GPU rank and return as a list.

        Args:
            export_config (ExportConfig): The export config with inference tp size, pp size etc.
            transformer_config (TransformerConfig): The transformer config
            dtype (DataType): The data type or model precision
            multi_query_mode (bool, optional): Defaults to False.
            activation (str, optional): Defaults to "gelu".
            scales (dict, optional): Dictionary with fp8 scaling factors.
        """
        if scales is None:
            scales = {}

        self.export_config = export_config
        self.transformer_config = transformer_config
        self.trtllm_model_weights = {}
        self.storage_type = str_dtype_to_torch(dtype)
        self.activation = activation
        self.scales = scales
        num_kv_heads = self.transformer_config.num_query_groups
        if num_kv_heads == 0:
            if multi_query_mode:
                num_kv_heads = 1
            else:
                num_kv_heads = self.transformer_config.num_attention_heads
        self.num_kv_heads = num_kv_heads

    def _convert_non_transformer_layer(self, model_state_dict: dict, layer_name: str):
        """Convert Non Transformer layers to TRTLLM weights

        Non transformer layers referes to layers that occur only once in the model (e.g Embedding , final output layer etc. ) They dont have any layer number associated with them. We remove this layer from the original state dict and cast it to storage type and convert to numpy and add it to trtllm_model_weights

        Args:
            model_state_dict (dict): The input model state dictionary (All collected on CPU)
            layer_name (str): The TRTLLM Layer name that we want to convert
        """
        if layer_name in model_state_dict:
            val = model_state_dict.pop(layer_name)
            val = val.to(self.storage_type).detach().contiguous()
            self.trtllm_model_weights[layer_name] = val

    def _cast_value(self, val: torch.Tensor, layer_name: str) -> torch.Tensor:
        """Casts weights to the expected datatype.
            When appropriate scaling factor is found inside self.scales, the weight gets scaled before the cast.

        Args:
            val (torch.Tensor): Model weight
            layer_name (str): Layer name, used for determining the scaling factor dictionary key
        Returns:
            torch.Tensor: The casted weight
        """
        storage = self.storage_type

        scale_key = '.'.join(layer_name.split('.')[:-1]) + '.weights_scaling_factor'
        if scale_key in self.scales and layer_name.endswith("weight"):
            storage = torch.float8_e4m3fn
            val = val * self.scales[scale_key]['weight_multiplier'].to(val.device)

        return val.to(storage)

    def _convert_transformer_layer(self, layer_name: str, val: torch.Tensor):
        """Convert Transformer layers to TRTLLM weights

        Transformer layers referes to layers within the transformber block. They have a layer number associated with them. Depending on the layer we either directly save it to trtllm_model_weights, or split it across some dimension and save the splits

        Args:
            model_state_dict (dict): The input model state dictionary (All collected on CPU)
            layer (TRTLLMLayerNames): The TRTLLM Layer that we want to change
        """

        def _add_to_trtllm_model_weights(val: torch.Tensor, layer_name: str, split_type=None):
            """Add the input weight to trtllm_model_weights

            Depending on split (Expert split/Tensor split/None) we split the input data and add accordingly

            Args:
                val (torch.Tensor): The model weight to be added
                layer_name (str): The TRTLLMlayername as a string
                split_type (str, optional): The split type. Defaults to None.
            """
            if split_type == 'expert_split':
                for split_num, split_val in enumerate(val):
                    self.trtllm_model_weights[f'{layer_name}.{split_num}.bin'] = (
                        self._cast_value(split_val, layer_name).detach().contiguous()
                    )
            elif split_type == 'tensor_split':
                for split_num, split_val in enumerate(val):
                    if split_val.ndim >= 2:
                        split_val = torch.transpose(split_val.reshape(split_val.shape[0], -1), 1, 0)

                    self.trtllm_model_weights[f'{layer_name}.{split_num}.bin'] = (
                        self._cast_value(split_val, layer_name).detach().contiguous()
                    )
            else:
                if val.ndim >= 2:
                    val = torch.transpose(val.reshape(val.shape[0], -1), 1, 0)

                self.trtllm_model_weights[layer_name] = (
                    self._cast_value(val, layer_name).detach().contiguous()
                )

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
        ):
            # Same as layernorm1p in NeMo
            if (
                self.transformer_config.layernorm_zero_centered_gamma
                and self.transformer_config.normalization == "LayerNorm"
                and 'layernorm.weight' in layer_name
            ):
                val = val + 1.0

            _add_to_trtllm_model_weights(val=val, layer_name=layer_name, split_type=None)

        elif (
            layer_name.endswith(suffix(TRTLLMLayers.attention_dense_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.mlp_projection_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.ffn_projection_weight))
        ):
            split_vals = torch.chunk(val, self.export_config.inference_tp_size, axis=0)
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='tensor_split'
            )

        elif (
            layer_name.endswith(suffix(TRTLLMLayers.mlp_fc_weight))
            or layer_name.endswith(suffix(TRTLLMLayers.mlp_fc_bias))
            or layer_name.endswith(suffix(TRTLLMLayers.ffn_fc_weight))
        ):
            split_gated_activation = is_gated_activation(self)
            if split_gated_activation:
                val, gate = torch.chunk(val, 2, axis=-1)
                gate_layer_name = layer_name.replace("fc", "gate")
                split_vals = torch.chunk(gate, self.export_config.inference_tp_size, axis=-1)
                _add_to_trtllm_model_weights(
                    val=split_vals, layer_name=gate_layer_name, split_type='tensor_split'
                )

            split_vals = torch.chunk(val, self.export_config.inference_tp_size, axis=-1)
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='tensor_split'
            )

        elif layer_name.endswith(suffix(TRTLLMLayers.ffn_linear_weight)) or layer_name.endswith(
            suffix(TRTLLMLayers.attention_linear_weight)
        ):
            split_vals = torch.chunk(val, self.export_config.inference_tp_size, axis=-1)
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='tensor_split'
            )

        elif layer_name.endswith(suffix(TRTLLMLayers.attention_qkv_bias)):
            qkv_hidden_dim = val.shape[0]
            size_per_head = qkv_hidden_dim // (
                self.transformer_config.num_attention_heads + 2 * self.num_kv_heads
            )
            q_num = self.transformer_config.num_attention_heads // self.num_kv_heads

            # We first concat all sub weights per tp rank together.
            val = val.reshape(self.num_kv_heads, q_num + 2, size_per_head)

            qkv = torch.split(val, [q_num, 1, 1], dim=1)
            q_split = torch.chunk(qkv[0], self.export_config.inference_tp_size, axis=0)
            k_split = torch.chunk(qkv[1], self.export_config.inference_tp_size, axis=0)
            v_split = torch.chunk(qkv[2], self.export_config.inference_tp_size, axis=0)

            # Concatenate Q, K, and V together
            split_vals = [
                torch.concatenate(
                    [q_split[i].reshape(-1), k_split[i].reshape(-1), v_split[i].reshape(-1)], dim=0
                )
                for i in range(self.export_config.inference_tp_size)
            ]
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='tensor_split'
            )

        # TODO : Should add a atten layer dimension "qkvqkv, qqkkvv etc to see how to reshape here"
        elif layer_name.endswith(suffix(TRTLLMLayers.attention_qkv_weight)):
            hidden_dim = val.shape[0]
            size_per_head = self.transformer_config.kv_channels
            if size_per_head is None:
                size_per_head = hidden_dim // self.transformer_config.num_attention_heads
            q_num = self.transformer_config.num_attention_heads // self.num_kv_heads

            # When the merge factor exceeds 1, the 'vals' list will have multiple entries.
            # Depending on the format, 'vals' can look like either [QQQQ..KV, QQQQ..KV, ...](for GQA) or [QKV, QKV, ...](for MHA).
            # We first concat all sub weights per tp rank together.
            val = val.reshape(hidden_dim, self.num_kv_heads, q_num + 2, size_per_head)

            # Split the QKV to separate variables.
            qkv = torch.split(val, [q_num, 1, 1], dim=2)

            query_groups_shape = qkv[0].shape
            if len(query_groups_shape) > 1:
                if (query_groups_shape[1] % self.export_config.inference_tp_size) != 0:
                    raise Exception(
                        "Number of query groups of the models is {0}. Please select tensor parallelism size "
                        "that can split the number of query groups to equal number of query matrices in the "
                        "each GPU.".format(query_groups_shape[1])
                    )

            q_split = torch.chunk(qkv[0], self.export_config.inference_tp_size, axis=1)
            k_split = torch.chunk(qkv[1], self.export_config.inference_tp_size, axis=1)
            v_split = torch.chunk(qkv[2], self.export_config.inference_tp_size, axis=1)

            # Concatenate Q, K, and V together
            split_vals = [
                torch.concatenate(
                    [
                        q_split[i].reshape(hidden_dim, -1),
                        k_split[i].reshape(hidden_dim, -1),
                        v_split[i].reshape(hidden_dim, -1),
                    ],
                    dim=1,
                )
                for i in range(self.export_config.inference_tp_size)
            ]
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='tensor_split'
            )

        elif layer_name.endswith(suffix(TRTLLMLayers.mlp_fc_weight_mixture_of_experts)):
            w1, w3 = torch.chunk(val, 2, axis=1)
            # w1 splits
            split_w1s = torch.chunk(w1, self.export_config.inference_tp_size, axis=1)
            # w3 splits
            split_w3s = torch.chunk(w3, self.export_config.inference_tp_size, axis=1)

            split_vals = [torch.concatenate(item, dim=1) for item in zip(split_w3s, split_w1s)]
            layer_name = layer_name.replace(".expert", "")  # Remove suffix .expert from key
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='expert_split'
            )

        elif layer_name.endswith(suffix(TRTLLMLayers.mlp_projection_weight_mixture_of_experts)):
            split_vals = torch.chunk(val, self.export_config.inference_tp_size, axis=-1)
            layer_name = layer_name.replace(".expert", "")  # Remove suffix .expert from key
            _add_to_trtllm_model_weights(
                val=split_vals, layer_name=layer_name, split_type='expert_split'
            )
        else:
            raise ValueError(f"{layer_name} cannot be handled by converter")

    @torch.no_grad()
    def convert(
        self, model_state_dict: dict, trtllm_conversion_dict, state_dict_split_by_layer_numbers=True
    ):
        """Convert model weights to trtllm model weights

        This method goes through each layer in the model state dict and converts to equivalent trtllm model weights. It also handles splitting across TP dimension , expert split etc.

        Args:
            model_state_dict (dict): The full model state dict (all on CPU)
            trtllm_conversion_dict (dict): The conversion dictionary used to convert model layer names to trtllm layer names
            state_dict_split_by_layer_numbers (bool, optional): Are the model layers split by layer numbers in state dict. For example : mlp.fc1.weight can be represented like mlp.fc1.weight of shape [num_layers, hidden_dim, ffn_hidden_dim]} or it can be like mlp.fc1.layers.0.weight of shape [hidden_dim, ffn_hidden_dim], then mlp.fc1.layers.1.weight ... for all layers. If you use represenation 2 set this to True. Defaults to True
        """

        # First step is to convert input model layer names to equivalent trtllm layer names
        model_state_dict = TRTLLMLayers.rename_input_layer_names_to_trtllm_layer_names(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=trtllm_conversion_dict,
            state_dict_split_by_layer_numbers=state_dict_split_by_layer_numbers,
        )

        # Convert the non transformer layers
        for layer_name in NON_TRANSFORMER_LAYERS_NAMES:
            # For vocab embedding layer alone we pad the weights to be divisible by inference tp size
            if (
                layer_name == TRTLLMLayers.vocab_embedding.value
                and self.export_config.use_parallel_embedding
            ):
                val = model_state_dict[TRTLLMLayers.vocab_embedding.value]
                vocab_size = val.shape[0]
                if vocab_size % self.export_config.inference_tp_size != 0:
                    vocab_size_padded = pad_vocab_size(
                        vocab_size, self.export_config.inference_tp_size
                    )
                    pad_width = vocab_size_padded - vocab_size
                    val = torch.nn.functional.pad(val, (0, 0, 0, pad_width), value=0)
                    model_state_dict[layer_name] = val
            if layer_name == TRTLLMLayers.final_layernorm_weight.value:
                # Same as layernorm1p in NeMo
                if (
                    self.transformer_config.layernorm_zero_centered_gamma
                    and self.transformer_config.normalization == "LayerNorm"
                ):
                    model_state_dict[layer_name] = model_state_dict[layer_name] + 1.0

            self._convert_non_transformer_layer(
                model_state_dict=model_state_dict, layer_name=layer_name
            )

        transformer_layers_dict = {}
        # Convert the transformer layers
        if state_dict_split_by_layer_numbers:
            # Already model dict is split by layer numbers
            transformer_layers_dict = model_state_dict
        else:
            # Here we split the model state dict into individual layers
            for layer_name in list(model_state_dict.keys()):
                value = model_state_dict.pop(layer_name)
                for layer_number in range(self.transformer_config.num_layers):
                    # e.g transformer.layers.mlp.fc.bias => transformer.layers.2.mlp.fc.bias
                    layer_name_with_layer_number = re.sub(
                        r'(?<=layers\.)', f'{layer_number}.', layer_name
                    )
                    transformer_layers_dict[layer_name_with_layer_number] = value[layer_number]

        for layer_name, value in tqdm(
            transformer_layers_dict.items(), desc="Converting to TRTLLM Weights"
        ):
            self._convert_transformer_layer(layer_name, value)

    def get_padded_vocab_size(self) -> int:
        """Return the paded vocab size

        We extract the lm head and vocab embedding and use that to determine padded_vocab_size

        Returns:
            int: Padded vocab size
        """
        lm_head_weight = self.trtllm_model_weights.get(TRTLLMLayers.lm_head.value, None)
        vocab_size = self.trtllm_model_weights[TRTLLMLayers.vocab_embedding.value].shape[0]
        vocab_size_padded = (
            vocab_size
            if lm_head_weight is None
            else pad_vocab_size(vocab_size, self.export_config.inference_tp_size)
        )
        return vocab_size_padded

    def get_local_model_weights_per_gpu(self, mapping, trtllm_model_config: dict):
        """Get the trtllm model weights split per gpu

        Given the trtllm mapping information (tp, pp rank etc) we split the model weights in a list, with each element of the list corresponding to the weights of each gpu rank

        Args:
            mapping : The trtllm mapping information
            trtllm_model_config (dict): The trtllm model config
        """

        def _split(torch_tensor, tp_size, idx, dim=0):
            """Splits the np tensor v on dim and return the idx's slice."""
            if tp_size == 1:
                return torch_tensor
            if len(torch_tensor.shape) == 1:
                return torch.chunk(torch_tensor, tp_size)[idx].contiguous()
            else:
                return torch.chunk(torch_tensor, tp_size, axis=dim)[idx].contiguous()

        pp_layer_range = mapping.pp_layers(self.transformer_config.num_layers)

        trtllm_model_weights_per_gpu = {}
        for layer_name, value in self.trtllm_model_weights.items():
            if layer_name in NON_TRANSFORMER_LAYERS_NAMES:
                continue

            # Happens in the case of TP split or expert split
            if layer_name.endswith(".bin"):
                if layer_name.endswith(f"{mapping.tp_rank}.bin"):
                    layer_name = layer_name.replace(f".{mapping.tp_rank}.bin", "")
                else:
                    continue

            layer_num = int(layer_name.split(".")[2])
            if layer_num in pp_layer_range:
                layer_name = layer_name.replace(
                    f"layers.{layer_num}", f"layers.{layer_num - pp_layer_range[0]}"
                )
            else:
                continue
            if (
                hasattr(trtllm_model_config, 'new_decoder_architecture')
                and trtllm_model_config.new_decoder_architecture
                and "post_layernorm" in layer_name
            ):
                layer_name = layer_name.replace("post_layernorm", "mlp_layernorm")

            trtllm_model_weights_per_gpu[layer_name] = value

        if mapping.is_first_pp_rank():
            embedding_weight = (
                _split(
                    self.trtllm_model_weights[TRTLLMLayers.vocab_embedding.value],
                    mapping.tp_size,
                    mapping.tp_rank,
                )
                if self.export_config.use_parallel_embedding
                else self.trtllm_model_weights[TRTLLMLayers.vocab_embedding.value]
            )

            trtllm_model_weights_per_gpu[TRTLLMLayers.vocab_embedding.value] = embedding_weight

            pos_embedding_weight = self.trtllm_model_weights.get(
                TRTLLMLayers.position_embedding.value
            )
            if pos_embedding_weight is not None:
                if self.export_config.use_parallel_embedding:
                    pos_embedding_weight = _split(
                        pos_embedding_weight, mapping.tp_size, mapping.tp_rank
                    )

                trtllm_model_weights_per_gpu[TRTLLMLayers.position_embedding.value] = (
                    pos_embedding_weight
                )

        if mapping.is_last_pp_rank():
            lm_head_weight = self.trtllm_model_weights.get(TRTLLMLayers.lm_head.value, None)
            if lm_head_weight is not None:
                trtllm_model_weights_per_gpu[TRTLLMLayers.lm_head.value] = _split(
                    lm_head_weight, mapping.tp_size, mapping.tp_rank
                )

            trtllm_model_weights_per_gpu[TRTLLMLayers.final_layernorm_weight.value] = (
                self.trtllm_model_weights[TRTLLMLayers.final_layernorm_weight.value]
            )

            ln_f_bias = self.trtllm_model_weights.get(TRTLLMLayers.final_layernorm_bias.value)
            if ln_f_bias is not None:
                trtllm_model_weights_per_gpu[TRTLLMLayers.final_layernorm_bias.value] = ln_f_bias

        return trtllm_model_weights_per_gpu
