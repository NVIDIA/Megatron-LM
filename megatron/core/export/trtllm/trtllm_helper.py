# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import tensorrt_llm
from tensorrt_llm.functional import non_gated_version
from tensorrt_llm.layers import MoeConfig

from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.model_type import ModelType
from megatron.core.export.trtllm.engine_builder.trtllm_engine_builder import TRTLLMEngineBuilder
from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
    DEFAULT_CONVERSION_DICT,
)
from megatron.core.export.trtllm.trt_model_config import TRT_MODEL_CONFIG
from megatron.core.export.trtllm.trt_model_type import TRT_MODEL_TYPE_STRING

# pylint: disable=line-too-long
from megatron.core.export.trtllm.trtllm_weights_converter.distributed_trtllm_model_weights_converter import (
    DistributedTRTLLMModelWeightsConverter,
)
from megatron.core.export.trtllm.trtllm_weights_converter.single_device_trtllm_model_weights_converter import (
    SingleDeviceTRTLLMModelWeightsConverter,
)
from megatron.core.transformer.transformer_config import TransformerConfig


class TRTLLMHelper:
    """TRTLLM Helper class to convert export and build TRTLLM model."""

    def __init__(
        self,
        transformer_config: TransformerConfig,
        model_type: ModelType,
        trtllm_conversion_dict: dict = {},
        position_embedding_type: str = 'learned_absolute',
        max_position_embeddings: int = None,
        rotary_percentage: int = 1.0,
        rotary_base: int = 10000,
        moe_tp_mode: int = 2,
        multi_query_mode: bool = False,
        activation: str = "gelu",
        seq_len_interpolation_factor: float = None,
        moe_renorm_mode=None,
        share_embeddings_and_output_weights=False,
    ):
        """Constructor for the TRTLLMHelper

        There are two public API's supported  by this helper.
        a) get_trtllm_pretrained_config_and_model_weights
        b) build_and_save_engine

        Args:
            transformer_config (TransformerConfig): The transformer config
            model_type (ModelType): The type of the input model. Enum (megatron.core.export.model_type.ModelType)
            conversion_dict (dict, optional): A conversion dictionary that will map your model layer names to trtllm equivalent layer names. Sample dictionaries are given megatron/core/export/model_mapping. NOTE: Ingore layer numbers in the model layer names. (e.g) decoder.layers.0.attention_qkv.weight will be decoder.layers.attention_qkv.weight in the mapping dictionary. Defaults to {}.
            position_embedding_type (str, optional): The position embedding type. Defaults to None.
            max_position_embeddings (int, optional): Max posistion embeddings value. Defaults to None.
            rotary_percentage (int, optional): The rotary percentage if using rope embedding. Defaults to 1.0.
            rotary_base (int, optional): The rotary base (theta value) if using rope embeddings. Defaults to 10000.
            moe_tp_mode (int, optional): TRTLLM Config. Defaults to 2.
            multi_query_mode (bool, optional): Defaults to False.
            activation (str, optional): Defaults to "gelu".
            seq_len_interpolation_factor (float, optional): The sequence length interpolation factor if using rope embeddings. Defaults to None.
            moe_renorm_mode (optional) : Renormalization mode if using mixture of experts. Defaults to None.
            share_embeddings_and_output_weights (bool, optional): True if input and output layers share weights. Defaults to False.
        """

        self.transformer_config = transformer_config
        self.model_type = model_type
        self.trtllm_conversion_dict = DEFAULT_CONVERSION_DICT[model_type]
        self.trtllm_conversion_dict.update(trtllm_conversion_dict)
        assert position_embedding_type in [
            'learned_absolute',
            'rope',
        ], f"Position embedding type should be one of learned_absolute, rope. You entered {position_embedding_type}"
        self.position_embedding_type = position_embedding_type
        self.max_position_embeddings = max_position_embeddings
        self.rotary_percentage = rotary_percentage
        self.rotary_base = rotary_base
        self.moe_tp_mode = moe_tp_mode
        self.multi_query_mode = multi_query_mode
        self.activation = activation
        self.seq_len_interpolation_factor = seq_len_interpolation_factor
        self.moe_renorm_mode = moe_renorm_mode
        self.share_embeddings_and_output_weights = share_embeddings_and_output_weights

    def _get_trtllm_config(
        self,
        export_config: ExportConfig,
        world_size: int,
        gpus_per_node: int,
        vocab_size_padded: int,
        dtype: DataType,
    ):
        """Get TRTLLM Config

        Returns appropriate TRTLLM PretrainedConfig used by TRTLLM for building engine

        Args:
            export_config (ExportConfig): The export config that defines inference tp , pp size etc.
            world_size (int): The number of gpus (Mostly TP * PP)
            gpus_per_node (int): Num gpus per node
            vocab_size_padded (int): Padded vocab size
            dtype (DataType): The datatype or model precision

        Returns:
            GPTConfig or the LLamaConfig or the PretrainedConfig constructed from your model config
        """
        hidden_act = self.activation
        hidden_act = (
            hidden_act.split("-")[-1]
            if self.transformer_config.num_moe_experts
            else non_gated_version(hidden_act)
        )

        config = {
            'architecture': TRT_MODEL_TYPE_STRING[self.model_type],
            'dtype': dtype.name,
            'num_hidden_layers': self.transformer_config.num_layers,
            'num_attention_heads': self.transformer_config.num_attention_heads,
            'num_key_value_heads': (
                self.transformer_config.num_query_groups
                if self.transformer_config.num_query_groups
                else self.transformer_config.num_attention_heads
            ),
            'head_size': self.transformer_config.kv_channels,
            'hidden_size': self.transformer_config.hidden_size,
            'intermediate_size': self.transformer_config.ffn_hidden_size,
            'norm_epsilon': self.transformer_config.layernorm_epsilon,
            'vocab_size': vocab_size_padded,
            'position_embedding_type': (
                "rope_gpt_neox" if self.position_embedding_type == "rope" else "learned_absolute"
            ),
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_act': hidden_act,
            'use_parallel_embedding': export_config.use_parallel_embedding,
            'embedding_sharding_dim': 0,
            'share_embedding_table': export_config.use_embedding_sharing,
            'quantization': {'quant_algo': None, 'kv_cache_quant_algo': None},
            'bias': self.transformer_config.add_bias_linear,
            'apply_query_key_layer_scaling': False,
            'rotary_pct': self.rotary_percentage,
            'rotary_base': self.rotary_base,
            'moe_num_experts': (
                0
                if self.transformer_config.moe_router_topk == 0
                else (self.transformer_config.num_moe_experts or 1)
            ),
            'moe_top_k': self.transformer_config.moe_router_topk,
            'moe_normalization_mode': self.moe_renorm_mode
            or MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
            'moe_tp_mode': self.moe_tp_mode,
            'logits_dtype': 'float32',
            'world_size': world_size,
            'tp_size': export_config.inference_tp_size,
            'pp_size': export_config.inference_pp_size,
            'gpus_per_node': gpus_per_node,
        }

        if self.model_type == ModelType.falcon:
            config["new_decoder_architecture"] = (
                False if self.transformer_config.num_layers == 32 else True
            )
            config["parallel_attention"] = True

        if self.seq_len_interpolation_factor is not None:
            config["rotary_scaling"] = {
                "type": "linear",
                "factor": float(self.seq_len_interpolation_factor),
            }

        config_cls = TRT_MODEL_CONFIG[self.model_type]
        return config_cls(**config)

    # pylint: disable=line-too-long
    def get_trtllm_pretrained_config_and_model_weights(
        self,
        model_state_dict,
        dtype: DataType,
        export_config: ExportConfig = None,
        on_device_distributed_conversion: bool = False,
        vocab_size: int = None,
        gpus_per_node: int = None,
        state_dict_split_by_layer_numbers: bool = True,
    ):
        """Get TRTLLM Config and Converted Model Weights

        This function returns the trtllm model weights as a list.
        There are two modes for conversion. The default is to use a single device cpu/gpu for conversion.
        NOTE: For faster performance, if your entire model will fit in memory, pre transfer the model state dict to cuda device and then call this function.
        For on device conversion it returns weights which will be used on the device itself.
        Same thing happens with the pretrained config

        Args:
            model_state_dict (dict, optional): The input model state dictionary (Entire model state loaded on CPU). Used only when on device conversion is set to False. Defaults to None.
            False, or the model state dict of each GPU in the case of on_device conversion)
            export_config (ExportConfig): The export config used to define inference tp size, pp size etc. Used only for on device conversion.
            dtype (DataType): The data type of model precision
            on_device_distributed_conversion (bool, optional): Convert on gpus in distributed setting. This assumes that the model state dict is sharded according to required inference model parallelism and that each gpu gets its part of the model state dict . Defaults to False.
            vocab_size (int, optional): The vocabulary size. Defaults to None.
            gpus_per_node (int, optional): The number of gpus per node. Used for on device conversion.
            state_dict_split_by_layer_numbers (bool, optional): Are the model layers split by layer numbers in state dict. For example : mlp.fc1.weight can be represented like mlp.fc1.weight of shape [num_layers, hidden_dim, ffn_hidden_dim]} or it can be like mlp.fc1.layers.0.weight of shape [hidden_dim, ffn_hidden_dim], then mlp.fc1.layers.1.weight ... for all layers. If you use represenation 2 set this to True. Defaults to True

        Returns:
            Two lists . First list of trtllm converted model weights(Either on device, or a list of weights for each gpu) and the trtllm_model_configs.
        """
        if on_device_distributed_conversion:
            assert (vocab_size is not None, "Need to pass in vocab_size for on device")
            assert (
                self.model_type in [ModelType.gpt, ModelType.gptnext, ModelType.llama],
                "On device conversion only supported for model types gptnext and llama",
            )
            assert (
                export_config is None,
                "Export config is inferred based on the parallel state. If you want to set inference tp 2, then load the model with this TP2 setting and just pass in the model state dict. ",
            )
            assert (
                gpus_per_node is not None
            ), "Need to pass in gpus_per_node for on device conversion"
            trtllm_model_weights_on_device, trtllm_model_config = (
                self._get_trtllm_pretrained_config_and_model_weights_in_distributed_setting(
                    model_state_dict, dtype, vocab_size, gpus_per_node
                )
            )
            return [trtllm_model_weights_on_device], [trtllm_model_config]

        else:
            assert not (
                self.share_embeddings_and_output_weights and not export_config.use_embedding_sharing
            ), "Found share_embeddings_and_output_weights is True in the model. So set export_config.use_embedding_sharing to True"
            assert (
                vocab_size is None
            ), "Vocab size is inferred from the input layer for cpu conversion. So leave it as None"
            trtllm_model_weights_list, trtllm_model_config_list = (
                self._get_trtllm_pretrained_config_and_model_weights_list_on_single_device(
                    export_config,
                    model_state_dict,
                    dtype,
                    gpus_per_node,
                    state_dict_split_by_layer_numbers,
                )
            )

            return trtllm_model_weights_list, trtllm_model_config_list

    def _get_trtllm_pretrained_config_and_model_weights_in_distributed_setting(
        self, model_state_dict: dict, dtype: DataType, vocab_size: int, gpus_per_node: int
    ):
        """Get the TRTLLM Pretrained config and model weights list in a distributed setting

        This function assumes the  model state dict is distributed according to model parallelism .
        Each device gets its own model state dict

        Args:
            export_config (ExportConfig): The export config to set inference tp, pp size etc.
            model_state_dict (dict): The model state dictionary (All collected on cpu)
            dtype (DataType): The data type or model precision
            vocab_size (int): Tokenizer vocab size
            gpus_per_node (int): The number of gpus per node

        Returns:
            Two lists . List of trtllm converted model weights and trtllm model configs (One for each gpu).
        """

        distributed_trtllm_model_weights_converter = DistributedTRTLLMModelWeightsConverter(
            transformer_config=self.transformer_config,
            dtype=dtype,
            multi_query_mode=self.multi_query_mode,
            activation=self.activation,
        )
        distributed_trtllm_model_weights_converter.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=self.trtllm_conversion_dict,
            tokenizer_vocab_size=vocab_size,
        )

        export_config = ExportConfig(
            inference_pp_size=distributed_trtllm_model_weights_converter.inference_pp_size,
            inference_tp_size=distributed_trtllm_model_weights_converter.inference_tp_size,
            use_parallel_embedding=True,
            use_embedding_sharing=self.share_embeddings_and_output_weights,
        )

        world_size = export_config.inference_tp_size * export_config.inference_pp_size

        trtllm_model_config = self._get_trtllm_config(
            export_config=export_config,
            world_size=world_size,
            gpus_per_node=gpus_per_node,
            vocab_size_padded=vocab_size,
            dtype=dtype,
        )

        model_parallel_rank = (
            distributed_trtllm_model_weights_converter.pp_rank
            * distributed_trtllm_model_weights_converter.inference_tp_size
            + distributed_trtllm_model_weights_converter.tp_rank
        )

        trtllm_model_config.mapping = tensorrt_llm.Mapping(
            world_size=world_size,
            rank=model_parallel_rank,
            tp_size=export_config.inference_tp_size,
            pp_size=export_config.inference_pp_size,
        )

        return distributed_trtllm_model_weights_converter.trtllm_model_weights, trtllm_model_config

    def _get_trtllm_pretrained_config_and_model_weights_list_on_single_device(
        self,
        export_config: ExportConfig,
        model_state_dict: dict,
        dtype: DataType,
        gpus_per_node=None,
        state_dict_split_by_layer_numbers=True,
    ):
        """Get the TRTLLM Pretrained config and model weights list (one per gpu rank) on single device (CPU/GPU)

        This function assumes the entire model state dict is present in CPU or on one GPU

        Args:
            export_config (ExportConfig): The export config to set inference tp, pp size etc.
            model_state_dict (dict): The model state dictionary (All collected on cpu)
            dtype (DataType): The data type or model precision
            gpus_per_node (int, optional): Number of gpus per node
            state_dict_split_by_layer_numbers (bool, optional): Are the model layers split by layer numbers in state dict. For example : mlp.fc1.weight can be represented like mlp.fc1.weight of shape [num_layers, hidden_dim, ffn_hidden_dim]} or it can be like mlp.fc1.layers.0.weight of shape [hidden_dim, ffn_hidden_dim], then mlp.fc1.layers.1.weight ... for all layers. If you use represenation 2 set this to True. Defaults to True

        Returns:
            Two lists . List of trtllm converted model weights and trtllm model configs (One for each gpu).
        """
        trtllm_model_configs_list = []
        trtllm_model_weights_list = []

        single_device_trtllm_model_weights_converter = SingleDeviceTRTLLMModelWeightsConverter(
            export_config=export_config,
            transformer_config=self.transformer_config,
            dtype=dtype,
            activation=self.activation,
            multi_query_mode=self.multi_query_mode,
        )
        # Convert the input model state dict to trtllm model weights dictionary
        single_device_trtllm_model_weights_converter.convert(
            model_state_dict=model_state_dict,
            trtllm_conversion_dict=self.trtllm_conversion_dict,
            state_dict_split_by_layer_numbers=state_dict_split_by_layer_numbers,
        )

        vocab_size_padded = single_device_trtllm_model_weights_converter.get_padded_vocab_size()
        world_size = export_config.inference_tp_size * export_config.inference_pp_size
        gpus_per_node = gpus_per_node or export_config.inference_tp_size

        for gpu_rank in range(world_size):
            mapping = tensorrt_llm.Mapping(
                world_size=world_size,
                rank=gpu_rank,
                tp_size=export_config.inference_tp_size,
                pp_size=export_config.inference_pp_size,
            )

            # Important to create a new instance everytime so that the list elements have differnt rank values in the mapping object
            trtllm_model_config = self._get_trtllm_config(
                export_config=export_config,
                world_size=world_size,
                gpus_per_node=gpus_per_node,
                vocab_size_padded=vocab_size_padded,
                dtype=dtype,
            )
            trtllm_model_config.mapping = mapping
            trtllm_model_configs_list.append(trtllm_model_config)

            # Get the model weights for each rank and append it to the trtllm_model_weights_list
            trtllm_model_weights_per_gpu = (
                single_device_trtllm_model_weights_converter.get_local_model_weights_per_gpu(
                    mapping, trtllm_model_config
                )
            )
            trtllm_model_weights_list.append(trtllm_model_weights_per_gpu)

        return trtllm_model_weights_list, trtllm_model_configs_list

    def build_and_save_engine(
        self,
        engine_dir: str,
        trtllm_model_weights: dict,
        trtllm_model_config,
        max_input_len: int = 1024,
        max_output_len: int = 1024,
        max_batch_size: int = 4,
        lora_ckpt_list=None,
        use_lora_plugin=None,
        max_lora_rank: int = 64,
        lora_target_modules=None,
        max_prompt_embedding_table_size: int = 0,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        paged_context_fmha: bool = False,
        use_refit: bool = False,
        max_num_tokens: int = None,
        max_seq_len: int = None,
        opt_num_tokens: int = None,
        max_beam_width: int = 1,
        tokens_per_block: int = 128,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
    ):
        """Method to build the TRTLLM Engine

        This method uses the TRTLLMEngineBuilder to build and save the engine to engine dir

        Args:
            engine_dir (str): The file path to save the engine
            trtllm_model_weights (dict): The TRTLLM converted model weights dict
            trtllm_model_config : The TRTLLM Config
            max_input_len (int, optional): Max input length. Defaults to 1024.
            max_output_len (int, optional): Max output length. Defaults to 1024.
            max_batch_size (int, optional): Max batch size. Defaults to 4.
            lora_ckpt_list (_type_, optional): Lora checkpoint list. Defaults to None.
            use_lora_plugin (_type_, optional): Use lora plugin. Defaults to None.
            max_lora_rank (int, optional): Max lora rank. Defaults to 64.
            lora_target_modules (_type_, optional): Lora target modules. Defaults to None.
            max_prompt_embedding_table_size (int, optional): Max size of prompt embedding table. Defaults to 0.
            paged_kv_cache (bool, optional): Use Paged KV cache. Defaults to True.
            remove_input_padding (bool, optional): Remove input padding. Defaults to True.
            paged_context_fmha (bool, optional): Paged context fmha. Defaults to False.
            use_refit (bool, optional): Use refit. Defaults to False.
            max_num_tokens (int, optional): Max num of tokens. Defaults to None.
            max_seq_len (int, optional): Max seq length. Defaults to None.
            opt_num_tokens (int, optional): Opt number of tokens. Defaults to None.
            max_beam_width (int, optional): Max beam width. Defaults to 1.
            tokens_per_block (int, optional): Nmber of tokens per block. Defaults to 128.
            multiple_profiles (bool, optional): Use multiple profiles. Defaults to False.
            gpt_attention_plugin (str, optional): Gpt attention plugin to use. Defaults to "auto".
            gemm_plugin (str, optional): Gemma plugin to use. Defaults to "auto".
        """

        TRTLLMEngineBuilder.build_and_save_engine(
            engine_dir,
            trtllm_model_weights,
            trtllm_model_config,
            max_input_len,
            max_output_len,
            max_batch_size,
            lora_ckpt_list,
            use_lora_plugin,
            max_lora_rank,
            lora_target_modules,
            max_prompt_embedding_table_size,
            paged_kv_cache,
            remove_input_padding,
            paged_context_fmha,
            use_refit,
            max_num_tokens,
            max_seq_len,
            opt_num_tokens,
            max_beam_width,
            tokens_per_block,
            multiple_profiles,
            gpt_attention_plugin,
            gemm_plugin,
        )
