# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import tensorrt_llm
from tensorrt_llm._common import check_max_num_tokens
from tensorrt_llm.builder import BuildConfig
from tensorrt_llm.commands.build import build as build_trtllm
from tensorrt_llm.logger import logger
from tensorrt_llm.lora_manager import LoraConfig
from tensorrt_llm.models.modeling_utils import optimize_model, preprocess_weights
from tensorrt_llm.plugin import PluginConfig


class TRTLLMEngineBuilder:
    """A utility class to build TRTLLM engine"""

    @staticmethod
    def build_and_save_engine(
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
            model_type (ModelType, optional): ModelType enum. Defaults to ModelType.gpt.
            lora_ckpt_list (_type_, optional): Lora checkpoint list. Defaults to None.
            use_lora_plugin (_type_, optional): Use lora plugin. Defaults to None.
            max_lora_rank (int, optional): Max lora rank. Defaults to 64.
            lora_target_modules (_type_, optional): Lora target modules. Defaults to None.
            max_prompt_embedding_table_size (int, optional): Defaults to 0.
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
        architecture = (
            "LLaMAForCausalLM"
            if trtllm_model_config.architecture == "LlamaForCausalLM"
            else trtllm_model_config.architecture
        )
        try:
            model_cls = getattr(tensorrt_llm.models, architecture)
        except:
            raise AttributeError(f"Could not find TRTLLM model for architecture: {architecture}!")

        logger.set_level("info")
        plugin_config = PluginConfig()
        plugin_config.gpt_attention_plugin = gpt_attention_plugin
        plugin_config.gemm_plugin = gemm_plugin
        if paged_kv_cache:
            plugin_config.enable_paged_kv_cache(tokens_per_block=tokens_per_block)
        else:
            plugin_config.paged_kv_cache = False
        plugin_config.remove_input_padding = remove_input_padding
        plugin_config.use_paged_context_fmha = paged_context_fmha
        plugin_config.multiple_profiles = multiple_profiles

        if max_seq_len is None:
            max_seq_len = max_input_len + max_output_len

        max_num_tokens, opt_num_tokens = check_max_num_tokens(
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_beam_width=max_beam_width,
            remove_input_padding=remove_input_padding,
            enable_context_fmha=plugin_config.context_fmha,
            tokens_per_block=tokens_per_block,
            multiple_profiles=multiple_profiles,
        )

        build_dict = {
            'max_input_len': max_input_len,
            'max_output_len': max_output_len,
            'max_batch_size': max_batch_size,
            'max_beam_width': max_beam_width,
            'max_seq_len': max_seq_len,
            'max_num_tokens': max_num_tokens,
            'opt_num_tokens': opt_num_tokens,
            'max_prompt_embedding_table_size': max_prompt_embedding_table_size,
            'gather_context_logits': False,
            'gather_generation_logits': False,
            'strongly_typed': False,
            'builder_opt': None,
            'use_refit': use_refit,
            'multiple_profiles': multiple_profiles,
        }
        build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)

        if use_lora_plugin is not None:
            # build_config.plugin_config.set_lora_plugin(use_lora_plugin)
            # build_config.plugin_config._lora_plugin = use_lora_plugin
            lora_config = LoraConfig(
                lora_dir=lora_ckpt_list,
                lora_ckpt_source='nemo',  # TODO : NEED TO SEE HOW TO HANDLE THIS FOR MCORE
                max_lora_rank=max_lora_rank,
                lora_target_modules=lora_target_modules,
            )
            build_config.lora_config = lora_config

        model = model_cls.from_config(trtllm_model_config)
        model = optimize_model(
            model,
            use_parallel_embedding=trtllm_model_config.use_parallel_embedding,
            share_embedding_table=trtllm_model_config.share_embedding_table,
        )
        preprocess_weights(trtllm_model_weights, trtllm_model_config)
        model.load(trtllm_model_weights)
        engine = build_trtllm(model, build_config)
        engine.save(engine_dir)
