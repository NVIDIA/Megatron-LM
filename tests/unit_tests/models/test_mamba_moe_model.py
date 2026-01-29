# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import hashlib
import inspect
import json
import os
import sys
from typing import Any, Dict, Mapping, Tuple

import pytest  # type: ignore[import]
import torch

from megatron.core.models.mamba.mamba_layer_specs import mamba_stack_spec
from megatron.core.models.mamba.mamba_model import MambaModel
from megatron.core.num_microbatches_calculator import destroy_num_microbatches_calculator
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.enums import AttnBackend
from megatron.training.arguments import core_transformer_config_from_args, parse_args, validate_args
from megatron.training.global_vars import (
    destroy_global_vars,
    get_args,
    set_args,
    set_global_variables,
)
from tests.unit_tests.test_utilities import Utils

GOLDEN_CONFIG: Dict[str, Any] = {
    "_cpu_offloading_context": None,
    "account_for_embedding_in_pipeline_split": False,
    "account_for_loss_in_pipeline_split": False,
    "activation_func": "megatron.core.activations.squared_relu",
    "activation_func_clamp_value": None,
    "activation_func_fp8_input_store": False,
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "apply_query_key_layer_scaling": False,
    "apply_residual_connection_post_layernorm": False,
    "apply_rope_fusion": False,
    "async_tensor_model_parallel_allreduce": True,
    "attention_backend": {
        "__objclass__": "megatron.core.transformer.enums.AttnBackend",
        "_name_": "flash",
        "_sort_order_": 0,
        "_value_": 1,
    },
    "attention_dropout": 0.0,
    "attention_output_gate": False,
    "attention_softmax_in_fp32": False,
    "autocast_dtype": "torch.bfloat16",
    "barrier_with_L1_time": True,
    "batch_invariant_mode": False,
    "batch_p2p_comm": True,
    "batch_p2p_sync": True,
    "bf16": True,
    "bias_activation_fusion": False,
    "bias_dropout_fusion": True,
    "calculate_per_token_loss": False,
    "clone_scatter_output_in_embedding": True,
    "config_logger_dir": "",
    "context_parallel_size": 1,
    "cp_comm_type": "p2p",
    "cpu_offloading": False,
    "cpu_offloading_activations": True,
    "cpu_offloading_double_buffering": False,
    "cpu_offloading_num_layers": 0,
    "cpu_offloading_weights": False,
    "cross_entropy_fusion_impl": "native",
    "cross_entropy_loss_fusion": True,
    "cuda_graph_impl": "none",
    "cuda_graph_retain_backward_graph": False,
    "cuda_graph_scope": [],
    "cuda_graph_use_single_mempool": False,
    "cuda_graph_warmup_steps": 3,
    "deallocate_pipeline_outputs": True,
    "defer_embedding_wgrad_compute": False,
    "delay_wgrad_compute": False,
    "deterministic_mode": False,
    "disable_bf16_reduced_precision_matmul": False,
    "disable_parameter_transpose_cache": False,
    "distribute_saved_activations": False,
    "dsa_indexer_head_dim": None,
    "dsa_indexer_loss_coeff": None,
    "dsa_indexer_n_heads": None,
    "dsa_indexer_topk": None,
    "dsa_indexer_use_sparse_loss": False,
    "embedding_init_method": {},
    "embedding_init_method_std": 0.014,
    "enable_autocast": False,
    "enable_cuda_graph": False,
    "ep_overlap_early_attn_memory_release": False,
    "experimental_attention_variant": None,
    "expert_model_parallel_size": 4,
    "expert_tensor_parallel_size": 1,
    "external_cuda_graph": False,
    "ffn_hidden_size": 1856,
    "finalize_model_grads_func": None,
    "first_last_layers_bf16": False,
    "flash_decode": False,
    "fp16": False,
    "fp32_residual_connection": False,
    "fp4": None,
    "fp4_param": False,
    "fp4_quantizer_factory": None,
    "fp4_recipe": "nvfp4",
    "fp8": None,
    "fp8_amax_compute_algo": "most_recent",
    "fp8_amax_history_len": 1,
    "fp8_dot_product_attention": False,
    "fp8_interval": 1,
    "fp8_margin": 0,
    "fp8_multi_head_attention": False,
    "fp8_param": False,
    "fp8_quantizer_factory": None,
    "fp8_recipe": "delayed",
    "fp8_wgrad": True,
    "fused_single_qkv_rope": False,
    "gated_linear_unit": False,
    "glu_linear_offset": 0.0,
    "grad_scale_func": None,
    "grad_sync_func": None,
    "gradient_accumulation_fusion": True,
    "hetereogenous_dist_checkpoint": False,
    "heterogeneous_block_specs": False,
    "hidden_dropout": 0.0,
    "hidden_size": 2688,
    "hierarchical_context_parallel_sizes": None,
    "inference_fuse_tp_communication": False,
    "inference_rng_tracker": False,
    "inference_sampling_seed": 42,
    "init_method": {},
    "init_method_std": 0.014,
    "init_model_with_meta_device": False,
    "is_hybrid_model": True,
    "kitchen_attention_backend": "sdpa",
    "kv_channels": 128,
    "layernorm_epsilon": 1e-05,
    "layernorm_zero_centered_gamma": False,
    "linear_attention_freq": None,
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 128,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 32,
    "linear_value_head_dim": 128,
    "log_max_attention_logit": False,
    "mamba_head_dim": 64,
    "mamba_num_groups": 8,
    "mamba_num_heads": 64,
    "mamba_state_dim": 128,
    "masked_softmax_fusion": True,
    "memory_efficient_layer_norm": False,
    "microbatch_group_size_per_vp_stage": 1,
    "mlp_chunks_for_prefill": 1,
    "moe_apply_probs_on_input": False,
    "moe_aux_loss_coeff": 0.0,
    "moe_deepep_num_sms": 20,
    "moe_enable_deepep": False,
    "moe_expert_capacity_factor": None,
    "moe_extended_tp": False,
    "moe_ffn_hidden_size": 1856,
    "moe_flex_dispatcher_backend": "deepep",
    "moe_grouped_gemm": True,
    "moe_hybridep_num_sms": 16,
    "moe_input_jitter_eps": None,
    "moe_latent_size": None,
    "moe_layer_freq": 1,
    "moe_layer_recompute": False,
    "moe_pad_expert_input_to_capacity": False,
    "moe_per_layer_logging": False,
    "moe_permute_fusion": False,
    "moe_router_bias_update_rate": 0.001,
    "moe_router_dtype": "fp64",
    "moe_router_enable_expert_bias": True,
    "moe_router_force_load_balancing": False,
    "moe_router_fusion": False,
    "moe_router_group_topk": None,
    "moe_router_load_balancing_type": "aux_loss",
    "moe_router_num_groups": None,
    "moe_router_padding_for_fp8": False,
    "moe_router_padding_for_quantization": False,
    "moe_router_pre_softmax": False,
    "moe_router_score_function": "sigmoid",
    "moe_router_topk": 6,
    "moe_router_topk_limited_devices": None,
    "moe_router_topk_scaling_factor": 2.5,
    "moe_shared_expert_gate": False,
    "moe_shared_expert_intermediate_size": 3712,
    "moe_shared_expert_overlap": False,
    "moe_token_dispatcher_type": "alltoall",
    "moe_token_drop_policy": "probs",
    "moe_token_dropping": False,
    "moe_use_legacy_grouped_gemm": False,
    "moe_z_loss_coeff": None,
    "moe_enable_routing_replay": False,
    "mrope_section": None,
    "mtp_loss_scaling_factor": 0.1,
    "mtp_num_layers": None,
    "mtp_standalone": False,
    "multi_latent_attention": False,
    "no_rope_freq": None,
    "no_sync_func": None,
    "normalization": "RMSNorm",
    "num_attention_heads": 32,
    "num_layers": 52,
    "num_layers_at_end_in_bf16": 1,
    "num_layers_at_start_in_bf16": 1,
    "num_layers_in_first_pipeline_stage": None,
    "num_layers_in_last_pipeline_stage": None,
    "num_microbatches_with_partial_activation_checkpoints": None,
    "num_moe_experts": 128,
    "num_query_groups": 2,
    "output_layer_init_method": {},
    "overlap_moe_expert_parallel_comm": False,
    "overlap_p2p_comm": False,
    "overlap_p2p_comm_warmup_flush": False,
    "param_sync_func": None,
    "params_dtype": "torch.bfloat16",
    "perform_initialization": True,
    "persist_layer_norm": True,
    "pipeline_dtype": "torch.bfloat16",
    "pipeline_model_parallel_comm_backend": None,
    "pipeline_model_parallel_layout": None,
    "pipeline_model_parallel_size": 1,
    "qk_clip": False,
    "qk_clip_alpha": 0.5,
    "qk_clip_threshold": 100,
    "qk_l2_norm": False,
    "qk_layernorm": False,
    "quant_recipe": None,
    "recompute_granularity": None,
    "recompute_method": None,
    "recompute_modules": ["core_attn"],
    "recompute_num_layers": None,
    "rotary_interleaved": False,
    "sequence_parallel": True,
    "softmax_scale": None,
    "softmax_type": "vanilla",
    "symmetric_ar_type": None,
    "tensor_model_parallel_size": 2,
    "test_mode": False,
    "timers": None,
    "tp_comm_atomic_ag": False,
    "tp_comm_atomic_rs": False,
    "tp_comm_bootstrap_backend": "nccl",
    "tp_comm_bulk_dgrad": True,
    "tp_comm_bulk_wgrad": True,
    "tp_comm_overlap": False,
    "tp_comm_overlap_ag": True,
    "tp_comm_overlap_disable_fc1": False,
    "tp_comm_overlap_disable_qkv": False,
    "tp_comm_overlap_rs": True,
    "tp_comm_overlap_rs_dgrad": False,
    "tp_comm_split_ag": True,
    "tp_comm_split_rs": True,
    "tp_only_amax_red": False,
    "transformer_impl": "transformer_engine",
    "use_cpu_initialization": None,
    "use_fused_weighted_squared_relu": False,
    "use_inference_optimized_layers": False,
    "use_kitchen": False,
    "use_kitchen_attention": False,
    "use_mamba_mem_eff_path": True,
    "use_ring_exchange_p2p": False,
    "use_te_activation_func": False,
    "use_te_rng_tracker": False,
    "variable_seq_lengths": False,
    "virtual_pipeline_model_parallel_size": None,
    "wgrad_deferral_limit": 0,
    "window_attn_skip_freq": None,
    "window_size": None,
    "fine_grained_activation_offloading": False,
    "min_offloaded_tensor_size": 1024 * 1024,
    "offload_modules": [],
    "hybrid_context_parallel": False,
    "max_seqlen_per_dp_cp_rank": None,
}
# Fields to ignore entirely (ephemeral, environment-specific, very large).
SKIP_FIELDS = set()
# Fields that are allowed to appear in the live config even if not yet in the golden.
ALLOW_ADDED_FIELDS = set()


def serialize_config(cfg: Any) -> Dict[str, Any]:
    """Normalize a config object into a JSON-serializable dict."""
    data = {k: v for k, v in vars(cfg).items() if k not in SKIP_FIELDS}
    return _ser(data)


def assert_config_matches_golden(cfg: Any) -> None:
    """Compare live config to golden snapshot with readable diffs."""
    current = serialize_config(cfg)
    golden = GOLDEN_CONFIG

    added, removed, changed = _diff_configs(golden, current)

    # Ignore added fields that are explicitly allowed.
    added = [k for k in added if k not in ALLOW_ADDED_FIELDS]

    if added or removed or changed:
        # Build actionable guidance for each type of drift
        guidance_parts = []

        if added:
            guidance_parts.append(
                f"\n\n[ADDED ARGS]: {sorted(added)}\n"
                "  → Update GOLDEN_CONFIG in this test file to include the new arg(s) with "
                "their default value(s).\n"
                "  ⚠️  CAUTION: Review any logic associated with new args to ensure it doesn't "
                "silently affect downstream model configs or behavior.\n"
            )

        if changed:
            guidance_parts.append(
                f"\n\n[CHANGED DEFAULTS]: {sorted(changed)}\n"
                "  → Please don't change the default values of existing args unless "
                "it is absolutely necessary for a bug fix.\n"
                "  → If you must change the default value, please update the GOLDEN_CONFIG "
                "in this test file to reflect the new default value.\n"
            )

        if removed:
            guidance_parts.append(
                f"\n\n[REMOVED ARGS]: {sorted(removed)}\n"
                "  → Do NOT remove args directly. Instead, deprecate them with a warning message "
                "to maintain backwards compatibility.\n"
            )

        guidance_parts.append(
            "Please contact NV-username @jbarker if you are unsure how to proceed.\n"
        )

        header = "Mamba MoE config drift detected!\n" "═" * 60 + "".join(guidance_parts)
        parts = [header]
        if changed:
            formatted = {k: {"expected": golden[k], "actual": current[k]} for k in sorted(changed)}
            parts.append(
                f"Changed field details:\n{json.dumps(formatted, indent=2, sort_keys=True)}"
            )
        pytest.fail("\n".join(parts))


def regenerate_mamba_moe_golden(cfg: Any) -> Dict[str, Any]:
    """Helper to regenerate the golden config; copy/paste into GOLDEN_CONFIG."""
    serialized = serialize_config(cfg)
    return serialized


def _ser(obj: Any) -> Any:
    """Recursively convert objects to JSON-friendly structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        return {k: _ser(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_ser(v) for v in obj]
    if inspect.isfunction(obj) or inspect.ismethod(obj):
        return f"{obj.__module__}.{obj.__name__}"
    if inspect.isclass(obj):
        return f"{obj.__module__}.{obj.__name__}"
    if hasattr(obj, "__dict__"):
        return {k: _ser(v) for k, v in vars(obj).items()}
    try:
        return str(obj)
    except Exception:
        return f"<unserializable:{type(obj).__name__}>"


def _diff_configs(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> Tuple[set, set, set]:
    """Return added, removed, and changed top-level keys between dicts."""
    expected_keys = set(expected)
    actual_keys = set(actual)
    added = actual_keys - expected_keys
    removed = expected_keys - actual_keys
    changed = {k for k in expected_keys & actual_keys if expected[k] != actual[k]}
    return added, removed, changed


class TestMambaMoEModel:
    """Test the initialization and use of an MoE Mamba model."""

    def create_test_args(self):
        destroy_global_vars()
        destroy_num_microbatches_calculator()

        sys.argv = ['test_mamba_moe_model.py']
        args = parse_args()

        # The following args would be set from the nano v3 checkpoint.
        args.num_layers = 52
        args.hidden_size = 2688
        args.ffn_hidden_size = 1856
        args.num_attention_heads = 32
        args.num_query_groups = 2
        args.group_query_attention = True
        args.kv_channels = 128
        args.position_embedding_type = 'none'
        args.add_position_embedding = True
        args.use_rotary_position_embeddings = False
        args.rotary_base = 10000
        args.rotary_percent = 1.0
        args.rotary_interleaved = False
        args.add_bias_linear = False
        args.add_qkv_bias = False
        args.squared_relu = True
        args.swiglu = False
        args.untie_embeddings_and_output_weights = True
        args.apply_layernorm_1p = False
        args.normalization = "RMSNorm"
        args.apply_query_key_layer_scaling = False
        args.attention_dropout = 0.0
        args.hidden_dropout = 0.0
        args.hybrid_override_pattern = "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEM*EMEMEMEME"
        args.spec = ["megatron.core.models.mamba.mamba_layer_specs", "mamba_stack_spec"]
        args.hybrid_attention_ratio = 0.0
        args.hybrid_mlp_ratio = 0.0
        args.num_experts = 128
        args.moe_layer_freq = 1
        args.moe_ffn_hidden_size = 1856
        args.moe_router_topk = 6
        args.moe_router_pre_softmax = False
        args.moe_grouped_gemm = True
        args.moe_shared_expert_intermediate_size = 3712
        args.moe_router_score_function = "sigmoid"
        args.moe_router_enable_expert_bias = True
        args.moe_router_topk_scaling_factor = 2.5
        args.mamba_state_dim = 128
        args.mamba_head_dim = 64
        args.mamba_num_groups = 8
        args.mamba_num_heads = 64
        args.is_hybrid_model = True
        args.tokenizer_type = "TikTokenizer"
        args.tiktoken_pattern = "v2"
        args.tokenizer_model = "/mnt/artifacts/model/nemotron6/tokenizers/multiMixV8.gpt4o_nc_sd.500000.128k.vocab.json"
        args.padded_vocab_size = 131072

        # The following args would be set in the user's nano v3 config.
        args.async_tensor_model_parallel_allreduce = True
        args.attention_backend = AttnBackend.flash
        args.bf16 = True
        args.ckpt_format = 'torch_dist'
        args.cross_entropy_loss_fusion = True
        args.cuda_graph_impl = "none"
        args.embedding_init_method_std = 0.014
        args.expert_model_parallel_size = 4
        args.expert_tensor_parallel_size = 1
        args.init_method_std = 0.014
        args.lr = 3e-5
        args.max_position_embeddings = 1024
        args.micro_batch_size = 2
        args.moe_aux_loss_coeff = 0.0
        args.moe_grouped_gemm = True
        args.moe_route_load_balancing_type = "aux_loss"
        args.moe_router_dtype = "fp64"
        args.moe_router_pre_softmax = False
        args.moe_token_dispatcher_type = "alltoall"
        args.no_load_optim = True
        args.no_load_rng = True
        args.no_save_optim = True
        args.pipeline_model_parallel_size = 1
        args.position_embedding_type = None
        args.recompute_granularity = None
        args.seed = 42
        args.seq_length = 1024
        args.sequence_parallel = True
        args.te_rng_tracker = True
        args.tensor_model_parallel_size = 2
        args.vocab_size = 131072

        validate_args(args)
        set_global_variables(args, False)
        return args

    def setup_method(self, method):

        os.environ['CUDA_DEVICE_MAX_CONNECTIONS'] = '1'
        args = self.create_test_args()
        set_args(args)

        Utils.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            expert_tensor_parallel_size=args.expert_tensor_parallel_size,
        )
        model_parallel_cuda_manual_seed(123)

        model_config = core_transformer_config_from_args(args, TransformerConfig)

        self.model = MambaModel(
            config=model_config,
            mamba_stack_spec=mamba_stack_spec,
            vocab_size=args.vocab_size,
            max_sequence_length=args.seq_length,
            hybrid_attention_ratio=args.hybrid_attention_ratio,
            hybrid_mlp_ratio=args.hybrid_mlp_ratio,
            hybrid_override_pattern=args.hybrid_override_pattern,
            position_embedding_type=args.position_embedding_type,
            rotary_base=args.rotary_base,
            rotary_percent=args.rotary_percent,
        )

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    def test_constructor(self):
        """Sanity check for the constructor of the Mamba MoE model."""

        args = get_args()

        assert_config_matches_golden(self.model.config)

        assert self.model.pre_process is True, "pre_process should be True"
        assert self.model.post_process is True, "post_process should be True"
        assert self.model.hybrid_attention_ratio == 0.0, "hybrid_attention_ratio should be 0.0"
        assert self.model.hybrid_mlp_ratio == 0.0, "hybrid_mlp_ratio should be 0.0"
        assert (
            self.model.hybrid_override_pattern == args.hybrid_override_pattern
        ), f"hybrid_override_pattern should be {args.hybrid_override_pattern}"
        num_weights = sum([p.numel() for p in self.model.parameters()])
        assert num_weights == 8449294624, f"Expected 8449294624 parameters, got {num_weights}"

    def test_set_input_tensor(self):

        args = get_args()

        config: TransformerConfig = self.model.config
        sequence_length = self.model.max_sequence_length
        micro_batch_size = args.micro_batch_size

        # [sequence length, batch size, hidden size]
        input_tensor = torch.ones((sequence_length, micro_batch_size, config.hidden_size))

        self.model.set_input_tensor(input_tensor)

        assert self.model.decoder.input_tensor.shape[0] == sequence_length
        assert self.model.decoder.input_tensor.shape[1] == micro_batch_size
        assert self.model.decoder.input_tensor.shape[2] == config.hidden_size

    def test_forward(self):
        """Basic smoke test for the forward pass of the Mamba MoE model."""

        args = get_args()

        # we must override this to avoid the need to initialize the optimizer
        for param in self.model.parameters():
            param.requires_grad = False

        sequence_length = self.model.max_sequence_length
        micro_batch_size = args.micro_batch_size

        self.model.cuda()

        data = list(range(sequence_length))
        input_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        position_ids = torch.tensor(data, dtype=torch.int64).repeat((micro_batch_size, 1)).cuda()
        attention_mask = torch.ones(
            (micro_batch_size, 1, sequence_length, sequence_length), dtype=bool
        ).cuda()

        logits = self.model.forward(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            runtime_gather_output=True,
        )

        assert logits.shape[0] == micro_batch_size
        assert logits.shape[1] == sequence_length
        assert logits.shape[2] == self.model.vocab_size
