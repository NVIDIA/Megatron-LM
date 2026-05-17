# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Computes theoretical memory footprint for model training."""


import math

from .utils import is_hybrid_model, print_rank_0

NUM_BYTES_IN_MEGABYTE = 1024 * 1024


def compute_weight_and_optimizer_memory(args, verbose=False):
    # Attention projection size.
    query_projection_size = args.kv_channels * args.num_attention_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_size
    # Group Query Attention.
    if not args.group_query_attention:
        args.num_query_groups = args.num_attention_heads
    # MoE.
    num_experts = 1 if args.num_experts is None else args.num_experts
    gated_linear_multiplier = 3 / 2 if args.swiglu else 1

    shared_expert_ffn_hidden_size = (
        0
        if args.moe_shared_expert_intermediate_size is None
        else args.moe_shared_expert_intermediate_size
    )

    if args.num_experts is not None:
        if isinstance(args.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % args.moe_layer_freq == 0) else 0 for i in range(args.num_layers)
            ]
        elif isinstance(args.moe_layer_freq, list):
            moe_layer_pattern = args.moe_layer_freq
            assert len(moe_layer_pattern) == args.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {args.num_layers}, "
                f"current moe layer pattern: {args.moe_layer_freq}"
            )

        num_dense_layers = args.num_layers - sum(moe_layer_pattern)
        num_moe_layers = sum(moe_layer_pattern)
        moe_ffn_hidden_size = args.moe_ffn_hidden_size
    else:
        moe_layer_pattern = [0] * args.num_layers
        num_dense_layers = args.num_layers
        num_moe_layers = 0
        moe_ffn_hidden_size = 0
    assert num_dense_layers + num_moe_layers == args.num_layers
    if args.mtp_num_layers is not None:
        mtp_layer_is_moe = moe_layer_pattern[-1]
        mtp_num_moe_layers = mtp_layer_is_moe * args.mtp_num_layers
        mtp_num_dense_layers = (1 - mtp_layer_is_moe) * args.mtp_num_layers
    else:
        mtp_num_moe_layers = 0
        mtp_num_dense_layers = 0

    # RMSNorm does not have bias, but LayerNorm has.
    norm_size = 1 if args.normalization == "RMSNorm" else 2

    if args.multi_latent_attention:
        assert not args.group_query_attention
        if args.q_lora_rank is None:
            q_term = (
                args.hidden_size
                * args.num_attention_heads
                * (args.qk_head_dim + args.qk_pos_emb_head_dim)
            )
        else:
            ## q lora + rope + q norm
            q_term = args.q_lora_rank * (
                args.hidden_size
                + args.num_attention_heads * (args.qk_head_dim + args.qk_pos_emb_head_dim)
                + norm_size
            )

        self_attn_term = (
            q_term
            ## kv lora + rope + kv norm
            + args.kv_lora_rank
            * (
                args.hidden_size
                + args.num_attention_heads * (args.qk_head_dim + args.v_head_dim)
                + norm_size
            )
            + args.hidden_size * args.qk_pos_emb_head_dim
            ## o proj
            + (args.num_attention_heads * args.v_head_dim) * args.hidden_size
        )
    else:
        self_attn_term = (
            2
            * args.hidden_size
            * args.hidden_size
            * (
                # Attention.
                (
                    (1 + (args.num_query_groups / args.num_attention_heads))
                    * query_projection_to_hidden_size_ratio
                )
            )
        )

    embedding_size = args.hidden_size * args.padded_vocab_size
    final_layernorm = norm_size * args.hidden_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_in_embedding_layers = 2 * embedding_size
    else:
        num_parameters_in_embedding_layers = embedding_size

    attention_params = self_attn_term
    dense_mlp_params = 2 * args.hidden_size * args.ffn_hidden_size * gated_linear_multiplier
    shared_expert_params = (
        2 * args.hidden_size * shared_expert_ffn_hidden_size * gated_linear_multiplier
    )
    routed_expert_params = (
        2 * args.hidden_size * moe_ffn_hidden_size * num_experts * gated_linear_multiplier
    )
    active_routed_expert_params = (
        2 * args.hidden_size * moe_ffn_hidden_size * args.moe_router_topk * gated_linear_multiplier
        if args.num_experts is not None
        else 0
    )
    layernorm_params = 2 * args.hidden_size * norm_size
    router_params = (
        (args.hidden_size * num_experts) + (num_experts if args.add_bias_linear else 0)
        if args.num_experts is not None
        else 0
    )
    shared_expert_gate_params = (
        args.hidden_size
        if shared_expert_ffn_hidden_size > 0 and getattr(args, "moe_shared_expert_gate", False)
        else 0
    )

    num_parameters_in_transformer_layer_dense = (
        attention_params + dense_mlp_params + layernorm_params
    )
    num_parameters_in_transformer_layer_moe = (
        attention_params
        + shared_expert_params
        + routed_expert_params
        + layernorm_params
        + router_params
        + shared_expert_gate_params
    )
    num_active_parameters_in_transformer_layer_moe = (
        attention_params
        + shared_expert_params
        + active_routed_expert_params
        + layernorm_params
        + router_params
        + shared_expert_gate_params
    )
    num_parameters_in_transformer_block = (
        num_parameters_in_transformer_layer_dense * num_dense_layers
        + num_parameters_in_transformer_layer_moe * num_moe_layers
        + final_layernorm
    )
    num_active_parameters_in_transformer_block = (
        num_parameters_in_transformer_layer_dense * num_dense_layers
        + num_active_parameters_in_transformer_layer_moe * num_moe_layers
        + final_layernorm
    )
    num_parameters_in_mtp_block = (
        num_parameters_in_transformer_layer_dense * mtp_num_dense_layers
        + num_parameters_in_transformer_layer_moe * mtp_num_moe_layers
    )
    num_total_parameters = (
        num_parameters_in_transformer_block
        + num_parameters_in_mtp_block
        + num_parameters_in_embedding_layers
    )
    num_active_parameters = (
        num_active_parameters_in_transformer_block
        + num_parameters_in_mtp_block
        + num_parameters_in_embedding_layers
    )
    if verbose:
        print(
            f"Number of parameters in transformer block in billions: "
            f"{num_parameters_in_transformer_block / 10**9: .2f}"
        )
        print(
            f"Number of active parameters in transformer block in billions: "
            f"{num_active_parameters_in_transformer_block / 10**9: .2f}"
        )
        if args.mtp_num_layers is not None:
            print(
                f"Number of parameters in mtp block in billions: "
                f"{num_parameters_in_mtp_block / 10**9: .2f}"
            )
        print(
            f"Number of parameters in embedding layers in billions: "
            f"{num_parameters_in_embedding_layers / 10**9:.2f}"
        )
        print(f"Total number of parameters in billions: {num_total_parameters / 10**9:.2f}")
        print(f"Total number of active parameters in billions: {num_active_parameters / 10**9:.2f}")

    expert_tensor_parallel_size = args.expert_tensor_parallel_size
    expert_model_parallel_size = args.expert_model_parallel_size
    expert_tensor_model_pipeline_parallel_size = (
        expert_tensor_parallel_size * expert_model_parallel_size * args.pipeline_model_parallel_size
    )
    expert_data_parallel_size = args.world_size // expert_tensor_model_pipeline_parallel_size

    # Split params by how they are held on each rank: regular TP, replicated, or EP/ETP.
    tp_sharded_params_in_transformer_block = (
        attention_params + dense_mlp_params
    ) * num_dense_layers + (attention_params + shared_expert_params) * num_moe_layers
    replicated_params_in_transformer_block = (
        layernorm_params * num_dense_layers
        + (layernorm_params + router_params + shared_expert_gate_params) * num_moe_layers
        + final_layernorm
    )
    expert_sharded_params_in_transformer_block = routed_expert_params * num_moe_layers
    tp_sharded_params_in_mtp_block = (
        attention_params + dense_mlp_params
    ) * mtp_num_dense_layers + (attention_params + shared_expert_params) * mtp_num_moe_layers
    replicated_params_in_mtp_block = (
        layernorm_params * mtp_num_dense_layers
        + (layernorm_params + router_params + shared_expert_gate_params) * mtp_num_moe_layers
    )
    expert_sharded_params_in_mtp_block = routed_expert_params * mtp_num_moe_layers

    # Most loaded model shard has 1/pp_size transformer layers, 1 mtp block, and 1 embedding layer.
    tp_sharded_params_on_most_loaded_shard = (
        (tp_sharded_params_in_transformer_block / args.pipeline_model_parallel_size)
        + tp_sharded_params_in_mtp_block
        + embedding_size
    ) / args.tensor_model_parallel_size
    replicated_params_on_most_loaded_shard = (
        replicated_params_in_transformer_block / args.pipeline_model_parallel_size
    ) + replicated_params_in_mtp_block
    expert_sharded_params_on_most_loaded_shard = (
        (expert_sharded_params_in_transformer_block / args.pipeline_model_parallel_size)
        + expert_sharded_params_in_mtp_block
    ) / (expert_tensor_parallel_size * expert_model_parallel_size)
    num_parameters_on_most_loaded_model_shard = (
        tp_sharded_params_on_most_loaded_shard
        + replicated_params_on_most_loaded_shard
        + expert_sharded_params_on_most_loaded_shard
    )
    if args.untie_embeddings_and_output_weights and args.pipeline_model_parallel_size == 1:
        tp_sharded_params_on_most_loaded_shard += embedding_size / args.tensor_model_parallel_size
        num_parameters_on_most_loaded_model_shard += (
            embedding_size / args.tensor_model_parallel_size
        )
    if verbose:
        print(
            f"Number of parameters in most loaded shard in billions: "
            f"{num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
        )

    if args.pipeline_model_parallel_size > 1:
        # Other shards just have 1/pp_size transformer layers.
        num_parameters_on_other_model_shards = (
            tp_sharded_params_in_transformer_block
            / (args.pipeline_model_parallel_size * args.tensor_model_parallel_size)
            + replicated_params_in_transformer_block / args.pipeline_model_parallel_size
            + expert_sharded_params_in_transformer_block
            / (
                args.pipeline_model_parallel_size
                * expert_tensor_parallel_size
                * expert_model_parallel_size
            )
        )
        if verbose:
            print(
                f"Number of parameters in other shards in billions: "
                f"{num_parameters_on_other_model_shards / 10**9:.4f}"
            )

    def num_bytes_per_parameter(data_parallel_size):
        # This estimator assumes bf16 training: bf16 model params, fp32 main gradients,
        # fp32 main params, and fp32 Adam states. See docs/user-guide/features/dist_optimizer.md.
        return 18 if not args.use_distributed_optimizer else 6 + (12 / data_parallel_size)

    weight_and_optimizer_memory = (
        tp_sharded_params_on_most_loaded_shard + replicated_params_on_most_loaded_shard
    ) * num_bytes_per_parameter(
        args.data_parallel_size
    ) + expert_sharded_params_on_most_loaded_shard * num_bytes_per_parameter(
        expert_data_parallel_size
    )

    return weight_and_optimizer_memory


def compute_activation_memory(args, num_microbatches, verbose=False):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    # We are trying to compute the maximum activation footprint, so all calculations in this
    # function are for the first pipeline stage.

    # TODO: This function needs to take into account query_projection_size potentially being
    # different from hidden_size.

    # Memory footprint from transformer layer (self-attention and MLP).
    activation_memory = (args.seq_length * args.micro_batch_size * args.hidden_size) * (
        18 + (4 * (args.ffn_hidden_size / args.hidden_size))
    )
    if verbose:
        print(
            f"Activation memory footprint per transformer layer: "
            f"{activation_memory / NUM_BYTES_IN_MEGABYTE / args.tensor_model_parallel_size:.1f} MB"
        )
    activation_memory *= args.num_layers

    # Now add activation memory required for input embeddings, last LayerNorm and output layer.

    # Input to embedding (pp_size microbatches in flight).
    activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight).
    activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # Multiply by interleaved PP memory factor.
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size,
    # so discount accordingly.
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    if args.pipeline_model_parallel_size == 1:
        # Inputs to output layer and CE loss.
        activation_memory += (
            args.seq_length
            * args.micro_batch_size
            * args.hidden_size
            * 4
            * (1 + (args.padded_vocab_size / args.hidden_size))
        )

    # Activation memory is partitioned by TP size due to tensor and sequence model parallelism.
    return activation_memory / args.tensor_model_parallel_size


def compute_activation_memory_without_sp(args, num_microbatches, verbose=False):
    """Compute activation memory without sequence parallelism"""

    # 4. Compute per-layer memory
    per_layer_memory = (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * (10 + (24 / args.tensor_model_parallel_size))
    )

    if verbose:
        print(
            f"Activation memory footprint per transformer layer (precise, without SP): "
            f"{per_layer_memory / NUM_BYTES_IN_MEGABYTE:.1f} MB"
        )

    # 5. Multiply by number of layers
    total_activation_memory = per_layer_memory * args.num_layers

    # 6. Add embedding activations
    # Input to embedding (pp_size microbatches in flight)
    total_activation_memory += (
        8 * args.seq_length * args.micro_batch_size * args.pipeline_model_parallel_size
    )
    # Dropout in embedding layer (pp_size microbatches in flight)
    total_activation_memory += (
        args.seq_length
        * args.micro_batch_size
        * args.hidden_size
        * args.pipeline_model_parallel_size
    )

    # 7. Handle pipeline parallelism schedules
    # Multiply by interleaved PP memory factor
    if args.virtual_pipeline_model_parallel_size is not None:
        interleaved_schedule_memory_penalty = 1 + (
            (args.pipeline_model_parallel_size - 1)
            / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
        )
        in_flight_microbatches = math.ceil(
            interleaved_schedule_memory_penalty * args.pipeline_model_parallel_size
        )
        if verbose:
            print(
                f"Memory penalty from interleaved schedule: {interleaved_schedule_memory_penalty:.2f}"
            )
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")
        total_activation_memory *= interleaved_schedule_memory_penalty

    # If using non-interleaved schedule, number of microbatches in pipeline can be less than pp_size
    if args.virtual_pipeline_model_parallel_size is None and args.pipeline_model_parallel_size > 1:
        if num_microbatches is not None:
            total_activation_memory *= min(1, num_microbatches / args.pipeline_model_parallel_size)
            in_flight_microbatches = min(num_microbatches, args.pipeline_model_parallel_size)
        else:
            in_flight_microbatches = args.pipeline_model_parallel_size
        if verbose:
            print(f"Number of in-flight microbatches: {in_flight_microbatches}")

    # 8. Add output layer memory if needed
    if args.pipeline_model_parallel_size == 1:
        # Logits calculation
        logits_size = args.seq_length * args.micro_batch_size * args.padded_vocab_size
        # The output projection is partitioned across TP
        logits_size /= args.tensor_model_parallel_size

        # Outputs from final layer norm
        final_ln_output = args.seq_length * args.micro_batch_size * args.hidden_size

        total_activation_memory += (logits_size + final_ln_output) * 2  # multiply by 2 for bytes

    # 9. Add buffer for optimizer and miscellaneous temporaries (5% overhead)
    overhead_factor = 1.05
    total_activation_memory *= overhead_factor

    return total_activation_memory


def report_theoretical_memory(args, num_microbatches=None, verbose=False):
    if is_hybrid_model(args):
        print(
            "Theoretical memory footprints not yet supported for hybrid Mamba-Transformer models."
        )
        return

    weight_and_optimizer_memory = (
        compute_weight_and_optimizer_memory(args, verbose=verbose) / NUM_BYTES_IN_MEGABYTE
    )

    # Choose the appropriate activation memory calculation based on parallelism strategy
    if args.sequence_parallel and args.recompute_granularity == 'selective':
        print_rank_0("compute_activation_memory with SP")
        activation_memory = (
            compute_activation_memory(args, num_microbatches=num_microbatches, verbose=verbose)
            / NUM_BYTES_IN_MEGABYTE
        )
    else:
        print_rank_0("compute_activation_memory_without_sp")
        activation_memory = (
            compute_activation_memory_without_sp(
                args, num_microbatches=num_microbatches, verbose=verbose
            )
            / NUM_BYTES_IN_MEGABYTE
        )

    total_memory = weight_and_optimizer_memory + activation_memory

    print(
        f"Theoretical memory footprints: weight and optimizer={weight_and_optimizer_memory:.2f} MB, "
        f"activation={activation_memory:.2f} MB, total={total_memory:.2f} MB\n"
    )

    return weight_and_optimizer_memory, activation_memory, total_memory
