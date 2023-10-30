from megatron.initialize import initialize_megatron
from megatron import get_args


def compute_weight_and_optimizer_memory(args):
    assert args.sequence_parallel
    num_parameters_in_transformer_layers = (
        10
        * args.num_layers
        * args.hidden_size
        * args.hidden_size
        * (
            1
            + (args.num_query_groups / (5.0 * args.num_attention_heads))
            + (2 / (5 * args.hidden_size))
            + (1 / (5 * args.num_layers * args.hidden_size))
        )
    )
    embedding_size = args.hidden_size * args.padded_vocab_size
    if args.untie_embeddings_and_output_weights:
        num_parameters_with_embeddings = num_parameters_in_transformer_layers + (2 * embedding_size)
    else:
        num_parameters_with_embeddings = num_parameters_in_transformer_layers + embedding_size
    print(f"Number of parameters in billions: {num_parameters_with_embeddings / 10**9:.2f}")

    # Most loaded model shard has (1/pp_size transformer layers + 1 embedding layer) / tp_size.
    num_parameters_on_most_loaded_model_shard = (
        (num_parameters_in_transformer_layers / args.pipeline_model_parallel_size) + embedding_size
    ) / args.tensor_model_parallel_size
    # Other shards just have (1/pp_size transformer layers) / tp_size.
    num_parameters_on_other_model_shards = num_parameters_in_transformer_layers / (
        args.pipeline_model_parallel_size * args.tensor_model_parallel_size
    )

    print(
        f"Number of parameters in most loaded shard in billions: {num_parameters_on_most_loaded_model_shard / 10**9:.4f}"
    )
    print(
        f"Number of parameters in other shards in billions: {num_parameters_on_other_model_shards / 10**9:.4f}"
    )

    num_bytes_per_parameter = (
        18 if not args.use_distributed_optimizer else 6 + (12 / args.data_parallel_size)
    )
    return num_parameters_on_most_loaded_model_shard * num_bytes_per_parameter


def compute_activation_memory(args):
    # Using formula in Table 2 of https://arxiv.org/pdf/2205.05198.pdf.
    assert args.recompute_granularity == 'selective'
    activation_memory = (
        args.seq_length * args.micro_batch_size * args.hidden_size * args.num_layers
    ) * 34

    # Multiply by interleaved PP memory factor.
    activation_memory *= 1 + (
        (args.pipeline_model_parallel_size - 2)
        / (args.pipeline_model_parallel_size * args.virtual_pipeline_model_parallel_size)
    )
    return activation_memory / args.tensor_model_parallel_size


def compute_total_memory(args):
    weight_and_optimizer_memory = compute_weight_and_optimizer_memory(args)
    activation_memory = compute_activation_memory(args)
    total_memory = weight_and_optimizer_memory + activation_memory
    print(
        f"(DP size, PP size, TP size) = {(args.data_parallel_size, args.pipeline_model_parallel_size, args.tensor_model_parallel_size)}, "
        f"Weight and optimizer memory: {weight_and_optimizer_memory / (1024 * 1024):.2f} MB, "
        f"Activation memory: {activation_memory / (1024 * 1024):.2f} MB, "
        f"Total memory: {total_memory / (1024 * 1024):.2f} MB\n"
    )


if __name__ == "__main__":
    initialize_megatron(allow_no_cuda=True, skip_mpu_initialization=True)
    args = get_args()

    compute_total_memory(args)
