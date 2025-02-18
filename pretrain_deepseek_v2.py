import torch

from megatron.training import print_rank_0,get_args,get_tokenizer,build_tokenizer,pretrain
from megatron.core.transformer.spec_utils import import_module
from megatron.training.arguments import core_transformer_config_from_arg
from megatron.core.models.gpt import GPTModel
from pretrain_gpt import loss_func, get_batch, forward_step,train_valid_test_datasets_provider
from megatron.training import pretrain

from megatron.core.models.deepseek import DeepSeekV2TransformerConfig, get_deepseek_layer_with_spec

def model_provider(
    pre_process: bool = True, post_process: bool = True, parallel_output: bool = True,
) -> GPTModel:
    """Builds the deepseek model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits

    Returns:
        GPTModel: The returned model
    """
    args = get_args()

    print_rank_0('building DeepseekV2 model ...')

    build_tokenizer(args)
    
    from megatron.training.arguments import core_transformer_config_from_args
    config = core_transformer_config_from_args(args, DeepSeekV2TransformerConfig)

    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")

    if args.spec is not None:
        transformer_layer_spec = import_module(args.spec)
    else:
        transformer_layer_spec = get_deepseek_layer_with_spec(args.num_experts,
                                                             args.moe_grouped_gemm,
                                                             args.qk_layernorm,
                                                             args.multi_latent_attention)

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=parallel_output,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
        rotary_base=args.rotary_base,
        seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor,
    )

    return model

if __name__ == "__main__":
    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
    )