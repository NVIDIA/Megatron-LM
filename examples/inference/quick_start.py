import os
import sys

import torch

from megatron.core.inference.backends.mcore_backend import MCoreBackend
from megatron.core.inference.common_generate_function import common_generate
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.inference_model_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.inference.text_generation_strategies.simple_text_generation_strategy import (
    SimpleTextGenerationStrategy,
)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))
)
from megatron import get_args, get_tokenizer, print_rank_0
from megatron.arguments import core_transformer_config_from_args
from megatron.checkpointing import load_checkpoint
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.initialize import initialize_megatron
from megatron.training import get_model


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(args)

    transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
        args.num_experts, args.moe_grouped_gemm
    )

    model = GPTModel(
        config=config,
        transformer_layer_spec=transformer_layer_spec,
        vocab_size=args.padded_vocab_size,
        max_sequence_length=args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
        position_embedding_type=args.position_embedding_type,
        rotary_percent=args.rotary_percent,
    )

    return model


def get_inference_backend():
    args = get_args()
    inference_wrapped_model = GPTInferenceWrapper(model, args)

    tokenizer = get_tokenizer()
    text_generation_strategy = SimpleTextGenerationStrategy(
        inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer
    )

    inference_backend = MCoreBackend(text_generation_strategy=text_generation_strategy)

    return inference_backend


if __name__ == "__main__":
    
    initialize_megatron(
        args_defaults={'no_load_rng': True, 'no_load_optim': True, 'micro_batch_size': 1}
    )

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]

    inference_backend = get_inference_backend()

    # Using default paramters
    common_inference_params = CommonInferenceParams()

    result = common_generate(
        inference_backend=inference_backend,
        prompts=["How large is the universe ?", "Where can you celebrate birthdays ? "],
        common_inference_params=common_inference_params,
    )

    if torch.distributed.get_rank() == 0:
        print(result['prompts_plus_generations_detokenized'])
