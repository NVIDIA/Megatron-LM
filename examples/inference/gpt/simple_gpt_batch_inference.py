import os
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import InferenceWrapperConfig
from pretrain_gpt import model_provider
import torch
import sys
from argparse import Namespace
from megatron.core.inference.engines.abstract_engine import AbstractEngine
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.inference_request import InferenceRequest
from megatron.core.inference.text_generation_controllers.simple_text_generation_controller import SimpleTextGenerationController
from megatron.core.transformer.module import MegatronModule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

from megatron.training import get_args
from megatron.training import get_tokenizer
from megatron.training.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.training.initialize import initialize_megatron
from megatron.training import get_model
from typing import List

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')

    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=1,
                       help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--return-log-probs", action='store_true', default=False,
                       help='Return the log probabilities of the final output tokens')
    group.add_argument("--num-tokens-to-generate", type=int, default=30,
                       help='Number of tokens to generate for each prompt')
    group.add_argument("--prompts", metavar='N', type=str, nargs='+',
                       help='Input prompts with each prompt within quotes and seperated by space')
    group.add_argument("--max-batch-size", type=int, default=1,
                       help='Max number of prompts to process at once')
    return parser


def get_inference_engine(args: Namespace, model: MegatronModule) -> AbstractEngine:
    """Utility to get the relevant backend for running inference

    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. TRT LLM Backend is not implmented yet. 

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model . 

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    inference_wrapper_config = InferenceWrapperConfig(
        hidden_size=args.hidden_size,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
        fp32_residual_connection=args.fp32_residual_connection,
        params_dtype=args.params_dtype,
        padded_vocab_size=args.padded_vocab_size
    )

    inference_wrapped_model = GPTInferenceWrapper(model, inference_wrapper_config)
    text_generation_controller = SimpleTextGenerationController(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
    return MCoreEngine(text_generation_controller=text_generation_controller, max_batch_size=args.max_batch_size)
            
def main():
    """Main program."""

    # Note: The default args passed here can be overwritten by using appropriate params (check arguments.py file)
    # Micro batch size is not needed to be set by user. (It is calculated based on inference-batch-times-seqlen-threshold argument)
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'no_load_rng': True,
                                       'no_load_optim': True,
                                       'micro_batch_size': 1, 
                                       'exit_on_missing_checkpoint': True})

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]

    args = get_args()

    inference_engine = get_inference_engine(args, model)

    common_inference_params = CommonInferenceParams(
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p, 
        return_log_probs=args.return_log_probs, 
        num_tokens_to_generate=args.num_tokens_to_generate)

    results: List[InferenceRequest] = inference_engine.generate(
        prompts=args.prompts, common_inference_params=common_inference_params
    )
    
    if torch.distributed.get_rank() == 0:
        for idx, result in enumerate(results):
            print(f' \n------------- RESULT FOR PROMPT {idx} --------------- ')
            result = {
                'id': result.request_id,
                'input_prompt': result.prompt, 
                'generated_text': result.generated_text,
                'generated_tokens' : result.generated_tokens
                }
            print(result)

if __name__ == "__main__":
    main()
