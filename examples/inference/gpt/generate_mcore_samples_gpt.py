from argparse import Namespace
import json
import os
import sys
import numpy as np 
from megatron.core.inference.backends.abstract_backend import AbstractBackend
from megatron.core.inference.backends.mcore_backend import MCoreBackend
from megatron.core.inference.backends.trt_llm_backend import TRTLLMBackend
from megatron.core.inference.common_inference_params import CommonInferenceParams
from megatron.core.inference.common_generate_function import common_generate
from megatron.core.inference.inference_model_wrappers.gpt.gpt_inference_wrapper import GPTInferenceWrapper
from megatron.core.inference.text_generation_strategies.simple_text_generation_strategy import SimpleTextGenerationStrategy
from megatron.core.transformer.module import MegatronModule
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir, os.path.pardir)))

import math
import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron import print_rank_0
from megatron.checkpointing import load_checkpoint
from megatron.core import mpu
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt import GPTModel
from typing import List, Union
import megatron.model
from megatron.core.transformer.spec_utils import import_module
from megatron.arguments import core_transformer_config_from_args
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec

GLOBAL_PROMPT_IDX = 0

def model_provider(pre_process=True, post_process=True) -> Union[GPTModel, megatron.model.GPTModel]:
    """Builds the model.

    If you set the use_mcore_models to True, it will return the mcore GPT model and if not the legacy GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.model.GPTModel]: The returned model
    """
    args = get_args()
    print_rank_0('building GPT model ...')
    config = core_transformer_config_from_args(args)

    if args.use_mcore_models:
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(args.num_experts, args.moe_grouped_gemm)

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent
        )
    else:
        assert(args.context_parallel_size == 1), "Context parallelism is only supported with Megatron Core!"

        model = megatron.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True, 
            pre_process=pre_process,
            post_process=post_process
        )

    return model

def add_text_generate_args(parser):
    """Text generation arguments."""
    group = parser.add_argument_group(title='text generation')


    group.add_argument("--greedy", action='store_true', default=False,
                       help='Use greedy sampling.')
    group.add_argument("--temperature", type=float, default=1.0,
                       help='Sampling temperature.')
    group.add_argument("--top_k", type=int, default=0,
                       help='Top k sampling.')
    group.add_argument("--top_p", type=float, default=0.0,
                       help='Top p sampling.')
    group.add_argument("--return-log-probs", action='store_true', default=False,
                       help='Return the log probabilities of the final output tokens')
    group.add_argument("--num-tokens-to-generate", type=int, default=30,
                       help='Number of tokens to generate for each prompt')
    group.add_argument("--prompts-input-file", type=str, default=None,
                       help='Get input from file instead of interactive mode, '
                       'each line is an input.')
    group.add_argument("--output-file", type=str, default=None,
                       help='If not given, output file name derived from --prompts-input-file')
    return parser


def get_inference_backend(args: Namespace, model: MegatronModule) -> AbstractBackend:
    """Utility to get the relevant backend for running inference

    This function will automatically chose the TRTLLMBackend when possible, and if not revert to Mcore backend if the user does not specify any backends. 

    Args:
        args (Namespace): The user arguments parsed from command line
        model (MegatronModule): The megatron model . 

    Returns:
        AbstractBackend: The chosen backend
    """
    tokenizer = get_tokenizer()

    if TRTLLMBackend.is_model_trt_llm_exportable(model):
        return TRTLLMBackend(model, tokenizer)
    else :
        inference_wrapped_model = GPTInferenceWrapper(model, args)
        text_generation_strategy = SimpleTextGenerationStrategy(inference_wrapped_model=inference_wrapped_model, tokenizer=tokenizer)
        return MCoreBackend(text_generation_strategy=text_generation_strategy)
          

def write_results_to_file(output_file:str, prompts:List[str], prompt_plus_generated_tokens:List , prompts_plus_generated_text: List, output_log_probs:List) -> None :
    """Utility to write the output results to a text file

    Args:
        output_file (str): The output file name
        prompts (List[str]): The list of input prompts of size global_batch_size
        prompt_plus_generated_tokens (List): The input prompt tokensa along with the generated tokens
        prompts_plus_generated_text (List): The input prompt along with generated text
        output_log_probs (List): The log probabilitites
    """
    with open(output_file, 'a') as f: 
        global GLOBAL_PROMPT_IDX
        for idx, prompt in enumerate(prompts):
            print(f' ------------- WRITING RESULT FOR PROMPT {GLOBAL_PROMPT_IDX} --------------- ')
            tokens = np.array2string(prompt_plus_generated_tokens[idx].cpu().numpy())
            generated_text = prompts_plus_generated_text[idx]
            output_log_probs_idx = None if output_log_probs is None else np.array2string(output_log_probs[idx].cpu().numpy())
            write_data = {'id': GLOBAL_PROMPT_IDX,'original_prompt': prompt, 'prompt_with_generated_text': generated_text, 'all_tokens' : tokens, 'output_log_probs': output_log_probs_idx}
            f.write(json.dumps(write_data) + '\n')
            GLOBAL_PROMPT_IDX += 1

def generate_and_write_results(model: MegatronModule, args:Namespace):
    """Generates the output text and writes it to a file

    Generates the output tokens for the input prompts which are read from the input prompts file. We store these outputs in a text file

    Args:
        model (MegatronModule): The transformer model on which generate function is called
        args (Namespace): The arguments prased from the command line and default arguments (arguments.py)
    """    
    inference_backend = get_inference_backend(args, model)

    common_inference_params = CommonInferenceParams(
        use_greedy=args.greedy, 
        temperature=args.temperature, 
        top_k=args.top_k, 
        top_p=args.top_p, 
        return_log_probs=args.return_log_probs, 
        num_tokens_to_generate=args.num_tokens_to_generate)


    if torch.distributed.get_rank() == 0:
        fname = open(args.prompts_input_file, "r")
        lines = fname.readlines()
        all_prompts = [json.loads(line)['prompt']['text'] for line in lines]
        output_file = args.prompts_input_file + ".out" if args.output_file is None else args.output_file
        print('`sample-output-file` not specified, setting ''it to {}'.format(output_file))
        total_number_of_prompts = len(all_prompts)

        # Broadcast num inference steps to other gpus
        num_inference_steps = math.ceil(total_number_of_prompts/args.global_batch_size)
        torch.distributed.broadcast(torch.tensor(num_inference_steps).cuda(), 0)

        # Iterate through the prompts passing global_batch_size prompts each time to the backend.
        for idx in range(num_inference_steps):
            start = args.global_batch_size * idx
            end = min(total_number_of_prompts, start + args.global_batch_size)
            prompts = all_prompts[start:end]
            output_dictionary  = common_generate(inference_backend=inference_backend, prompts=prompts, common_inference_params=common_inference_params)
            
            write_results_to_file(output_file, prompts, output_dictionary['prompts_tokens_with_generations'], output_dictionary['prompts_plus_generations_detokenized'], output_dictionary['output_log_probs'])
    else:
        # The num inference steps is obtained from GPU 0 as shown above
        num_inference_steps_tensor = torch.tensor(0).cuda()
        torch.distributed.broadcast(num_inference_steps_tensor, 0)

        for _ in range(num_inference_steps_tensor.item()):
            common_generate(inference_backend=inference_backend, common_inference_params=common_inference_params)

def main():
    """Main program."""

    # Note: The default args passed here can be overwridden by using appropriate params (check arguments.py file)
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True,
                                       'seq_length': 2048})

    # Set up model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)
    load_checkpoint(model, None, None)
    model = model[0]

    args = get_args()

    generate_and_write_results(model, args)

if __name__ == "__main__":
    main()
