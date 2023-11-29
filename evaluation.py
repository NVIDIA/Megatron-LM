# File containing evaluation scripts and evaluator object from lm eval harness

import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron import is_last_rank
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch
import os
import sys
import lm_eval 
from lm_eval.base import BaseLM
from megatron import get_tokenizer
from megatron.text_generation.forward_step import ForwardStep
from megatron.utils import get_ltor_masks_and_position_ids
from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from lm_eval.tasks import ALL_TASKS
import json
import argparse


def _build_attention_mask_and_position_ids(tokens):
    """Build the attention mask and postition ids for the input tokens."""
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False)

    return attention_mask, position_ids


def extract_keyword_args(filestr, keyword):
    gpt_split = filestr.split(keyword)
    if len(gpt_split) <=1:
        raise ValueError("Config provided does not have a GPT_ARGS variable provided")
    arg_splits = gpt_split[1].split("\"")
    gpt_args = arg_splits[1]
    gpt_args = gpt_args.replace("\n","").replace("\\","").replace("\t","")
    gpt_args = ' '.join(gpt_args.split())
    return gpt_args.strip().split(" ")

def extract_data_paths(filestr, checkpoint_path):
    vocab_file = filestr.split("VOCAB_FILE=")[1].split("\n")[0]
    merge_file = filestr.split("MERGE_FILE=")[1].split("\n")[0]
    #vocab_file = "/workspace/gpt-neox/data/gpt2-vocab.json"
    #merge_file = "/workspace/gpt-neox/data/gpt2-merges.txt"
    #checkpoint_path = "/workspace/ckpts_bf16_125m"
    data_path = filestr.split("DATA_PATH=")[1].split("\n")[0]
    return ["--data-path", data_path, "--vocab-file" , vocab_file, "--merge-file" , merge_file,"--load" , checkpoint_path]
    

def parse_config_file_update_argv(config_path, checkpoint_path):
    with open(config_path,"r") as f:
        filestr = f.read()
    
    sys.argv = ["checkpointing_tests.py"] # a hack to get around the jupyter issues in the sys.argc which we are messing with
    sys.argv += extract_keyword_args(filestr, "GPT_ARGS")
    sys.argv += extract_data_paths(filestr, checkpoint_path)
    #print(sys.argv)

def model_provider(pre_process=True, post_process=True):
    """Build the model."""
    args = get_args()
    args.recompute_granularity = None # enforce for inference
    config = core_transformer_config_from_args(args)

    print_rank_0('building GPT model ...')
    model = GPTModel(config, num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def add_text_generate_args(parser):
    group = parser.add_argument_group(title='text generation')
    group.add_argument("--port", type=int, default=5000,
                       help='port for text generation server to run on')
    return parser


def init_megatron():
    #if args.tokenizer_type == "GPT2BPETokenizer":
    #    tokenizer_args = {'tokenizer_type': 'GPT2BPETokenizer',
    #                                    'no_load_rng': True,
    #                                    'no_load_optim': True}
    #else:
    #    tokenizer_args = {'tokenizer_type': 'HFAutoTokenizer',
    #                'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
    #                'no_load_rng': True,
    #                'no_load_optim': True}
        
    initialize_megatron(extra_args_provider=add_text_generate_args,
                        args_defaults = {'tokenizer_type': 'HFAutoTokenizer',
                        'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
                        'no_load_rng': True,
                        'no_load_optim': True})



class MegatronEvaluateHarness(BaseLM):
    def __init__(self,model, tokenizer, max_batch_size = 512, max_length=1024):
        super(MegatronEvaluateHarness, self).__init__()
        self.max_batch_size = max_batch_size
        self._max_length = max_length
        self.tokenizer = tokenizer
        self.model = model
        
        self.vocab_size = self.tokenizer.vocab_size
        
    @property
    def eot_token_id(self):
        self.tokenizer.eos_token_id
        
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def max_gen_toks(self):
        return self.max_length
    
    @property
    def device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"
    
    @property
    def batch_size(self):
        return self.max_batch_size
        
    
    def tok_encode(self, string):
        return self.tokenizer.tokenize(string)
    
    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens.cpu().numpy())
    
    def _model_call(self, inps):
        self.forward_step = ForwardStep(self.model, max_batch_size = self.max_batch_size, max_sequence_length = self.max_length)
        with torch.no_grad():
            attention_mask, position_ids = _build_attention_mask_and_position_ids(inps)
            logits = self.forward_step(inps, position_ids, attention_mask)
            return logits
        
    def _model_generate(self,context, max_length, eos_token_id):
        args = get_args()
        response, response_seg, response_logprobs, _ = \
        generate_and_post_process(
        self.model,
        prompts=context,
        tokens_to_generate=max_length,
        return_output_log_probs=True,
        top_k_sampling=0.0,
        top_p_sampling=0.0,
        top_p_decay=0.0,
        top_p_bound=0.0,
        temperature=1.0,
        add_BOS=True,
        use_eod_token_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        prevent_newline_after_colon=False,
        random_seed=random_seed)
        return response, response_seg, response_logprobs
            
            
class Evaluator():
    def __init__(self, checkpoint_path=None, model=None, results_path = "./results.json", tokenizer = None, task_list = None):
        
        if tokenizer is None:
            try:
                self.tokenizer = get_tokenizer()
            except Exception as e:
                print("Failed to create a tokenizer and none provided: ", e)
                
        if checkpoint_path is None and model is None:
            raise ValueError("Either a model object or a checkpoint file must be provided to the evaluator.")
        
        if model is not None:
            self.model = model
        else: 
            model = get_model(model_provider, wrap_with_ddp=False)
            _ = load_checkpoint(model, None, None)
            assert len(model) == 1, "Above condition should have caught this"
            self.model = model[0]
            
        print("TASK LIST: ", task_list)
        if task_list is None:
            self.task_list = ["lambada_openai","hellaswag"]
        else:
            self.task_list = task_list.split(",")
            
        self.task_dict = tasks.get_task_dict(self.task_list)
        self.results_path = results_path
        
        
    def evaluate(self, max_batch_size = 64, max_length = 2048, adaptive_seqlen = False, num_fewshot = 0, eval_fp32 = False):
        
        adaptor = MegatronEvaluateHarness(self.model, self.tokenizer,max_batch_size=max_batch_size, max_length = max_length)
        
        results = lm_eval.evaluator.evaluate(adaptor, self.task_dict, False, num_fewshot, None)
        print("RESULTS: ", results)
        return results
    
    def write_results(self, results, results_path = None):
        if results_path is None:
            results_path = self.results_path
            
        if is_last_rank():
            with open(results_path, "w") as f:
                json.dump(results, f)
        


if __name__ == '__main__':
    # pass in a config name and checkpoint path
    #config_path = "examples/bf16_125m_8E.sh"
    #checkpoint_path = "/workspace/ckpts_bf16_125m"
    
    # EXAMPLE COMMAND:
    # torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000 evaluation.py --config /opt/Megatron-LM/examples/megarun_slurm/moe_1p3B_8E_bare.sh --checkpoint /checkpoints/megarun/ckpts_1p3b_bf16 --task-list openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai
    # task list openbookqa,arc_easy,winogrande,hellaswag,arc_challenge,piqa,boolq,lambada_openai,lambada_standard
    parser = argparse.ArgumentParser(description='Download evaluation harness', allow_abbrev=False)
    parser.add_argument('--config', type=str, help='Path to the model config file.')
    parser.add_argument('--checkpoint', type=str, help='Path to the model config file.')
    parser.add_argument('--task-list', type=str, default ="", help = "Pass in a comma separated task list")
    
    args = parser.parse_args()
    
    os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
    
    config_path = args.config
    checkpoint_path = args.checkpoint
    task_list = args.task_list
    if task_list == "":
        task_list = None
    if task_list == "all":
        task_list = ALL_TASKS
    # parse the config file
    parse_config_file_update_argv(config_path, checkpoint_path)
    # initialize megatron with the correct args
    init_megatron()
    # begin evaluation
    evaluator = Evaluator(checkpoint_path = checkpoint_path, task_list = task_list)
    results = evaluator.evaluate()
    print("RESULTS: ", results)
    evaluator.write_results(results, "./results.json")
    
    
    