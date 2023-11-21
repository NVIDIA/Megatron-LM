# %%
%load_ext autoreload
%autoreload 2
"""Sample Generate GPT"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import socket
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.arguments import core_transformer_config_from_args
from megatron.text_generation_server import MegatronServer
from megatron.text_generation import generate_and_post_process
from megatron.text_generation import beam_search_and_post_process
import torch

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "6000"

config_path = "examples/bf16_125m_8E.sh"

with open(config_path,"r") as f:
    filestr = f.read()
    
#print(file_str)

def extract_keyword_args(filestr, keyword):
    gpt_split = filestr.split(keyword)
    if len(gpt_split) <=1:
        raise ValueError("Config provided does not have a GPT_ARGS variable provided")
    arg_splits = gpt_split[1].split("\"")
    gpt_args = arg_splits[1]
    gpt_args = gpt_args.replace("\n","").replace("\\","").replace("\t","")
    gpt_args = ' '.join(gpt_args.split())
    return gpt_args.strip().split(" ")
print(extract_keyword_args(filestr, "GPT_ARGS"))

def extract_data_paths(filestr):
    #vocab_file = filestr.split("VOCAB_FILE=")[1].split("\n")[0]
    #merge_file = filestr.split("MERGE_FILE=")[1].split("\n")[0]
    vocab_file = "/workspace/gpt-neox/data/gpt2-vocab.json"
    merge_file = "/workspace/gpt-neox/data/gpt2-merges.txt"
    checkpoint_path = "/workspace/ckpts_bf16_125m"
    
    data_path = filestr.split("DATA_PATH=")[1].split("\n")[0]
    return ["--data-path", data_path, "--vocab-file" , vocab_file, "--merge-file" , merge_file,"--load" , checkpoint_path]
    
print(extract_data_paths(filestr))


sys.argv = ["checkpointing_tests.py"] # a hack to get around the jupyter issues in the sys.argc which we are messing with
sys.argv += extract_keyword_args(filestr, "GPT_ARGS")
sys.argv += extract_data_paths(filestr)
print(sys.argv)
#sys.argv += extract_keyword_args(filestr, "DATA_ARGS")

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

initialize_megatron(extra_args_provider=add_text_generate_args,
                    args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                    'no_load_rng': True,
                                    'no_load_optim': True})

#args_defaults={'tokenizer_type': 'HFAutoTokenizer',
#                'hf_autotokenizer_model': 'EleutherAI/gpt-neox-20b',
#                'no_load_rng': True,
#                'no_load_optim': True})

args = get_args()
print("ARGS: ", args)

model = get_model(model_provider, wrap_with_ddp=False)
_ = load_checkpoint(model, None, None)
assert len(model) == 1, "Above condition should have caught this"
model = model[0]

# %%
#print("MODEL: ", model)
prompts = ["How is your day?"]
tokens_to_generate = 100
logprobs = True
top_k = 0.0
top_p = 0.0
temperature = 1.0
top_p_decay = 0.0
top_p_bound = 0.0
add_BOS = False
stop_on_double_eol = False
stop_on_eol = False
random_seed = -1
prevent_newline_after_colon = False

response, response_seg, response_logprobs, _ = \
    generate_and_post_process(
    model,
    prompts=prompts,
    tokens_to_generate=tokens_to_generate,
    return_output_log_probs=logprobs,
    top_k_sampling=top_k,
    top_p_sampling=top_p,
    top_p_decay=top_p_decay,
    top_p_bound=top_p_bound,
    temperature=temperature,
    add_BOS=add_BOS,
    use_eod_token_for_early_termination=True,
    stop_on_double_eol=stop_on_double_eol,
    stop_on_eol=stop_on_eol,
    prevent_newline_after_colon=prevent_newline_after_colon,
    random_seed=random_seed)
    
print("DONE")
    
 # %%

print("SUCCESSFULLY GENERATED:")
print("RESPONSE: ", response)



# %%
from megatron import get_tokenizer
from megatron.text_generation.forward_step import ForwardStep
from megatron.utils import get_ltor_masks_and_position_ids

def _build_attention_mask_and_position_ids(tokens):
    """Build the attention mask and postition ids for the input tokens."""

    # Since we are not interested in loss-mask and reset attention/position
    # is also False, eod_token is not used so it is safe to set it to None.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False)

    return attention_mask, position_ids

tokenizer = get_tokenizer()
print(tokenizer)
prompt = "Hello world 123"
tokens = torch.tensor(tokenizer.tokenize(prompt)).unsqueeze(0).cuda()


print(tokens)
forward_step = ForwardStep(model, max_batch_size = 100, max_sequence_length = 100)

with torch.no_grad():
    attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
    logits = forward_step(tokens, position_ids, attention_mask)
    print("OUTPUTS: ", logits)
    


import lm_eval 
from lm_eval.base import BaseLM
    
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
        return self.max_batch_size # this doesn't work ? for miultiGPU
        
    
    def tok_encode(self, string):
        return self.tokenizer.tokenize(string)
    
    def tok_decode(self, tokens):
        return self.tokenizer.detokenize(tokens.cpu().numpy())
    
    def _model_call(self, inps):
        self.forward_step = ForwardStep(self.model, max_batch_size = self.max_batch_size, max_sequence_length = self.max_length)
        #print("IN MODEL CALL", inps.shape)
        #print(self.tok_decode(inps[0,:]))
        #print("IN MODEL CALL: ", inps.shape)
        with torch.no_grad():
            attention_mask, position_ids = _build_attention_mask_and_position_ids(inps)
            logits = self.forward_step(inps, position_ids, attention_mask)
            return logits
        
    def _model_generate(self,context, max_length, eos_token_id):
        print("IN MODEL GENERATE: ")
        args = get_args()
        print("ARGS EOS: ", args.eos_token_id)
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
            
    
tokenizer = get_tokenizer()
eval_harness = MegatronEvaluateHarness(model,tokenizer)
print("EVAL HARNESS: ", eval_harness)
    

print("LM EVAL DONE")

# %%


task_list = ["lambada_openai","hellaswag"]
results_path = "./results.json"
adaptive_seq_len = False
num_fewshot = 0
eval_fp32 = False


from lm_eval.models.gpt2 import GPT2LM
from lm_eval import evaluator, tasks, utils
from lm_eval.base import CacheHook
from lm_eval.tasks import ALL_TASKS

#task_list = ALL_TASKS if args.task_list == 'all' else args.task_list.split(',')
task_dict = tasks.get_task_dict(task_list)
print("TASK DICT: ", task_dict)

tokenizer = get_tokenizer()
adaptor = MegatronEvaluateHarness(model, tokenizer,max_batch_size=8)
results = evaluator.evaluate(adaptor, task_dict, False, num_fewshot, None)
print("RESULTS: ", results)

# %%

tokenizer = get_tokenizer()
print(tokenizer)
prompt = "Hello world 123"
tokens = torch.tensor(tokenizer.tokenize(prompt)).unsqueeze(0).cuda()
print(tokens)
string = tokenizer.detokenize(tokens[0,:].cpu().numpy())
print(string)