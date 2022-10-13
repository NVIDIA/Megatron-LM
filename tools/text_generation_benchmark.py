
"""Sample Generate GPT"""
import os
import sys
import re
sys.path.append(os.path.abspath(os.path.join(
    os.getcwd(),
    "Megatron-LM",
)))
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import mpu
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.text_generation import generate_and_post_process
import torch
from human_eval.data import write_jsonl, read_problems
from tqdm import tqdm


GENERATE_NUM = 0

# End on unindented code
# EOF_STRINGS = ["\nclass", "\ndef", "\n#", "\n@", "\nprint", "\nif"]


BATCH_SIZE = 512
TOKENS_TO_GENERATE = 128
PROMPT_LENGTH = 128
NUM_BATCHES = 8


# NUM_SAMPLES_PER_TASK = 5
# # Number of human-eval tasks
# NUM_TASKS = 200

def send_do_generate():
        choice = torch.cuda.LongTensor([GENERATE_NUM])
        torch.distributed.broadcast(choice, 0)


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

    return model

def get_batches(prompts, batch_size):
    for start_idx in tqdm(range(0, len(prompts), batch_size)):
        actual_batch_size = min(batch_size, len(prompts) - start_idx)
        yield prompts[start_idx: start_idx + actual_batch_size]


def unbatch(d: dict):
    return [dict(zip(d.keys(), t)) for t in zip(*d.values())]


# Use fixed-length prompts
def load_evaluation_data(args):
    # HumanEval data
    # problems = read_problems()

    # batches = get_batches(
    #     [
    #         problems[task_id]["prompt"]
    #         for task_id in problems
    #         for _ in range(5)
    #     ],
    #     BATCH_SIZE
    # )
    # return batches

    prompt = " ".join(["one"] * PROMPT_LENGTH)
    prompts = [prompt] * (BATCH_SIZE * NUM_BATCHES)

    batches = get_batches(prompts, BATCH_SIZE)
    return batches


if __name__ == "__main__":
    # Initialize Megatron
    initialize_megatron(extra_args_provider=None,
                        args_defaults={'tokenizer_type': 'GPT2BPETokenizer',
                                       'no_load_rng': True,
                                       'no_load_optim': True})

    args = get_args()
    timers = get_timers()

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()
    # Setup model and load checkpoint
    model = get_model(model_provider, wrap_with_ddp=False)

    if args.load is not None:
        iteration = load_checkpoint(model, None, None, iteration=None)
    else:
        iteration = None

    assert len(model) == 1
    model = model[0]

    def generate(prompts):
        response, response_seg, response_logprobs, tokens = \
                generate_and_post_process(
                    model,
                    prompts=prompts,
                    tokens_to_generate=TOKENS_TO_GENERATE,
                    return_output_log_probs=True,
                    use_eod_token_for_early_termination=False)
        
        assert all([r.startswith(p) for r, p in zip(response, prompts)])
        result = {
            "response": response, 
            "response_seg": response_seg,
            "raw_completion": [r[len(p):] for r, p in zip(response, prompts)]
        }
        # The "completion" field contains the string that is actually going to be evaluated by the HumanEval script
        # result["completion"] = [post_process_completion(c) for c in result["raw_completion"]]
        # Return a list of dicts
        return unbatch(result)

    # if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
    #     server = MegatronServer(model)
    #     server.run("0.0.0.0")

    # while True:
    #     choice = torch.cuda.LongTensor(1)
    #     torch.distributed.broadcast(choice, 0)
    #     if choice[0].item() == 0:
    #         generate_and_post_process(model)


    # Evaluation data iterator
    batches = load_evaluation_data(args)

    timers('generate').start()
    # Generate
    samples = [
        generate_dict
        for batch in batches
        for generate_dict in generate(batch)
    ]
    timers('generate').stop()

    elapsed = timers.timers['generate'].elapsed(reset=False)
    num_tokens = TOKENS_TO_GENERATE * NUM_BATCHES * BATCH_SIZE
    print(f"{elapsed * 1000 / (num_tokens)} ms per token")
    timers.log(['generate'])
    if args.transformer_timers: 
        timers.log(["Transformer forward"])
    print("DONE")

    # Write results to file
    # if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
    #     write_jsonl(args.output_file.format(iteration), samples)

