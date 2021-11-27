
import json
import torch
from nltk import word_tokenize
from megatron import mpu
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from tasks.knwl_dialo.utils import get_token_stream


def model_provider(pre_process=True, post_process=True):
    """Build the model."""

    print_rank_0('building GPT model ...')
    model = GPTModel(
        num_tokentypes=0,
        parallel_output=True,
        pre_process=pre_process,
        post_process=post_process
    )
    return model


def generate_samples_by_prompting_input_from_file(model):

    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        input_pos = 0
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                    'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file

        fname_out = open(sample_output_file, "w")

    # Read the prompt file
    if args.dynamic_prompt:
        prompt_examples_dict = {}
        with open(args.prompt_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                line_dict = json.loads(line)
                key = list(line_dict.keys())[0]
                
                if key not in prompt_examples_dict:
                    prompt_examples = line_dict[key]

                    prompt = ""
                    for instance in prompt_examples:
                        instance = instance.strip()
                        prompt += instance + " \n"

                    prompt_examples_dict[key] = prompt

    else:
        with open(args.prompt_file, "r") as f:
            prompt_examples = f.readlines()
            prompt_examples = prompt_examples[:args.num_prompt_examples]

            prompt = ""
            for instance in prompt_examples:
                instance = instance.strip()
                prompt += instance + " \n"

    assert args.prompt_type in ["knowledge", "response"]
    context_count = 0
    model.eval()
    with torch.no_grad():
        while True:
            raw_text_len = 0

            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                input_str = all_raw_text[input_pos]
                input_str = input_str.strip()
                splits = input_str.split("\t")
                control_codes = splits[0].split(" [CTRL] ")
                topic = control_codes[0]

                if args.dynamic_prompt:
                    turns = splits[1].split(" [SEP] ")
                    last_turn = turns[-1]
                    key = topic + " " + last_turn
                    raw_text = prompt_examples_dict[key]

                else:
                    raw_text = prompt

                if args.prompt_type == "knowledge":
                    turns = splits[1].split(" [SEP] ")
                    context = turns[-1]
                    if " -> " in raw_text and " => " not in raw_text:
                        raw_text += "( " + context + " ) " + topic + " ->"
                    else:
                        raw_text += "( " + context + " ) " + topic + " =>"
                
                else:
                    # args.prompt_type == "response":
                    turns = splits[1].split(" [SEP] ")
                    knowledge = splits[2]
                    knowledge = " ".join(word_tokenize(knowledge))

                    last_turn = turns[-1]
                    knowledge = knowledge.strip()
                    last_turn = last_turn.strip()
                    raw_text += "Topic: " + topic + ". "
                    raw_text += "User says: " + last_turn + " "
                    raw_text += "We know that: " + knowledge + " "
                    raw_text += "System replies:"

                input_pos += 1
                raw_text_len = len(raw_text)
                context_tokens = tokenizer.tokenize(raw_text)
            
            else:
                context_tokens = tokenizer.tokenize("EMPTY TEXT")

            if input_pos % 100 == 0:
                print_rank_0("input_pos: %d" % input_pos)

            token_stream = get_token_stream(model, [context_tokens])
            for _, decode_tokens in enumerate(token_stream):
                pass
            
            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():

                    decode_tokens, _ = decode_tokens
                    decode_tokens = decode_tokens[0].cpu().numpy().tolist()
                    trim_decode_tokens = tokenizer.detokenize(
                        decode_tokens)[raw_text_len:]
                    
                    generated_output = trim_decode_tokens.split("\n")[0]
                    generated_output = generated_output.strip()

                    fname_out.write(generated_output)
                    fname_out.write("\n")

            raw_text = None
            context_count += 1

            if input_pos == input_count:
                return


def main():

    args = get_args()
    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider)

    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    generate_samples_by_prompting_input_from_file(model)
