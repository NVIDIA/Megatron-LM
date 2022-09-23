# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Prompting the pretrained language model to generate knowledge/response"""

import json
import torch
import requests
from nltk import word_tokenize
from megatron import get_args
from megatron import print_rank_0
from megatron import get_tokenizer
from megatron.core import mpu
from megatron.model import GPTModel
from megatron.training import get_model
from megatron.checkpointing import load_checkpoint
from megatron.initialize import initialize_megatron
from megatron.text_generation import generate_and_post_process


def call_model_api(inputs, tokens_to_generate):
    """Calling the model api to get the output generations"""
    
    args = get_args()

    # The following is an example of using the Megatron API
    # You can also implement your own API function to place this part
    headers = {'Content-Type': 'application/json; charset=UTF-8'}
    data = {"prompts": [inputs], "tokens_to_generate": tokens_to_generate, "top_k": 1}
    data_json = json.dumps(data)
    outputs = requests.put(args.megatron_api_url, headers=headers, data=data_json).json()["text"][0]

    input_len = len(inputs)
    outputs = outputs[input_len:]
    outputs = outputs.split("\n")[0].strip()
    
    return outputs


def read_prompts(prompt_path, prompt_type, n_example):
    """Read prompt data"""

    if prompt_type == "knowledge":
        # prompts for the knowledge generation
        prompt_examples_dict = {}
        # read prompt_path
        with open(prompt_path, "r") as f:
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

        return prompt_examples_dict

    else:
        # prompts for the response generation
        # read prompt_path
        prompt = ""
        with open(prompt_path, "r") as f:
            prompt_examples = f.readlines()
            prompt_examples = prompt_examples[:n_example]
            for instance in prompt_examples:
                instance = instance.strip()
                prompt += instance + " \n"

        return prompt


def generate_samples_by_calling_api():
    """ Generate outputs by calling"""
    args = get_args()
    assert args.prompt_type in ["knowledge", "response"], \
                "Please input a correct prompt type!"

    if args.prompt_type == "knowledge":
        # read knowledge generation prompts
        knwl_gen_prompt_dict = read_prompts(
            args.prompt_file, args.prompt_type, args.num_prompt_examples)
        
    else:
        resp_gen_prompt = read_prompts(
            args.prompt_file, args.prompt_type, args.num_prompt_examples)

    # read the test data
    fname = open(args.sample_input_file, "r")
    test_sample_list = fname.readlines()
    # create output file
    fname_out = open(args.sample_output_file, "w")

    # call the api to get the output generations
    for test_sample in test_sample_list:
        test_sample = test_sample.strip()
        splits = test_sample.split("\t")
        topic = splits[0]

        # prepare the inputs for the api
        if args.prompt_type == "knowledge":
            ## inputs = prompt + current test
            # get the prompt
            turns = splits[1].split(" [SEP] ")
            last_turn = turns[-1]
            key = topic + " " + last_turn
            inputs = knwl_gen_prompt_dict[key]

            # add current test
            inputs += "( " + last_turn + " ) " + topic + " =>"

        else:
            # inputs = prompt + current test
            # get the prompt
            inputs = resp_gen_prompt

            # add current test
            turns = splits[1].split(" [SEP] ")
            knowledge = splits[2]
            last_turn = turns[-1]
            last_turn = " ".join(word_tokenize(last_turn))
            knowledge = " ".join(word_tokenize(knowledge))
            knowledge = knowledge.strip()
            last_turn = last_turn.strip()
            inputs += "Topic: " + topic + ". "
            inputs += "User says: " + last_turn + " "
            inputs += "We know that: " + knowledge + " "
            inputs += "System replies:"

        # get the output generations from the api, 
        # and write to the output file
        generations = call_model_api(inputs, args.out_seq_length)
        fname_out.write(generations)
        fname_out.write("\n")

    fname.close()
    fname_out.close()


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
    """Prompt a pretrained language model to generate knowledge/response"""
    
    # get tokenizer
    args = get_args()
    tokenizer = get_tokenizer()

    # Read the sample file and open the output file.
    assert args.sample_input_file is not None, \
        'sample input file is not provided.'
    if mpu.is_pipeline_first_stage() and mpu.get_tensor_model_parallel_rank() == 0:
        fname = open(args.sample_input_file, "r")
        all_raw_text = fname.readlines()
        input_count = len(all_raw_text)
        if args.sample_output_file is None:
            sample_output_file = args.sample_input_file + ".out"
            print('`sample-output-file` not specified, setting '
                    'it to {}'.format(sample_output_file))
        else:
            sample_output_file = args.sample_output_file

        fname_out = open(sample_output_file, "w")

    # only two prompt types (i.e., knowledge and response) are allowed
    assert args.prompt_type in ["knowledge", "response"], \
                "Please input a correct prompt type!"

    # Read the prompt file
    if args.prompt_type == "knowledge":
        # read the prompts for the knowledge generation
        prompt_examples_dict = {}
        with open(args.prompt_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                line_dict = json.loads(line)
                key = list(line_dict.keys())[0]

                # get the prompt examples based on the key
                if key not in prompt_examples_dict:
                    prompt_examples = line_dict[key]
                    prompt = ""
                    for instance in prompt_examples:
                        instance = instance.strip()
                        prompt += instance + " \n"
                    prompt_examples_dict[key] = prompt

    else:
        # read the prompts for the response generation
        # prompts are fixed for all test samples
        with open(args.prompt_file, "r") as f:
            prompt_examples = f.readlines()
            prompt_examples = prompt_examples[:args.num_prompt_examples]

            prompt = ""
            for instance in prompt_examples:
                instance = instance.strip()
                prompt += instance + " \n"

    input_pos = 0
    model.eval()
    # perform prompting
    with torch.no_grad():
        while True:
            raw_text_len = 0
            if mpu.is_pipeline_first_stage() \
               and mpu.get_tensor_model_parallel_rank() == 0:
                input_str = all_raw_text[input_pos]
                input_str = input_str.strip()
                splits = input_str.split("\t")
                topic = splits[0]

                if args.prompt_type == "knowledge":
                    # first add the prompt into the raw_text
                    turns = splits[1].split(" [SEP] ")
                    last_turn = turns[-1]
                    key = topic + " " + last_turn
                    raw_text = prompt_examples_dict[key]

                    # construct inputs for knowledge generation
                    # then add the constructed inputs into the raw_text
                    raw_text += "( " + last_turn + " ) " + topic + " =>"
                
                else:
                    # first add the prompt into the raw_text
                    raw_text = prompt

                    # construct inputs for response generation
                    # then add the constructed inputs into the raw_text
                    turns = splits[1].split(" [SEP] ")
                    knowledge = splits[2]
                    last_turn = turns[-1]
                    last_turn = " ".join(word_tokenize(last_turn))
                    knowledge = " ".join(word_tokenize(knowledge))
                    knowledge = knowledge.strip()
                    last_turn = last_turn.strip()
                    raw_text += "Topic: " + topic + ". "
                    raw_text += "User says: " + last_turn + " "
                    raw_text += "We know that: " + knowledge + " "
                    raw_text += "System replies:"

                input_pos += 1
                raw_text_len = len(raw_text)
            
            else:
                raw_text = "EMPTY TEXT"

            if input_pos % 100 == 0:
                print_rank_0("input_pos: %d" % input_pos)

            outputs = generate_and_post_process(
                        model=model, 
                        prompts=[raw_text], 
                        tokens_to_generate=args.out_seq_length,
                        top_k_sampling=1)
            prompts_plus_generations = outputs[0]
            prompts_plus_generations = prompts_plus_generations[0]

            # write the generated output to the output file
            if mpu.get_tensor_model_parallel_rank() == 0:
                if mpu.is_pipeline_first_stage():

                    generations = prompts_plus_generations[raw_text_len:]
                    generations = generations.split("\n")[0]
                    generations = generations.strip()
                    fname_out.write(generations)
                    fname_out.write("\n")

            raw_text = None
            if input_pos == input_count:
                return


def main():

    args = get_args()
    if args.api_prompt:
        # obtain the generations by calling the api
        generate_samples_by_calling_api()
        return

    if args.num_layers_per_virtual_pipeline_stage is not None:
        print("Interleaved pipeline schedule is not yet supported for text generation.")
        exit()

    # Set up model and load checkpoint.
    model = get_model(model_provider, wrap_with_ddp=False)
    if args.load is not None:
        _ = load_checkpoint(model, None, None)

    assert len(model) == 1, "Above condition should have caught this"
    model = model[0]

    # perform the prompting
    generate_samples_by_prompting_input_from_file(model)
