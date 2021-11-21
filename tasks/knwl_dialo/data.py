
"""Build Dataset for Controllable Coversational Model"""

import os
import torch
import numpy as np

from megatron import get_tokenizer
from megatron import print_rank_0


def read_data_for_finetuning(tokenizer, data_path, module):
    """
    Data Format: topic \t dialog context \t knowledge \t response.
    """
    
    data_list = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            splits = line.split("\t")
            assert len(splits) == 4

            topic = splits[0].split(" [CTRL] ")[0]
            dialog_context = splits[1]
            knowledge = splits[2]
            response = splits[3]

            turns = dialog_context.split(" [SEP] ")
            turns = turns[-3:]

            if module == "response":
                # input_ids
                input_ids = tokenizer.tokenize("( " + topic + " )")
                if knowledge != "no_passages_used":
                    input_ids.extend(tokenizer.tokenize("( " + knowledge + " )")[:256])
                
                for turn in turns:
                    turn = "<< " + turn + " >>"
                    input_ids.extend(tokenizer.tokenize(turn))
                input_ids.extend(tokenizer.tokenize(":"))

                # output_ids
                output_ids = tokenizer.tokenize(response)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})
                
            elif module == "knowledge":
                # skip example without knowledge sentences
                if knowledge == "no_passages_used":
                    continue

                input_ids = []
                input_ids.extend(tokenizer.tokenize("( " + topic + " )"))
                
                for turn in turns:
                    turn = "<< " + turn + " >>"
                    input_ids.extend(tokenizer.tokenize(turn))
                input_ids.extend(tokenizer.tokenize(":"))

                output_ids = tokenizer.tokenize(knowledge)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            else:
                raise ValueError("Please input a correct module name! " \
                                 "(either dialog or cnotrol))")
    
    return data_list


def read_data_for_prompting(tokenizer, test_data_path, prompt_file, 
                            module, num_prompt_examples, dynamic_prompt):
    
    # get prompts
    if dynamic_prompt:
        import json
        prompt_examples_dict = {}
        with open(prompt_file, "r") as f:
            for i, line in enumerate(f):
                line = line.strip()
                line_dict = json.loads(line)
                key = list(line_dict.keys())[0]
                
                if key not in prompt_examples_dict:
                    prompt_examples = line_dict[key]
                    prompt_examples = prompt_examples[:num_prompt_examples]
                    prompt = ""
                    for instance in prompt_examples:
                        instance = instance.strip()
                        prompt += instance + " \n"

                    prompt_examples_dict[topic] = prompt

    else:
        with open(prompt_file, "r") as f:
            prompt_examples = f.readlines()
    
            prompt_examples = prompt_examples[:num_prompt_examples]
            prompt = ""
            for instance in prompt_examples:
                instance = instance.strip()
                prompt += instance + " \n"

    data_list = []
    with open(test_data_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")

            topic = splits[0].split(" [CTRL] ")[0]
            turns = splits[1].split(" [SEP] ")[-3:]
            last_turn = turns[-1]
            ctrl_sent = splits[2]
            response = splits[3]

            if dynamic_prompt:
                prompt = prompt_examples_dict[topic]

            if module == "response":
                # input seq
                input_seq = prompt

                input_seq += "Topic: " + topic + ". "
                input_seq += "User says: " + last_turn + " "
                input_seq += "We know that: " + ctrl_sent + " "
                input_seq += "System replies:"

                # output seq
                output_seq = response

                input_ids = tokenizer.tokenize(input_seq)
                output_ids = tokenizer.tokenize(output_seq)
                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            elif module == "knowledge":
                # input seq
                input_seq = prompt
                input_seq += "( " + last_turn + " ) " + topic + " =>"

                # output seq
                output_seq = ctrl_sent

                input_ids = tokenizer.tokenize(input_seq)
                output_ids = tokenizer.tokenize(output_seq)
                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            else:
                raise ValueError("Please input a correct module name! " \
                                 "(either dialog or cnotrol))")

    return data_list


def data_shuffle(data, seed):
    # set random seed to make the shuffling reproducible
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


class KnwlDialoDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_seq_len, pad_id, eod_id):
        # need to deal with padding, label masking
        self.data = data
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.eod_id = eod_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        input_ids, output_ids = data_dict["input_ids"], data_dict["output_ids"]
        
        text = input_ids + output_ids + [self.eod_id]
        loss_mask = [0]*(len(input_ids)-1) + [1]*(len(output_ids)+1)

        text_len = len(text)
        if text_len > self.max_seq_len+1:
            text = text[:self.max_seq_len+1]
            loss_mask = loss_mask[:self.max_seq_len]
        else:
            text += [self.pad_id] * (self.max_seq_len+1 - text_len)
            loss_mask += [0] * (self.max_seq_len+1 - text_len)

        return {"text": np.array(text, dtype=np.int64), \
                "loss_mask": np.array(loss_mask, dtype=np.int64)}


def build_train_valid_datasets(train_data_path, valid_data_path, module,
                               max_seq_len, seed):
    """Build train, valid, and test datasets."""

    tokenizer = get_tokenizer()
    train_data_list = read_data_for_finetuning(tokenizer, train_data_path, module)
    valid_data_list = read_data_for_finetuning(tokenizer, valid_data_path, module)

    # shuffle the training data
    train_data_list = data_shuffle(train_data_list, seed)

    # build train, valid datasets
    train_dataset = KnwlDialoDataset(train_data_list, 
                                     max_seq_len, 
                                     pad_id=tokenizer.pad_id, 
                                     eod_id=tokenizer.eod_id)

    valid_dataset = KnwlDialoDataset(valid_data_list, 
                                     max_seq_len, 
                                     pad_id=tokenizer.pad_id, 
                                     eod_id=tokenizer.eod_id)

    return train_dataset, valid_dataset


def build_test_dataset(test_data_path, module, max_seq_len):
    tokenizer = get_tokenizer()

    test_data_list = read_data_for_finetuning(tokenizer, test_data_path, module)

    test_dataset = KnwlDialoDataset(test_data_list, 
                                    max_seq_len, 
                                    pad_id=tokenizer.pad_id, 
                                    eod_id=tokenizer.eod_id)

    return test_dataset


def build_test_dataset_for_prompting(test_data_path, prompt_file, module, max_seq_len, 
                                     num_prompt_examples, dynamic_prompt):
    tokenizer = get_tokenizer()

    test_data_list = read_data_for_prompting(tokenizer, test_data_path, prompt_file, module, \
                                             num_prompt_examples, dynamic_prompt)

    test_dataset = KnwlDialoDataset(test_data_list,
                                    max_seq_len,
                                    pad_id=tokenizer.pad_id, 
                                    eod_id=tokenizer.eod_id)

    return test_dataset
