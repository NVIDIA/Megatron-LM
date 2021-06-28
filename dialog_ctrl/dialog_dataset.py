
"""Build Dataset for Controllable Coversational Model"""

import os
import torch
import numpy as np

from megatron import get_tokenizer

def read_data(tokenizer, data_path, train_module):
    """read and tokenize dialog data"""

    data_list = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            splits = line.split("\t")
            length_split = len(splits)
            assert length_split == 2 or length_split == 3 or length_split == 4

            if train_module == "dialog":
                dialog_context = splits[0]
                response = splits[-1]
                # only take the last three turns in the dialog context
                turns = dialog_context.split(" [SEP] ")
                turns = turns[-3:]
                context = " [SEP] ".join(turns)

                input_ids = tokenizer.tokenize(context)
                output_ids = tokenizer.tokenize(response)
                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            elif train_module == "control":
                if length_split == 2:
                    continue
                dialog_context = splits[0]
                ctrl_sent = splits[-2]
                ctrl_code = splits[1] if length_split == 4 else None

                turns = dialog_context.split(" [SEP] ")
                last_turn = turns[-1]
                
                if ctrl_code:
                    inputs = last_turn + " [CTRL] " + ctrl_code
                else:
                    inputs = last_turn
                outputs = ctrl_sent

                input_ids = tokenizer.tokenize(inputs)
                output_ids = tokenizer.tokenize(outputs)
                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            else:
                raise ValueError("Please input a correct train-module name! (either dialog or cnotrol))")
    
    return data_list


def data_shuffle(data, seed):
    # set random seed to make the shuffling reproducible
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


class ControlDialogDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_seq_len, pad_id, eod_id):
        # need to deal with padding, label masking
        self.data = data
        self.max_seq_len
        self.pad_id = pad_id
        self.eod_id = eod_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        input_ids, output_ids = data_dict["input_ids"], data_dict["output_ids"]
        
        assert len(input_ids) < self.max_seq_len, "Set a larger max_seq_len!"

        # length_of_loss_mask == length_of_text - 1
        text = input_ids + [self.pad_id] + output_ids + [self.eod_id]
        loss_mask = [0]*len(input_ids) + [1]*(len(output_ids)+1)

        text_len = len(text)
        if text_len > self.max_seq_len:
            text = text[:self.max_seq_len]
            loss_mask = loss_mask[:self.max_seq_len-1]
        else:
            text += [self.pad_id] * (self.max_seq_len - text_len)
            loss_mask += [0] * (self.max_seq_len - text_len)

        return {"text": np.array(text, dtype=np.int64), "loss_mask": np.array(loss_mask, dtype=np.int64)}


def build_train_valid_test_datasets(data_folder, dataset_name, train_module, max_seq_len, seed):
    """Build train, valid, and test datasets."""

    dataname_dict = {"wizard_of_wikipedia": {"train": "train_entity_based_control.txt", "valid": "valid_random_split_entity_based_control.txt", "test": "test_random_split_entity_based_control.txt"}}
    
    train_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["train"])
    valid_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["valid"])
    test_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["test"])

    tokenizer = get_tokenizer()
    train_data_list = read_data(tokenizer, train_data_path, train_module)
    valid_data_list = read_data(tokenizer, valid_data_path, train_module)
    test_data_list = read_data(tokenizer, test_data_path, train_module)

    # shuffle the training data
    train_data_list = data_shuffle(train_data_list, seed)

    # build train, valid, and test datasets
    train_dataset = ControlDialogDataset(train_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)
    valid_dataset = ControlDialogDataset(valid_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)
    test_dataset = ControlDialogDataset(test_data_list, max_seq_len, tokenizer.pad_id, tokenizer.eod_id)

    return (train_dataset, valid_dataset, test_dataset)
