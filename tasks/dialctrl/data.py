
"""Build Dataset for Controllable Coversational Model"""

import os
import torch
import numpy as np

from megatron import get_tokenizer
from megatron import print_rank_0

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
                # if length_split == 2:
                #     continue

                dialog_context = splits[0]
                if length_split > 2:
                    ctrl_sent = splits[-2]
                response = splits[-1]
                # only take the last three turns in the dialog context
                turns = dialog_context.split(" [SEP] ")
                turns = turns[-3:]

                # input_ids
                input_ids = []
                if length_split > 2:
                    input_ids.extend(tokenizer.tokenize("( " + ctrl_sent + " )"))

                for idx, turn in enumerate(turns):
                    if not (turn.endswith("?") or turn.endswith(".") or turn.endswith("!")):
                        turn = turn + " ."
                    input_ids.extend(tokenizer.tokenize(turn))
                
                # output_ids
                output_ids = tokenizer.tokenize(response)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            elif train_module == "control":
                if length_split == 2:
                    continue
                dialog_context = splits[0]
                ctrl_sent = splits[-2]
                ctrl_code = splits[1] if length_split == 4 else None

                turns = dialog_context.split(" [SEP] ")
                # put control code at the begginning
                input_ids = []
                if ctrl_code:
                    ctrl_code_list = ctrl_code.split(" [CTRL] ")
                    for code in ctrl_code_list:
                        input_ids.extend(tokenizer.tokenize("( " + code + " )"))
                
                turns = turns[-3:]
                for turn in turns:
                    if not (turn.endswith("?") or turn.endswith(".") or turn.endswith("!")):
                        turn = turn + " ."
                    input_ids.extend(tokenizer.tokenize(turn))

                # output_ids
                outputs = ctrl_sent
                output_ids = tokenizer.tokenize(outputs)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            else:
                raise ValueError("Please input a correct train-module name! " \
                                 "(either dialog or cnotrol))")
                
    return data_list


def read_data_v2(tokenizer, data_path, train_module, 
                 last_turn=False, no_control_code=False, add_separator=False, 
                 add_ctrl_code_to_dialog=False, remove_ctrl_sent=False):
    """
    Read and tokenize data for version 2 (v2) data files.
    Format: control code \t dialog context \t control sentence \t response.
    Response only comes from the wizard.
    Currently, this function is used to build test dataset for calculating PPL.
    """
    
    data_list = []
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            line = line.rstrip()
            splits = line.split("\t")
            assert len(splits) == 4

            control_code = splits[0]
            dialog_context = splits[1]
            control_sent = splits[2]
            response = splits[3]

            turns = dialog_context.split(" [SEP] ")
            turns = turns[-3:]

            if train_module == "dialog":
                # input_ids
                if add_ctrl_code_to_dialog:
                    ctrl_code = control_code.split(" [CTRL] ")[0]
                    input_ids = tokenizer.tokenize("( " + ctrl_code + " )")
                    if not remove_ctrl_sent and control_sent != "no_passages_used":
                        input_ids.extend(tokenizer.tokenize("( " + control_sent + " )")[:256])
                
                else:
                    if remove_ctrl_sent or control_sent == "no_passages_used":
                        input_ids = []
                    else:
                        input_ids = tokenizer.tokenize("( " + control_sent + " )")[:256]
                
                for turn in turns:
                    if add_separator:
                        turn = "<< " + turn + " >>"
                    input_ids.extend(tokenizer.tokenize(turn))

                if add_separator:
                    input_ids.extend(tokenizer.tokenize(":"))

                # output_ids
                output_ids = tokenizer.tokenize(response)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})
                
            elif train_module == "control":
                # skip example without control sentences
                if control_sent == "no_passages_used":
                    continue

                input_ids = []
                if not no_control_code:
                    ctrl_code_list = control_code.split(" [CTRL] ")[:3]
                    # only choose maximum three control codes
                    for code in ctrl_code_list:
                        if len(code) > 0:
                            input_ids.extend(tokenizer.tokenize("( " + code + " )"))
                
                if last_turn:
                    input_ids.extend(tokenizer.tokenize(turns[-1]))
                else:
                    for turn in turns:
                        if add_separator:
                            turn = "<< " + turn + " >>"
                        input_ids.extend(tokenizer.tokenize(turn))
                
                if add_separator:
                    input_ids.extend(tokenizer.tokenize(":"))

                output_ids = tokenizer.tokenize(control_sent)

                data_list.append({"input_ids": input_ids, "output_ids": output_ids})

            else:
                raise ValueError("Please input a correct train-module name! " \
                                 "(either dialog or cnotrol))")
    
    return data_list


def data_shuffle(data, seed):
    # set random seed to make the shuffling reproducible
    np.random.seed(seed)
    np.random.shuffle(data)
    return data


class ControlDialogDataset(torch.utils.data.Dataset):

    def __init__(self, data, max_seq_len, sep_id, pad_id, eod_id):
        # need to deal with padding, label masking
        self.data = data
        self.max_seq_len = max_seq_len
        self.sep_id = sep_id
        self.pad_id = pad_id
        self.eod_id = eod_id

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data_dict = self.data[idx]
        input_ids, output_ids = data_dict["input_ids"], data_dict["output_ids"]
        
        # assert len(input_ids) < self.max_seq_len, "Set a larger max-seq-len!"

        # length_of_loss_mask == length_of_text - 1
        # text = input_ids + [self.sep_id] + output_ids + [self.eod_id]
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


def build_train_valid_datasets(train_data_path, valid_data_path, train_module,
                               max_seq_len, seed, last_turn, no_control_code, 
                               add_separator, add_ctrl_code_to_dialog, remove_ctrl_sent):
    """Build train, valid, and test datasets."""

    # dataname_dict = {"wizard_of_wikipedia": {"train": "train_entity_based_control.txt", "valid": "valid_random_split_entity_based_control.txt", "test": "test_random_split_entity_based_control.txt"}}
    
    # train_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["train"])
    # valid_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["valid"])
    # test_data_path = os.path.join(data_folder, dataset_name+"/processed/"+dataname_dict[dataset_name]["test"])

    tokenizer = get_tokenizer()
    # train_data_list = read_data(tokenizer, train_data_path, train_module)
    train_data_list = read_data_v2(tokenizer, train_data_path, train_module, 
                                   last_turn, no_control_code, add_separator, 
                                   add_ctrl_code_to_dialog, remove_ctrl_sent)
    valid_data_list = read_data_v2(tokenizer, valid_data_path, train_module,
                                   last_turn, no_control_code, add_separator, 
                                   add_ctrl_code_to_dialog, remove_ctrl_sent)

    # shuffle the training data
    train_data_list = data_shuffle(train_data_list, seed)

    # build train, valid datasets
    train_dataset = ControlDialogDataset(train_data_list, 
                                         max_seq_len, 
                                         sep_id=tokenizer.sep_id, 
                                         pad_id=tokenizer.pad_id, 
                                         eod_id=tokenizer.eod_id)

    valid_dataset = ControlDialogDataset(valid_data_list, 
                                         max_seq_len, 
                                         sep_id=tokenizer.sep_id, 
                                         pad_id=tokenizer.pad_id, 
                                         eod_id=tokenizer.eod_id)

    return train_dataset, valid_dataset


def build_test_dataset(test_data_path, train_module, max_seq_len, 
                       last_turn, no_control_code, add_separator,
                       add_ctrl_code_to_dialog, remove_ctrl_sent):
    tokenizer = get_tokenizer()

    test_data_list = read_data_v2(tokenizer, test_data_path, train_module,
                                  last_turn, no_control_code, add_separator,
                                  add_ctrl_code_to_dialog, remove_ctrl_sent)

    test_dataset = ControlDialogDataset(test_data_list, 
                                        max_seq_len, 
                                        sep_id=tokenizer.sep_id, 
                                        pad_id=tokenizer.pad_id, 
                                        eod_id=tokenizer.eod_id)

    return test_dataset
