
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import os
from tqdm import tqdm
import logging
logger = logging.getLogger()
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

label_set = ["O", "B-ENTITY", "I-ENTITY"]

def read_ner(tokenizer, datapath):
    inputs, labels = [], []
    with open(datapath, "r") as fr:
        token_list, label_list = [], []
        for i, line in enumerate(fr):
            line = line.strip()
            if line == "":
                if len(token_list) > 0:
                    assert len(token_list) == len(label_list)
                    inputs.append([tokenizer.cls_token_id] + token_list + [tokenizer.sep_token_id])
                    labels.append([pad_token_label_id] + label_list + [pad_token_label_id])
                
                token_list, label_list = [], []
                continue
            
            splits = line.split("\t")
            token = splits[0]
            label = splits[1]
            if label.startswith("B-"):
                label = "B-ENTITY"
            elif label.startswith("I-"):
                label = "I-ENTITY"

            subs_ = tokenizer.tokenize(token)
            if len(subs_) > 0:
                label_list.extend([label_set.index(label)] + [pad_token_label_id] * (len(subs_) - 1))
                token_list.extend(tokenizer.convert_tokens_to_ids(subs_))
            else:
                print("length of subwords for %s is zero; its label is %s" % (token, label))

    return inputs, labels

class Dataset(data.Dataset):
    def __init__(self, tokenizer, inputs, labels):
        self.X = inputs
        self.y = labels
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return len(self.X)

    def collate_fn(self, data):
        X, y = zip(*data)
        lengths = [len(bs_x) for bs_x in X]
        max_lengths = max(lengths)
        padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(self.tokenizer.pad_token_id)
        padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
        for i, (seq, y_) in enumerate(zip(X, y)):
            length = lengths[i]
            padded_seqs[i, :length] = torch.LongTensor(seq)
            padded_y[i, :length] = torch.LongTensor(y_)

        return padded_seqs, padded_y

def get_dataloader(model_name, batch_size, data_folder):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    inputs_train, labels_train = read_ner(tokenizer, os.path.join(data_folder, "train.txt"))
    inputs_dev, labels_dev = read_ner(tokenizer, os.path.join(data_folder, "dev.txt"))
    inputs_test, labels_test = read_ner(tokenizer, os.path.join(data_folder, "test.txt"))

    logger.info("conll2003 dataset: train size: %d; dev size %d; test size: %d" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

    dataset_train = Dataset(tokenizer, inputs_train, labels_train)
    dataset_dev = Dataset(tokenizer, inputs_dev, labels_dev)
    dataset_test = Dataset(tokenizer, inputs_test, labels_test)
    
    dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=dataset_train.collate_fn)
    dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=dataset_dev.collate_fn)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=dataset_test.collate_fn)

    return dataloader_train, dataloader_dev, dataloader_test

