import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from copy import copy
import os
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
import numpy as np
from functools import partial

class BlipImageBaseProcessor():
    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms.Normalize(mean, std)

class BlipImageEvalProcessor(BlipImageBaseProcessor):
    def __init__(self, image_size=384, mean=None, std=None):
        super().__init__(mean=mean, std=std)

        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (image_size, image_size), interpolation=InterpolationMode.BICUBIC
                ),
                transforms.ToTensor(),
                self.normalize,
            ]
        )

    def __call__(self, item):
        return self.transform(item)

def blip2_image_processor_func_megatron(image_seq_length, image_processor, image):
    return {'external_images': image_processor(image).unsqueeze(0), 'external_input_ids': torch.zeros(1, image_seq_length, dtype=torch.long), 'external_position_ids': torch.arange(image_seq_length, dtype=torch.long).unsqueeze(0)}

def _history_to_prompt(self, history, query, add_eoi_first=False):
    ret = []
    for i, (old_query, response) in enumerate(history):
        ret.append({"user": old_query, "assistant": response})
    ret.append({"user": query})
    return ret

def format_conversation(conversations, tokenizer, image_length, is_inference=False, is_text_only=False):
    # Note: `loss_mask` here means whether *the prediction* of the token should take loss
    tokens = (([0]*image_length) if not is_text_only else []) + [tokenizer.bos_token_id] # For simplicify, just insert image at the begining for now.
    loss_masks = [0] * len(tokens)
    def _update(_tokens, value):
        value = int(value)
        tokens.extend(_tokens)
        loss_masks.extend([value] * len(_tokens))
    context_length = len(tokens)
    for idx, conv in enumerate(conversations):
        no_training_tokens = []
        # prompt
        if idx == 0:
            no_training_tokens.extend(tokenizer.encode("[INST] <<SYS>>\nYou are a helpful assistant.\n<</SYS>>\n", add_special_tokens=False))
        no_training_tokens.extend(tokenizer.encode("{} [/INST]\n".format(conv["user"]), add_special_tokens=False))
        _update(no_training_tokens, 0)
        # context_length
        if idx == len(conversations) - 1:
            context_length = len(tokens)
        # answer
        if not (is_inference and idx == len(conversations) - 1):
            # update answer
            ans_tokens = tokenizer.encode(conv["assistant"], add_special_tokens=False)
            _update(ans_tokens, 1)
            _update([tokenizer.eos_token_id], 1)
            suffix_tokens = tokenizer.encode("\n", add_special_tokens=False)
            _update(suffix_tokens, 0)
    assert len(tokens) == len(loss_masks), f"length mismatch: {len(tokens)} vs {len(loss_masks)}"
    return tokens, loss_masks, context_length

from transformers import AutoTokenizer

def llama2_tokenizer(tokenizer_path):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return tokenizer

import re
import numpy as np

class llama2_text_processor:
    def __init__(self, tokenizer, max_target_length=1024, image_length=256, model=None):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.image_length = image_length
        self.model = model

    def __call__(self, caption, prompt="", history=[], is_text_only=False):
        if is_text_only:
            cut = 5
        else:
            cut = self.image_length + 5
        if len(prompt) > self.max_target_length - cut:
            prompt = prompt[:self.max_target_length - cut]
        ret = self.history_to_prompt(history, prompt)
        ret[-1].update({"assistant": caption})
        input_ids, loss_masks, context_length = format_conversation(ret, self.tokenizer, self.image_length, is_text_only=is_text_only)

        if context_length >= self.max_target_length - 5:
            return None
        elif len(input_ids) > self.max_target_length:
            input_ids = input_ids[:self.max_target_length]
            loss_masks = loss_masks[:self.max_target_length]

        attention_mask = [1] * len(input_ids)
        pad_len = self.max_target_length - len(input_ids)
        input_ids = input_ids + [0] * pad_len
        loss_masks = loss_masks + [0] * pad_len
        labels = input_ids[1:] + [0]
        loss_masks = loss_masks[1:] + [0] # !!!!!
        attention_mask = attention_mask + [1] * pad_len # no need to pad for mask
        np_mask = np.tril(np.expand_dims(np.array(attention_mask), 0).repeat(len(attention_mask), 0))
        input_ids = torch.tensor(input_ids)
        loss_masks = torch.tensor(loss_masks)
        labels = torch.tensor(labels)
        attention_mask = torch.from_numpy(np_mask).unsqueeze(0)
        position_ids = torch.arange(input_ids.shape[-1])

        return {'tokens': input_ids, 'labels': labels, 'loss_mask': loss_masks.float(), 'attention_mask': attention_mask < 0.5, 'position_ids': position_ids}

    def history_to_prompt(self, history, query):
        return _history_to_prompt(self, history, query)

class ImageJsonDataset(Dataset):
    def __init__(self, json_path, single_process_fn=None):
        """
        Initializes the ImageJsonlDataset.

        Args:
        jsonl_path (str): Path to the JSONL file where each line is a JSON object.
        single_process_fn (callable, optional): A callable function to process data items in __getitem__.
        """
        self.json_path = json_path
        self.image_path = '/'.join(json_path.split('/')[:-1])
        self.single_process_fn = single_process_fn
        
        # Load JSONL file and store each line
        self.data = []
        with open(json_path, 'r') as file:
            self.data = json.load(file)
    
    def __len__(self):
        """
        Returns the number of entries in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset by index.

        Args:
        idx (int): Index of the data item in the dataset.

        Returns:
        Processed data item, which includes the image as a PIL object and other key-value pairs.
        """
        item = copy(self.data[idx])
        # Load the image
        image_path = item.get('image')
        try:
            image = Image.open(os.path.join(self.image_path, "images", image_path))
            item['image'] = image
        except IOError:
            print(f"Warning: Failed to open image at {image_path}. Returning None for 'image'.")
            item['image'] = None  # Indicate failure to load image
        
        # Apply the processing function if specified
        if self.single_process_fn:
            item = self.single_process_fn(item)
        
        return item
