# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

import numpy as np
import torch
from tqdm import tqdm

from .external_libs import transformers


class IterableTextDataset(torch.utils.data.IterableDataset):
    '''Iterable over a text dataset.'''

    def __init__(self, text_dataset):
        self.text_dataset = text_dataset

    def __iter__(self):
        '''Remove 'endoftext' string.'''
        for sample_idx in range(len(self.text_dataset)):
            sample = self.text_dataset[sample_idx]
            text = sample["text"].replace("<|endoftext|>", "")
            yield text


class MyFeatureExtractionPipeline(transformers.FeatureExtractionPipeline):
    def _forward(self, model_inputs):

        # Embed inputs.
        model_outputs = self.model(**model_inputs)

        # Attention mask.
        embeddings = model_outputs[0]
        masks = torch.sum(model_inputs['attention_mask'], dim=1)

        # Collect embeddings & check for nan.
        outputs = []
        for embedding, mask in zip(embeddings, masks):
            output = torch.mean(embedding[1: mask - 1], dim=0)

            # Nans due to empty input sequences; so only check first element.
            if torch.isnan(output.view(-1)[0]).any():
                output.zero_()

            outputs.append(output)

        # Sample.
        data = {
            "input" : model_inputs["input_ids"],
            "output" : outputs,
        }

        return data

    def postprocess(self, model_outputs):
        # Return input for analysis.
        return {
            "input" : model_outputs["input"].numpy(),
            "output" : model_outputs["output"].numpy(),
        }


class HuggingfaceEmbedder:

    def __init__(self, batch_size, max_seq_length):

        # Model, tokenizer.
        self.model = transformers.BertModel.from_pretrained("bert-large-cased")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            "bert-large-cased", model_max_length=max_seq_length)

        # Feature extraction pipeline.
        self.pipe = MyFeatureExtractionPipeline(
            model=self.model,
            tokenizer=self.tokenizer,
            device=torch.cuda.current_device(),
            truncation=True,
            max_length=max_seq_length,
        )

        self.batch_size = batch_size

    def embed_text_dataset(self, text_dataset, verbose=True):

        # Wrap dataset in iterable.
        dataset = IterableTextDataset(text_dataset)

        # Allocate output array.
        n_samples = len(text_dataset)
        embeddings = np.zeros((n_samples, 1024), dtype="f4")
        start_idx = 0

        # Wrap iterator in tqdm for verbose output.
        _iter = self.pipe(dataset, batch_size=self.batch_size)
        if verbose:
            _iter = tqdm(_iter, "hf embed", total=n_samples)

        # Embed dataset.
        for idx, out_dict in enumerate(_iter):
            inp = out_dict["input"]
            out = out_dict["output"]
            embeddings[start_idx] = out
            start_idx += 1

        return embeddings

    def embed_text(self, text):
        '''Embed a single text string.

        Primarily used for on-the-fly embeddings, particularly during
        analysis or debugging. For large scale, use 'embed_text_dataset()'.
        '''

        class SingleTextDataset(torch.utils.data.Dataset):
            '''Dataset that holds single string.'''
            def __init__(self, text):
                assert isinstance(text, str)
                self.text = text
            def __len__(self):
                return 1
            def __getitem__(self, i):
                return {"text": self.text}

        # Embed text.
        text_ds = SingleTextDataset(text)
        embed = self.embed_text_dataset(text_ds, verbose=False)[0]

        return embed
