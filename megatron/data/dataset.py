"""TO BE ADDED """

import random
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset_utils import build_training_sample


class AlbertDataSet(Dataset):

    def __init__(self, indexed_dataset, tokenizer, num_epochs,
                 masked_lm_prob, max_seq_length, short_seq_prob, seed):

        # Params to store.
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length

        # Indexed dataset.
        self.indexed_dataset = indexed_dataset

        # Build the samples mapping.
        self.samples_mapping = build_training_samples_mapping(
            indexed_dataset,
            num_epochs,
            self.max_seq_length,
            short_seq_prob,
            self.seed)

        # Vocab stuff.
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.vocab['[CLS]']
        self.sep_id = tokenizer.vocab['[SEP]']
        self.mask_id = tokenizer.vocab['[MASK]']
        self.pad_id = tokenizer.vocab['[PAD]']


    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        rng = random.Random(self.seed + idx)
        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length,
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, rng)


def get_target_seq_length(max_num_tokens, short_seq_prob, np_rng):
    """With probability `short_seq_prob` generate a smaller sequence lenght."""
    if np_rng.random() < short_seq_prob:
        return np_rng.randint(2, max_num_tokens + 1)
    return max_num_tokens


def build_training_samples_mapping(indexed_dataset, num_epochs, max_seq_length,
                                   short_seq_prob, seed):
    """Build a mapping to reconstruct training samples."""

    start_time = time.time()
    print('> building training samples mapping ...')

    # RNG:
    np_rng = np.random.RandomState(seed=seed)

    # List of start sentence index and end sentence index (end is exclusive)
    # to retrieve.
    samples = []

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # Number of documents processed:
    total_docs = 0
    # Number of documents that are skipped:
    skipped_docs = 0
    # Number of empty documents:
    empty_docs = 0

    # For each epoch:
    for epoch in range(num_epochs):
        # For each document:
        for doc_index in range(indexed_dataset.num_docs):
            if epoch == 0:
                total_docs += 1

            # Document sentences are in [sent_index_first, sent_index_last).
            sent_index_first = indexed_dataset.doc_idx[doc_index]
            sent_index_last = indexed_dataset.doc_idx[doc_index+1]
            assert sent_index_last >= sent_index_first

            # Empty docs.
            if (sent_index_last - sent_index_first) == 0:
                if epoch == 0:
                    print('***WARNING*** document {} is empty'.format(
                        doc_index))
                    empty_docs += 1
                continue
            # Skip documents that only have one sentences.
            if (sent_index_last - sent_index_first) == 1:
                if epoch == 0:
                    print('***WARNING*** document {} has only one sentnece, '
                          'skipping ...'.format(doc_index))
                    skipped_docs += 1
                continue

            # Loop through sentences.
            sent_index = sent_index_first
            target_seq_length = get_target_seq_length(max_num_tokens,
                                                      short_seq_prob, np_rng)
            size = 0
            while sent_index < sent_index_last:

                # Get the size.
                assert indexed_dataset.sizes[sent_index] > 0
                size += indexed_dataset.sizes[sent_index]
                sent_index += 1

                # If we have reached the target length.
                exceeded_target_size = (size >= target_seq_length)
                # If only one sentence is left in the document.
                only_one_sent_left = (sent_index == (sent_index_last - 1))
                # If we have at least two sentneces.
                have_more_than_one_sent = (sent_index - sent_index_first) > 1
                # If we have reached end of the document.
                reached_end_of_doc = (sent_index == sent_index_last)
                if (exceeded_target_size and not only_one_sent_left and
                    have_more_than_one_sent) or reached_end_of_doc:
                    assert (sent_index - sent_index_first) > 1
                    assert size > 1
                    # Add the sample.
                    samples.append([sent_index_first, sent_index,
                                    target_seq_length])
                    # Reset indices
                    sent_index_first = sent_index
                    target_seq_length = get_target_seq_length(max_num_tokens,
                                                              short_seq_prob,
                                                              np_rng)
                    size = 0
                    num_sentences = 0

    # Convert to numpy array.
    samples_np = np.array(samples, dtype=np.int64)
    # Shuffle.
    np_rng.shuffle(samples_np)
    elapsed_time = time.time() - start_time

    # Print some stats:
    print('\n***************************** info *****************************')
    print('   elapsed time (sec) ..................... {}'.format(elapsed_time))
    print('   number of epochs ....................... {}'.format(num_epochs))
    print('   number of samples ...................... {}'.format(
        samples_np.shape[0]))
    print('   number of documents .................... {}'.format(total_docs))
    print('   number of empty documents .............. {}'.format(empty_docs))
    print('   number of documents with one sentence .. {}'.format(skipped_docs))
    print('****************************************************************\n')

    return samples_np


# WILL BE REPLACED WITH JARED'S
class JaredDataset(object):

    def __init__(self, doc_idx, sizes, sentences):
        self.doc_idx = doc_idx
        self.num_docs = len(self.doc_idx) - 1
        self.sizes = sizes
        self.sentences = sentences

    def __getitem__(self, idx):
        return self.sentences[idx]



if __name__ == '__main__':

    print('dataset ...')

    from bert_tokenization import FullTokenizer
    import json
    import nltk
    nltk.download('punkt')

    def document_generator_provider(input_file):
        with open(input_file, 'r') as ifile:
            for document in ifile:
                data = json.loads(document)
                text = data['text']
                sentences = []
                for line in text.split('\n'):
                    if line != '\n':
                        sent = nltk.tokenize.sent_tokenize(line)
                        if sent:
                            sentences.extend(sent)
                yield sentences

    input_file = '/raid/mshoeybi/data/albert/sample/samples_1000.json'
    vocab_file = '/raid/mshoeybi/data/albert/bert_vocab/vocab.txt'

    tokenizer = FullTokenizer(vocab_file, do_lower_case=True)
    document_generator = document_generator_provider(input_file)

    doc_idx = [0]
    sizes = []
    sentences_list = []

    for sentences in document_generator:
        num_sent = 0
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            if tokens:
                ids = tokenizer.convert_tokens_to_ids(tokens)
                if len(ids) == 0:
                    print('****************')
                    print(sentence)
                    print(tokens)
                    print(ids)
                    print('****************')
                sizes.append(len(ids))
                sentences_list.append(ids)
                num_sent += 1
        doc_idx.append(num_sent)
    for i in range(1, len(doc_idx)):
        doc_idx[i] += doc_idx[i-1]

    indexed_dataset = JaredDataset(doc_idx, sizes, sentences_list)
    dataset = AlbertDataSet(indexed_dataset=indexed_dataset,
                            tokenizer=tokenizer,
                            num_epochs=10,
                            masked_lm_prob=0.15,
                            max_seq_length=512,
                            short_seq_prob=0.1,
                            seed=1234)
