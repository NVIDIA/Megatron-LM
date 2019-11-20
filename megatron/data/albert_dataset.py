"""TO BE ADDED """

import random
import time
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset_utils import build_training_sample
#from data.mapping import build_training_samples_mapping

from . import helpers
from megatron.data import FullBertTokenizer
from megatron.data.indexed_dataset import make_dataset as make_indexed_dataset
from megatron.utils import print_rank_0


class AlbertDataset(Dataset):


    def __init__(self,
                 vocab_file, data_prefix, data_impl, skip_warmup,
                 num_epochs, max_num_samples,
                 masked_lm_prob, max_seq_length, short_seq_prob, seed):

        # Params to store.
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=True)

        # Indexed dataset.
        self.indexed_dataset = self._get_indexed_dataset(data_prefix, data_impl,
                                                         skip_warmup)

        # Build the samples mapping.
        self.samples_mapping = self._get_samples_mapping(self.indexed_dataset,
                                                         data_prefix,
                                                         num_epochs,
                                                         max_num_samples,
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
        exit()


    def num_tokens(self):
        return self.tokenizer.vocab_size()


    def __len__(self):
        return self.samples_mapping.shape[0]


    def __getitem__(self, idx):

        rng = random.Random(self.seed + idx)
        start_index, end_index, seq_length = self.samples_mapping[idx]
        sample = []
        for index in range(start_index, end_index):
            sample.append(self.indexed_dataset[index])
        for s in sample:
            if len(s) > 1000:
                print(self.tokenizer.convert_ids_to_tokens(s))
        return build_training_sample(sample, seq_length,
                                     self.max_seq_length, # needed for padding
                                     self.vocab_id_list,
                                     self.vocab_id_to_token_dict,
                                     self.cls_id, self.sep_id,
                                     self.mask_id, self.pad_id,
                                     self.masked_lm_prob, rng)



    def _get_indexed_dataset(self, data_prefix, data_impl, skip_warmup):
        start_time = time.time()
        print_rank_0("> Reading dataset index ...")
        indexed_dataset = make_indexed_dataset(data_prefix,
                                               data_impl,
                                               skip_warmup)
        print_rank_0("> Finished creating indexed dataset in {:4f} "
                     "seconds".format(time.time() - start_time))
        return indexed_dataset


    def _get_samples_mapping(self,
                             indexed_dataset,
                             data_prefix,
                             num_epochs,
                             max_num_samples,
                             max_seq_length,
                             short_seq_prob,
                             seed):
        if not num_epochs:
            if not max_num_samples:
                raise ValueError("Need to specify either max_num_samples "
                                 "or num_epochs")
            num_epochs = np.iinfo(np.int32).max - 1
        if not max_num_samples:
            max_num_samples = np.iinfo(np.int64).max - 1

        # Filename of the index mapping
        indexmap_filename = data_prefix
        indexmap_filename += '_indexmap'
        indexmap_filename += '_{}ep'.format(num_epochs)
        indexmap_filename += '_{}mns'.format(max_num_samples)
        indexmap_filename += '_{}msl'.format(max_seq_length)
        indexmap_filename += '_{:0.2f}ssp'.format(short_seq_prob)
        indexmap_filename += '_{}s'.format(seed)
        indexmap_filename += '.npy'

        # Build the indexed mapping if not exist.
        if torch.distributed.get_rank() == 0 and \
           not os.path.isfile(indexmap_filename):
            print('WARNING: could not find index map file {}, building '
                  'the indices on rank 0 ...'.format(indexmap_filename))
            # Make sure the types match the helpers input types.
            assert indexed_dataset.doc_idx.dtype == np.int64
            assert indexed_dataset.sizes.dtype == np.int32

            # Build samples mapping
            verbose = torch.distributed.get_rank()==0
            start_time = time.time()
            samples_mapping = helpers.build_mapping(
                indexed_dataset.doc_idx,
                indexed_dataset.sizes,
                num_epochs,
                max_num_samples,
                max_seq_length-3, # account for added tokens
                short_seq_prob,
                seed,
                verbose)
            np.save(indexmap_filename, samples_mapping, allow_pickle=True)
            # Make sure all the ranks have built the mapping
            print_rank_0('> elasped time to build and save samples mapping '
                         '(seconds): {:4f}'.format(
                             time.time() - start_time))
        torch.distributed.barrier()

        # Load indexed dataset.
        print_rank_0('> loading indexed mapping from {}'.format(
            indexmap_filename))
        start_time = time.time()
        samples_mapping = np.load(indexmap_filename, allow_pickle=True)
        print_rank_0('  loaded indexed file in {:3.3f} seconds'.format(
            time.time() - start_time))
        print_rank_0('  total number of samples: {}'.format(
            samples_mapping.shape[0]))

        return samples_mapping


'''
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
'''

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

    input_file = 'test/samples_10000.json'
    vocab_file = 'test/vocab.txt'

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

    #max_size = np.iinfo(np.int32).max // 32

    import time

    docs_np = np.array(doc_idx, dtype=np.uint32)
    sizes_np = np.array(sizes, dtype=np.uint16)

    start_time = time.time()
    max_seq_length = 512
    max_size = docs_np.shape[0]
    lens = np.full(max_size, max_seq_length-3, dtype=np.uint16)
    lens_rand = np.random.randint(low=2, high=(max_seq_length-2),
                                  size=max_size//10, dtype=np.uint16)
    lens_view = lens[:max_size//10]
    np.copyto(lens_view, lens_rand)
    np.random.shuffle(lens)
    print('num docs', max_size)
    print('lens time', time.time() - start_time)

    import helpers
    start_time = time.time()
    maps = helpers.build_mapping(docs_np, sizes_np, 10, 100, 509, 0.1, 1234)
    print('maps time', time.time() - start_time)
    print(maps)
    exit()

    start_time = time.time()
    max_size = 10 #np.iinfo(np.int32).max 32
    docs = np.arange(10, dtype=np.uint32)
    print(docs)

    a = example.doit(docs, max_size)
    print(type(a))
    print(a.shape)
    print(a)
    print(time.time() - start_time)
    exit()


    #start_time = time.time()
    count = doit(maps, docs_np, sizes_np, lens,docs_np.shape[0]-1, 10)
    print(count)
    maps = maps[:count]
    np.random.shuffle(maps)
    print(time.time() - start_time)


    exit()

    indexed_dataset = JaredDataset(doc_idx, sizes, sentences_list)
    dataset = AlbertDataSet(indexed_dataset=indexed_dataset,
                            tokenizer=tokenizer,
                            num_epochs=10,
                            masked_lm_prob=0.15,
                            max_seq_length=512,
                            short_seq_prob=0.1,
                            seed=1234)
