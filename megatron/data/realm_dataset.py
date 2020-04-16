import numpy as np
from torch.utils.data import Dataset

from megatron import get_tokenizer
from megatron.data.bert_dataset import get_samples_mapping_
from megatron.data.dataset_utils import build_simple_training_sample


class RealmDataset(Dataset):
    """Dataset containing sentences and their blocks for an inverse cloze task."""
    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed):

        # Params to store.
        self.name = name
        self.seed = seed
        self.masked_lm_prob = masked_lm_prob
        self.max_seq_length = max_seq_length

        # Dataset.
        self.indexed_dataset = indexed_dataset


        # Build the samples mapping.
        self.samples_mapping = get_samples_mapping_(self.indexed_dataset,
                                                    data_prefix,
                                                    num_epochs,
                                                    max_num_samples,
                                                    self.max_seq_length,
                                                    short_seq_prob,
                                                    self.seed,
                                                    self.name)

        # Vocab stuff.
        tokenizer = get_tokenizer()
        self.vocab_id_list = list(tokenizer.inv_vocab.keys())
        self.vocab_id_to_token_dict = tokenizer.inv_vocab
        self.cls_id = tokenizer.cls
        self.sep_id = tokenizer.sep
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad

    def __len__(self):
        return self.samples_mapping.shape[0]

    def __getitem__(self, idx):
        start_idx, end_idx, seq_length = self.samples_mapping[idx]
        sample = [self.indexed_dataset[i] for i in range(start_idx, end_idx)]
        # Note that this rng state should be numpy and not python since
        # python randint is inclusive whereas the numpy one is exclusive.
        np_rng = np.random.RandomState(seed=(self.seed + idx))
        return build_simple_training_sample(sample, seq_length,
                                            self.max_seq_length,  # needed for padding
                                            self.vocab_id_list,
                                            self.vocab_id_to_token_dict,
                                            self.cls_id, self.sep_id,
                                            self.mask_id, self.pad_id,
                                            self.masked_lm_prob, np_rng)

