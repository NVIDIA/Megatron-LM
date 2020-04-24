import itertools

import numpy as np
import spacy

from megatron import get_tokenizer
from megatron.data.bert_dataset import BertDataset, get_samples_mapping_
from megatron.data.dataset_utils import create_masked_lm_predictions, pad_and_convert_to_numpy

qa_nlp = spacy.load('en_core_web_lg')


class RealmDataset(BertDataset):
    """Dataset containing simple masked sentences for masked language modeling.

    The dataset should yield sentences just like the regular BertDataset
    However, this dataset also needs to be able to return a set of blocks
    given their start and end indices.

    Presumably

    """
    def __init__(self, name, indexed_dataset, data_prefix,
                 num_epochs, max_num_samples, masked_lm_prob,
                 max_seq_length, short_seq_prob, seed):
        super(RealmDataset, self).__init__(name, indexed_dataset, data_prefix,
                                           num_epochs, max_num_samples, masked_lm_prob,
                                           max_seq_length, short_seq_prob, seed)
        self.build_sample_fn = build_simple_training_sample


def build_simple_training_sample(sample, target_seq_length, max_seq_length,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 cls_id, sep_id, mask_id, pad_id,
                                 masked_lm_prob, np_rng):

    tokens = list(itertools.chain(*sample))[:max_seq_length - 2]
    tokens, tokentypes = create_single_tokens_and_tokentypes(tokens, cls_id, sep_id)

    max_predictions_per_seq = masked_lm_prob * max_seq_length
    (tokens, masked_positions, masked_labels, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    # REALM true sequence length is twice as long but none of that is to be predicted with LM
    loss_mask_np = np.concatenate((loss_mask_np, np.ones(loss_mask_np.shape)), -1)

    train_sample = {
        'tokens': tokens_np,
        'labels': labels_np,
        'loss_mask': loss_mask_np,
        'pad_mask': padding_mask_np
    }
    return train_sample


def create_single_tokens_and_tokentypes(_tokens, cls_id, sep_id):
    tokens = []
    tokens.append(cls_id)
    tokens.extend(list(_tokens))
    tokens.append(sep_id)
    tokentypes = [0] * len(tokens)
    return tokens, tokentypes


def spacy_ner(block_text):
    candidates = {}
    block = qa_nlp(block_text)
    starts = []
    answers = []
    for ent in block.ents:
        starts.append(int(ent.start_char))
        answers.append(str(ent.text))
    candidates['starts'] = starts
    candidates['answers'] = answers
