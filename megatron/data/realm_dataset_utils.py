import itertools
import os
import random
import time

import numpy as np
import spacy
import torch
try:
    import stanza
    processors_dict = {'tokenize': 'default', 'mwt': 'default', 'ner': 'conll03'}
    stanza_pipeline = stanza.Pipeline('en', processors=processors_dict, use_gpu=True)
except:
    pass

from megatron.data.dataset_utils import create_masked_lm_predictions, pad_and_convert_to_numpy
from megatron import get_args, get_tokenizer, print_rank_0, mpu

SPACY_NER = spacy.load('en_core_web_lg')


def build_realm_training_sample(sample, max_seq_length,
                                vocab_id_list, vocab_id_to_token_dict,
                                cls_id, sep_id, mask_id, pad_id,
                                masked_lm_prob, block_ner_mask, cased_tokens,
                                cased_tokenizer, np_rng):
    tokens = list(itertools.chain(*sample))[:max_seq_length - 2]
    tokens, tokentypes = create_single_tokens_and_tokentypes(tokens, cls_id, sep_id)

    args = get_args()
    if args.use_regular_masking:
        max_predictions_per_seq = masked_lm_prob * max_seq_length
        masked_tokens, masked_positions, masked_labels, _ = create_masked_lm_predictions(
            tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
            cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)
    elif block_ner_mask is not None:
        block_ner_mask = list(itertools.chain(*block_ner_mask))[:max_seq_length - 2]
        if args.use_random_spans:
            rand_idx = np.random.randint(len(block_ner_mask))
            block_ner_mask = block_ner_mask[rand_idx:] + block_ner_mask[:rand_idx]
        block_ner_mask = [0] + block_ner_mask + [0]
        masked_tokens, masked_positions, masked_labels = get_arrays_using_ner_mask(tokens, block_ner_mask, mask_id)
    else:
        try:
            if args.cased_data_path is not None:
                total_len = sum(len(l) for l in sample)
                # truncate the last sentence to make it so that the whole thing has length max_seq_length - 2
                if total_len > max_seq_length - 2:
                    offset = -(total_len - (max_seq_length - 2))
                    sample[-1] = sample[-1][:offset]
                masked_tokens, masked_positions, masked_labels = get_stanza_ner_mask(sample, cased_tokens, cased_tokenizer,
                                                                                     cls_id, sep_id, mask_id)
            else:
                masked_tokens, masked_positions, masked_labels = salient_span_mask(tokens, mask_id)
        except:
            # print("+" * 100, flush=True)
            # print('could not create salient span', flush=True)
            # print("+" * 100, flush=True)
            # this means the above returned None, and None isn't iterable.
            # TODO: consider coding style.
            max_predictions_per_seq = masked_lm_prob * max_seq_length
            masked_tokens, masked_positions, masked_labels, _ = create_masked_lm_predictions(
                tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
                cls_id, sep_id, mask_id, max_predictions_per_seq, np_rng)

    tokens_np, tokentypes_np, labels_np, padding_mask_np, loss_mask_np \
        = pad_and_convert_to_numpy(masked_tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    train_sample = {
        'tokens': tokens_np,
        'labels': labels_np,
        'loss_mask': loss_mask_np,
        'pad_mask': padding_mask_np
    }
    return train_sample


def get_stanza_ner_mask(tokens, cased_tokens, cased_tokenizer, cls_id, sep_id, mask_id):
    """Use stanza to generate NER salient span masks in the loop"""
    # assuming that the default tokenizer is uncased.
    uncased_tokenizer = get_tokenizer()
    block_ner_mask = []

    for cased_sent_ids, uncased_sent_ids in zip(cased_tokens, tokens):
        # print('>')
        token_pos_map = id_to_str_pos_map(uncased_sent_ids, uncased_tokenizer)

        # get the cased string and do NER with both toolkits
        cased_sent_str = join_str_list(cased_tokenizer.tokenizer.convert_ids_to_tokens(cased_sent_ids))
        entities = stanza_pipeline(cased_sent_str).ents
        spacy_entities = SPACY_NER(cased_sent_str).ents

        # CoNLL doesn't do dates, so we scan with spacy to get the dates.
        entities = [e for e in entities if e.text != 'CLS']
        entities.extend([e for e in spacy_entities if (e.text != 'CLS' and e.label_ == 'DATE')])

        # randomize which entities to look at, and set a target of 12% of tokens being masked
        entity_indices = np.arange(len(entities))
        np.random.shuffle(entity_indices)
        target_num_masks = int(len(cased_sent_ids) * 0.12)

        masked_positions = []
        for entity_idx in entity_indices[:3]:

            # if we have enough masks then break.
            if len(masked_positions) > target_num_masks:
                break

            selected_entity = entities[entity_idx]
            # print(">> selected entity: {}".format(selected_entity.text), flush=True)

            mask_start = mask_end = 0
            set_mask_start = False
            # loop for checking where mask should start and end.
            while mask_end < len(token_pos_map) and token_pos_map[mask_end] < selected_entity.end_char:
                if token_pos_map[mask_start] > selected_entity.start_char:
                    set_mask_start = True
                if not set_mask_start:
                    mask_start += 1
                mask_end += 1

            # add offset to indices since our input was list of sentences
            masked_positions.extend(range(mask_start - 1, mask_end))

        ner_mask = [0] * len(uncased_sent_ids)
        for pos in masked_positions:
            ner_mask[pos] = 1
        block_ner_mask.extend(ner_mask)

    # len_tokens = [len(l) for l in tokens]
    # print(len_tokens, flush=True)
    # print([sum(len_tokens[:i + 1]) for i in range(len(tokens))], flush=True)
    tokens = list(itertools.chain(*tokens))
    tokens = [cls_id] + tokens + [sep_id]
    block_ner_mask = [0] + block_ner_mask + [0]
    return get_arrays_using_ner_mask(tokens, block_ner_mask, mask_id)


def get_arrays_using_ner_mask(tokens, block_ner_mask, mask_id):
    tokenizer = get_tokenizer()
    tokens_str = join_str_list(tokenizer.tokenizer.convert_ids_to_tokens(tokens))

    masked_tokens = tokens.copy()
    masked_positions = []
    masked_labels = []

    for i in range(len(tokens)):
        if block_ner_mask[i] == 1:
            masked_positions.append(i)
            masked_labels.append(tokens[i])
            masked_tokens[i] = mask_id

    # print("\nTOKEN STR\n", tokens_str + '\n',
    #     "OUTPUT\n", join_str_list(tokenizer.tokenizer.convert_ids_to_tokens(masked_tokens)) + '\n',
    #     "FRAC_MASKED: {}\n".format(len(masked_labels) / len(tokens)),
    #     "-" * 100 + '\n',
    #     flush=True)

    return masked_tokens, masked_positions, masked_labels


def create_single_tokens_and_tokentypes(_tokens, cls_id, sep_id):
    tokens = []
    tokens.append(cls_id)
    tokens.extend(list(_tokens))
    tokens.append(sep_id)
    tokentypes = [0] * len(tokens)
    return tokens, tokentypes


def join_str_list(str_list):
    """Join a list of strings, handling spaces appropriately"""
    result = ""
    for s in str_list:
        if s.startswith("##"):
            result += s[2:]
        else:
            result += " " + s
    return result


def id_to_str_pos_map(token_ids, tokenizer):
    """Given a list of ids, return a list of integers which correspond to the starting index
    of the corresponding token in the original string (with spaces, without artifacts e.g. ##)"""
    token_strs = tokenizer.tokenizer.convert_ids_to_tokens(token_ids)
    pos_map = [0]
    for i in range(len(token_strs) - 1):
        len_prev = len(token_strs[i])
        # do not add the length of the "##"
        if token_strs[i].startswith("##"):
            len_prev -= 2

        # add the length of the space if needed
        if token_strs[i + 1].startswith("##"):
            pos_map.append(pos_map[-1] + len_prev)
        else:
            pos_map.append(pos_map[-1] + len_prev + 1)

    # make sure total size is correct
    offset = -2 if token_strs[-1].startswith("##") else 0
    total_len = pos_map[-1] + len(token_strs[-1]) + offset
    assert total_len == len(join_str_list(token_strs)) - 1, (total_len, len(join_str_list(token_strs)))

    return pos_map


def salient_span_mask(tokens, mask_id):
    """Creates the predictions for the masked LM objective.
    Note: Tokens here are vocab ids and not text tokens."""
    tokenizer = get_tokenizer()
    tokens_str = join_str_list(tokenizer.tokenizer.convert_ids_to_tokens(tokens))

    # need to get all named entities
    entities = SPACY_NER(tokens_str).ents
    undesired_types = ['CARDINAL', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL']
    entities = [e for e in entities if e.text != "CLS" and e.label_ not in undesired_types]
    if len(entities) == 0:
        return None
    entity_idx = np.random.randint(0, len(entities))
    selected_entity = entities[entity_idx]

    token_pos_map = id_to_str_pos_map(tokens, tokenizer)
    mask_start = mask_end = 0
    set_mask_start = False
    while mask_end < len(token_pos_map) and token_pos_map[mask_end] < selected_entity.end_char:
        if token_pos_map[mask_start] > selected_entity.start_char:
            set_mask_start = True
        if not set_mask_start:
            mask_start += 1
        mask_end += 1
    masked_positions = list(range(mask_start - 1, mask_end))

    labels = []
    output_tokens = tokens.copy()
    for id_idx in masked_positions:
        labels.append(tokens[id_idx])
        output_tokens[id_idx] = mask_id
    # print("-" * 100 + '\n',
    #       "TOKEN STR\n", tokens_str + '\n',
    #       "SELECTED ENTITY\n", selected_entity.text + '\n',
    #       "OUTPUT\n", join_str_list(tokenizer.tokenizer.convert_ids_to_tokens(output_tokens)), flush=True)

    return output_tokens, masked_positions, labels


def get_block_samples_mapping(block_dataset, title_dataset, data_prefix, num_epochs,
                              max_num_samples, max_seq_length, seed, name):
    if not num_epochs:
        if not max_num_samples:
            raise ValueError("Need to specify either max_num_samples "
                             "or num_epochs")
        num_epochs = np.iinfo(np.int32).max - 1
    if not max_num_samples:
        max_num_samples = np.iinfo(np.int64).max - 1

    # Filename of the index mapping
    indexmap_filename = data_prefix
    indexmap_filename += '_{}_indexmap'.format(name)
    if num_epochs != (np.iinfo(np.int32).max - 1):
        indexmap_filename += '_{}ep'.format(num_epochs)
    if max_num_samples != (np.iinfo(np.int64).max - 1):
        indexmap_filename += '_{}mns'.format(max_num_samples)
    indexmap_filename += '_{}msl'.format(max_seq_length)
    indexmap_filename += '_{}s'.format(seed)
    indexmap_filename += '.npy'

    # Build the indexed mapping if not exist.
    if mpu.get_data_parallel_rank() == 0 and \
            not os.path.isfile(indexmap_filename):
        print(' > WARNING: could not find index map file {}, building '
              'the indices on rank 0 ...'.format(indexmap_filename))

        # Make sure the types match the helpers input types.
        assert block_dataset.doc_idx.dtype == np.int64
        assert block_dataset.sizes.dtype == np.int32

        # Build samples mapping
        verbose = torch.distributed.get_rank() == 0
        start_time = time.time()
        print_rank_0(' > building samples index mapping for {} ...'.format(
            name))
        from megatron.data.dataset_utils import compile_helper
        compile_helper()
        from megatron.data import helpers
        samples_mapping = helpers.build_blocks_mapping(
            block_dataset.doc_idx,
            block_dataset.sizes,
            title_dataset.sizes,
            num_epochs,
            max_num_samples,
            max_seq_length-3,  # account for added tokens
            seed,
            verbose)
        print_rank_0(' > done building samples index mapping')
        np.save(indexmap_filename, samples_mapping, allow_pickle=True)
        print_rank_0(' > saved the index mapping in {}'.format(
            indexmap_filename))
        # Make sure all the ranks have built the mapping
        print_rank_0(' > elapsed time to build and save samples mapping '
                     '(seconds): {:4f}'.format(
            time.time() - start_time))
    # This should be a barrier but nccl barrier assumes
    # device_index=rank which is not the case for model
    # parallel case
    counts = torch.cuda.LongTensor([1])
    torch.distributed.all_reduce(counts, group=mpu.get_data_parallel_group())
    assert counts[0].item() == torch.distributed.get_world_size(
        group=mpu.get_data_parallel_group())

    # Load indexed dataset.
    print_rank_0(' > loading indexed mapping from {}'.format(
        indexmap_filename))
    start_time = time.time()
    samples_mapping = np.load(indexmap_filename, allow_pickle=True)
    print_rank_0('    loaded indexed file in {:3.3f} seconds'.format(
        time.time() - start_time))
    print_rank_0('    total number of samples: {}'.format(
        samples_mapping.shape[0]))

    return samples_mapping
