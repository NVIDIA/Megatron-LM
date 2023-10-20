# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

""" Tasks data utility."""

import re
import numpy as np


def clean_text(text):
    """Remove new lines and multiple spaces and adjust end of sentence dot."""

    text = text.replace("\n", " ")
    text = re.sub(r'\s+', ' ', text)
    for _ in range(3):
        text = text.replace(' . ', '. ')

    return text


def build_sample(ids, types, paddings, label, unique_id):
    """Convert to numpy and return a sample consumed by the batch producer."""

    ids_np = np.array(ids, dtype=np.int64)
    types_np = np.array(types, dtype=np.int64)
    paddings_np = np.array(paddings, dtype=np.int64)
    sample = ({'text': ids_np,
               'types': types_np,
               'padding_mask': paddings_np,
               'label': int(label),
               'uid': int(unique_id)})

    return sample


def build_tokens_types_paddings_from_text(text_a, text_b,
                                          tokenizer, max_seq_length):
    """Build token types and paddings, trim if needed, and pad if needed."""

    text_a_ids = tokenizer.tokenize(text_a)
    text_b_ids = None
    if text_b is not None:
        text_b_ids = tokenizer.tokenize(text_b)

    return build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids,
                                                max_seq_length, tokenizer.cls,
                                                tokenizer.sep, tokenizer.pad)


def build_tokens_types_paddings_from_ids(text_a_ids, text_b_ids, max_seq_length,
                                         cls_id, sep_id, pad_id):
    """Build token types and paddings, trim if needed, and pad if needed."""

    ids = []
    types = []
    paddings = []

    # [CLS].
    ids.append(cls_id)
    types.append(0)
    paddings.append(1)

    # A.
    len_text_a = len(text_a_ids)
    ids.extend(text_a_ids)
    types.extend([0] * len_text_a)
    paddings.extend([1] * len_text_a)

    # [SEP].
    ids.append(sep_id)
    types.append(0)
    paddings.append(1)

    # B.
    if text_b_ids is not None:
        len_text_b = len(text_b_ids)
        ids.extend(text_b_ids)
        types.extend([1] * len_text_b)
        paddings.extend([1] * len_text_b)

    # Cap the size.
    trimmed = False
    if len(ids) >= max_seq_length:
        max_seq_length_m1 = max_seq_length - 1
        ids = ids[0:max_seq_length_m1]
        types = types[0:max_seq_length_m1]
        paddings = paddings[0:max_seq_length_m1]
        trimmed = True

    # [SEP].
    if (text_b_ids is not None) or trimmed:
        ids.append(sep_id)
        if text_b_ids is None:
            types.append(0)
        else:
            types.append(1)
        paddings.append(1)

    # Padding.
    padding_length = max_seq_length - len(ids)
    if padding_length > 0:
        ids.extend([pad_id] * padding_length)
        types.extend([pad_id] * padding_length)
        paddings.extend([0] * padding_length)

    return ids, types, paddings
