"""TO BE ADDED"""


import collections
import numpy as np


def build_training_sample(sample, vocab_id_list, vocab_id_to_token_dict,
                          cls_id, sep_id, mask_id, pad_id,
                          masked_lm_prob, max_seq_length, rng):
    """Biuld training sample.

    Arguments:
        sample: A list of sentences in which each sentence is a list token ids.
        vocab_id_list: List of vocabulary ids. Used to pick a random id.
        vocab_id_to_token_dict: A dictionary from vocab ids to text tokens.
        cls_id: Start of example id.
        sep_id: Separator id.
        mask_id: Mask token id.
        pad_id: Padding token id.
        masked_lm_prob: Probability to mask tokens.
        max_seq_length: Maximum length of the sequence. All values are padded to
            this length.
        rng: Random number genenrator.
    """

    # We assume that we have at least two sentences in the sample
    assert len(sample) > 1

    # Divide sample into two segments (A and B).
    tokens_a, tokens_b, is_next_random = get_a_and_b_segments(sample, rng)

    # Truncate to `max_sequence_length`.
    # Note that we have account for [CLS] A [SEP] B [SEP]
    max_num_tokens = max_seq_length - 3
    truncate_segments(tokens_a, tokens_b, len(tokens_a), len(tokens_b),
                      max_num_tokens, rng)

    # Build tokens and toketypes.
    tokens, tokentypes = create_tokens_and_tokentypes(tokens_a, tokens_b,
                                                      cls_id, sep_id)

    # Masking.
    max_predictions_per_seq = masked_lm_prob * max_num_tokens
    (tokens, masked_positions, masked_labels, _) = create_masked_lm_predictions(
        tokens, vocab_id_list, vocab_id_to_token_dict, masked_lm_prob,
        cls_id, sep_id, mask_id, max_predictions_per_seq)

    # Padding.
    tokens_np, tokentypes_np, labels, padding_mask, loss_mask \
        = pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                                   masked_labels, pad_id, max_seq_length)

    train_sample = {
        'text': tokens_np,
        'types': tokentypes_np,
        'labels': labels,
        'is_random': int(is_next_random),
        'loss_mask': loss_mask,
        'padding_mask': padding_mask}
    return train_sample


def get_a_and_b_segments(sample, rng):
    """Divide sample into a and b segments."""

    # Number of sentences in the sample.
    n_sentences = len(sample)
    # Make sure we always have two sentences.
    assert n_sentences > 1, 'make sure each sample has at least two sentences.'

    # First part:
    # `a_end` is how many sentences go into the `A`.
    a_end = 1
    if n_sentences >= 3:
        # Note that randin in python is inclusive.
        a_end = rng.randint(1, n_sentences - 1)
    tokens_a = []
    for j in range(a_end):
        tokens_a.extend(sample[j])

    # Second part:
    tokens_b = []
    for j in range(a_end, n_sentences):
        tokens_b.extend(sample[j])

    # Random next:
    is_next_random = False
    if rng.random() < 0.5:
        is_next_random = True
        tokens_a, tokens_b = tokens_b, tokens_a

    return tokens_a, tokens_b, is_next_random


def truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    assert len_a > 0
    assert len_b > 0
    if (len_a + len_b) <= max_num_tokens:
        return
    else:
        if len_a > len_b:
            len_a -= 1
            tokens = tokens_a
        else:
            len_b -= 1
            tokens = tokens_b
        if rng.random() < 0.5:
            del tokens[0]
        else:
            tokens.pop()
        truncate_segments(tokens_a, tokens_b, len_a, len_b, max_num_tokens, rng)


def create_tokens_and_tokentypes(tokens_a, tokens_b, cls_id, sep_id):
    """Merge segments A and B, add [CLS] and [SEP] and build tokentypes."""

    tokens = []
    tokentypes = []
    # [CLS].
    tokens.append(cls_id)
    tokentypes.append(0)
    # Segment A.
    for token in tokens_a:
        tokens.append(token)
        tokentypes.append(0)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(0)
    # Segment B.
    for token in tokens_b:
        tokens.append(token)
        tokentypes.append(1)
    # [SEP].
    tokens.append(sep_id)
    tokentypes.append(1)

    return tokens, tokentypes


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def is_start_piece(piece):
  """Check if the current word piece is the starting piece (BERT)."""
  # When a word has been split into
  # WordPieces, the first token does not have any marker and any subsequence
  # tokens are prefixed with ##. So whenever we see the ## token, we
  # append it to the previous set of word indexes.
  return not piece.startswith("##")


def create_masked_lm_predictions(tokens,
                                 vocab_id_list, vocab_id_to_token_dict,
                                 masked_lm_prob,
                                 cls_id, sep_id, mask_id,
                                 max_predictions_per_seq,
                                 max_ngrams=3,
                                 do_whole_word_mask=True,
                                 favor_longer_ngram=False,
                                 do_permutation=False):
  """Creates the predictions for the masked LM objective.
  Note: Tokens here are vocab ids and not text tokens."""

  cand_indexes = []
  # Note(mingdachen): We create a list for recording if the piece is
  # the starting piece of current token, where 1 means true, so that
  # on-the-fly whole word masking is possible.
  token_boundary = [0] * len(tokens)

  for (i, token) in enumerate(tokens):
    if token == cls_id or token == sep_id:
      token_boundary[i] = 1
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (do_whole_word_mask and len(cand_indexes) >= 1 and
        not is_start_piece(vocab_id_to_token_dict[token])):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])
      if is_start_piece(vocab_id_to_token_dict[token]):
        token_boundary[i] = 1

  output_tokens = list(tokens)

  masked_lm_positions = []
  masked_lm_labels = []

  if masked_lm_prob == 0:
    return (output_tokens, masked_lm_positions,
            masked_lm_labels, token_boundary)

  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  # Note(mingdachen):
  # By default, we set the probilities to favor shorter ngram sequences.
  ngrams = np.arange(1, max_ngrams + 1, dtype=np.int64)
  pvals = 1. / np.arange(1, max_ngrams + 1)
  pvals /= pvals.sum(keepdims=True)

  if favor_longer_ngram:
    pvals = pvals[::-1]

  ngram_indexes = []
  for idx in range(len(cand_indexes)):
    ngram_index = []
    for n in ngrams:
      ngram_index.append(cand_indexes[idx:idx+n])
    ngram_indexes.append(ngram_index)

  rng.shuffle(ngram_indexes)

  masked_lms = []
  covered_indexes = set()
  for cand_index_set in ngram_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    if not cand_index_set:
      continue
    # Note(mingdachen):
    # Skip current piece if they are covered in lm masking or previous ngrams.
    for index_set in cand_index_set[0]:
      for index in index_set:
        if index in covered_indexes:
          continue

    n = np.random.choice(ngrams[:len(cand_index_set)],
                         p=pvals[:len(cand_index_set)] /
                         pvals[:len(cand_index_set)].sum(keepdims=True))
    index_set = sum(cand_index_set[n - 1], [])
    n -= 1
    # Note(mingdachen):
    # Repeatedly looking for a candidate that does not exceed the
    # maximum number of predictions by trying shorter ngrams.
    while len(masked_lms) + len(index_set) > num_to_predict:
      if n == 0:
        break
      index_set = sum(cand_index_set[n - 1], [])
      n -= 1
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = mask_id
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_id_list[rng.randint(0, len(vocab_id_list) - 1)]

      output_tokens[index] = masked_token

      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict

  rng.shuffle(ngram_indexes)

  select_indexes = set()
  if do_permutation:
    for cand_index_set in ngram_indexes:
      if len(select_indexes) >= num_to_predict:
        break
      if not cand_index_set:
        continue
      # Note(mingdachen):
      # Skip current piece if they are covered in lm masking or previous ngrams.
      for index_set in cand_index_set[0]:
        for index in index_set:
          if index in covered_indexes or index in select_indexes:
            continue

      n = np.random.choice(ngrams[:len(cand_index_set)],
                           p=pvals[:len(cand_index_set)] /
                           pvals[:len(cand_index_set)].sum(keepdims=True))
      index_set = sum(cand_index_set[n - 1], [])
      n -= 1

      while len(select_indexes) + len(index_set) > num_to_predict:
        if n == 0:
          break
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
      # If adding a whole-word mask would exceed the maximum number of
      # predictions, then just skip this candidate.
      if len(select_indexes) + len(index_set) > num_to_predict:
        continue
      is_any_index_covered = False
      for index in index_set:
        if index in covered_indexes or index in select_indexes:
          is_any_index_covered = True
          break
      if is_any_index_covered:
        continue
      for index in index_set:
        select_indexes.add(index)
    assert len(select_indexes) <= num_to_predict

    select_indexes = sorted(select_indexes)
    permute_indexes = list(select_indexes)
    rng.shuffle(permute_indexes)
    orig_token = list(output_tokens)

    for src_i, tgt_i in zip(select_indexes, permute_indexes):
      output_tokens[src_i] = orig_token[tgt_i]
      masked_lms.append(MaskedLmInstance(index=src_i, label=orig_token[src_i]))

  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)
  return (output_tokens, masked_lm_positions, masked_lm_labels, token_boundary)


def pad_and_convert_to_numpy(tokens, tokentypes, masked_positions,
                             masked_labels, pad_id, max_seq_length):
    """Pad sequences and convert them to numpy."""

    # Some checks.
    num_tokens = len(tokens)
    padding_length = max_seq_length - num_tokens
    assert padding_length >= 0
    assert len(tokentypes) == num_tokens
    assert len(masked_positions) == len(masked_labels)

    # Tokens and token types.
    filler = [pad_id]*padding_length
    tokens_np = np.array(tokens + filler, dtype=np.int64)
    tokentypes_np = np.array(tokentypes + filler, dtype=np.int64)

    # Padding mask.
    padding_mask = np.array([1]*num_tokens + [0]*padding_length, dtype=np.int64)

    # Lables and loss mask.
    labels = [-1] * max_seq_length
    loss_mask = [0] * max_seq_length
    for i in range(len(masked_positions)):
        assert masked_positions[i] < num_tokens
        labels[masked_positions[i]] = masked_labels[i]
        loss_mask[masked_positions[i]] = 1
    labels_np = np.array(labels, dtype=np.int64)
    loss_mask_np = np.array(loss_mask, dtype=np.int64)

    return tokens_np, tokentypes_np, labels, padding_mask, loss_mask




if __name__ == '__main__':


    print('building the dataset ...')

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
                        sentences.extend(nltk.tokenize.sent_tokenize(line))
                yield sentences

    input_file = '/raid/mshoeybi/data/albert/sample/samples_11.json'
    vocab_file = '/raid/mshoeybi/data/albert/bert_vocab/vocab.txt'

    tokenizer = FullTokenizer(vocab_file, do_lower_case=True)

    document_generator = document_generator_provider(input_file)
    samples = []
    sizes = []
    for sentences in document_generator:
        tokens_list = []
        size = 0
        for sentence in sentences:
            tokens = tokenizer.tokenize(sentence)
            tokens_list.append(tokens)
            size += len(tokens)
        samples.append(tokens_list)
        sizes.append(size)
    print(sizes)

    import random
    rng = random.Random(123567)
    vocab_id_list = list(tokenizer.inv_vocab.keys())
    cls_id = tokenizer.vocab['[CLS]']
    sep_id = tokenizer.vocab['[SEP]']
    mask_id = tokenizer.vocab['[MASK]']
    pad_id = tokenizer.vocab['[PAD]']
    vocab_id_to_token_dict = tokenizer.inv_vocab
    sample = []
    for s in samples[0]:
        sample.append(tokenizer.convert_tokens_to_ids(s))
    max_seq_length = 512
    masked_lm_prob = 0.15
    example = build_training_sample(sample,
                                    vocab_id_list, vocab_id_to_token_dict,
                                    cls_id, sep_id, mask_id, pad_id,
                                    masked_lm_prob, max_seq_length, rng)

    orig_tokens = []
    for s in samples[0]:
        orig_tokens.extend(s)
    is_random = example['is_random']
    if is_random:
        print('random')
    else:
        print('not-random')
    #exit()
    ii = 0
    for i in range(max_seq_length):
        token = tokenizer.inv_vocab[example['text'][i]]
        if token in ['[CLS]', '[SEP]'] :
            orig_token = token
        elif ii < len(orig_tokens):
            orig_token = orig_tokens[ii]
            ii += 1
        else:
            orig_token = 'EMPTY'
        tokentype = example['types'][i]
        label_id = example['labels'][i]
        label = 'NONE'
        if label_id >= 0:
            label = tokenizer.inv_vocab[label_id]
        loss_mask = example['loss_mask'][i]
        padding_mask = example['padding_mask'][i]

        string = ''
        string += '{:15s}'.format(orig_token)
        string += '{:15s}'.format(token)
        string += '{:15s}'.format(label)
        string += '{:5d}'.format(loss_mask)
        string += '{:5d}'.format(tokentype)
        string += '{:5d}'.format(padding_mask)
        print(string)

