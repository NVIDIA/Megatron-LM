# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer
import sentencepiece
import tokenizers
import youtokentome as yttm


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    assert args.vocab_file is not None
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'YTTMTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _YTTMTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'ByteLevelBPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _ByteLevelBPETokenizer(args.vocab_file, args.merge_file, vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size.
    args.padded_vocab_size = _vocab_size_with_padding(tokenizer.vocab_size,
                                                      args)

    return tokenizer


def _vocab_size_with_padding(orig_vocab_size, args):
    """Pad vocab size so it is divisible by model parallel size and
    still having GPU friendly size."""

    after = orig_vocab_size
    multiple = args.make_vocab_size_divisible_by * \
        args.tensor_model_parallel_size
    while (after % multiple) != 0:
        after += 1
    if args.rank == 0:
        print(' > padded vocab (size: {}) with {} dummy tokens '
              '(new size: {})'.format(
                  orig_vocab_size, after - orig_vocab_size, after), flush=True)
    return after


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    @abstractmethod
    def vocab(self):
        """Dictionary from vocab text token to id token."""
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        """Dictionary from vocab id token to text token."""
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

    def detokenize(self, token_ids):
        raise NotImplementedError('detokenizer is not implemented for {} '
                                  'tokenizer'.format(self.name))

    @property
    def cls(self):
        raise NotImplementedError('CLS is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def sep(self):
        raise NotImplementedError('SEP is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def pad(self):
        raise NotImplementedError('PAD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def eod(self):
        raise NotImplementedError('EOD is not provided for {} '
                                  'tokenizer'.format(self.name))

    @property
    def mask(self):
        raise NotImplementedError('MASK is not provided for {} '
                                  'tokenizer'.format(self.name))


class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']
        self.mask_id = self.tokenizer.vocab['[MASK]']
        self._additional_special_tokens = []

        # (dsachan) Add BOS and EOS tokens
        SPECIAL_TOKENS = {'eos_token': '[EOS]',
                          'bos_token': '[BOS]'}
        self._bos_token = '[BOS]'
        self.add_token(self._bos_token)
        self._bos_token_id = self.vocab.get(self._bos_token)

        self._eos_token = '[EOS]'
        self.add_token(self._eos_token)
        self._eos_token_id = self.vocab.get(self._eos_token)

        # (dsachan) Add additional special tokens
        # These can be used as sentinel tokens in T5 model inputs
        additional_special_tokens = []
        additional_special_tokens.extend(
            ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)])
        self.add_additional_special_tokens(additional_special_tokens)

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            # self.vocab_size comes from len(vocab)
            # and it will increase as we add elements
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens_list):
        setattr(self, "additional_special_tokens", tokens_list)
        for value in tokens_list:
            self.add_token(value)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    @property
    def vocab(self):
        return self.tokenizer.vocab

    @property
    def inv_vocab(self):
        return self.tokenizer.inv_vocab

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    def decode(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return self.tokenizer.convert_tokens_to_string(tokens)

    def decode_token_ids(self, token_ids):
        tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
        exclude_list = ['[PAD]', '[CLS]']
        non_pads = [t for t in tokens if t not in exclude_list]

        result = ""
        for s in non_pads:
            if s.startswith("##"):
                result += s[2:]
            else:
                result += " " + s

        return result

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def bos_token(self):
        """ Beginning of sentence token id """
        return self._bos_token

    @property
    def eos_token(self):
        """ End of sentence token id """
        return self._eos_token

    @property
    def additional_special_tokens(self):
        """ All the additional special tokens you may want to use (list of strings)."""
        return self._additional_special_tokens

    @property
    def bos_token_id(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos_token_id(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

    @property
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _GPT2BPETokenizer(AbstractTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        name = 'GPT2 BPE'
        super().__init__(name)

        self.tokenizer = GPT2Tokenizer(vocab_file, merge_file, errors='replace',
                                       special_tokens=[], max_len=None)
        self.eod_id = self.tokenizer.encoder['<|endoftext|>']

    @property
    def vocab_size(self):
        return len(self.tokenizer.encoder)

    @property
    def vocab(self):
        return self.tokenizer.encoder

    @property
    def inv_vocab(self):
        return self.tokenizer.decoder

    def tokenize(self, text):
        return self.tokenizer.encode(text)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids)

    @property
    def eod(self):
        return self.eod_id


class _YTTMTokenizer(AbstractTokenizer):
    """ YTTM tokenizer."""

    def __init__(self, model_path, vocab_extra_ids=0):
        name = 'YTTM'
        super().__init__(name)
        self.bpe = yttm.BPE(model=model_path)

        self.vocab_ = {}
        self.inv_vocab_ = {}
        self._additional_special_tokens = []

        self._initalize(vocab_extra_ids)

    def _initalize(self, vocab_extra_ids):
        for subword in self.bpe.vocab():
            self.add_token(subword)
        self.add_token('<CLS>'); self.cls_id = self.vocab_['<CLS>']
        self.add_token('<SEP>'); self.sep_id = self.vocab_['<SEP>']
        self.add_token('<PAD>'); self.pad_id = self.vocab_['<PAD>']
        self.add_token('<BOS>'); self.bos_id = self.vocab_['<BOS>']
        self.add_token('<EOS>'); self.eos_id = self.vocab_['<EOS>']
        self.add_token('<EOD>'); self.eod_id = self.vocab_['<EOD>']
        self.add_token('<MASK>'); self.mask_id = self.vocab_['<MASK>']
        self.special_token_ids = [self.cls_id, self.sep_id, self.pad_id,
                                  self.bos_id, self.eos_id, self.eod_id,
                                  self.mask_id]

        self.add_additional_special_tokens([
            "<extra_id_{}>".format(i) for i in range(vocab_extra_ids)
        ])

    def add_token(self, token):
        if token not in self.vocab:
            self.inv_vocab[self.vocab_size] = token
            self.vocab[token] = self.vocab_size

    def add_additional_special_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                self._additional_special_tokens.append(token)
                self.special_token_ids.append(token)
                self.add_token(token)

    @property
    def vocab_size(self):
        return len(self.vocab_)

    @property
    def vocab(self):
        return self.vocab_

    @property
    def inv_vocab(self):
        return self.inv_vocab_

    def tokenize(self, text):
        return self.bpe.encode([text], output_type=yttm.OutputType.ID)[0]

    def detokenize(self, token_ids):
        return self.bpe.decode([token_ids], ignore_ids=self.special_token_ids)[0]

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def bos_token_id(self):
        return self.bos_id

    @property
    def bos(self):
        return self.bos_id

    @property
    def eod(self):
        return self.eod_id

    @property
    def eos_token_id(self):
        return self.eos_id

    @property
    def eos(self):
        return self.eos_id

    @property
    def mask(self):
        return self.mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab.get(token) for token in self._additional_special_tokens]


class _ByteLevelBPETokenizer(AbstractTokenizer):
    """ByteLevelBPETokenizer that can support T5 pretraining."""

    def __init__(self, vocab_file, merges_file, vocab_extra_ids=0):
        name = 'ByteLevelBPETokenizer'
        super().__init__(name)
        self._bpe = tokenizers.ByteLevelBPETokenizer(vocab=vocab_file, merges=merges_file)
        self._inv_vocab = {}
        self._additional_special_tokens = []
        self._initalize(vocab_extra_ids)

    def _initalize(self, vocab_extra_ids):

        self._bpe.add_special_tokens(['<CLS>', '<SEP>', '<PAD>', '<BOS>', '<EOS>', '<EOD>', '<MASK>'])

        self._cls_id = self.vocab['<CLS>']
        self._sep_id = self.vocab['<SEP>']
        self._pad_id = self.vocab['<PAD>']
        self._bos_id = self.vocab['<BOS>']
        self._eos_id = self.vocab['<EOS>']
        self._eod_id = self.vocab['<EOD>']
        self._mask_id = self.vocab['<MASK>']

        t5_tokens = ["<extra_id_{}>".format(i) for i in range(vocab_extra_ids)]
        self._bpe.add_special_tokens(t5_tokens)
        self._additional_special_tokens = t5_tokens

    @property
    def vocab_size(self):
        return self._bpe.get_vocab_size()

    @property
    def vocab(self):
        return self._bpe.get_vocab()

    @property
    def inv_vocab(self):
        vocab = self.vocab
        if len(self._inv_vocab) != len(vocab):
            self._inv_vocab = {}
            for (k, v) in vocab.items():
                self._inv_vocab[v] = k
        return self._inv_vocab

    def tokenize(self, text):
        return self._bpe.encode(text).ids

    def detokenize(self, token_ids):
        return self._bpe.decode(token_ids)

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab.get(token) for token in self._additional_special_tokens]


class _SentencePieceTokenizer(AbstractTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        name = 'SentencePieceTokenizer'
        super().__init__(name)

        self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _initalize(self, vocab_extra_ids):
        self._vocab = {}
        self._inv_vocab = {}

        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        for i in range(len(self._tokenizer)):
            t = self._tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>'); self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>'); self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>'); self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>'); self._mask_id = self._vocab['<MASK>']

        pad_id = self._tokenizer.pad_id()
        try:
            pad_token = self._tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token); self._pad_id = self._vocab[pad_token]

        bos_id = self._tokenizer.bos_id()
        try:
            bos_token = self._tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token); self._bos_id = self._vocab[bos_token]

        eos_id = self._tokenizer.eos_id()
        try:
            eos_token = self._tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token); self._eos_id = self._vocab[eos_token]

        for i in range(vocab_extra_ids):
            t = "<extra_id_{}>".format(i)
            _add_special_token(t)
            self._t5_tokens += [t]

    @property
    def vocab_size(self):
        return len(self._vocab)

    @property
    def vocab(self):
        return self._vocab

    @property
    def inv_vocab(self):
        return self._inv_vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0
        last_idx = 0

        while 1:
            indices = {}
            for token in self._special_tokens:
                try:
                    indices[token] = text[idx:].index(token)
                except ValueError:
                    continue
            if len(indices) == 0:
                break

            next_token = min(indices, key=indices.get)
            next_idx = idx + indices[next_token]

            ids.extend(self._tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self._tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self._tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self._tokenizer.decode_ids(ids[last_i:])
        return text.strip()

    @property
    def cls(self):
        return self._cls_id

    @property
    def sep(self):
        return self._sep_id

    @property
    def pad(self):
        return self._pad_id

    @property
    def bos_token_id(self):
        return self._bos_id

    @property
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos_token_id(self):
        return self._eos_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]

