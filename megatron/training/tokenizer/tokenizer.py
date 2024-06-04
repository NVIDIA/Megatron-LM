# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.

"""Megatron tokenizers."""

from abc import ABC
from abc import abstractmethod

from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer

from .bert_tokenization import FullTokenizer as FullBertTokenizer
from .gpt2_tokenization import GPT2Tokenizer


def build_tokenizer(args):
    """Initialize tokenizer."""
    if args.rank == 0:
        print('> building {} tokenizer ...'.format(args.tokenizer_type),
              flush=True)

    # Select and instantiate the tokenizer.
    if args.tokenizer_type == 'BertWordPieceLowerCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=True,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'BertWordPieceCase':
        assert args.vocab_file is not None
        tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab_file,
                                            lower_case=False,
                                            vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPT2BPETokenizer':
        assert args.vocab_file is not None
        assert args.merge_file is not None
        tokenizer = _GPT2BPETokenizer(args.vocab_file, args.merge_file)
    elif args.tokenizer_type == 'SentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _SentencePieceTokenizer(args.tokenizer_model, vocab_extra_ids=args.vocab_extra_ids)
    elif args.tokenizer_type == 'GPTSentencePieceTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _GPTSentencePieceTokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'Llama2Tokenizer':
        assert args.tokenizer_model is not None
        tokenizer = _Llama2Tokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'Llama3Tokenizer':
        assert args.tokenizer_model is not None
        tokenizer = create_llama3_tokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'MistralTokenizer':
        assert args.tokenizer_model is not None
        tokenizer = create_mistral_tokenizer(args.tokenizer_model)
    elif args.tokenizer_type == 'NullTokenizer':
        assert args.vocab_size is not None
        tokenizer = _NullTokenizer(args.vocab_size)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(args.tokenizer_type))

    # Add vocab size (if not already set from a checkpoint).
    if getattr(args, "padded_vocab_size", None) is None:
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


class _BertWordPieceTokenizer(MegatronTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True, vocab_extra_ids=0):
        super().__init__(vocab_file, lower_case=lower_case, vocab_extra_ids=vocab_extra_ids)
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
    def bos(self):
        """ Id of the beginning of sentence token in the vocabulary."""
        return self._bos_token_id

    @property
    def eos(self):
        """ Id of the end of sentence token in the vocabulary."""
        return self._eos_token_id

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
    def additional_special_tokens_ids(self):
        """ Ids of all the additional special tokens in the vocabulary (list of integers)."""
        return [self.vocab.get(token) for token in self._additional_special_tokens]

    @additional_special_tokens.setter
    def additional_special_tokens(self, value):
        self._additional_special_tokens = value


class _GPT2BPETokenizer(MegatronTokenizer):
    """Original GPT2 BPE tokenizer."""

    def __init__(self, vocab_file, merge_file):
        super().__init__(vocab_file, merge_file)

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


class _SentencePieceTokenizer(MegatronTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file, vocab_extra_ids=0):
        super().__init__(model_file, vocab_extra_ids=vocab_extra_ids)

        import sentencepiece
        self.tokenizer = sentencepiece.SentencePieceProcessor(model_file=model_file)
        self._initalize(vocab_extra_ids)

    def _populate_vocab(self):
        self._vocab = {}
        self._inv_vocab = {}

        for i in range(len(self.tokenizer)):
            t = self.tokenizer.id_to_piece(i)
            self._inv_vocab[i] = t
            self._vocab[t] = i

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()
        self._special_tokens = {}
        self._inv_special_tokens = {}

        self._t5_tokens = []

        def _add_special_token(t):
            if t not in self._vocab:
                next_id = len(self._vocab)
                self._vocab[t] = next_id
                self._inv_vocab[next_id] = t
            self._special_tokens[t] = self._vocab[t]
            self._inv_special_tokens[self._vocab[t]] = t

        _add_special_token('<CLS>')
        self._cls_id = self._vocab['<CLS>']
        _add_special_token('<SEP>')
        self._sep_id = self._vocab['<SEP>']
        _add_special_token('<EOD>')
        self._eod_id = self._vocab['<EOD>']
        _add_special_token('<MASK>')
        self._mask_id = self._vocab['<MASK>']

        pad_id = self.tokenizer.pad_id()
        try:
            pad_token = self.tokenizer.id_to_piece(pad_id)
        except IndexError:
            pad_token = '<PAD>'
        _add_special_token(pad_token)
        self._pad_id = self._vocab[pad_token]

        bos_id = self.tokenizer.bos_id()
        try:
            bos_token = self.tokenizer.id_to_piece(bos_id)
        except IndexError:
            bos_token = '<BOS>'
        _add_special_token(bos_token)
        self._bos_id = self._vocab[bos_token]

        eos_id = self.tokenizer.eos_id()
        try:
            eos_token = self.tokenizer.id_to_piece(eos_id)
        except IndexError:
            eos_token = '<EOS>'
        _add_special_token(eos_token)
        self._eos_id = self._vocab[eos_token]

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

    @property
    def decoder(self):
        return self._inv_vocab

    @property
    def encoder(self):
        return self._vocab

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L89
    def tokenize(self, text):
        ids = []
        idx = 0

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

            ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
            ids.append(self._special_tokens[next_token])
            idx = next_idx + len(next_token)

        ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
        return ids

    # From:
    # https://github.com/NVIDIA/NeMo/blob/c8fa217e811d60d11d014827c7f3845ff6c99ae7/nemo/collections/common/tokenizers/sentencepiece_tokenizer.py#L125
    def detokenize(self, ids):
        text = ""
        last_i = 0

        for i, id in enumerate(ids):
            if id in self._inv_special_tokens:
                text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                text += self._inv_special_tokens[id] + " "
                last_i = i + 1

        text += self.tokenizer.decode_ids(ids[last_i:])
        return text

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
    def bos(self):
        return self._bos_id

    @property
    def eod(self):
        return self._eod_id

    @property
    def eos(self):
        return self._eos_id

    @property
    def mask(self):
        return self._mask_id

    @property
    def additional_special_tokens_ids(self):
        return [self.vocab[k] for k in self._t5_tokens]


class _GPTSentencePieceTokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file,):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        self._pad_id = self.tokenizer.pad_id()
        self._bos_id = self.tokenizer.bos_id()
        self._eos_id = self.tokenizer.eos_id()

    def tokenize(self, text):
        return self.tokenizer.encode_as_ids(text)

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


class _Llama2Tokenizer(_SentencePieceTokenizer):
    """SentencePieceTokenizer-Megatron wrapper"""

    def __init__(self, model_file,):
        super().__init__(model_file, vocab_extra_ids=0)

    def _initalize(self, vocab_extra_ids):
        self._populate_vocab()

        # BOS / EOS token IDs
        self.n_words: int = self.tokenizer.vocab_size()
        self.bos_id: int = self.tokenizer.bos_id()
        self.eos_id: int = self.tokenizer.eos_id()
        self.pad_id: int = self.tokenizer.pad_id()
        assert self.tokenizer.vocab_size() == self.tokenizer.get_piece_size()

    def tokenize(self, s: str, bos=True, eos=False):
        '''Default args for text completion, not chat/dialog.'''
        assert type(s) is str
        t = self.tokenizer.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def detokenize(self, ids):
        return self.tokenizer.decode_ids(ids)

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self.eos_id

    @property
    def additional_special_tokens_ids(self):
        return None


def create_llama3_tokenizer(*args, **kwargs):

    try:
        from llama.tokenizer import Tokenizer as Llama3Tokenizer
    except ImportError:
        raise ImportError("Module 'llama' is required but not installed.")

    class _Llama3Tokenizer(Llama3Tokenizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def tokenize(self, s: str, bos=True, eos=False):
            '''Default args for text completion, not chat/dialog.'''

            assert type(s) is str

            t = self.encode(s, bos=False, eos=eos, allowed_special='all')
            return t

        def detokenize(self, ids):
            return self.decode(ids)

        @property
        def cls(self):
            return -1

        @property
        def sep(self):
            return -1

        @property
        def mask(self):
            return -1

        @property
        def eod(self):
            return self.eos_id

        @property
        def additional_special_tokens_ids(self):
            return None

        @property
        def vocab_size(self):
            return self.model.n_vocab

    return _Llama3Tokenizer(*args, **kwargs)


def create_mistral_tokenizer(*args, **kwargs):
    try:
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError:
        raise ImportError("Module 'mistral-common' is required but not installed.")

    class _MistralTokenizer(MistralTokenizer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    return _MistralTokenizer.from_file(*args, **kwargs)


class _NullTokenizer(MegatronTokenizer):
    def __init__(self, vocab_size):
        super().__init__(None, vocab_size=vocab_size)
        self._vocab_size_without_eod = int(vocab_size)
        self._eod_id = self._vocab_size_without_eod

    def tokenize(self, text):
        return [int(x) for x in text.split(' ')]

    def detokenize(self, ids):
        text = [str(x) for x in ids]
        return ' '.join(text)

    @property
    def vocab_size(self):
        return self._vocab_size_without_eod + 1

    @property
    def vocab(self):
        raise NotImplementedError

    @property
    def inv_vocab(self):
        raise NotImplementedError

    @property
    def cls(self):
        return -1

    @property
    def sep(self):
        return -1

    @property
    def mask(self):
        return -1

    @property
    def eod(self):
        return self._eod_id

    @property
    def additional_special_tokens_ids(self):
        return None
