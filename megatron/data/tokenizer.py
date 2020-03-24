
"""Megatron tokenizer."""


from abc import ABC
from abc import abstractmethod

from megatron.utils import vocab_size_with_padding
from .bert_tokenization import FullTokenizer as FullBertTokenizer


def add_tokenizer_to_args(args, tokenizer_type):
    """Instantiate tokenizer based on input type and add it to args."""

    # Make sure we have not already called this method.
    if hasattr(args, 'tokenizer'):
        raise Exception('args already has a tokenizer')
    # Select and instantiate the tokenizer.
    if tokenizer_type == 'BertWordPieceLowerCase':
        args.tokenizer = _BertWordPieceTokenizer(vocab_file=args.vocab,
                                                 lower_case=True)
    else:
        raise NotImplementedError('{} tokenizer is not '
                                  'implemented.'.format(tokenizer_type))

    # Add vocab size.
    args.vocab_size = vocab_size_with_padding(args.tokenizer.vocab_size, args)


class AbstractTokenizer(ABC):
    """Abstract class for tokenizer."""

    def __init__(self, name):
        self.name = name
        super().__init__()

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @abstractmethod
    def tokenize(self, text):
        pass

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



class _BertWordPieceTokenizer(AbstractTokenizer):
    """Original BERT wordpiece tokenizer."""

    def __init__(self, vocab_file, lower_case=True):
        if lower_case:
            name = 'BERT Lower Case'
        else:
            name = 'BERT Upper Case'
        super().__init__(name)
        self.tokenizer = FullBertTokenizer(vocab_file, do_lower_case=lower_case)
        self.cls_id = self.tokenizer.vocab['[CLS]']
        self.sep_id = self.tokenizer.vocab['[SEP]']
        self.pad_id = self.tokenizer.vocab['[PAD]']

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size()

    def tokenize(self, text):
        text_tokens = self.tokenizer.tokenize(text)
        return self.tokenizer.convert_tokens_to_ids(text_tokens)

    @property
    def cls(self):
        return self.cls_id

    @property
    def sep(self):
        return self.sep_id

    @property
    def pad(self):
        return self.pad_id
