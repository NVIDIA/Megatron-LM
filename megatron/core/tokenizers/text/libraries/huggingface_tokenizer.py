# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import json
import logging
import re
from functools import cached_property
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Union

try:
    from transformers import AutoTokenizer, BatchEncoding, PreTrainedTokenizerFast
    from transformers.utils import cached_file

    HAVE_TRANSFORMERS = True
except ModuleNotFoundError:
    HAVE_TRANSFORMERS = False

from megatron.core.utils import log_single_rank

from .abstract_tokenizer import MegatronTokenizerTextAbstract

if TYPE_CHECKING:
    from gigatoken import HFCompat
    from numpy.typing import NDArray
    from torch import Tensor


logger = logging.getLogger(__name__)


def _load_generation_config(tokenizer_path: str) -> Optional[dict]:
    """Load the model's generation_config.json, if present.

    HF tokenizers do not load generation_config.json themselves -- it is a
    model-level file -- but it holds `eos_token_id`, which may be a LIST of stop
    tokens (e.g. [2, 11]). Reading it here lets termination honor every declared
    eos token.

    `tokenizer_path` can be a local directory OR a Hub model id (the
    AutoTokenizer.from_pretrained call in __init__ already handles both). A
    plain `os.path.join` + `os.path.isfile` only ever resolves the
    local-directory case, silently finding nothing for a Hub id with no error
    raised -- so this uses HF's own `cached_file` helper, which resolves and
    caches from either source the same way `from_pretrained` does.

    Returns None when the file is missing or unreadable (graceful, logged at
    WARNING -- not every model ships a generation_config.json).
    """
    try:
        gc_path = cached_file(
            tokenizer_path, "generation_config.json", _raise_exceptions_for_missing_entries=False
        )
        if gc_path is None:
            return None
        with open(gc_path) as gc_file:
            return json.load(gc_file)
    except Exception as gc_e:
        log_single_rank(
            logger,
            logging.WARNING,
            f"Could not read generation_config.json from {tokenizer_path}: {gc_e}",
        )
        return None


class HuggingFaceTokenizer(MegatronTokenizerTextAbstract):
    """
    Wrapper of HuggingFace AutoTokenizer
        https://huggingface.co/transformers/model_doc/auto.html#autotokenizer.
    """

    def __init__(
        self,
        tokenizer_path: str,
        vocab_file: Optional[str] = None,
        merges_file: Optional[str] = None,
        mask_token: Optional[str] = None,
        bos_token: Optional[str] = None,
        eos_token: Optional[str] = None,
        pad_token: Optional[str] = None,
        sep_token: Optional[str] = None,
        cls_token: Optional[str] = None,
        unk_token: Optional[str] = None,
        additional_special_tokens: Optional[List] = [],
        use_fast: Optional[bool] = True,
        trust_remote_code: Optional[bool] = False,
        include_special_tokens: bool = True,
        chat_template: str = None,
        use_gigatoken: bool = False,
    ):
        """
        Args:
            tokenizer_path: corresponds to HuggingFace-AutoTokenizer's
                'pretrained_model_name_or_path' input argument.
                For more details please refer to
                https://huggingface.co/transformers/_modules/transformers/tokenization_auto.html#AutoTokenizer.from_pretrained.
            vocab_file: path to file with vocabulary which consists
                of characters separated by newlines.
            mask_token: mask token
            bos_token: the beginning of sequence token
            eos_token: the end of sequence token. Usually equal to sep_token
            pad_token: token to use for padding
            sep_token: token used for separating sequences
            cls_token: class token. Usually equal to bos_token
            unk_token: token to use for unknown tokens
            additional_special_tokens: list of other tokens beside standard special tokens
                (bos, eos, pad, etc.).
                For example, sentinel tokens for T5 (<extra_id_0>, <extra_id_1>, etc.)
            use_fast: whether to use fast HuggingFace tokenizer
            include_special_tokens: when True, converting text to ids will include special
                tokens / prompt tokens (if any), yielding self.tokenizer(text).input_ids
            use_gigatoken: whether to use GigaToken implementation
        """

        try:
            # this logic deals with different huggingface tokenizers having different args
            if vocab_file is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
            elif merges_file is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    vocab_file=vocab_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path,
                    vocab_file=vocab_file,
                    merges_file=merges_file,
                    use_fast=use_fast,
                    trust_remote_code=trust_remote_code,
                )
        except Exception as e:
            raise ValueError(
                'Unable to instantiate HuggingFace AutoTokenizer '
                f'for {tokenizer_path}. Exception: {e}'
            )

        self.generation_config = _load_generation_config(tokenizer_path)

        # Store the tokenizer's existing chat template if the user does not provide
        # a custom chat template. Otherwise, override the default chat template with
        # the user-provided template.
        if chat_template is None:
            chat_template = self.tokenizer.chat_template
        else:
            self.tokenizer.chat_template = chat_template

        self.use_gigatoken = use_gigatoken
        self.include_special_tokens = include_special_tokens
        self.original_vocab_size = len(self.tokenizer)
        self.eos_token = eos_token
        self.chat_template = chat_template
        special_tokens_dict = {}

        # # setting special tokens, by default the default model's special tokens will be preserved
        # # unless passes new values to the special tokens
        if unk_token is not None:
            special_tokens_dict["unk_token"] = unk_token
        if mask_token is not None:
            special_tokens_dict["mask_token"] = mask_token
        if pad_token is not None:
            special_tokens_dict["pad_token"] = pad_token

        # if the model does not have eos_token but has sep_token,
        if sep_token is not None:
            special_tokens_dict["sep_token"] = sep_token
        if eos_token is not None:
            special_tokens_dict["eos_token"] = eos_token
        elif self.tokenizer.sep_token is None and self.tokenizer.eos_token:
            special_tokens_dict["sep_token"] = self.tokenizer.eos_token
        elif self.tokenizer.eos_token is None and self.tokenizer.sep_token:
            special_tokens_dict["eos_token"] = self.tokenizer.sep_token

        # if the model does not have bos_token but has cls_token,
        # set bos_token = cls_token, and vice versa
        if bos_token is not None:
            special_tokens_dict["bos_token"] = bos_token
        elif self.tokenizer.bos_token is None and self.tokenizer.cls_token:
            special_tokens_dict["bos_token"] = self.tokenizer.cls_token
        if cls_token is not None:
            special_tokens_dict["cls_token"] = cls_token
        elif self.tokenizer.cls_token is None and self.tokenizer.bos_token:
            special_tokens_dict["cls_token"] = self.tokenizer.bos_token

        # add additional special tokens (not standard special tokens such as bos, eod, sep)
        if additional_special_tokens is not None:
            special_tokens_dict["additional_special_tokens"] = additional_special_tokens

        # Remember the requested list so we can return its IDs reliably even
        # if the underlying HF tokenizer doesn't surface it via
        # ``additional_special_tokens`` after ``add_special_tokens``.
        self._additional_special_tokens = list(additional_special_tokens or [])

        new_tokens_in_vocab = []
        for token in [mask_token, bos_token, eos_token, pad_token, sep_token, cls_token, unk_token]:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)
        for token in additional_special_tokens:
            if token is not None and token not in self.tokenizer.get_vocab():
                new_tokens_in_vocab.append(token)

        if len(new_tokens_in_vocab) > 0:
            """
            Special tokens that were not previously included in the tokenizer's vocabulary file
            will be added to the vocabulary and, as a result, the model should be resized,
            for example:

            # define your model
            tokenizer_path = 'roberta-base'
            tokenizer = MegatronTokenizer.from_pretrained(tokenizer_path=tokenizer_path)

            special_tokens = {'bos_token': '<BOS>',
                              'cls_token': '<CSL>',
                              'additional_special_tokens': ['<MY_NER_TOKEN>', '<ANOTHER_TOKEN>']}
            tokenizer.add_special_tokens(special_tokens_dict=special_tokens)

            # resize your model so that the embeddings for newly added tokens
            tokenizer.resize_token_embeddings(tokenizer_default.vocab_size)
            """

            log_single_rank(
                logger,
                logging.WARNING,
                f'{new_tokens_in_vocab} \n will be added to the vocabulary.\n'
                f'Please resize your model accordingly.',
            )
        self.add_special_tokens(special_tokens_dict)
        self.space_sensitive = self.text_to_tokens('x y') != self.text_to_tokens(
            'x'
        ) + self.text_to_tokens('y')
        self._inv_vocab_dict = {}

        self._hf_tokenizer = self.tokenizer
        if self.use_gigatoken:
            self.tokenizer = _GigatokenTokenizer(self.tokenizer)

    def add_special_tokens(self, special_tokens_dict: dict) -> int:
        """
        Adds a dictionary of special tokens (eos, pad, cls...).
        If special tokens are NOT in the vocabulary, they are added
        to it (indexed starting from the last index of the current vocabulary).

        Args:
            special_tokens_dict: dict of string.
                Keys should be in the list of predefined special attributes:
                [``bos_token``, ``eos_token``, ``unk_token``, ``sep_token``, ``pad_token``,
                ``cls_token``, ``mask_token``, ``additional_special_tokens``].
                Tokens are only added if they are not already in the vocabulary.

        Returns:
            Number of tokens added to the vocabulary.
        """

        num_tokens_added = self.tokenizer.add_special_tokens(special_tokens_dict)

        if num_tokens_added > 0:
            log_single_rank(
                logger,
                logging.INFO,
                f'{num_tokens_added} special tokens added, resize your model accordingly.',
            )
        for k in self.tokenizer.SPECIAL_TOKENS_ATTRIBUTES:
            setattr(self, k, getattr(self.tokenizer, k, None))
        return num_tokens_added

    @property
    def additional_special_tokens_ids(self):
        """
        Returns a list of the additional special tokens (excluding bos, eos, pad, unk).
        Used to return sentinel tokens for e.g. T5.
        """
        # ``additional_special_tokens_ids`` was removed from HF's
        # ``SpecialTokensMixin`` in recent transformers releases, and the
        # ``additional_special_tokens`` list on the underlying tokenizer is
        # not reliably populated after ``add_special_tokens`` on those
        # versions. Use the constructor-supplied list and convert via the
        # tokenizer's own ``convert_tokens_to_ids``.
        tokens = (
            self._additional_special_tokens
            or getattr(self.tokenizer, "additional_special_tokens", None)
            or []
        )
        if not tokens:
            return []
        return self.tokenizer.convert_tokens_to_ids(tokens)

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts text to tokens."""
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[str]) -> str:
        """Converts list of tokens text."""
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text

    def token_to_id(self, token: str) -> int:
        """Converts a single token to it's id."""
        return self.tokens_to_ids([token])[0]

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts list of tokens to it's ids."""
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return ids

    def ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Converts list of tokens ids to it's token values."""
        tokens = self.tokenizer.convert_ids_to_tokens(ids)
        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to tokens ids."""
        return self.tokenizer(text, add_special_tokens=self.include_special_tokens).input_ids

    def ids_to_text(self, ids: List[int], remove_special_tokens: Optional[bool] = None) -> str:
        """Converts list of ids to text.

        When remove_special_tokens is None, uses not self.include_special_tokens.
        """
        if remove_special_tokens is None:
            remove_special_tokens = not self.include_special_tokens
        tokens = self.ids_to_tokens(ids)
        if remove_special_tokens:
            tokens_clean = [
                t for t in tokens if t is not None and t not in self.tokenizer.all_special_tokens
            ]
        else:
            tokens_clean = [t for t in tokens if t is not None]
        text = self.tokens_to_text(tokens_clean)
        return text

    def apply_chat_template(self, conversation, chat_template, **kwargs):
        """Applies chat template and tokenizes results"""
        return self.tokenizer.apply_chat_template(
            conversation=conversation, chat_template=chat_template, **kwargs
        )

    def encode_files(self, paths: list[str], field: str = "text") -> "ak.Array":
        """Encodes whole jsonl file."""
        if self.use_gigatoken:
            from megatron.core.tokenizers.utils import has_gigatoken_support

            if has_gigatoken_support():
                import gigatoken as gt

                return self.tokenizer.tokenizer.encode_files(
                    gt.JsonlFileSource(paths, field=field), parallel=True
                )
            else:
                raise ModuleNotFoundError(
                    "gigatoken library is not installed. "
                    "Please, install gigatoken to use fast tokenizers: `pip install gigatoken`."
                )
        else:
            raise NotImplementedError(
                "This method is supported only for gigatoken tokenizers. "
                "Please, set `use_gigatoken=True`."
            )

    @property
    def vocab(self) -> list:
        """Returns tokenizer vocab values."""
        id2vocab = {v: k for k, v in self.tokenizer.vocab.items()}
        return [id2vocab[i] for i in range(len(id2vocab))]

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        if self._inv_vocab_dict == {}:
            self._inv_vocab_dict = {v: k for k, v in self.tokenizer.vocab.items()}
        return self._inv_vocab_dict

    @property
    def vocab_size(self) -> int:
        """Returns size of tokenizer vocabulary."""
        return len(self.tokenizer)

    @property
    def pad_id(self) -> int:
        """Returns id of padding token."""
        if getattr(self, 'pad_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'pad_token')])[0]

    @property
    def bos_id(self) -> int:
        """Returns id of beggining of sentence token."""
        if getattr(self, 'bos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'bos_token')])[0]

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def eod(self) -> int:
        """Returns EOD token id."""
        if getattr(self, 'eos_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'eos_token')])[0]

    @property
    def sep_id(self) -> int:
        """Returns id of SEP token."""
        if getattr(self, 'sep_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'sep_token')])[0]

    @property
    def cls_id(self) -> int:
        """Returns id of classification token."""
        if getattr(self, 'cls_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'cls_token')])[0]

    @property
    def unk_id(self) -> int:
        """Returns id of unknown tokens."""
        if getattr(self, 'unk_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'unk_token')])[0]

    @property
    def mask_id(self) -> int:
        """Returns id of mask token."""
        if getattr(self, 'mask_token') is None:
            return None
        return self.tokens_to_ids([getattr(self, 'mask_token')])[0]

    def save_vocabulary(self, save_directory: str, filename_prefix: str = None):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self._hf_tokenizer.save_vocabulary(
            save_directory=save_directory, filename_prefix=filename_prefix
        )

    def save_pretrained(self, save_directory: str):
        """Saves tokenizer's vocabulary and other artifacts to the specified directory"""
        return self._hf_tokenizer.save_pretrained(save_directory)


if HAVE_TRANSFORMERS:

    class _GigatokenTokenizer(PreTrainedTokenizerFast):
        """Keep HF rendering, metadata and decoding while accelerating text encoding."""

        # Gigatoken's ICU 2.2 uses Unicode 17; HF tokenizers 0.22 uses Unicode 16.
        # These ranges cover all differences in letter case, marks, numbers and whitespace.
        _unicode_fallback_re = re.compile(
            r'[\u0295\u088f\u0c5c\u0cdc\u1acf-\u1add\u1ae0-\u1aeb\ua7ce-\ua7cf\ua7d2\ua7d4'
            r'\ua7f1\U00010940-\U00010959\U00010ec5-\U00010ec7\U00010efa-\U00010efb'
            r'\U00011b60-\U00011b67\U00011db0-\U00011ddb\U00011de0-\U00011de9'
            r'\U00016ea0-\U00016eb8\U00016ebb-\U00016ed3\U00016ff2-\U00016ff6'
            r'\U000187f8-\U000187ff\U00018d09-\U00018d1e\U00018d80-\U00018df2'
            r'\U0001e6c0-\U0001e6de\U0001e6e0-\U0001e6f5\U0001e6fe-\U0001e6ff'
            r'\U0002b73a-\U0002b73f\U0002cea2-\U0002cead\U000323b0-\U00033479]'
        )

        def __init__(self, tokenizer: PreTrainedTokenizerFast) -> None:
            from megatron.core.tokenizers.utils import init_gigatoken_from_hf

            # Preserve the loaded HF configuration and native decoder for DecodeStream.
            self.__dict__.update(tokenizer.__dict__)
            self._hf_tokenizer = tokenizer
            self._gigatoken = init_gigatoken_from_hf(tokenizer, tokenizer.name_or_path)
            self.tokenizer = self._gigatoken.tokenizer

        def __reduce__(
            self,
        ) -> tuple[
            Callable[..., "_GigatokenTokenizer"], tuple[PreTrainedTokenizerFast], dict[str, Any]
        ]:
            # Spawned serving workers rebuild native encoders from the retained HF tokenizer.
            state = self.__dict__.copy()
            for name in ('tokenizer', '_gigatoken', '_split_special_tokenizer'):
                state.pop(name, None)
            return type(self), (self._hf_tokenizer,), state

        @cached_property
        def _split_special_tokenizer(self) -> "HFCompat":
            import gigatoken as gt

            # Gigatoken's matcher is immutable. Ordinary text needs a separate matcher
            # without special added tokens; non-special added tokens still match.
            config = json.loads(self.backend_tokenizer.to_str())
            config['added_tokens'] = [
                t for t in config.get('added_tokens', []) if not t.get('special')
            ]
            return gt.Tokenizer.from_json(json.dumps(config)).as_hf()

        def __call__(
            self, text: Any = None, text_pair: Any = None, *args: Any, **kwargs: Any
        ) -> BatchEncoding:
            supported = {
                'add_special_tokens',
                'split_special_tokens',
                'return_attention_mask',
                'return_tensors',
                'padding',
                'truncation',
                'max_length',
            }
            if (
                args
                or text_pair is not None
                or not (
                    isinstance(text, str)
                    or (
                        isinstance(text, (list, tuple))
                        and text
                        and all(isinstance(t, str) for t in text)
                    )
                )
                or kwargs.keys() - supported
                or kwargs.get('padding')
                or kwargs.get('truncation')
                or kwargs.get('max_length') is not None
                or set(self.model_input_names) - {'input_ids', 'attention_mask'}
            ):
                return super().__call__(text, text_pair, *args, **kwargs)

            # Keep ASCII cheap; fall back for the whole input to preserve BPE boundaries.
            texts = (text,) if isinstance(text, str) else text
            if any(not t.isascii() and self._unicode_fallback_re.search(t) for t in texts):
                return super().__call__(text, text_pair, *args, **kwargs)

            tokenizer = (
                self._split_special_tokenizer
                if kwargs.get('split_special_tokens', self.split_special_tokens)
                else self._gigatoken
            )
            encoded = tokenizer(
                text,
                add_special_tokens=kwargs.get('add_special_tokens', True),
                return_attention_mask=kwargs.get(
                    'return_attention_mask', 'attention_mask' in self.model_input_names
                ),
            )
            return BatchEncoding(
                dict(encoded),
                tensor_type=kwargs.get('return_tensors'),
                prepend_batch_axis=isinstance(text, str),
            )

        def apply_chat_template(
            self, *args: Any, **kwargs: Any
        ) -> Union[
            str, List[str], List[int], List[List[int]], BatchEncoding, "Tensor", "NDArray[Any]"
        ]:
            """Preserve HF's character-to-token alignment when returning assistant masks."""
            if kwargs.get('return_assistant_tokens_mask'):
                # Assistant masks require HF's character-to-token alignment data.
                return self._hf_tokenizer.apply_chat_template(*args, **kwargs)
            return super().apply_chat_template(*args, **kwargs)
