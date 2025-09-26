# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

import base64
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    import tiktoken
except ImportError:
    pass

from .abstract_tokenizer import MegatronTokenizerTextAbstract
from .chat_template import MegatronTokenizerChatTemplate

PATTERN_TIKTOKEN_V1 = (
    r"[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"
)
PATTERN_TIKTOKEN_V2 = "[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]*[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]+|[^\\r\\n\\p{L}\\p{N}]?[\\p{Lu}\\p{Lt}\\p{Lm}\\p{Lo}\\p{M}]+[\\p{Ll}\\p{Lm}\\p{Lo}\\p{M}]*|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n/]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"  # pylint: disable=line-too-long
DEFAULT_TIKTOKEN_MAX_VOCAB = 2**17  # 131072
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<mask>", "<pad>", "<cls>", "<sep>"]
SPECIAL_TOKEN_TEMPLATE = "<SPECIAL_{id}>"


def reload_mergeable_ranks(
    path: str, max_vocab: Optional[int] = None, num_special_tokens: Optional[int] = None
) -> Dict[bytes, int]:
    """
    Reload the tokenizer JSON file and convert it to Tiktoken format.

    Args:
        path (str): path to the tokenizer.
        max_vocab (Optional[int]): maximum size of vocabulary.
        num_special_tokens (Optional[int]): number of added special tokens.

    Returns:
        Dict[bytes, int]: reloaded tokenizer vocab.
    """

    assert path.endswith(".json")

    # reload vocab
    with open(path, "r") as f:
        vocab = json.load(f)
    assert isinstance(vocab, list)
    print(f"Vocab size: {len(vocab)}")
    if max_vocab is not None:
        vocab = vocab[:max_vocab]
        print(f"Cutting vocab to first {len(vocab)} tokens.")

    # build ranks
    ranks: Dict[bytes, int] = {}
    for i, x in enumerate(vocab):
        assert x.keys() == {"rank", "token_bytes", "token_str"}
        assert x["rank"] == i
        merge = base64.b64decode(x["token_bytes"])
        assert i >= 256 or merge == bytes([i])
        ranks[merge] = x["rank"] + num_special_tokens

    # sanity check
    assert len(ranks) == len(vocab)
    assert set(ranks.values()) == set(range(num_special_tokens, len(ranks) + num_special_tokens))

    return ranks


class TikTokenTokenizer(MegatronTokenizerTextAbstract, MegatronTokenizerChatTemplate):
    """TikTokenTokenizer https://github.com/openai/tiktoken."""

    def __init__(
        self,
        tokenizer_path: str,
        special_tokens: Optional[List[str]] = None,
        num_special_tokens: Optional[int] = 1000,
        chat_template: Optional[str] = None,
        pattern: Optional[str] = "v2",
        vocab_size: Optional[int] = DEFAULT_TIKTOKEN_MAX_VOCAB,
    ):
        """
        Args:
            tokenizer_path (str): path to tokenizer vocabulary.
            special_tokens (Optional[List[str]]): template for user-defined special tokens.
            num_special_tokens (int): number of special tokens to generate.
            chat_template (Optional[str]): tokenizer chat template in jinja format.
            pattern (Optional[str]): regex pattern to split the text.
            vocab_size (Optional[int]): size of vocabulary.
        """

        if not tokenizer_path or not os.path.exists(tokenizer_path):
            raise ValueError(f"tokenizer_path: {tokenizer_path} is invalid")

        if special_tokens is None:
            special_tokens = SPECIAL_TOKENS.copy()

        if pattern == "v1":
            pattern = PATTERN_TIKTOKEN_V1
        elif pattern == "v2":
            pattern = PATTERN_TIKTOKEN_V2
        else:
            raise ValueError(f"Expected tiktoken pattern to be `v1` or `v2`, but got {pattern}.")

        assert len(special_tokens) == len(
            set(special_tokens)
        ), f"Special tokens should be unique: {special_tokens}"
        assert len(special_tokens) <= num_special_tokens < vocab_size
        assert set(SPECIAL_TOKENS) <= set(
            special_tokens
        ), f"Custom special tokens should include {SPECIAL_TOKENS}"

        self._unk_id = special_tokens.index("<unk>")
        self._bos_id = special_tokens.index("<s>")
        self._eos_id = special_tokens.index("</s>")
        self._mask_id = special_tokens.index("<mask>")
        self._pad_id = special_tokens.index("<pad>")
        self._cls_id = special_tokens.index("<cls>")
        self._sep_id = special_tokens.index("<sep>")

        self._vocab_size = vocab_size
        self.chat_template = chat_template
        self.num_special_tokens = num_special_tokens
        special_filler = [
            SPECIAL_TOKEN_TEMPLATE.format(id=i)
            for i in range(len(special_tokens), num_special_tokens)
        ]
        self.special_filler = special_filler
        if special_filler:
            print(
                "Adding special tokens: "
                f"{', '.join(special_tokens)}, {special_filler[0]}, ..., {special_filler[-1]}"
            )
        self.special_tokens = special_tokens + special_filler
        assert (
            len(set(self.special_tokens)) == len(self.special_tokens) == num_special_tokens
        ), self.special_tokens
        self.inner_vocab_size = vocab_size - num_special_tokens

        # reload vocab
        self.token2id = reload_mergeable_ranks(
            tokenizer_path, max_vocab=self.inner_vocab_size, num_special_tokens=num_special_tokens
        )

        self.id2token = {v: k for k, v in self.token2id.items()}
        assert set(range(num_special_tokens, vocab_size)) == set(self.id2token.keys())

        self.shifted_id2token = {i: tok for i, tok in enumerate(self.special_tokens)}
        for key, value in self.id2token.items():
            self.shifted_id2token[key + self.num_special_tokens] = value.decode(
                'utf-8', errors='replace'
            )

        special_tokens_dict = {t: i for i, t in enumerate(self.special_tokens)}
        self.tokenizer = tiktoken.Encoding(
            name=Path(tokenizer_path).parent.name,
            pat_str=pattern,
            mergeable_ranks=self.token2id,
            special_tokens=special_tokens_dict,  # special tokens are handled manually
        )

    def text_to_tokens(self, text: str) -> List[str]:
        """Converts text to tokens."""
        token_ids = self.tokenizer.encode(text)
        return [self.tokenizer.decode_single_token_bytes(token) for token in token_ids]

    def tokens_to_text(self, tokens: List[int]) -> str:
        """Converts list of tokens to text."""
        token_ids = [self.tokenizer.encode_single_token(tokens) for tokens in tokens]
        return self.tokenizer.decode(token_ids)

    def token_to_id(self, token: str) -> int:
        """Converts a single token to it's id."""
        if token in self.special_tokens:
            return self.special_tokens.index(token)
        else:
            return self.tokenizer.encode_single_token(token) + self.num_special_tokens

    def tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Converts list of tokens to list of it's ids."""
        return [self.token_to_id(token) for token in tokens]

    def id_to_token(self, token_id: int) -> str:
        """Converts token id to token."""
        if token_id < self.num_special_tokens:
            return self.special_tokens[token_id]
        else:
            token_bytes = self.tokenizer.decode_single_token_bytes(token_id)
            return token_bytes.decode('utf-8', errors='replace')

    def ids_to_tokens(self, token_ids: List[int]) -> List[str]:
        """Converts list of tokens ids to list of tokens."""
        tokens = []
        for token_id in token_ids:
            tokens.append(self.id_to_token(token_id))

        return tokens

    def text_to_ids(self, text: str) -> List[int]:
        """Converts text to list of ids."""
        tokens = self.tokenizer.encode(text, allowed_special="all")
        return tokens

    def ids_to_text(self, tokens: List[int], remove_special_tokens: bool = False) -> str:
        """Converts list of ids to text."""
        # Filter out special tokens and adjust the remaining tokens
        if remove_special_tokens:
            adjusted_tokens = [
                t
                for t in tokens
                if t not in {self.bos_id, self.eos_id} and t >= self.num_special_tokens
            ]
        else:
            adjusted_tokens = tokens

        # Decode only if there are tokens left after filtering
        if adjusted_tokens:
            return "".join(self.ids_to_tokens(adjusted_tokens))
        else:
            return ""  # Return an empty string if all tokens were filtered out

    def add_special_tokens(self, special_tokens_dict: dict):
        """Adds special tokens to the tokenizer."""
        raise NotImplementedError("This method is not supported for TikToken tokenizers.")

    @property
    def additional_special_tokens_ids(self) -> list:
        """
        Returns a list of the additional special tokens, excluding [bos, eos, pad, unk]
        and special_filler. Used to return sentinel tokens for e.g. T5.
        """
        excluding_tokens = (
            self.ids_to_tokens([self._unk_id, self._bos_id, self._eos_id]) + self.special_filler
        )
        result = [
            self.token_to_id(token)
            for token in self.special_tokens
            if token not in excluding_tokens
        ]
        return result

    @property
    def bos_id(self) -> int:
        """Returns id of beginning of sentence token."""
        return self._bos_id

    @property
    def eos_id(self) -> int:
        """Returns id of end of sentence token."""
        return self._eos_id

    @property
    def eod(self) -> int:
        """Returns id of end of document token."""
        return self._eos_id

    @property
    def unk_id(self) -> int:
        """Returns id of unknown tokens."""
        return self._unk_id

    @property
    def mask_id(self) -> int:
        """Returns id of mask token."""
        return self._mask_id

    @property
    def pad_id(self) -> int:
        """Returns id of padding token."""
        return self._pad_id

    @property
    def cls_id(self) -> int:
        """Returns id of classification token."""
        return self._cls_id

    @property
    def sep_id(self) -> int:
        """Returns id of SEP token."""
        return self._sep_id

    @property
    def vocab(self):
        """Returns tokenizer vocab."""
        return self.token2id

    @property
    def decoder(self):
        """ """
        return self.shifted_id2token

    @property
    def encoder(self):
        """ """
        return self.vocab

    @property
    def vocab_size(self) -> int:
        """Returns tokenizer vocab size."""
        return self._vocab_size

    @property
    def inv_vocab(self) -> dict:
        """Returns tokenizer vocab with reversed keys and values."""
        return self.shifted_id2token
