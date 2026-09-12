# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest

from megatron.core.tokenizers.text.libraries.bytelevel_tokenizer import ByteLevelTokenizer
from megatron.core.tokenizers.text.libraries.huggingface_tokenizer import HuggingFaceTokenizer
from megatron.core.tokenizers.text.libraries.megatron_hf_tokenizer import MegatronHFTokenizer
from megatron.core.tokenizers.text.libraries.sft_tokenizer import SFTTokenizer


@pytest.fixture
def local_tokenizer():
    """Build local HF tokenizers without model downloads."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")

    def build(kind="bpe"):
        if kind == "byte":
            alphabet = sorted(tokenizers.pre_tokenizers.ByteLevel.alphabet())
            backend = tokenizers.Tokenizer(
                tokenizers.models.BPE(vocab=dict(zip(alphabet, range(256))), merges=[])
            )
            backend.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
            backend.decoder = tokenizers.decoders.ByteLevel()
        elif kind in ('fallback', 'llama'):
            pieces = [f'<0x{byte:02X}>' for byte in range(256)] + ['a', 'b', 'c', 'ab', 'bc', '▁']
            backend = tokenizers.Tokenizer(
                tokenizers.models.BPE(
                    vocab=dict(zip(pieces, range(len(pieces)))),
                    merges=[('b', 'c'), ('a', 'b')],
                    byte_fallback=True,
                )
            )
            decoders = [tokenizers.decoders.ByteFallback(), tokenizers.decoders.Fuse()]
            if kind == 'llama':
                backend.normalizer = tokenizers.normalizers.Sequence(
                    [tokenizers.normalizers.Prepend('▁'), tokenizers.normalizers.Replace(' ', '▁')]
                )
                decoders = [tokenizers.decoders.Replace('▁', ' ')] + decoders
                decoders.append(tokenizers.decoders.Strip(' ', 1, 0))
            backend.decoder = tokenizers.decoders.Sequence(decoders)
        elif kind == "wordpiece":
            backend = tokenizers.Tokenizer(
                tokenizers.models.WordPiece(
                    vocab={
                        "[UNK]": 0,
                        "hello": 1,
                        "##s": 2,
                        "world": 3,
                        "!": 4,
                        "i": 5,
                        "'": 6,
                        "m": 7,
                    },
                    unk_token="[UNK]",
                )
            )
            backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
            backend.decoder = tokenizers.decoders.WordPiece()
        else:
            backend = tokenizers.Tokenizer(
                tokenizers.models.BPE(
                    vocab={"a": 0, "b": 1, "c": 2, "ab": 3, "bc": 4},
                    merges=[("b", "c"), ("a", "b")],
                )
            )
            backend.decoder = tokenizers.decoders.Fuse()
        return transformers.PreTrainedTokenizerFast(
            tokenizer_object=backend, clean_up_tokenization_spaces=False
        )

    return build


@pytest.fixture(params=[HuggingFaceTokenizer, MegatronHFTokenizer, SFTTokenizer])
def wrap_tokenizer(request):
    def wrap(backend):
        # Avoid constructor I/O.
        tokenizer = request.param.__new__(request.param)
        if isinstance(tokenizer, HuggingFaceTokenizer):
            tokenizer.tokenizer = backend
            tokenizer._hf_tokenizer = backend
            tokenizer.include_special_tokens = True
        else:
            tokenizer._tokenizer = backend
        return tokenizer

    return wrap


class _WithoutOffsetMapping:
    """Disable fast mappings while preserving the tokenizer's decoder."""

    is_fast = False

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


@pytest.mark.parametrize(
    "kind,text,expected",
    [("wordpiece", "hellos world!", [0, 5, 7, 12]), ("byte", "A€B", [0, 1, 1, 1, 2])],
)
def test_offsets_fast_mapping(local_tokenizer, wrap_tokenizer, kind, text, expected):
    backend = local_tokenizer(kind)
    tokenizer = wrap_tokenizer(backend)
    ids = backend.encode(text, add_special_tokens=False)

    assert tokenizer.ids_to_text(ids) == text
    assert tokenizer.offsets(ids, text) == expected
    assert tokenizer.offsets([], "") == []


def test_offsets_preserve_whitespace_trimmed_by_fast_mapping(local_tokenizer, wrap_tokenizer):
    tokenizers = pytest.importorskip("tokenizers")
    backend = local_tokenizer("byte")
    backend.backend_tokenizer.post_processor = tokenizers.processors.ByteLevel(
        trim_offsets=True, add_prefix_space=False
    )
    tokenizer = wrap_tokenizer(backend)
    text = " A B"
    encoding = backend(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoding["input_ids"]

    assert encoding["offset_mapping"][0][0] == 1
    assert tokenizer.ids_to_text(ids) == text
    offsets = tokenizer.offsets(ids, text)
    assert offsets == [0, 1, 2, 3]
    segments = [text[start:end] for start, end in zip(offsets, offsets[1:] + [len(text)])]
    assert "".join(segments) == text


def test_offsets_preserve_supplied_tokens_when_reencoding_changes_ids(
    local_tokenizer, wrap_tokenizer
):
    backend = local_tokenizer()
    tokenizer = wrap_tokenizer(backend)
    ids = backend.convert_tokens_to_ids(["ab", "c"])
    text = tokenizer.ids_to_text(ids)

    assert text == "abc"
    assert backend.convert_ids_to_tokens(backend.encode(text)) == ["a", "bc"]
    assert tokenizer.offsets(ids, text) == [0, 2]


@pytest.mark.parametrize(
    "kind,text,expected",
    [
        ("byte", "A€B", [0, 1, 1, 1, 2]),
        ("byte", "�B", [0, 0, 0, 1]),
        ("wordpiece", "hellos world!", [0, 5, 6, 12]),
    ],
)
def test_offsets_without_fast_mapping_preserve_decoder_context(
    local_tokenizer, wrap_tokenizer, kind, text, expected
):
    backend = local_tokenizer(kind)
    tokenizer = wrap_tokenizer(_WithoutOffsetMapping(backend))
    ids = backend.encode(text, add_special_tokens=False)

    assert tokenizer.ids_to_text(ids) == text
    assert tokenizer.offsets(ids, text) == expected


def test_offsets_malformed_utf8_replacement_span(local_tokenizer, wrap_tokenizer):
    backend = local_tokenizer("byte")
    tokenizer = wrap_tokenizer(_WithoutOffsetMapping(backend))
    ids = backend.encode("€A", add_special_tokens=False)
    assert len(ids) == 4
    del ids[2]  # Remove the final continuation byte of E2 82 AC.

    assert tokenizer.ids_to_text(ids) == "�A"
    assert tokenizer.offsets(ids, "�A") == [0, 0, 1]


@pytest.mark.parametrize('kind', ['fallback', 'llama'])
@pytest.mark.parametrize('text,expected', [('éé', [0, 0, 1, 1]), ('�', [0, 0, 0])])
def test_offsets_byte_fallback_preserve_sampled_tokens(
    local_tokenizer, wrap_tokenizer, kind, text, expected
):
    backend = local_tokenizer(kind)
    tokenizer = wrap_tokenizer(backend)
    pieces = [f'<0x{byte:02X}>' for byte in text.encode('utf-8')]
    if kind == 'llama':
        pieces = ['▁'] + pieces + ['▁']
        expected = [0] + expected + [len(text), len(text) + 1, len(text) + 3]
        text += ' abc'
    else:
        expected = expected + [len(text), len(text) + 2]
        text += 'abc'
    ids = backend.convert_tokens_to_ids(pieces + ['ab', 'c'])
    canonical_ids = backend.encode(text, add_special_tokens=False)

    assert tokenizer.ids_to_text(ids) == text
    assert canonical_ids != ids
    assert len(canonical_ids) == len(ids)
    assert tokenizer.offsets(ids, text) == expected


def test_offsets_byte_fallback_invalid_run(local_tokenizer, wrap_tokenizer):
    backend = local_tokenizer('fallback')
    tokenizer = wrap_tokenizer(backend)
    ids = backend.convert_tokens_to_ids(['<0xC3>', '<0xA9>', '<0xC3>', 'ab', 'c'])
    text = tokenizer.ids_to_text(ids)

    assert text == '���abc'
    assert tokenizer.offsets(ids, text) == [0, 1, 2, 3, 5]


@pytest.mark.parametrize('keep_leading_eod', [False, True])
def test_offsets_follow_removed_trailing_eod(local_tokenizer, wrap_tokenizer, keep_leading_eod):
    backend = local_tokenizer('fallback')
    backend.add_special_tokens({'eos_token': '<eos>'})
    tokenizer = wrap_tokenizer(backend)
    ids = backend.convert_tokens_to_ids(['<0xC3>', '<0xA9>', '<0xC3>', '<0xA9>'])
    expected = [0, 0, 1, 1]
    if keep_leading_eod:
        ids = [backend.eos_token_id] + ids
        expected = [0] + [offset + len('<eos>') for offset in expected]
    text = tokenizer.ids_to_text(ids)
    ids += [backend.eos_token_id] * 2

    offsets = tokenizer.offsets(ids, text)
    assert offsets == expected + [len(text)] * 2
    segments = [text[start:end] for start, end in zip(offsets, offsets[1:] + [len(text)])]
    assert ''.join(segments) == text


def test_sft_offsets_follow_full_sequence_whitespace_cleanup(local_tokenizer):
    backend = local_tokenizer("wordpiece")
    backend.clean_up_tokenization_spaces = True
    tokenizer = SFTTokenizer.__new__(SFTTokenizer)
    tokenizer._tokenizer = _WithoutOffsetMapping(backend)
    ids = backend.convert_tokens_to_ids(["i", "'", "m"])

    assert tokenizer.ids_to_text(ids[:2]) == "i '"
    assert tokenizer.ids_to_text(ids) == "i'm"
    assert tokenizer.offsets(ids, "i'm") == [0, 1, 2]


def test_offsets_slow_bert_tokenizer(tmp_path, wrap_tokenizer):
    transformers = pytest.importorskip("transformers")
    if int(transformers.__version__.split(".")[0]) >= 5:
        pytest.skip("Transformers 5 replaced the slow BertTokenizer with a fast backend")
    vocab = tmp_path / "vocab.txt"
    vocab.write_text("[UNK]\n[CLS]\n[SEP]\n[PAD]\n[MASK]\nhello\n##s\nworld\n!\n")
    backend = transformers.BertTokenizer(vocab_file=str(vocab), do_lower_case=False)
    tokenizer = wrap_tokenizer(backend)
    ids = backend.encode("hellos world", add_special_tokens=False)

    assert not backend.is_fast
    assert tokenizer.ids_to_text(ids) == "hellos world"
    assert tokenizer.offsets(ids, "hellos world") == [0, 5, 6]


@pytest.mark.parametrize("include_special_tokens", [True, False])
@pytest.mark.parametrize("remove_special_tokens", [True, False])
def test_hf_offsets_follow_supplied_text_special_token_policy(
    local_tokenizer, include_special_tokens, remove_special_tokens
):
    backend = local_tokenizer()
    backend.add_special_tokens({"pad_token": "<pad>"})
    tokenizer = HuggingFaceTokenizer.__new__(HuggingFaceTokenizer)
    tokenizer.tokenizer = backend
    tokenizer._hf_tokenizer = backend
    tokenizer.include_special_tokens = include_special_tokens
    ids = backend.convert_tokens_to_ids(["<pad>", "a", "<pad>", "b", "<pad>"])
    text = tokenizer.ids_to_text(ids, remove_special_tokens=remove_special_tokens)

    if remove_special_tokens:
        assert text == "ab"
        expected = [0, 0, 1, 1, 2]
    else:
        assert text == "<pad>a<pad>b<pad>"
        expected = [0, 5, 6, 11, 12]
    assert tokenizer.offsets(ids, text) == expected


@pytest.mark.parametrize(
    "ids,text,expected",
    [
        ([], "", []),
        ([65, 66], "AB", [0, 1]),
        (list("A€🙂B".encode("utf-8")), "A€🙂B", [0, 1, 1, 1, 2, 2, 2, 2, 3]),
        ([65, 255, 66], "AB", [0, 1, 1]),
        ([65, 226, 130], "A", [0, 1, 1]),
        ([226, 65], "A", [0, 0]),
        ([65, 32, 9, 10], "A", [0, 1, 1, 1]),
    ],
)
def test_byte_offsets_match_detokenization(ids, text, expected):
    tokenizer = ByteLevelTokenizer()

    assert tokenizer.ids_to_text(ids) == text
    assert tokenizer.offsets(ids, text) == expected


def test_byte_offsets_keep_positions_for_removed_special_tokens():
    tokenizer = ByteLevelTokenizer(special_tokens=["<special>"])
    special = tokenizer.special_token_to_id["<special>"]
    ids = [special, 65, 226, special, 130, 172, 66, special]

    assert tokenizer.ids_to_text(ids) == "A€B"
    assert tokenizer.offsets(ids, "A€B") == [0, 0, 1, 1, 1, 1, 2, 3]
