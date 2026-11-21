# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Stopping once a single token cannot fit must keep the masked samples.

``_reference`` keeps drawing an N for every remaining candidate. With permutation those draws
stay, because the permutation pass continues on the same RandomState.
"""

from types import SimpleNamespace

import numpy
import pytest

from megatron.core.datasets.masked_dataset import MaskedWordPieceDataset

CLS, SEP, MASK = 1, 2, 3
VOCAB_SIZE = 300


class _Tokenizer:
    cls, sep, mask = CLS, SEP, MASK

    def __init__(self, rng):
        self.inv_vocab = {
            i: f"##w{i}" if i > 10 and rng.random() < 0.3 else f"w{i}" for i in range(VOCAB_SIZE)
        }


class _Dataset(MaskedWordPieceDataset):
    """The masking logic only; the token mask draws from the RNG like BERT's."""

    def __init__(self, config):
        self.config = config

    def __getitem__(self, idx):
        raise NotImplementedError

    def _get_token_mask(self, numpy_random_state):
        if numpy_random_state.random() < 0.8:
            return MASK
        if numpy_random_state.random() >= 0.5:
            return int(numpy_random_state.randint(4, VOCAB_SIZE))
        return None


def _reference(dataset, token_ids, target_sequence_length, numpy_random_state):
    config = dataset.config
    boundaries = []
    candidates = []
    for i, token_id in enumerate(token_ids):
        if token_id == config.tokenizer.cls or token_id == config.tokenizer.sep:
            boundaries.append(1)
        elif not config.tokenizer.inv_vocab[token_id].startswith("##"):
            boundaries.append(1)
            candidates.append([i])
        else:
            boundaries.append(0)
            if config.masking_do_full_word and len(candidates) > 0:
                candidates[-1].append(i)
            else:
                candidates.append([i])

    n_maskings = min(
        config.masking_probability * target_sequence_length,
        max(1, int(round(len(token_ids) * config.masking_probability))),
    )
    ngram_nvals = numpy.arange(config.masking_max_ngram, dtype=numpy.int64) + 1
    nprobs = 1.0 / ngram_nvals
    nprobs = nprobs / nprobs.sum(keepdims=True)
    if config.masking_use_longer_ngrams:
        nprobs = nprobs[::-1]
    candidate_ngrams = [
        [candidates[idx : idx + n] for n in ngram_nvals] for idx in range(len(candidates))
    ]
    numpy_random_state.shuffle(candidate_ngrams)

    masked_token_ids = list(token_ids)
    masked_positions_and_labels = []
    masked_spans = []
    masked_indices = set()
    for candidate_idx in range(len(candidate_ngrams)):
        n_ngrams = len(candidate_ngrams[candidate_idx])
        if len(masked_positions_and_labels) >= n_maskings:
            break
        if not candidate_ngrams[candidate_idx]:
            continue
        if config.masking_use_geometric_distribution:
            n = min(numpy_random_state.geometric(0.2), config.masking_max_ngram)
        else:
            p = nprobs[:n_ngrams] / nprobs[:n_ngrams].sum(keepdims=True)
            n = numpy_random_state.choice(ngram_nvals[:n_ngrams], p=p)
        while True:
            ngram_indices = sum(candidate_ngrams[candidate_idx][n - 1], [])
            n = n - 1
            if n_maskings >= len(masked_positions_and_labels) + len(ngram_indices):
                skip_candidate = False
                break
            if n == 0:
                skip_candidate = True
                break
        if skip_candidate:
            continue
        if any(idx in masked_indices for idx in ngram_indices):
            continue
        for index in ngram_indices:
            masked_indices.add(index)
            mask = dataset._get_token_mask(numpy_random_state)
            masked_token_ids[index] = token_ids[index] if mask is None else mask
            masked_positions_and_labels.append((index, token_ids[index]))
        masked_spans.append((ngram_indices, [token_ids[index] for index in ngram_indices]))

    numpy_random_state.shuffle(candidate_ngrams)

    if config.masking_do_permutation:
        n_swappings = n_maskings
        permuted_indices = set()
        for candidate_idx in range(len(candidate_ngrams)):
            n_ngrams = len(candidate_ngrams[candidate_idx])
            if len(permuted_indices) >= n_swappings:
                break
            if not candidate_ngrams[candidate_idx]:
                continue
            p = nprobs[:n_ngrams] / nprobs[:n_ngrams].sum(keepdims=True)
            n = numpy.random.choice(ngram_nvals[:n_ngrams], p=p)
            while True:
                ngram_indices = sum(candidate_ngrams[candidate_idx][n - 1], [])
                n = n - 1
                if n_swappings >= len(permuted_indices) + len(ngram_indices):
                    skip_candidate = False
                    break
                if n == 0:
                    skip_candidate = True
                    break
            if skip_candidate:
                continue
            if any(idx in masked_indices or idx in permuted_indices for idx in ngram_indices):
                continue
            for index in ngram_indices:
                permuted_indices.add(index)

        permuted_indices = sorted(permuted_indices)
        permuted_indices_copy = list(permuted_indices)
        numpy_random_state.shuffle(permuted_indices_copy)
        masked_token_ids_copy = list(masked_token_ids)
        for idx, idx_copy in zip(permuted_indices, permuted_indices_copy):
            masked_token_ids[idx] = masked_token_ids_copy[idx_copy]
            masked_positions_and_labels.append((idx, masked_token_ids_copy[idx]))

    masked_positions_and_labels = sorted(masked_positions_and_labels, key=lambda x: x[0])
    masked_positions = [position for position, _ in masked_positions_and_labels]
    masked_labels = [label for _, label in masked_positions_and_labels]
    masked_spans = sorted(masked_spans, key=lambda x: x[0][0])
    return (masked_token_ids, masked_positions, masked_labels, boundaries, masked_spans)


def _same_rng_state(left, right):
    left, right = left.get_state(), right.get_state()
    return left[0] == right[0] and numpy.array_equal(left[1], right[1]) and left[2:] == right[2:]


def _random_tokens(rng, length):
    tokens = [CLS]
    tokens.extend(
        SEP if rng.random() < 0.02 else int(rng.integers(4, VOCAB_SIZE)) for _ in range(length)
    )
    tokens.append(SEP)
    return tokens


@pytest.fixture
def restore_global_numpy_rng():
    """The permutation pass draws from NumPy's global RNG, which the test seeds."""
    state = numpy.random.get_state()
    yield
    numpy.random.set_state(state)


@pytest.mark.usefixtures("restore_global_numpy_rng")
@pytest.mark.parametrize(
    "masking_do_full_word, masking_do_permutation, masking_use_geometric_distribution",
    [
        (True, False, False),  # BERT
        (True, False, True),  # T5
        (False, False, False),
        (False, False, True),
        (True, True, False),
        (False, True, False),
    ],
)
@pytest.mark.parametrize("masking_use_longer_ngrams", [False, True])
def test_masked_lm_predictions_match_reference(
    masking_do_full_word,
    masking_do_permutation,
    masking_use_geometric_distribution,
    masking_use_longer_ngrams,
):
    rng = numpy.random.default_rng(1234)
    for _ in range(40):
        dataset = _Dataset(
            SimpleNamespace(
                tokenizer=_Tokenizer(rng),
                masking_probability=float(rng.choice([0.05, 0.15, 0.3])),
                masking_max_ngram=int(rng.choice([1, 3, 10])),
                masking_do_full_word=masking_do_full_word,
                masking_do_permutation=masking_do_permutation,
                masking_use_longer_ngrams=masking_use_longer_ngrams,
                masking_use_geometric_distribution=masking_use_geometric_distribution,
            )
        )
        length = int(rng.choice([0, 3, 60, 509]))
        token_ids = _random_tokens(rng, length)
        target_sequence_length = max(1, length + int(rng.integers(-3, 4)))
        seed, global_seed = (int(x) for x in rng.integers(2**32, size=2))

        numpy.random.seed(global_seed)
        ref_rng = numpy.random.RandomState(seed)
        expected = _reference(dataset, token_ids, target_sequence_length, ref_rng)
        numpy.random.seed(global_seed)
        act_rng = numpy.random.RandomState(seed)
        actual = dataset._create_masked_lm_predictions(token_ids, target_sequence_length, act_rng)
        assert actual == expected
        if masking_do_permutation:
            assert _same_rng_state(ref_rng, act_rng)


class _CountingRNG:
    """Counts choice and geometric draws. RandomState rejects new attributes."""

    def __init__(self, seed):
        self._rng = numpy.random.RandomState(seed)
        self.draws = 0

    def choice(self, *args, **kwargs):
        self.draws += 1
        return self._rng.choice(*args, **kwargs)

    def geometric(self, *args, **kwargs):
        self.draws += 1
        return self._rng.geometric(*args, **kwargs)

    def shuffle(self, values):
        self._rng.shuffle(values)

    def random(self, *args, **kwargs):
        return self._rng.random(*args, **kwargs)

    def randint(self, *args, **kwargs):
        return self._rng.randint(*args, **kwargs)

    def get_state(self):
        return self._rng.get_state()


def _one_sample(dataset, token_ids):
    numpy.random.seed(7)
    ref_rng = _CountingRNG(11)
    expected = _reference(dataset, token_ids, len(token_ids), ref_rng)
    numpy.random.seed(7)
    act_rng = _CountingRNG(11)
    actual = dataset._create_masked_lm_predictions(token_ids, len(token_ids), act_rng)
    assert actual == expected
    return ref_rng, act_rng


def test_masking_loop_skips_draws_once_a_token_cannot_fit():
    rng = numpy.random.default_rng(5)
    dataset = _Dataset(
        SimpleNamespace(
            tokenizer=_Tokenizer(rng),
            masking_probability=0.15,
            masking_max_ngram=3,
            masking_do_full_word=True,
            masking_do_permutation=False,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=False,
        )
    )
    ref_rng, act_rng = _one_sample(dataset, _random_tokens(rng, 509))
    assert act_rng.draws < ref_rng.draws
    dataset.config.masking_use_geometric_distribution = True
    dataset.config.masking_max_ngram = 10
    ref_rng, act_rng = _one_sample(dataset, _random_tokens(rng, 509))
    assert act_rng.draws < ref_rng.draws


def test_permutation_keeps_every_draw():
    rng = numpy.random.default_rng(5)
    dataset = _Dataset(
        SimpleNamespace(
            tokenizer=_Tokenizer(rng),
            masking_probability=0.15,
            masking_max_ngram=3,
            masking_do_full_word=True,
            masking_do_permutation=True,
            masking_use_longer_ngrams=False,
            masking_use_geometric_distribution=False,
        )
    )
    ref_rng, act_rng = _one_sample(dataset, _random_tokens(rng, 509))
    assert act_rng.draws == ref_rng.draws
    assert _same_rng_state(ref_rng, act_rng)
