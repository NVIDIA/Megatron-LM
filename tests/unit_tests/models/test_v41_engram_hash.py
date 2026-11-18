# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""EngramHasher's fused Triton hash must reproduce its PyTorch arithmetic bit for bit.

Covers random tokens over the whole raw and compressed vocabularies (negative ids and int32
ids included), token masks, THD documents of random lengths (single-token documents and an
unassigned ``-1`` tail included), batch > 1, outputs shorter than the history halo,
max_ngram_size 2..5, 1..8 heads, one to three hashed layers, multipliers whose products
overflow int64, context parallelism over 1/2/4 ranks (neighbor halo exchange and the tiny-shard
all-gather), the MCORE_ENGRAM_FUSED_HASH and MCORE_ENGRAM_HASH_CHECKS switches (and their README
rows), token ids outside the token map, CUDA-graph capture and replay with new tokens, one
Triton specialization for any lengths, strides and alignments, and the warned PyTorch fallbacks
for cold specializations inside capture and for kernels that fail to compile.
"""

import copy
import itertools
import logging
import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.models.deepseek_v41 import engram as engram_module
from megatron.core.models.deepseek_v41 import engram_hash_kernels
from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout
from tests.unit_tests.test_utilities import Utils

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and engram_hash_kernels.HAVE_TRITON),
    reason="needs a GPU and Triton",
)

# The released tokenizer's raw and compressed vocabulary sizes (realistic multiplier magnitudes).
RAW_VOCAB = 129280
COMPRESSED_VOCAB = 98627


@pytest.fixture(autouse=True)
def _rank_device():
    """Each torch.distributed.run rank uses its own GPU, also in single-GPU tests."""
    torch.cuda.set_device(Utils.local_rank % torch.cuda.device_count())


@pytest.fixture
def kernel_calls(monkeypatch):
    """Switch the fused hash on and record, per EngramHasher call, whether the kernel ran."""
    calls = []
    original = engram_module.fused_engram_hash

    def spy(*args, **kwargs):
        result = original(*args, **kwargs)
        calls.append(result is not None)
        return result

    monkeypatch.setattr(engram_module, "fused_engram_hash", spy)
    monkeypatch.setattr(engram_module, "_FUSED_HASH", True)
    return calls


@pytest.fixture
def fresh_kernel_state(monkeypatch):
    """Empty per-process warm, failed and warned sets for this test only."""
    for name in ("_WARM_KEYS", "_FAILED_KEYS", "_WARNED"):
        monkeypatch.setattr(engram_hash_kernels, name, set())


class _FailingKernel:
    """Stands in for the Triton kernel: every launch raises, like a failed compilation."""

    def __init__(self):
        self.launches = 0

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.launches += 1
            raise RuntimeError("synthetic Triton compilation failure")

        return launch


def _token_map(generator, raw_vocab=RAW_VOCAB, compressed_vocab=COMPRESSED_VOCAB):
    """A random map of the raw ids onto every compressed id."""
    values = torch.cat(
        (
            torch.randperm(compressed_vocab, generator=generator),
            torch.randint(
                0, compressed_vocab, (raw_vocab - compressed_vocab,), generator=generator
            ),
        )
    )
    return values[torch.randperm(raw_vocab, generator=generator)]


def _hasher(max_ngram_size, n_heads, layer_ids=(0,), seed=0):
    generator = torch.Generator().manual_seed(seed)
    args = SimpleNamespace(
        engram_layer_ids=list(layer_ids),
        engram_num_embeddings=[0] * len(layer_ids),
        engram_max_ngram_size=max_ngram_size,
        engram_vocab_size=97,
        engram_n_heads=n_heads,
        engram_head_dim=4,
    )
    rows = [sum(map(sum, primes)) for primes in EngramLayout.from_args(args).primes]
    pad_token_id = int(torch.randint(0, RAW_VOCAB, (), generator=generator))
    options = EngramConfig(
        list(layer_ids), rows, max_ngram_size, 97, n_heads, 4, pad_token_id, COMPRESSED_VOCAB
    )
    return EngramHasher(options, _token_map(generator)).cuda()


def _documents(batch, length, generator):
    """THD sequence ids: random-length documents (30% single-token) and a ``-1`` tail."""
    rows = []
    for _ in range(batch):
        ids, document = [], 0
        while len(ids) < length:
            single = torch.rand((), generator=generator) < 0.3
            size = 1 if single else int(torch.randint(2, 40, (), generator=generator))
            ids += [document] * size
            document += 1
        row = torch.tensor(ids[:length], dtype=torch.long)
        tail = int(torch.randint(0, length // 8 + 1, (), generator=generator))
        if tail:
            row[length - tail :] = -1
        rows.append(row)
    return torch.stack(rows)


def _inputs(batch, length, generator, *, mask, documents, dtype=torch.int64, negative=False):
    """Token ids (optionally also negative), an optional mask and optional THD documents."""
    low = -RAW_VOCAB if negative else 0
    tokens = torch.randint(low, RAW_VOCAB, (batch, length), generator=generator).to(dtype)
    sequence_ids = _documents(batch, length, generator) if documents else None
    token_mask = None
    if mask:
        token_mask = torch.rand(batch, length, generator=generator) < 0.9
        if sequence_ids is not None:
            token_mask &= sequence_ids >= 0
    return tuple(None if t is None else t.cuda() for t in (tokens, token_mask, sequence_ids))


def _fused(hasher, tokens, token_mask, sequence_ids, local_length):
    return engram_hash_kernels.fused_engram_hash(
        tokens,
        token_mask,
        sequence_ids,
        hasher.token_map,
        hasher.multipliers,
        hasher.primes,
        hasher.offsets,
        hasher.pad_id,
        local_length,
    )


def _key(hasher, tokens, token_mask, sequence_ids):
    """The kernel specialization (warm/failed key) of hashing these inputs with ``hasher``."""
    ngram, heads = hasher.multipliers.shape[1], hasher.primes.shape[2]
    width = hasher.offsets.numel()
    return engram_hash_kernels._specialization_key(
        tokens, token_mask, sequence_ids, ngram, heads, width
    )


def _warnings(caplog, text):
    return [r.getMessage() for r in caplog.records if text in r.getMessage()]


def _assert_same(actual, expected, what):
    assert actual is not None, f"{what}: the fused kernel did not run"
    assert actual.dtype == expected.dtype == torch.int64, what
    assert actual.shape == expected.shape, (what, actual.shape, expected.shape)
    assert torch.equal(actual, expected), f"{what}: {int((actual != expected).sum())} differ"


def _assert_fused_matches(hasher, tokens, token_mask, sequence_ids, local_length, what):
    expected = hasher._reference_hash(tokens, token_mask, sequence_ids, local_length)
    actual = _fused(hasher, tokens, token_mask, sequence_ids, local_length)
    _assert_same(actual, expected, what)
    return actual


@pytest.mark.parametrize("max_ngram_size", [2, 3, 4, 5])
@pytest.mark.parametrize("n_heads,layer_ids", [(1, (0,)), (3, (0, 2, 5)), (8, (1, 14))])
def test_fused_hash_matches_reference(max_ngram_size, n_heads, layer_ids):
    hasher = _hasher(max_ngram_size, n_heads, layer_ids, seed=max_ngram_size * 10 + n_heads)
    generator = torch.Generator().manual_seed(max_ngram_size + 100 * n_heads)
    for mask, documents in itertools.product((False, True), repeat=2):
        for batch, length in ((1, 1), (1, 517), (3, 300)):
            inputs = _inputs(batch, length, generator, mask=mask, documents=documents)
            what = f"mask={mask} documents={documents} batch={batch} length={length}"
            _assert_fused_matches(hasher, *inputs, length, what)
    # int32 ids, negative ids (PyTorch indexing wraps them) and non-contiguous views.
    tokens, token_mask, sequence_ids = _inputs(
        2, 640, generator, mask=True, documents=True, negative=True
    )
    _assert_fused_matches(hasher, tokens, token_mask, sequence_ids, 640, "negative ids")
    _assert_fused_matches(
        hasher, tokens.int(), token_mask, sequence_ids.int(), 640, "int32 ids and documents"
    )
    _assert_fused_matches(
        hasher, tokens[:, ::2], token_mask[:, 1::2], sequence_ids[:, ::2], 320, "strided views"
    )
    wide = torch.stack((tokens, sequence_ids), -1).transpose(0, 1).contiguous().transpose(0, 1)
    _assert_fused_matches(hasher, wide[..., 0], None, wide[..., 1], 640, "interleaved columns")


@pytest.mark.parametrize("max_ngram_size", [2, 4, 5])
def test_outputs_shorter_than_the_halo(max_ngram_size):
    """CP-style calls: the input carries a history halo and only the last rows are returned."""
    hasher = _hasher(max_ngram_size, 4, (3, 7))
    generator = torch.Generator().manual_seed(7)
    halo = max_ngram_size - 1
    for local_length in range(1, halo + 3):
        for mask, documents in itertools.product((False, True), repeat=2):
            inputs = _inputs(2, halo + local_length, generator, mask=mask, documents=documents)
            what = f"local_length={local_length} mask={mask} documents={documents}"
            hashes = _assert_fused_matches(hasher, *inputs, local_length, what)
            assert hashes.shape[1] == local_length
    # ``[:, -0:]`` keeps every row, like the reference's slice, and so does a local length above
    # the input length.
    inputs = _inputs(2, halo, generator, mask=True, documents=True)
    assert _assert_fused_matches(hasher, *inputs, 0, "local_length=0").shape[1] == halo
    for local_length in (halo + 1, halo + 5):
        what = f"local_length={local_length} > length={halo}"
        assert _assert_fused_matches(hasher, *inputs, local_length, what).shape[1] == halo
    empty = _inputs(2, 0, generator, mask=True, documents=True)
    assert _assert_fused_matches(hasher, *empty, 0, "empty").shape == (2, 0, 2, halo * 4)
    assert _assert_fused_matches(hasher, *empty, 3, "empty, local 3").shape == (2, 0, 2, halo * 4)


def test_int64_overflowing_multipliers():
    """Wrapping products, negative XORs and floored modulo of negative values match PyTorch."""
    hasher = _hasher(5, 8, (0, 1))
    generator = torch.Generator().manual_seed(3)
    count = hasher.multipliers.numel()
    random_bits = torch.randint(0, 256, (8 * count,), dtype=torch.uint8, generator=generator)
    hasher.multipliers.copy_(random_bits.view(torch.int64).view(hasher.multipliers.shape))
    for mask, documents in itertools.product((False, True), repeat=2):
        inputs = _inputs(2, 1000, generator, mask=mask, documents=documents)
        hashes = _assert_fused_matches(
            hasher, *inputs, 1000, f"overflow mask={mask} documents={documents}"
        )
        assert (hashes >= hasher.offsets).all()
    # Products wrap and some XORs are negative, so the floored modulo is exercised.
    products = hasher.token_map[inputs[0]].unsqueeze(-1) * hasher.multipliers[0]
    assert ((products[..., 0] ^ products[..., 1]) < 0).any()


def test_dispatch_follows_switch_and_device(monkeypatch, kernel_calls):
    hasher = _hasher(4, 2, (0, 1))
    tokens, token_mask, sequence_ids = _inputs(
        2, 200, torch.Generator().manual_seed(5), mask=True, documents=True
    )
    fused = hasher(tokens, token_mask, sequence_ids)
    assert kernel_calls == [True]
    monkeypatch.setattr(engram_module, "_FUSED_HASH", False)
    reference = hasher(tokens, token_mask, sequence_ids)
    assert kernel_calls == [True]
    _assert_same(fused, reference, "switch")
    # CPU tensors keep the PyTorch arithmetic with the switch on.
    monkeypatch.setattr(engram_module, "_FUSED_HASH", True)
    host = copy.deepcopy(hasher).cpu()
    _assert_same(host(tokens.cpu(), token_mask.cpu(), sequence_ids.cpu()), reference.cpu(), "cpu")
    assert kernel_calls == [True, False]


def test_switch_is_read_at_import():
    code = (
        "from megatron.core.models.deepseek_v41 import engram, engram_hash_kernels\n"
        "assert engram._FUSED_HASH is False, engram._FUSED_HASH\n"
        "assert engram_hash_kernels.DEVICE_CHECKS is True, engram_hash_kernels.DEVICE_CHECKS\n"
    )
    environment = dict(os.environ, MCORE_ENGRAM_FUSED_HASH="0", MCORE_ENGRAM_HASH_CHECKS="1")
    result = subprocess.run(
        [sys.executable, "-c", code], env=environment, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr[-2000:]
    if "MCORE_ENGRAM_FUSED_HASH" not in os.environ:
        assert engram_module._FUSED_HASH is True
    if "MCORE_ENGRAM_HASH_CHECKS" not in os.environ:
        assert engram_hash_kernels.DEVICE_CHECKS is False


def test_switches_are_documented():
    """Every switch the Engram hash modules read has a row with its default in the V4.1 README."""
    readme = Path(__file__).resolve().parents[3] / "examples" / "deepseek_v41" / "README.md"
    if not readme.exists():
        pytest.skip("the examples are not part of this checkout")
    rows = dict(re.findall(r"^\| `(MCORE_\w+)` \| (\w+) \|", readme.read_text(), re.MULTILINE))
    switches = []
    for module in (engram_module, engram_hash_kernels):
        source = Path(module.__file__).read_text()
        switches += re.findall(r'env_flag\(\s*"(MCORE_\w+)",\s*(True|False)\s*\)', source)
    for name, default in switches:
        assert rows.get(name) == ("on" if default == "True" else "off"), (name, rows.get(name))
    names = {name for name, _ in switches}
    assert names == {"MCORE_ENGRAM_FUSED_HASH", "MCORE_ENGRAM_HASH_CHECKS"}, names


def test_out_of_range_ids_hash_as_compressed_zero_without_checks(monkeypatch):
    """By default the kernel reads nothing for an id the reference rejects and hashes it as
    compressed id 0 (the documented behavior; compute-sanitizer found no out-of-bounds read)."""
    monkeypatch.setattr(engram_hash_kernels, "DEVICE_CHECKS", False)
    hasher = _hasher(4, 8, (1, 14))
    generator = torch.Generator().manual_seed(41)
    tokens, token_mask, sequence_ids = _inputs(2, 300, generator, mask=True, documents=True)
    size = hasher.token_map.numel()
    tokens[0, 10], tokens[1, 200], tokens[1, 201], tokens[1, 299] = size, -size - 1, 2**40, -(2**40)
    out_of_range = (tokens >= size) | (tokens < -size)
    zero = int((hasher.token_map == 0).nonzero()[0, 0])
    substituted = tokens.masked_fill(out_of_range, zero)
    expected = hasher._reference_hash(substituted, token_mask, sequence_ids, 300)
    _assert_same(_fused(hasher, tokens, token_mask, sequence_ids, 300), expected, "out of range")
    host = copy.deepcopy(hasher).cpu()
    with pytest.raises(IndexError):
        host._reference_hash(tokens.cpu(), token_mask.cpu(), sequence_ids.cpu(), 300)


# Runs with MCORE_ENGRAM_HASH_CHECKS=1 in a fresh process (a device assert ends the CUDA context).
_CHECKS_PROGRAM = """
import os
from types import SimpleNamespace

import torch

from megatron.core.models.deepseek_v41 import engram_hash_kernels
from megatron.core.models.deepseek_v41.config import EngramConfig
from megatron.core.models.deepseek_v41.engram import EngramHasher
from megatron.core.models.deepseek_v41.engram_hash import EngramLayout

print("DEVICE_CHECKS", getattr(engram_hash_kernels, "DEVICE_CHECKS", None), flush=True)
torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")) % torch.cuda.device_count())
torch.manual_seed(0)
args = SimpleNamespace(
    engram_layer_ids=[1, 14],
    engram_num_embeddings=[0, 0],
    engram_max_ngram_size=4,
    engram_vocab_size=97,
    engram_n_heads=8,
    engram_head_dim=4,
)
rows = [sum(map(sum, primes)) for primes in EngramLayout.from_args(args).primes]
token_map = torch.cat((torch.arange(1000), torch.randint(0, 1000, (500,))))[torch.randperm(1500)]
hasher = EngramHasher(EngramConfig([1, 14], rows, 4, 97, 8, 4, 3, 1000), token_map).cuda()
tokens = torch.randint(-1500, 1500, (2, 777), device="cuda")
token_mask = torch.rand(2, 777, device="cuda") < 0.9
sequence_ids = (torch.rand(2, 777, device="cuda") < 0.05).long().cumsum(1)
static = (tokens, token_mask, sequence_ids)


def hash_static():
    return engram_hash_kernels.fused_engram_hash(
        *static, hasher.token_map, hasher.multipliers, hasher.primes, hasher.offsets,
        hasher.pad_id, 777,
    )


assert torch.equal(hash_static(), hasher._reference_hash(*static, 777))
print("EAGER_OK", flush=True)
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(stream):
    hash_static()
torch.cuda.current_stream().wait_stream(stream)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
    captured = hash_static()
tokens.copy_(torch.randint(-1500, 1500, (2, 777), device="cuda"))
graph.replay()
torch.cuda.synchronize()
assert torch.equal(captured, hasher._reference_hash(*static, 777))
print("GRAPH_OK", flush=True)
tokens[1, 500] = 1500  # one id past the token map
graph.replay()
torch.cuda.synchronize()
print("NOT_REACHED", flush=True)
"""


def test_device_checks_assert_on_out_of_range_ids():
    """MCORE_ENGRAM_HASH_CHECKS=1: in-range ids hash as before (eager and in a CUDA graph); an
    id past the token map fails a device assert at the next replay, like the reference."""
    environment = dict(os.environ, MCORE_ENGRAM_HASH_CHECKS="1")
    result = subprocess.run(
        [sys.executable, "-c", _CHECKS_PROGRAM],
        env=environment,
        capture_output=True,
        text=True,
        timeout=300,
    )
    output = result.stdout + result.stderr
    assert "EAGER_OK" in result.stdout and "GRAPH_OK" in result.stdout, output[-3000:]
    assert "NOT_REACHED" not in result.stdout, output[-3000:]
    assert result.returncode != 0, output[-3000:]
    assert engram_hash_kernels._OUT_OF_RANGE_MESSAGE in output, output[-3000:]
    assert "DEVICE_CHECKS True" in result.stdout, output[-3000:]


def _capture(function):
    """Capture ``function()`` on a side stream after it ran once eagerly there.

    Like full-iteration CUDA graphs, capture in thread-local mode: this thread may not make
    unsafe CUDA calls (such as loading a module), while NCCL's watchdog thread keeps polling.
    """
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        function()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode="thread_local"):
        output = function()
    return graph, output


def test_cuda_graph_replays_new_tokens(kernel_calls):
    hasher = _hasher(4, 8, (1, 14))
    generator = torch.Generator().manual_seed(11)
    static = _inputs(1, 4096, generator, mask=True, documents=True)
    fallbacks = engram_hash_kernels.capture_fallbacks
    graph, captured = _capture(lambda: hasher(*static))
    # The eager warm-up and the captured call both ran the kernel.
    assert kernel_calls == [True, True]
    assert engram_hash_kernels.capture_fallbacks == fallbacks
    for trial in range(3):
        fresh = _inputs(1, 4096, generator, mask=True, documents=True)
        for target, source in zip(static, fresh):
            target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        expected = hasher._reference_hash(*fresh, 4096)
        _assert_same(captured, expected, f"replay {trial}")


def test_cold_specialization_inside_capture_uses_reference(
    kernel_calls, fresh_kernel_state, caplog
):
    """Capture never compiles: a specialization first seen inside capture records the PyTorch
    arithmetic with one warning per specialization, and once warm the captured kernel replays
    new tokens bit-exactly."""
    hasher = _hasher(3, 5, (2,))
    generator = torch.Generator().manual_seed(13)
    static = _inputs(2, 333, generator, mask=False, documents=True, dtype=torch.int32)
    key = _key(hasher, *static)
    hasher._reference_hash(*static, 333)  # load the PyTorch kernels before capturing them
    torch.cuda.synchronize()
    fallbacks = engram_hash_kernels.capture_fallbacks

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    cold_graphs, colds = [], []
    with caplog.at_level(logging.WARNING, logger=engram_hash_kernels.__name__):
        for _ in range(2):
            cold_graphs.append(torch.cuda.CUDAGraph())
            with torch.cuda.graph(
                cold_graphs[-1], stream=stream, capture_error_mode="thread_local"
            ):
                colds.append(hasher(*static))
    assert len(_warnings(caplog, "first used inside CUDA-graph capture")) == 1, caplog.text
    assert engram_hash_kernels.capture_fallbacks == fallbacks + 2
    assert key not in engram_hash_kernels._WARM_KEYS and kernel_calls == [False, False]

    _assert_same(hasher(*static), hasher._reference_hash(*static, 333), "eager")
    assert key in engram_hash_kernels._WARM_KEYS
    warm_graph, warm = _capture(lambda: hasher(*static))
    assert engram_hash_kernels.capture_fallbacks == fallbacks + 2
    assert kernel_calls == [False, False, True, True, True]

    fresh = _inputs(2, 333, generator, mask=False, documents=True, dtype=torch.int32)
    for target, source in zip(static, fresh):
        if target is not None:
            target.copy_(source)
    for graph in (*cold_graphs, warm_graph):
        graph.replay()
    torch.cuda.synchronize()
    expected = hasher._reference_hash(*fresh, 333)
    for index, cold in enumerate(colds):
        _assert_same(cold, expected, f"cold graph {index}")
    _assert_same(warm, expected, "warm graph")


def test_failed_specialization_uses_reference_with_one_warning(
    monkeypatch, kernel_calls, fresh_kernel_state, caplog
):
    """A kernel that fails to compile or launch outside capture is not retried: the hasher keeps
    the PyTorch arithmetic with one warning, also inside a later capture."""
    hasher = _hasher(4, 2, (0, 1))
    generator = torch.Generator().manual_seed(17)
    static = _inputs(2, 200, generator, mask=True, documents=True)
    failing = _FailingKernel()
    monkeypatch.setattr(engram_hash_kernels, "_engram_hash_kernel", failing)
    with caplog.at_level(logging.WARNING, logger=engram_hash_kernels.__name__):
        for call in range(2):
            expected = hasher._reference_hash(*static, 200)
            _assert_same(hasher(*static), expected, f"eager call {call}")
        graph, captured = _capture(lambda: hasher(*static))
    assert len(_warnings(caplog, "kernel unavailable")) == 1, caplog.text
    assert failing.launches == 1 and kernel_calls == [False] * 4
    key = _key(hasher, *static)
    assert key in engram_hash_kernels._FAILED_KEYS and key not in engram_hash_kernels._WARM_KEYS
    fresh = _inputs(2, 200, generator, mask=True, documents=True)
    for target, source in zip(static, fresh):
        target.copy_(source)
    graph.replay()
    torch.cuda.synchronize()
    _assert_same(captured, hasher._reference_hash(*fresh, 200), "graph")


def test_launch_failure_inside_capture_raises(monkeypatch, kernel_calls, fresh_kernel_state):
    """Inside capture a failing launch of a warm specialization propagates (the graph may be
    invalid) and does not mark the specialization as failed."""
    hasher = _hasher(4, 2, (0, 1))
    static = _inputs(2, 200, torch.Generator().manual_seed(19), mask=True, documents=True)
    hasher(*static)
    key = _key(hasher, *static)
    assert key in engram_hash_kernels._WARM_KEYS
    failing = _FailingKernel()
    monkeypatch.setattr(engram_hash_kernels, "_engram_hash_kernel", failing)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with pytest.raises(RuntimeError, match="synthetic Triton compilation failure"):
        with torch.cuda.graph(
            torch.cuda.CUDAGraph(), stream=stream, capture_error_mode="thread_local"
        ):
            hasher(*static)
    assert failing.launches == 1 and key not in engram_hash_kernels._FAILED_KEYS
    torch.cuda.synchronize()


def _compiled_specializations():
    """Variants of the hash kernel compiled in this process (Triton's in-memory JIT cache)."""
    caches = getattr(engram_hash_kernels._engram_hash_kernel, "device_caches", None)
    if caches is None:
        pytest.skip("this Triton version does not expose its JIT cache")
    return sum(len(entry[0] if isinstance(entry, tuple) else entry) for entry in caches.values())


def test_new_lengths_strides_and_alignment_reuse_one_specialization(fresh_kernel_state):
    """The capture guard keys only on dtypes and the hasher's shape, so nothing else may select
    another Triton variant: batch, length, strides, pointer alignment, token-map length and pad
    id reuse the variant of the first call, eagerly and inside a CUDA graph."""
    hasher = _hasher(4, 8, (1, 14), seed=23)
    generator = torch.Generator().manual_seed(29)
    _assert_fused_matches(
        hasher, *_inputs(1, 4096, generator, mask=True, documents=True), 4096, "aligned warm-up"
    )
    compiled = _compiled_specializations()
    fallbacks = engram_hash_kernels.capture_fallbacks

    def views(batch, length):
        """Strided views at 8-, 3- and 8-byte offsets (no 16-byte aligned pointer)."""
        tokens, token_mask, sequence_ids = _inputs(
            batch, 2 * length + 3, generator, mask=True, documents=True
        )
        return (
            tokens[:, 1::2][:, :length],
            token_mask[:, 3 : 3 + length],
            sequence_ids[:, 1 : 1 + length],
        )

    for batch, length, local_length in ((3, 1500, 1500), (2, 4099, 4096), (1, 17, 2)):
        _assert_fused_matches(hasher, *views(batch, length), local_length, f"views {length}")
    other = copy.deepcopy(hasher)
    other.token_map = hasher.token_map[:50021].clone()
    other.pad_id = 12345
    tokens, token_mask, sequence_ids = views(2, 999)
    tokens.remainder_(50021)
    _assert_fused_matches(other, tokens, token_mask, sequence_ids, 999, "shorter map, new pad")
    assert _compiled_specializations() == compiled

    static = views(2, 1234)
    graph, captured = _capture(lambda: _fused(hasher, *static, 1234))
    assert engram_hash_kernels.capture_fallbacks == fallbacks
    assert _compiled_specializations() == compiled
    for trial in range(2):
        for target, source in zip(static, views(2, 1234)):
            target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        _assert_same(captured, hasher._reference_hash(*static, 1234), f"replay {trial}")


def _global_inputs(cp_size, local_length, seed, *, batch, mask, documents):
    """The same global batch on every rank; each CP rank hashes its contiguous slice."""
    generator = torch.Generator().manual_seed(seed)
    return _inputs(batch, cp_size * local_length, generator, mask=mask, documents=documents)


def _local(tensor, rank, local_length):
    if tensor is None:
        return None
    return tensor[:, rank * local_length : (rank + 1) * local_length].contiguous()


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_context_parallel_halo_matches_full_sequence(cp_size, monkeypatch, kernel_calls):
    if Utils.world_size < cp_size or Utils.world_size % cp_size:
        pytest.skip(f"needs a multiple of {cp_size} ranks")
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        group = parallel_state.get_context_parallel_group()
        rank = group.rank()
        cases = [(1, True, True), (2, False, False), (2, True, False), (1, False, True)]
        for max_ngram_size in (2, 4, 5):
            hasher = _hasher(max_ngram_size, 3, (0, 2), seed=max_ngram_size)
            halo = max_ngram_size - 1
            # Lengths below the halo take the tiny-shard all-gather when cp_size > 1.
            for local_length in sorted({1, max(halo - 1, 1), halo, 37}):
                for index, (batch, mask, documents) in enumerate(cases):
                    seed = 1000 * max_ngram_size + 10 * local_length + index
                    full = _global_inputs(
                        cp_size, local_length, seed, batch=batch, mask=mask, documents=documents
                    )
                    expected = hasher._reference_hash(*full, full[0].shape[1])
                    expected = expected[:, rank * local_length : (rank + 1) * local_length]
                    local = [_local(t, rank, local_length) for t in full]
                    what = (
                        f"cp={cp_size} rank={rank} ngram={max_ngram_size} "
                        f"local={local_length} batch={batch} mask={mask} documents={documents}"
                    )
                    results = {}
                    for fused in (True, False):
                        monkeypatch.setattr(engram_module, "_FUSED_HASH", fused)
                        calls = len(kernel_calls)
                        results[fused] = hasher(*local, cp_group=group)
                        # With the switch on, the kernel hashed the haloed rows.
                        assert kernel_calls[calls:] == ([True] if fused else []), what
                    _assert_same(results[True], expected, what + " fused")
                    _assert_same(results[False], expected, what + " reference")
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.parametrize("cp_size,local_length", [(1, 512), (4, 512), (4, 2)])
def test_context_parallel_cuda_graph_replays_new_tokens(cp_size, local_length, kernel_calls):
    """The halo exchange (or tiny-shard all-gather) and the kernel replay from one graph."""
    if Utils.world_size < cp_size or Utils.world_size % cp_size:
        pytest.skip(f"needs a multiple of {cp_size} ranks")
    Utils.initialize_model_parallel(context_parallel_size=cp_size)
    try:
        group = parallel_state.get_context_parallel_group()
        rank = group.rank()
        hasher = _hasher(4, 8, (1, 14), seed=21)
        options = dict(batch=1, mask=True, documents=True)
        full = _global_inputs(cp_size, local_length, 0, **options)
        static = [_local(t, rank, local_length) for t in full]
        fallbacks = engram_hash_kernels.capture_fallbacks
        graph, captured = _capture(lambda: hasher(*static, cp_group=group))
        assert kernel_calls == [True, True], "the kernel was not captured"
        assert engram_hash_kernels.capture_fallbacks == fallbacks
        for trial in range(1, 4):
            full = _global_inputs(cp_size, local_length, trial, **options)
            for target, source in zip(static, full):
                target.copy_(_local(source, rank, local_length))
            graph.replay()
            torch.cuda.synchronize()
            expected = hasher._reference_hash(*full, full[0].shape[1])
            expected = expected[:, rank * local_length : (rank + 1) * local_length]
            _assert_same(captured, expected, f"cp={cp_size} rank={rank} replay {trial}")
        # Release the graph (and its NCCL work) before the process groups go away.
        del graph, captured
        torch.cuda.synchronize()
    finally:
        Utils.destroy_model_parallel()
