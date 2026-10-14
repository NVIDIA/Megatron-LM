# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Bitwise parity between the Mamba training forward and the generation kernels.

In batch-invariant mode the training forward runs the chunk scan through
``mamba_chunk_scan_combined_train``, which drives the same Triton kernels as dynamic
prefill (``mamba_chunk_scan_combined_varlen``) and the buffered decode replay. The
scan tests below force every candidate tile config of those kernels in turn: parity
has to hold for each of them, not only for the one autotuning or pinning happens to
choose. A control shows that the comparison catches a tile mismatch, which is what
pairing mamba_ssm's copy of the kernels with generation's left to chance.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.mamba_mixer import MambaMixer
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    from megatron.core.ssm.ops.mamba2 import (
        ssd_bmm,
        ssd_chunk_scan,
        ssd_chunk_state,
        ssd_state_passing,
    )
    from megatron.core.ssm.ops.mamba2.batch_invariant_decode import (
        BatchInvariantDecodeBuffers,
        batch_invariant_decode_buffered_scan,
    )
    from megatron.core.ssm.ops.mamba2.ssd_combined import (
        mamba_chunk_scan_combined_train,
        mamba_chunk_scan_combined_varlen,
    )

    HAVE_MAMBA = True
except ImportError:
    HAVE_MAMBA = False

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = causal_conv1d_fn is not None
except ImportError:
    HAVE_CAUSAL_CONV1D = False


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(not HAVE_MAMBA, reason="mamba_ssm required"),
]

NHEADS = 8
HEADDIM = 64
NGROUPS = 2
DSTATE = 128
CHUNK_SIZE = 128
SEQLEN = 300  # two full chunks and a partial one
# Prompt lengths of the generation requests: shorter than a chunk, exactly one chunk,
# between chunks, and two full chunks; each continues by decoding up to SEQLEN.
PROMPT_LENS = (77, 128, 200, 256)
D_CONV = 4


def _forward_autotuners():
    """The SSD forward kernels shared by training, prefill and decode."""
    return {
        "bmm_chunk": ssd_bmm._bmm_chunk_fwd_kernel,
        "chunk_cumsum": ssd_chunk_state._chunk_cumsum_fwd_kernel,
        "chunk_state": ssd_chunk_state._chunk_state_fwd_kernel,
        "state_passing": ssd_state_passing._state_passing_fwd_kernel,
        "chunk_scan": ssd_chunk_scan._chunk_scan_fwd_kernel,
    }


def _conv_autotuner():
    from megatron.core.ssm.ops.common import causal_conv1d_varlen

    return causal_conv1d_varlen._causal_conv1d_varlen_kernel


# Every candidate each forward kernel was decorated with, captured before any pinning.
_CANDIDATES = (
    {name: list(kernel.configs) for name, kernel in _forward_autotuners().items()}
    if HAVE_MAMBA
    else {}
)
_CONV_CANDIDATES = list(_conv_autotuner().configs) if HAVE_MAMBA else []


@pytest.fixture(autouse=True)
def _pinned_like_batch_invariant_mode():
    """Pin the autotuners as enable_batch_invariant_mode does.

    Decode launches the kernels with different optional arguments than prefill, which
    Triton keys as a separate tuning entry; only a pinned config holds across those.
    """
    from megatron.core.transformer.custom_layers.batch_invariant_kernels import (
        _pin_mamba_autotuners,
        _unpin_mamba_autotuners,
    )

    _pin_mamba_autotuners()
    yield
    _unpin_mamba_autotuners()


@contextlib.contextmanager
def _forced_config(kernel, config):
    """Leave ``kernel`` a single candidate, so Triton runs it without timing."""
    saved = kernel.configs
    kernel.configs = [config]
    kernel.cache.clear()
    try:
        yield
    finally:
        kernel.configs = saved
        kernel.cache.clear()


def _report(label, a, b):
    """Log a pair's agreement and return whether it is bitwise identical."""
    same = torch.equal(a, b)
    delta = (a.float() - b.float()).pow(2).mean().sqrt()
    scale = b.float().pow(2).mean().sqrt().clamp_min(1e-12)
    print(f"[mamba-parity] {label}: bitwise={same} rel_rms={(delta / scale).item():.3e}")
    return same


def _make_inputs(batch, dtype, seed=0, seqlen=SEQLEN, dims=(NHEADS, HEADDIM, NGROUPS, DSTATE)):
    """SSM inputs for ``batch`` training sequences of ``seqlen`` tokens.

    ``dims`` is (heads, headdim, groups, d_state) of one tensor-parallel rank.
    """
    nheads, headdim, ngroups, dstate = dims
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape, dt=dtype, scale=1.0):
        return torch.randn(*shape, device=device, dtype=dt, generator=generator) * scale

    return dict(
        x=randn(batch, seqlen, nheads, headdim),
        z=randn(batch, seqlen, nheads, headdim),
        dt=randn(batch, seqlen, nheads),
        B=randn(batch, seqlen, ngroups, dstate, scale=0.3),
        C=randn(batch, seqlen, ngroups, dstate, scale=0.3),
        A=-torch.rand(nheads, device=device, generator=generator).mul(2).exp(),
        D=torch.rand(nheads, device=device, generator=generator),
        dt_bias=torch.randn(nheads, device=device, generator=generator),
    )


def _train(inp):
    """The batch-invariant training scan, as MambaMixer._static_prefill calls it."""
    return mamba_chunk_scan_combined_train(
        inp["x"],
        inp["dt"],
        inp["A"],
        inp["B"],
        inp["C"],
        CHUNK_SIZE,
        D=inp["D"],
        z=inp["z"],
        dt_bias=inp["dt_bias"],
        dt_softplus=True,
        state_dtype=torch.float32,
    )


def _prefill(inp, rows, starts, ends, ssm_state, buffers, out):
    """One dynamic-engine prefill step: tokens [starts[i], ends[i]) of each row in ``rows``.

    One varlen call over all requests, chunked from each slice's start and cached as
    ``MambaMixer.ssm_prefill`` does in batch-invariant mode: the state at the last full
    chunk boundary goes to ``ssm_state``, the unfinished chunk to the decode buffers.
    """
    device = inp["x"].device

    def tokens(key):
        return torch.cat([inp[key][i, a:b] for i, a, b in zip(rows, starts, ends)]).contiguous()

    x, z, dt, B, C = (tokens(k) for k in ("x", "z", "dt", "B", "C"))
    lens = [b - a for a, b in zip(starts, ends)]
    cu_seqlens = torch.tensor([0, *lens], device=device).cumsum(0).to(torch.int32)
    chunk_bounds, last_chunk, seq_of_chunk = [0], [], []
    for n, (start, end) in enumerate(zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())):
        chunk_bounds += list(range(start + CHUNK_SIZE, end, CHUNK_SIZE)) + [end]
        seq_of_chunk += [n] * (len(chunk_bounds) - 1 - len(seq_of_chunk))
        last_chunk.append(len(chunk_bounds) - 2)
    slots = torch.tensor(rows, dtype=torch.int32, device=device)
    initial_states = ssm_state[slots.long()]
    y = torch.zeros_like(x)
    _, raw_states = mamba_chunk_scan_combined_varlen(
        x=x,
        dt=dt,
        A=inp["A"],
        B=B,
        C=C,
        chunk_size=CHUNK_SIZE,
        cu_chunk_seqlens=torch.tensor(chunk_bounds, dtype=torch.int32, device=device),
        last_chunk_indices=torch.tensor(last_chunk, dtype=torch.int32, device=device),
        seq_idx=torch.tensor(seq_of_chunk, dtype=torch.int32, device=device),
        out=y,
        D=inp["D"],
        z=z,
        dt_bias=inp["dt_bias"],
        initial_states=initial_states,
        return_raw_states=True,
        dt_softplus=True,
        state_dtype=torch.float32,
    )
    for n, (i, length) in enumerate(zip(rows, lens)):
        if length >= CHUNK_SIZE:
            ssm_state[i] = raw_states[last_chunk[n] - (1 if length % CHUNK_SIZE else 0)]
        out[i, starts[n] : ends[n]] = y[cu_seqlens[n] : cu_seqlens[n + 1]]
    buffers.seed(x, z, dt, B, C, cu_seqlens, slots)


def _generate(inp, prompt_lens, prefill_step=None):
    """Dynamic-engine generation of every row of ``inp``, all requests in one batch.

    Prefill of ``prompt_lens[i]`` tokens of row i, in one step or, with
    ``prefill_step``, in chunked-prefill steps of that many tokens (a multiple of the
    chunk size, as ``ssm_chunk_alignment`` keeps them), then batch-invariant decode of
    the remaining tokens of every row. Returns the per-token outputs as a
    (batch, seqlen, nheads, headdim) tensor.
    """
    batch = len(prompt_lens)
    device = inp["x"].device
    rows = range(batch)
    _, seqlen, nheads, headdim = inp["x"].shape
    ngroups, dstate = inp["B"].shape[2:]
    ssm_state = torch.zeros(batch, nheads, headdim, dstate, device=device)
    buffers = BatchInvariantDecodeBuffers.allocate(
        batch, CHUNK_SIZE, nheads, headdim, ngroups, dstate, device, inp["x"].dtype
    )
    out = torch.zeros_like(inp["x"])
    piece = prefill_step or max(prompt_lens)
    for start in range(0, max(prompt_lens), piece):
        live = [i for i in rows if prompt_lens[i] > start]
        ends = [min(start + piece, prompt_lens[i]) for i in live]
        _prefill(inp, live, [start] * len(live), ends, ssm_state, buffers, out)

    # Every request decodes in the same step, each at its own position.
    for step in range(seqlen - min(prompt_lens)):
        live = [i for i in rows if prompt_lens[i] + step < seqlen]
        positions = [prompt_lens[i] + step for i in live]
        index = torch.tensor(live, dtype=torch.int32, device=device)

        def token(key):
            return torch.stack([inp[key][i, p] for i, p in zip(live, positions)]).unsqueeze(1)

        y_step = batch_invariant_decode_buffered_scan(
            buffers,
            token("x"),
            token("z"),
            token("dt"),
            token("B"),
            token("C"),
            inp["A"],
            inp["D"],
            inp["dt_bias"],
            index,
            ssm_state,
        )
        for row, (i, p) in enumerate(zip(live, positions)):
            out[i, p] = y_step[row, 0]
    return out


def _assert_train_matches_generation(inp, prompt_lens, label):
    y_train = _train(inp)
    y_gen = _generate(inp, prompt_lens)
    same = True
    for i, length in enumerate(prompt_lens):
        same &= _report(
            f"{label} row {i} prefill[:{length}]", y_train[i, :length], y_gen[i, :length]
        )
        same &= _report(
            f"{label} row {i} decode[{length}:]", y_train[i, length:], y_gen[i, length:]
        )
    assert same, f"{label}: generation does not reproduce the training forward bitwise"


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
class TestScanParity:
    """Training scan against prefill and prefill + decode, for every tile config."""

    @pytest.mark.parametrize(
        "kernel_name,config_index",
        [(name, i) for name, configs in _CANDIDATES.items() for i in range(len(configs))],
    )
    def test_training_matches_generation_for_every_tile_config(
        self, dtype, kernel_name, config_index
    ):
        """One kernel forced to one candidate; training and generation must agree."""
        from triton.runtime.errors import OutOfResources

        kernel = _forward_autotuners()[kernel_name]
        with _forced_config(kernel, _CANDIDATES[kernel_name][config_index]):
            try:
                _assert_train_matches_generation(
                    _make_inputs(len(PROMPT_LENS), dtype),
                    PROMPT_LENS,
                    f"{kernel_name}[{config_index}]",
                )
            except OutOfResources as error:
                # Autotuning prunes such a candidate; it can never be selected here.
                pytest.skip(f"candidate does not fit on this GPU: {error}")

    @pytest.mark.parametrize(
        "dims",
        [(32, 64, 2, 128), (128, 80, 8, 128), (32, 64, 1, 256)],
        ids=["super-tp4", "nano-v2-headdim80", "dstate256-tp8"],
    )
    def test_scale_shapes(self, dtype, dims):
        """Per-rank shapes of the larger models, with a long prompt and a varlen batch.

        Ultra's Mamba dimensions are not public here; d_state 256 with one group per rank
        stands in for a wide model sharded eight ways.
        """
        prompt_lens = (8100, 7937, 8064)
        inp = _make_inputs(len(prompt_lens), dtype, seqlen=8192, dims=dims)
        _assert_train_matches_generation(inp, prompt_lens, f"{dims}")

    def test_chunked_prefill(self, dtype):
        """Prompts prefilled over several steps, each continuing from the cached state."""
        inp = _make_inputs(len(PROMPT_LENS), dtype)
        y_train = _train(inp)
        y_gen = _generate(inp, PROMPT_LENS, prefill_step=CHUNK_SIZE)
        for i, length in enumerate(PROMPT_LENS):
            assert _report(f"chunked prefill row {i} (prompt {length})", y_train[i], y_gen[i])

    def test_independent_of_batch_composition(self, dtype):
        """A row's training output does not depend on the rows trained beside it."""
        inp = _make_inputs(len(PROMPT_LENS), dtype)
        y_batch = _train(inp)
        for i in range(len(PROMPT_LENS)):
            alone = {k: (v[i : i + 1] if v.dim() > 1 else v) for k, v in inp.items()}
            assert _report(f"row {i} alone vs in batch", _train(alone)[0], y_batch[i])
        # And generation of one request alone matches its training row as well.
        alone = {k: (v[2:3] if v.dim() > 1 else v) for k, v in inp.items()}
        _assert_train_matches_generation(alone, PROMPT_LENS[2:3], "single request")

    def test_pinned_configs_survive_a_cleared_cache(self, dtype):
        """Every autotuner cache dropped between the training and the generation side."""
        inp = _make_inputs(len(PROMPT_LENS), dtype)
        y_train = _train(inp)
        for kernel in _forward_autotuners().values():
            kernel.cache.clear()
        y_gen = _generate(inp, PROMPT_LENS)
        for i, length in enumerate(PROMPT_LENS):
            assert _report(f"pinned row {i}", y_train[i], y_gen[i])

    def test_detects_a_tile_mismatch(self, dtype):
        """Control: the comparison catches two sides on different tiles.

        mamba_ssm's scan (the training kernel before this path existed) against
        generation's, with the dA cumsum on a different head tiling in each. That pairing
        only agreed when the two libraries happened to select matching tiles; the head
        tiling changes the cumsum's reduction tree (measured on GB200, bf16 and fp32).
        """
        import triton
        from mamba_ssm.ops.triton import ssd_chunk_state as mamba_ssm_chunk_state

        ours = ssd_chunk_state._chunk_cumsum_fwd_kernel
        theirs = mamba_ssm_chunk_state._chunk_cumsum_fwd_kernel
        inp = _make_inputs(len(PROMPT_LENS), dtype)
        with _forced_config(ours, triton.Config({"BLOCK_SIZE_H": 2})):
            y_gen = _generate(inp, PROMPT_LENS)
        agree = []
        for block_h in (1, 4, 8):
            with _forced_config(theirs, triton.Config({"BLOCK_SIZE_H": block_h})):
                y_old = mamba_chunk_scan_combined(
                    inp["x"],
                    inp["dt"],
                    inp["A"],
                    inp["B"],
                    inp["C"],
                    CHUNK_SIZE,
                    D=inp["D"],
                    z=inp["z"],
                    dt_bias=inp["dt_bias"],
                    dt_softplus=True,
                )
            agree.append(_report(f"mamba_ssm BLOCK_SIZE_H={block_h} vs generation", y_old, y_gen))
        assert not all(agree), "no head tiling changed the bits; the control is vacuous"

    def test_gradients_match_mamba_ssm(self, dtype):
        """The backward is mamba_ssm's; only the saved pre-gate output comes from here."""
        inp = _make_inputs(2, dtype)
        names = ("x", "dt", "A", "B", "C", "D", "z", "dt_bias")

        def grads(scan):
            leaves = {k: inp[k].detach().clone().requires_grad_(True) for k in names}
            y = scan(leaves)
            return torch.autograd.grad(y, list(leaves.values()), torch.ones_like(y))

        def ssm(t):
            return mamba_chunk_scan_combined(
                t["x"],
                t["dt"],
                t["A"],
                t["B"],
                t["C"],
                CHUNK_SIZE,
                D=t["D"],
                z=t["z"],
                dt_bias=t["dt_bias"],
                dt_softplus=True,
            )

        tol = dict(atol=2e-2, rtol=2e-2) if dtype == torch.bfloat16 else dict(atol=1e-3, rtol=1e-3)
        for name, ours, ref in zip(names, grads(_train), grads(ssm)):
            torch.testing.assert_close(ours, ref, **tol, msg=f"d{name}")


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal-conv1d required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
class TestMixerParity:
    """The mixer's own training forward against its own prefill and decode.

    Covers what the scan tests cannot: the split of the projection, the conv, the gate
    placement and the norm, on the code paths the mixer actually dispatches to.
    """

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @staticmethod
    def _mixer(dtype, tp_size):
        if Utils.world_size < tp_size:
            pytest.skip(f"requires {tp_size} ranks, got {Utils.world_size}")
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp_size)
        model_parallel_cuda_manual_seed(123)
        config = TransformerConfig(
            hidden_size=256,
            num_layers=1,
            num_attention_heads=tp_size,
            use_cpu_initialization=True,
            tensor_model_parallel_size=tp_size,
        )
        mixer = MambaMixer(
            config,
            hybrid_stack_spec.submodules.mamba_layer.submodules.mixer.submodules,
            config.hidden_size,
            layer_number=1,
            pg_collection=ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp", "cp"]),
        )
        # The checks batch_invariant_mode adds to TransformerConfig concern other
        # layers (attention backend, GEMMs); the mixer only reads the flag.
        config.batch_invariant_mode = True
        return mixer.to(device="cuda", dtype=dtype)

    @pytest.mark.parametrize("tp_size", [1, 2])
    def test_training_forward_matches_prefill_and_decode(self, dtype, tp_size):
        """Per rank: with TP the heads, groups and conv channels are this rank's shard."""
        mixer = self._mixer(dtype, tp_size)
        device = torch.device("cuda")
        batch = len(PROMPT_LENS)
        proj_dim = mixer.in_proj.weight.shape[0]
        generator = torch.Generator(device=device).manual_seed(0)
        zxBCdt = torch.randn(SEQLEN, batch, proj_dim, device=device, generator=generator)
        zxBCdt = zxBCdt.to(dtype)

        # Training: the dispatch _mamba_chunk makes in this mode, even on the fused default.
        assert mixer.use_mem_eff_path
        y_train = mixer._static_prefill(zxBCdt, conv_state=None, ssm_state=None)

        # Generation: every request prefilled in one varlen step, then decoded together.
        conv_dim = mixer.conv1d_weight.shape[0]
        conv_state = torch.zeros(batch, conv_dim, mixer.d_conv, device=device, dtype=dtype)
        ssm_state = torch.zeros(
            batch, mixer.nheads_local_tp, mixer.headdim, mixer.d_state, device=device
        )
        lens = torch.tensor(PROMPT_LENS, device=device)
        cu_seqlens = torch.cat([lens.new_zeros(1), lens.cumsum(0)]).to(torch.int32)
        batch_indices = torch.arange(batch, dtype=torch.int32, device=device)
        context = SimpleNamespace(
            mamba_slot_allocator=None,
            mamba_metadata=SimpleNamespace(
                seq_idx=torch.repeat_interleave(batch_indices, lens).unsqueeze(0),
                cu_seqlens=cu_seqlens,
                batch_indices_prefill=batch_indices,
                intermediate_chunk_indices=None,
                intermediate_abs_positions=None,
                intermediate_real_count=None,
                cu_chunk_seqlens=None,
                last_chunk_indices=None,
                seq_idx_for_varlen=None,
                conv_seq_idx=None,
                conv_seq_start=None,
            ),
        )
        prompts = torch.cat([zxBCdt[:n, i] for i, n in enumerate(PROMPT_LENS)]).unsqueeze(1)
        y_prefill = mixer.ssm_prefill(prompts, conv_state, ssm_state, context)
        y_gen = torch.zeros_like(y_train)
        for i, n in enumerate(PROMPT_LENS):
            y_gen[:n, i] = y_prefill[cu_seqlens[i] : cu_seqlens[i + 1], 0]
        for step in range(SEQLEN - min(PROMPT_LENS)):
            live = [i for i, n in enumerate(PROMPT_LENS) if n + step < SEQLEN]
            tokens = torch.stack([zxBCdt[PROMPT_LENS[i] + step, i] for i in live]).unsqueeze(1)
            y_step = mixer.ssm_decode(
                tokens,
                conv_state,
                ssm_state,
                batch_indices=torch.tensor(live, dtype=torch.int32, device=device),
            )
            for row, i in enumerate(live):
                y_gen[PROMPT_LENS[i] + step, i] = y_step[row, 0]

        same = True
        for i, n in enumerate(PROMPT_LENS):
            same &= _report(f"mixer row {i} prefill[:{n}]", y_train[:n, i], y_gen[:n, i])
            same &= _report(f"mixer row {i} decode[{n}:]", y_train[n:, i], y_gen[n:, i])
        assert same, "the mixer's generation paths do not reproduce its training forward"


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal-conv1d required")
class TestConvParity:
    """The depthwise conv, whose existing test only requires atol=1e-2 for bf16."""

    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32], ids=["bf16", "fp32"])
    @pytest.mark.parametrize("config_index", range(len(_CONV_CANDIDATES)))
    def test_training_conv_matches_varlen_prefill_conv(self, dtype, config_index):
        """Bitwise, not close, on every tile config of the varlen conv."""
        from megatron.core.ssm.ops.common.causal_conv1d_varlen import causal_conv1d_varlen_fn

        device = torch.device("cuda")
        conv_dim = NHEADS * HEADDIM + 2 * NGROUPS * DSTATE
        generator = torch.Generator(device=device).manual_seed(1)
        xBC = torch.randn(SEQLEN, conv_dim, device=device, dtype=dtype, generator=generator)
        weight = torch.randn(conv_dim, D_CONV, device=device, dtype=dtype, generator=generator)
        bias = torch.randn(conv_dim, device=device, dtype=dtype, generator=generator)
        cu_seqlens = torch.tensor([0, SEQLEN], dtype=torch.int32, device=device)

        y_train = causal_conv1d_fn(
            x=xBC.t().unsqueeze(0).contiguous(), weight=weight, bias=bias, activation="silu"
        )
        with _forced_config(_conv_autotuner(), _CONV_CANDIDATES[config_index]):
            y_prefill = causal_conv1d_varlen_fn(
                x=xBC, weight=weight, bias=bias, cu_seqlens=cu_seqlens, activation="silu"
            )

        same = _report(
            f"training conv vs varlen prefill conv [{config_index}]",
            y_train.squeeze(0).t().reshape(-1),
            y_prefill.reshape(-1),
        )
        assert same, (
            "the varlen conv fork is not bitwise-equal to the pip conv the training "
            "forward runs; its own test only requires atol=1e-2"
        )
