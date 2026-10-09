# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Bit-exact replay of the shared-prefix Mamba kernels.

Covers the ragged state-fork forest (the ``_RaggedGather`` Triton kernels in
``megatron/core/ssm/mamba_ragged.py`` and the forest state-passing kernels plus the reused
``mamba_ssm`` SSD kernels in ``mamba_ragged_scan.py``) and the chunk-aligned packed recurrence
in ``mamba_sequence_packing.py``, first kernel by kernel and then through a real
``MambaMixer``'s parameters.

The ragged gather backward sums every copy of an input row in a fixed order in FP32, and the
forest state kernels walk siblings in a fixed order; neither uses atomics. The intra-chunk SSD
backward is the installed ``mamba_ssm`` one, which reduces ``ddt``/``dD``/``ddA`` with atomics
unless ``MAMBA_DETERMINISTIC=1`` selects its per-tile workspace, and the channel-last
``causal_conv1d`` weight gradient needs ``CAUSAL_CONV1D_DETERMINISTIC=1``. Both are pinned
here, as ``--deterministic-mode`` does.
"""

import pytest
import torch

from megatron.core.models.hybrid.shared_prefix_layout import (
    SharedPrefixForestLayout,
    SharedPrefixLayout,
)
from tests.unit_tests.determinism.correctness.test_ssm_conv1d import (
    _build_mixer,
    requires_deterministic_conv1d,
)
from tests.unit_tests.determinism.kernels.harness import (
    assert_module_replays_bit_exact,
    assert_replays_bit_exact,
    bytes_equal,
    count_differing_replays,
    seeded,
)
from tests.unit_tests.test_utilities import Utils

try:
    import mamba_ssm  # noqa: F401
    import triton  # noqa: F401

    HAVE_KERNELS = True
except ImportError:
    HAVE_KERNELS = False

pytestmark = [
    pytest.mark.skipif(
        not (torch.cuda.is_available() and HAVE_KERNELS), reason="needs a GPU, Triton and mamba_ssm"
    ),
    requires_deterministic_conv1d,
]

_CHUNK = 128
_CONV_WIDTH = 4
# (prefix, sibling lengths) per root. The star has 16 siblings around every chunk boundary
# class (1, cs-1, cs, cs+1, ...) after an unaligned prefix (2 * cs + 37); the forest mixes an
# aligned prefix, a sub-chunk prefix and an unaligned one. Each lays out ~4k scan tokens.
_STAR = ((293, (1, 7, 127, 128, 129, 300, 41, 260, 515, 2, 128, 64, 255, 90, 333, 17)),)
_FOREST = ((256, (7, 129, 400)), (50, (1, 127, 128, 900)), (389, (300, 513)))
_ROOTS = [pytest.param(_STAR, id="star"), pytest.param(_FOREST, id="forest")]


@pytest.fixture(autouse=True)
def mamba_deterministic(monkeypatch):
    """Select the deterministic ``mamba_ssm`` and ``causal_conv1d`` backward reductions."""
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "1")
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")


def _layout(roots):
    layouts = [SharedPrefixLayout(prefix_len=p, completion_lens=lens) for p, lens in roots]
    return layouts[0] if len(layouts) == 1 else SharedPrefixForestLayout(roots=tuple(layouts))


# --- kernels ---------------------------------------------------------------------------------


@pytest.mark.parametrize("roots", _ROOTS)
def test_ragged_gather_replays(roots):
    """The halo/tail gather is a pure copy; its backward owns each input element once."""
    from megatron.core.ssm.mamba_ragged import _RaggedGather, ragged_mamba_layout

    seeded()
    meta = ragged_mamba_layout(roots, _CHUNK, _CONV_WIDTH, torch.device("cuda"))
    width = 2568
    value = torch.randn(
        meta.input_tokens, 1, width, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad = torch.randn(
        meta.convolution_indices.numel(), 1, width, device="cuda", dtype=torch.bfloat16
    )

    def fn(value):
        return _RaggedGather.apply(value, meta.convolution_indices, meta.contributors)

    out, grads = assert_replays_bit_exact(
        fn, (value,), grad_outputs={"out": grad}, contention=True, what="ragged gather"
    )
    assert int((meta.contributors >= 0).sum(0).max()) > 1, "no row is copied to a sibling"
    index = meta.convolution_indices.long()
    expected = torch.where(
        (index >= 0)[:, None, None], value.detach()[index.clamp_min(0)], value.new_zeros(())
    )
    assert bytes_equal(out["out"], expected)
    assert grads["in[0]"].shape == value.shape


def _forest_scan_case(roots, d_has_hdim=False):
    """Seeded forest-scan inputs on ~4k scan tokens, plus the layout boundaries they use."""
    from megatron.core.ssm.mamba_ragged import ragged_mamba_layout

    seeded()
    meta = ragged_mamba_layout(roots, _CHUNK, _CONV_WIDTH, torch.device("cuda"))
    tokens, heads, headdim, groups, dstate = meta.scan_tokens, 32, 64, 1, 128
    bf16 = dict(device="cuda", dtype=torch.bfloat16)
    inputs = dict(
        x=torch.randn(1, tokens, heads, headdim, **bf16),
        dt=torch.randn(1, tokens, heads, **bf16) * 0.5,
        # Mamba2's A in [-4, -1] and softplus(dt + dt_bias) around 0.01-0.07 keep the forked
        # states alive across chunks instead of decaying to zero.
        A=-(torch.rand(heads, device="cuda") * 3 + 1),
        B=torch.randn(1, tokens, groups, dstate, **bf16),
        C=torch.randn(1, tokens, groups, dstate, **bf16),
        D=torch.randn(*((heads, headdim) if d_has_hdim else (heads,)), device="cuda"),
        dt_bias=torch.rand(heads, device="cuda") * 2 - 4.6,
    )
    for tensor in inputs.values():
        tensor.requires_grad_(True)
    return meta, inputs


def _forest_scan(meta, save_intermediates=False):
    from megatron.core.ssm.mamba_ragged_scan import mamba_chunk_scan_forest

    def fn(x, dt, A, B, C, D, dt_bias):
        return mamba_chunk_scan_forest(
            x,
            dt,
            A,
            B,
            C,
            _CHUNK,
            meta.segment_chunks,
            meta.root_segments,
            D=D,
            dt_bias=dt_bias,
            dt_softplus=True,
            save_intermediates=save_intermediates,
        )

    return fn


@pytest.mark.parametrize("d_has_hdim", [False, True], ids=["D-heads", "D-hdim"])
@pytest.mark.parametrize("roots", _ROOTS)
def test_forest_scan_replays(roots, d_has_hdim):
    """Forest SSD scan replays bitwise, and retaining its intermediates changes no bit."""
    meta, inputs = _forest_scan_case(roots, d_has_hdim)
    grad = torch.randn_like(inputs["x"])
    results = {
        save: assert_replays_bit_exact(
            _forest_scan(meta, save),
            inputs,
            grad_outputs={"out": grad},
            contention=True,
            what=f"forest scan (save_intermediates={save})",
        )
        for save in (False, True)
    }

    (recomputed_out, recomputed_grads), (saved_out, saved_grads) = results[False], results[True]
    assert recomputed_grads.keys() == saved_grads.keys() == {f"in.{k}" for k in inputs}
    mismatches = [
        name
        for ref, got in ((recomputed_out, saved_out), (recomputed_grads, saved_grads))
        for name in ref
        if not bytes_equal(ref[name], got[name])
    ]
    assert not mismatches, f"save_intermediates changed {mismatches}"


def test_forest_scan_default_backward_races(monkeypatch):
    """Negative control: the reused mamba_ssm backward reduces ``dA``/``ddt_bias`` with atomics.

    Without ``MAMBA_DETERMINISTIC`` the strict replay above must fail, so it is known to be
    sensitive. Measured on GB200: every replay differs in 24-30 of 32 ``dA`` / ``ddt_bias``
    entries at this size, for both the star and the forest.
    """
    monkeypatch.setenv("MAMBA_DETERMINISTIC", "0")
    meta, inputs = _forest_scan_case(_STAR)
    assert count_differing_replays(_forest_scan(meta), inputs, replays=8) > 0


# --- through a real MambaMixer's parameters ---------------------------------------------------


# The input and every mixer parameter the scans read (in_proj, norm and out_proj are not).
_SCAN_GRADS = {"in[0]"} | {
    f"chunk0.mixer.{name}" for name in ("conv1d_weight", "conv1d_bias", "A_log", "D", "dt_bias")
}


class _Scan(torch.nn.Module):
    """Expose a mixer-level scan as a module so parameter gradients are replayed too."""

    def __init__(self, mixer, scan):
        super().__init__()
        self.mixer = mixer
        self.scan = scan

    def forward(self, recurrent):
        return self.scan(self.mixer, recurrent)


def _recurrent_inputs(mixer, tokens):
    """The x/B/C/dt projection slice the scans consume, plus a fixed output gradient."""
    cp = mixer.cp
    channels = cp.d_inner_local_tpcp + 2 * cp.ngroups_local_tpcp * mixer.d_state
    channels += cp.nheads_local_tpcp
    seeded(7)
    recurrent = torch.randn(tokens, 1, channels, device="cuda", dtype=torch.bfloat16)
    grad = torch.randn(tokens, 1, cp.d_inner_local_tpcp, device="cuda", dtype=torch.bfloat16)
    return recurrent.requires_grad_(True), grad


class TestSharedPrefixMambaScans:

    def teardown_method(self, method):
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("roots,padding", [(_STAR, 0), (_FOREST, 5)], ids=["star", "forest"])
    def test_ragged_forest_replays(self, roots, padding):
        """Gather, channel-last conv, forest scan and output map, with parameter gradients.

        ``padding`` appends physical tokens past the layout, which the scan folds into the
        last sibling.
        """
        from megatron.core.ssm.mamba_ragged import scan_mamba_ragged_forest

        mixer = _build_mixer()
        layout = _layout(roots)
        recurrent, grad = _recurrent_inputs(mixer, layout.total_len + padding)
        module = _Scan(mixer, lambda m, r: scan_mamba_ragged_forest(m, r, layout))
        _, grads = assert_module_replays_bit_exact(
            module,
            (recurrent,),
            grad_output=grad,
            contention=True,
            restore_rng=False,
            what="scan_mamba_ragged_forest",
        )
        assert _SCAN_GRADS <= grads.keys()

    def test_packed_recurrence_replays(self):
        """Every sequence starts on a chunk boundary; conv and scan reset on ``seq_idx``."""
        from megatron.core.ssm.mamba_sequence_packing import scan_mamba_packed_recurrence

        lengths = (300, 1, 129, 1024, 1500, 7, 128, 1000)
        mixer = _build_mixer()
        recurrent, grad = _recurrent_inputs(mixer, sum(lengths))
        module = _Scan(mixer, lambda m, r: scan_mamba_packed_recurrence(m, r, lengths))
        _, grads = assert_module_replays_bit_exact(
            module,
            (recurrent,),
            grad_output=grad,
            contention=True,
            restore_rng=False,
            what="scan_mamba_packed_recurrence",
        )
        assert _SCAN_GRADS <= grads.keys()
