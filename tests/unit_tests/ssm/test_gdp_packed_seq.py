# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""GDP v4 packed-sequence + context-parallel equivalence tests.

Verifies that ``GatedDeltaProductMixer`` (v4) produces forward outputs and
parameter gradients under ``cp_size=2`` that match a ``cp_size=1`` reference
run on the same packed (THD/SFT-format) input.

Style mirrors ``test_mamba_context_parallel.py``: ``Utils.initialize_model_parallel``,
``@pytest.mark.internal``, fixed-seed bf16 tensors, tolerance via
``torch.testing.assert_close``.

Run with::

    torchrun --nproc_per_node=2 -m pytest \\
        tests/unit_tests/ssm/test_gdp_packed_seq.py -m internal -v
"""

from __future__ import annotations

import os
from typing import List

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.extensions.transformer_engine import (
    TELayerNormColumnParallelLinear,
    TERowParallelLinear,
)
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.ssm.gated_delta_product import (
    GatedDeltaProductMixer,
    GatedDeltaProductMixerSubmodules,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer import TransformerConfig
from tests.unit_tests.test_utilities import Utils

try:
    import einops  # noqa: F401
    import mamba_ssm  # noqa: F401

    HAVE_MAMBA_DEPS = True
except ImportError:
    HAVE_MAMBA_DEPS = False

try:
    import fla  # noqa: F401

    HAVE_FLA = True
except ImportError:
    HAVE_FLA = False


# Skip the whole file when bare ``pytest`` is invoked outside torchrun. The
# CP=2 reference setup needs a multi-rank world to be meaningful; without a
# distributed launcher the fixture would either fail loudly or worse, hang.
# Reported as SKIPPED with a clear ``reason`` so contributors running the
# unit suite locally see why and how to invoke it properly. The torchrun
# command sets WORLD_SIZE in the environment for every worker process.
_WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
pytestmark = [
    pytest.mark.internal,
    pytest.mark.skipif(
        _WORLD_SIZE < 2,
        reason=(
            "CP=2 equivalence test requires a multi-rank world; run via "
            "``torchrun --nproc_per_node=2 -m pytest ... -m internal``."
        ),
    ),
]


# Pack shapes used to parametrize both forward and backward equivalence tests.
# Each segment length must be a multiple of ``2 * cp_size = 4`` so that
# ``tex.thd_get_partitioned_indices`` can split the pack evenly across CP
# ranks (mirrors ``sft_dataset.py``'s pad_granularity).
PACK_SHAPES = [
    pytest.param([16, 8, 24], id="headline"),
    pytest.param([48], id="single-long"),
    pytest.param([8, 8, 8, 8, 8, 8], id="many-short"),
    pytest.param([4, 20, 12, 12], id="mixed-short-long"),
    pytest.param([40, 4, 4], id="head-heavy"),
    pytest.param([4, 4, 40], id="tail-heavy"),
]


def _make_packed_seq_params(seq_lens: List[int]) -> PackedSeqParams:
    """Build a PackedSeqParams for a single THD pack with these segment lengths."""
    cu = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(seq_lens), 0).tolist()),
        dtype=torch.int32,
        device="cuda",
    )
    total = int(cu[-1].item())
    return PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu,
        cu_seqlens_kv=cu,
        cu_seqlens_q_padded=None,
        cu_seqlens_kv_padded=None,
        max_seqlen_q=total,
        max_seqlen_kv=total,
    )


def _make_config(cp_size: int) -> TransformerConfig:
    """Small-but-shape-valid TransformerConfig for the v4 GDP mixer."""
    return TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=4,
        ffn_hidden_size=128,
        normalization="RMSNorm",
        bf16=True,
        mamba_num_heads=4,
        mamba_head_dim=16,
        mamba_num_groups=4,
        mamba_state_dim=16,
        tensor_model_parallel_size=1,
        sequence_parallel=False,
        context_parallel_size=cp_size,
    )


def _build_mixer(cp_group):
    """Construct a v4 GDP mixer wired to the given CP group."""
    config = _make_config(cp_group.size())
    pg = ProcessGroupCollection(tp=parallel_state.get_tensor_model_parallel_group(), cp=cp_group)
    submodules = GatedDeltaProductMixerSubmodules(
        in_proj=TELayerNormColumnParallelLinear, out_proj=TERowParallelLinear
    )
    mixer = GatedDeltaProductMixer(
        config=config,
        submodules=submodules,
        d_model=config.hidden_size,
        layer_number=1,
        pg_collection=pg,
        name="decoder.layers.0.mixer",
    )
    return mixer.cuda().bfloat16(), config


def _sync_weights_from_rank0(mixer):
    """Broadcast every parameter from global rank 0 so all ranks share weights."""
    for p in mixer.parameters():
        torch.distributed.broadcast(p.data, src=0)


def _build_cp_pair():
    """Build CP=2 + per-rank CP=1 reference mixers with identical weights.

    Returns ``(mixer_cp2, mixer_cp1, config, cp_group, cp_rank)``. The cp=1
    instance lives in a 1-rank subgroup (containing only this rank), so the
    same mixer code path runs in cp=1 mode and provides a numerical reference.
    """
    cp_group = parallel_state.get_context_parallel_group()
    cp_rank = parallel_state.get_context_parallel_rank()
    global_rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    cp1_groups = [torch.distributed.new_group(ranks=[r]) for r in range(world_size)]
    cp1_group = cp1_groups[global_rank]

    mixer_cp2, _ = _build_mixer(cp_group)
    mixer_cp1, config = _build_mixer(cp1_group)
    _sync_weights_from_rank0(mixer_cp2)
    mixer_cp1.load_state_dict(mixer_cp2.state_dict())
    return mixer_cp2, mixer_cp1, config, cp_group, cp_rank


def _make_hidden_packed(seq_lens, hidden_size):
    """Build a packed [total_tokens, 1, hidden] input + matching PackedSeqParams.

    The tensor is broadcast from rank 0 so cp=1 reference and cp=2 sliced
    paths see bit-identical input.
    """
    psp = _make_packed_seq_params(seq_lens)
    total_tokens = sum(seq_lens)
    torch.manual_seed(0)
    hidden_full = torch.randn(total_tokens, 1, hidden_size, device="cuda", dtype=torch.bfloat16)
    torch.distributed.broadcast(hidden_full, src=0)
    return hidden_full, psp


@pytest.mark.internal
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA + NCCL")
@pytest.mark.skipif(
    torch.cuda.device_count() < 2 if torch.cuda.is_available() else True,
    reason="CP=2 test requires at least 2 GPUs",
)
@pytest.mark.skipif(not HAVE_MAMBA_DEPS, reason="GDP mixer requires mamba_ssm + einops")
@pytest.mark.skipif(not HAVE_FLA, reason="GDP mixer requires fla")
class TestGDPPackedSequence:
    """v4 GDP forward + backward equivalence under CP=2 with packed (THD) input."""

    @pytest.fixture(autouse=True)
    def setup_method(self):
        """Initialize TP=1 PP=1 CP=2 model parallel state for every test."""
        Utils.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1, context_parallel_size=2
        )
        model_parallel_cuda_manual_seed(123)
        yield
        Utils.destroy_model_parallel()

    @pytest.mark.parametrize("seq_lens", PACK_SHAPES)
    def test_forward_equivalence(self, seq_lens):
        """CP=2 forward output (sliced+gathered) matches CP=1 reference on the
        same packed input to bf16 tolerance.
        """
        import transformer_engine_torch as tex

        mixer_cp2, mixer_cp1, config, cp_group, cp_rank = _build_cp_pair()
        mixer_cp2.eval()
        mixer_cp1.eval()

        hidden_full, psp = _make_hidden_packed(seq_lens, config.hidden_size)
        total_tokens = hidden_full.shape[0]

        with torch.no_grad():
            ref_out = mixer_cp1(hidden_full, packed_seq_params=psp)
        ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        assert ref_out.shape == hidden_full.shape

        idx = tex.thd_get_partitioned_indices(
            psp.cu_seqlens_q, total_tokens, cp_group.size(), cp_rank
        )
        hidden_local = hidden_full.index_select(0, idx)

        with torch.no_grad():
            cp2_out_local = mixer_cp2(hidden_local, packed_seq_params=psp)
        cp2_out_local = cp2_out_local[0] if isinstance(cp2_out_local, tuple) else cp2_out_local

        # Scatter the per-rank slice back to its original positions, then
        # all-reduce(SUM) across the CP group to reconstruct the full output.
        cp2_out_full = torch.zeros_like(ref_out)
        scatter_index = idx.long().view(-1, 1, 1).expand_as(cp2_out_local)
        cp2_out_full.scatter_(0, scatter_index, cp2_out_local)
        torch.distributed.all_reduce(cp2_out_full, group=cp_group)

        torch.testing.assert_close(cp2_out_full, ref_out, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("seq_lens", PACK_SHAPES)
    def test_split_compute_interface_matches_forward(self, seq_lens):
        """GDP's shortcut-MoE split interface preserves its regular forward output."""
        _, mixer, config, _, _ = _build_cp_pair()
        assert mixer.supports_split_output_projection()
        mixer.eval()
        hidden_full, psp = _make_hidden_packed(seq_lens, config.hidden_size)

        with torch.no_grad():
            forward_output = mixer(hidden_full, packed_seq_params=psp)
            split_output = mixer.forward_output_proj(
                mixer.forward_pre_output_proj(hidden_full, packed_seq_params=psp)
            )

        forward_output = forward_output[0] if isinstance(forward_output, tuple) else forward_output
        split_output = split_output[0] if isinstance(split_output, tuple) else split_output
        torch.testing.assert_close(split_output, forward_output, atol=5e-2, rtol=5e-2)

    @pytest.mark.parametrize("seq_lens", PACK_SHAPES)
    def test_backward_equivalence(self, seq_lens):
        """CP=2 parameter gradients (after all-reduce across CP) match CP=1
        reference gradients on the same packed input.

        Mechanics: weights are replicated across CP ranks, so for any scalar
        loss ``L_full`` computed over the full output,
        ``dL_full/dW = sum_{r in cp_ranks} dL_local/dW`` where ``L_local`` is
        the same loss restricted to the rank's local output slice.

        Loss = ``out.float().pow(2).sum()`` so every output element
        contributes — exercises every weight that feeds the output.
        Tolerance is looser than the forward test: bf16 backward accumulates
        error through the chain of matmuls and the all-to-all forward+backward
        in CP.
        """
        import transformer_engine_torch as tex

        mixer_cp2, mixer_cp1, config, cp_group, cp_rank = _build_cp_pair()
        mixer_cp2.eval()
        mixer_cp1.eval()
        for p in mixer_cp1.parameters():
            p.grad = None
        for p in mixer_cp2.parameters():
            p.grad = None

        hidden_full, psp = _make_hidden_packed(seq_lens, config.hidden_size)
        total_tokens = hidden_full.shape[0]

        # CP=1 reference: full-sequence forward + backward.
        ref_out = mixer_cp1(hidden_full, packed_seq_params=psp)
        ref_out = ref_out[0] if isinstance(ref_out, tuple) else ref_out
        ref_loss = ref_out.float().pow(2).sum()
        ref_loss.backward()

        # CP=2: local slice forward + backward. ``L_full = sum_{r in cp_ranks} L_local``
        # by construction (sum of squares is additive across token partitions),
        # so the all-reduced grad equals the reference grad up to bf16 noise.
        idx = tex.thd_get_partitioned_indices(
            psp.cu_seqlens_q, total_tokens, cp_group.size(), cp_rank
        )
        hidden_local = hidden_full.index_select(0, idx)
        out_local = mixer_cp2(hidden_local, packed_seq_params=psp)
        out_local = out_local[0] if isinstance(out_local, tuple) else out_local
        loss_local = out_local.float().pow(2).sum()
        loss_local.backward()

        cp1_params = dict(mixer_cp1.named_parameters())
        cp2_params = dict(mixer_cp2.named_parameters())
        assert set(cp1_params) == set(cp2_params)

        mismatches = []
        n_compared = 0
        for name in sorted(cp1_params):
            g1 = cp1_params[name].grad
            g2 = cp2_params[name].grad
            if g1 is None and g2 is None:
                continue
            assert g1 is not None, f"cp=1 has no grad for {name} but cp=2 does"
            assert g2 is not None, f"cp=2 has no grad for {name} but cp=1 does"
            g2_reduced = g2.clone().contiguous()
            torch.distributed.all_reduce(g2_reduced, group=cp_group)
            try:
                torch.testing.assert_close(g2_reduced, g1, atol=8e-2, rtol=8e-2)
                n_compared += 1
            except AssertionError as e:
                mismatches.append((name, tuple(g1.shape), str(e).splitlines()[0]))
        assert not mismatches, (
            f"{len(mismatches)} parameter(s) mismatched out of "
            f"{n_compared + len(mismatches)} compared:\n"
            + "\n".join(f"  {n} {s}: {m}" for n, s, m in mismatches)
        )
        assert n_compared > 0, "no parameters received a gradient — test setup is wrong"
