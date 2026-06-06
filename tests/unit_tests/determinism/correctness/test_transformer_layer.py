# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism check for a ``TransformerBlock`` (stack of TransformerLayers).

Using a stack instead of a single layer gives the runner something PP / VPP
can actually split — every chunk is a uniform hidden-state in/out module
(no embedding / no logits asymmetry), which keeps the pipeline schedule
happy and is enough to exercise the per-chunk grad determinism path.

Three sub-tests:

1. ``test_bit_exact_under_parallelism`` — runner-driven, covers every entry
   in the filtered parallelism matrix (TP / EP / FSDP and composites; PP
   composites are covered by ``test_gpt_model``).
2. ``test_bit_exact_under_racing_streams`` — TP=4 + side-stream contention.
   ``skipif(CUDA_DEVICE_MAX_CONNECTIONS=='1')`` because side streams can't
   actually race when serialised through a single hardware queue (Hopper
   determinism env); fires on Blackwell when the launcher sets ``=32``.
3. ``test_bit_exact_under_jitter`` — TP=4 + cuda._sleep jitter. Perturbs
   per-submodule launch timing on the default stream, so it stresses
   cross-rank NCCL race ordering even under ``CUDA_DEVICE_MAX_CONNECTIONS=1``.
"""

import os

import pytest
import torch

from megatron.core.enums import ModelType
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_layer_with_transformer_engine_spec,
)
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import GPT_CONFIGS, gpt_base, parallelism_configs

# Layer-stack determinism: drop PP composites — the full PP path (embedding +
# block + logits through the schedule) is exhaustively covered by
# ``test_gpt_model``. Keep TP / EP / FSDP cells that exercise the layer's
# own parallelism plumbing.
_LAYER_PARALLELISM_CONFIGS = parallelism_configs(exclude=("pp2", "pp4", "tp2-pp2", "pp2-vpp2"))
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.utils import (
    CudaSleepJitter,
    RacingStreams,
    assert_bit_exact,
    capture_rng_state,
    restore_rng_state,
)
from tests.unit_tests.test_utilities import Utils

_SEQ_LEN = 32
_MICRO_BATCH = 2
_DTYPE = torch.bfloat16


class _LayerStackWrapper(torch.nn.Module):
    """Thin wrapper around ``TransformerBlock`` for the bit-exact runner.

    Mirrors ``GPTModel.set_input_tensor`` semantics: the pipeline schedule
    always wraps the recv'd activation in a list before calling
    ``set_input_tensor`` (schedules.py:424-425), and the underlying
    ``TransformerBlock.set_input_tensor`` stores whatever it gets verbatim.
    Without unwrapping, ``self.input_tensor`` ends up as a list and the
    block's forward path uses a list as ``hidden_states`` — which mis-shapes
    the next P2P send and hangs NCCL.

    This wrapper unwraps the list (like GPTModel does) before delegating to
    the block, so PP fwd+bwd works end-to-end.
    """

    model_type = ModelType.encoder_or_decoder

    def __init__(self, block: TransformerBlock):
        super().__init__()
        self.block = block
        self.config = block.config
        self.pre_process = block.pre_process
        self.post_process = block.post_process

    def set_input_tensor(self, input_tensor):
        if isinstance(input_tensor, (list, tuple)):
            assert len(input_tensor) == 1
            input_tensor = input_tensor[0]
        self.block.set_input_tensor(input_tensor)

    def set_is_first_microbatch(self):
        # The pipeline schedule calls ``set_is_first_microbatch`` on the
        # top-level model (schedules.py guards with ``hasattr``) so TE's
        # per-iteration amax/scale recompute path fires at the start of
        # every iteration. Without this forwarder, the schedule's
        # ``hasattr`` returns False and the inner TransformerBlock never
        # gets the signal — silently exercising a non-production path.
        fn = getattr(self.block, "set_is_first_microbatch", None)
        if fn is not None:
            fn()

    def forward(self, hidden_states=None, attention_mask=None, **kwargs):
        return self.block(hidden_states=hidden_states, attention_mask=attention_mask)


def _build_layer(
    overrides: dict,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
):
    """Build a ``TransformerBlock`` — a real stack of TransformerLayers —
    wrapped so PP set_input_tensor list-unwrap matches the schedule contract.

    ``vp_stage`` is forwarded to TransformerBlock for VPP layer slicing.
    """
    cfg_kwargs = gpt_base() | overrides
    cfg_kwargs.setdefault("deterministic_mode", True)
    config = TransformerConfig(**cfg_kwargs)
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=cfg_kwargs.get("num_moe_experts"),
    )
    block = TransformerBlock(
        config=config,
        spec=spec,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    return _LayerStackWrapper(block).cuda().to(_DTYPE)


def _layer_inputs() -> dict:
    """Stack consumes (hidden_states, attention_mask)."""
    hidden_size = gpt_base()["hidden_size"]
    hidden = torch.randn(_SEQ_LEN, _MICRO_BATCH, hidden_size, dtype=_DTYPE, device="cuda")
    hidden.requires_grad_(True)
    mask = torch.ones(1, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda")
    return {"hidden_states": hidden, "attention_mask": mask}


RUNNER = BitExactRunner(
    build_model=_build_layer,
    make_inputs=_layer_inputs,
    base_config=gpt_base,
    # _LayerStackWrapper fixes the set_input_tensor list-unwrap so the
    # pipeline schedule's P2P recv shape matches and PP works end-to-end.
    supports_pp=True,
    seq_len=_SEQ_LEN,
    micro_batch=_MICRO_BATCH,
)


# ---------------------------------------------------------------------------
# Helpers reused by the specialty (perf/racing/jitter) sub-tests below.
# ---------------------------------------------------------------------------
def _fwd_bwd(layer, hidden, mask):
    out = layer(hidden_states=hidden, attention_mask=mask)
    # TransformerBlock returns a tensor; some layer types return a tuple.
    out = out[0] if isinstance(out, tuple) else out
    loss = out.float().pow(2).mean()
    loss.backward()
    grads = {
        name: p.grad.detach().clone()
        for name, p in layer.named_parameters()
        if p.grad is not None
    }
    return out.detach().clone(), grads


def _make_inputs():
    hidden_size = gpt_base()["hidden_size"]
    hidden = torch.randn(_SEQ_LEN, _MICRO_BATCH, hidden_size, dtype=_DTYPE, device="cuda")
    hidden.requires_grad_(True)
    mask = torch.ones(1, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda")
    return hidden, mask


def _run_twice_with_state_capture(layer):
    state = capture_rng_state()
    h_a, m_a = _make_inputs()
    out_a, g_a = _fwd_bwd(layer, h_a, m_a)
    restore_rng_state(state)
    layer.zero_grad(set_to_none=True)
    h_b, m_b = _make_inputs()
    out_b, g_b = _fwd_bwd(layer, h_b, m_b)
    return out_a, g_a, out_b, g_b


class TestTransformerLayerDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", _LAYER_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)

    @pytest.mark.internal
    @pytest.mark.skipif(
        os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS", "1") == "1",
        reason=(
            "RacingStreams is a no-op when CUDA_DEVICE_MAX_CONNECTIONS=1 "
            "— all CUDA streams serialise through one hardware queue so the "
            "side-stream GEMMs cannot run concurrently with the model's "
            "default-stream fwd/bwd. Effective on Blackwell where the =1 "
            "requirement was dropped (arguments.py:1299) and the launcher "
            "leaves the value at the multi-queue default (e.g. =32)."
        ),
    )
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_racing_streams(self, cfg_overrides):
        # NB: CUDA_DEVICE_MAX_CONNECTIONS is captured by the driver at CUDA
        # context creation (~import time). Mid-test ``os.environ`` writes
        # are no-ops; setting it to a non-1 value would have to happen in
        # the SLURM launcher / shell before CUDA was first touched. The
        # stress harness relies on the connection count the launcher
        # picked; the launcher pins it to 1 for non-FSDP Hopper which is
        # the conservative determinism value.
        if Utils.world_size < 4:
            pytest.skip("Requires at least 4 GPUs for TP=4 racing-stream test")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=4)
        torch.manual_seed(99)
        model_parallel_cuda_manual_seed(123)
        layer = _build_layer(cfg_overrides)
        state = capture_rng_state()
        with RacingStreams(num_streams=4):
            h_a, m_a = _make_inputs()
            out_a, g_a = _fwd_bwd(layer, h_a, m_a)
        restore_rng_state(state)
        layer.zero_grad(set_to_none=True)
        with RacingStreams(num_streams=4):
            h_b, m_b = _make_inputs()
            out_b, g_b = _fwd_bwd(layer, h_b, m_b)
        assert_bit_exact(out_a, g_a, out_b, g_b)

    @pytest.mark.internal
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_jitter(self, cfg_overrides):
        if Utils.world_size < 4:
            pytest.skip("Requires at least 4 GPUs for TP=4 jitter test")
        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(tensor_model_parallel_size=4)
        torch.manual_seed(2024)
        model_parallel_cuda_manual_seed(123)
        layer = _build_layer(cfg_overrides)
        state = capture_rng_state()
        with RacingStreams(num_streams=4), CudaSleepJitter(layer):
            h_a, m_a = _make_inputs()
            out_a, g_a = _fwd_bwd(layer, h_a, m_a)
        restore_rng_state(state)
        layer.zero_grad(set_to_none=True)
        with RacingStreams(num_streams=4), CudaSleepJitter(layer):
            h_b, m_b = _make_inputs()
            out_b, g_b = _fwd_bwd(layer, h_b, m_b)
        assert_bit_exact(out_a, g_a, out_b, g_b)
