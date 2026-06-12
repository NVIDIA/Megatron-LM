# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Unit tests for the PP / VPP plumbing landed in Phase C and Phase D.

These are *single-process* tests (world_size=1) that exercise the
stage-gating contract added to ``BagelMCoreModel`` and ``BagelMimoModel``
without spinning up real pipeline-parallel communication.  They
specifically guard against the smoking-gun bug we hit in Phase C — the
non-last PP stage returning a 2-D ``[seq, hidden]`` activation while
Megatron's ``_communicate_shapes`` allocates a 3-element shape buffer
expecting 3-D ``[seq, batch, hidden]`` (see
``examples/mimo_bagel/model/mcore_bagel_llm.py:254-263``).

What this module covers:

  * ``test_post_process_true_returns_dict``: at PP=1 (post_process=True),
    ``BagelMCoreModel.forward`` returns a dict with ``last_hidden_state``
    and ``ce``.
  * ``test_post_process_false_returns_3d_tensor``: at non-last PP stage
    (post_process=False), forward returns a 3-D Tensor of shape
    ``[seq, 1, hidden]`` — the contract Megatron's PP shape exchange
    requires.
  * ``test_vp_stage_propagation``: ``vp_stage`` passed to
    ``BagelMCoreModel`` reaches ``GPTModel.vp_stage`` (and thus the
    inner ``TransformerMoTBlock``) — the Phase D plumbing.

Run::

    PYTHONPATH=/workspace/megatron-lm-bage_m4:/workspace/megatron-lm-bage_m4/bagel-package:\
/workspace/megatron-lm-bage_m4/bagel-package/bagel:/workspace/megatron-lm-bage_m4/examples/mimo_bagel \
        WORLD_SIZE=1 LOCAL_RANK=0 RANK=0 MASTER_ADDR=localhost MASTER_PORT=29900 \
        pytest examples/mimo_bagel/unit_test/test_pp_vpp_unit.py -v
"""

import os
import sys
import types

import pytest
import torch
import torch.distributed as dist

_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")
sys.path.insert(0, _ROOT)
sys.path.insert(0, _BAGEL_PKG)
sys.path.insert(0, _BAGEL_SRC)

from megatron.core.models.bagel.mcore_bagel_llm import BagelMCoreModel
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams

from megatron.core.transformer.transformer_config import TransformerConfig

# Tiny config — keeps the model construction cheap. Same scale as
# test_bagel_mimo.py's micro setup so we can reuse the helpers.
HIDDEN_SIZE = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS = 4
NUM_KV_HEADS = 4
NUM_LAYERS = 2
VOCAB_SIZE = 256
MAX_SEQ_LEN = 128
ROPE_THETA = 10000.0


@pytest.fixture(scope="module", autouse=True)
def _init_dist():
    """Initialise a 1-process distributed group for the whole module.

    BagelMCoreModel uses ``ProcessGroupCollection.use_mpu_process_groups()``
    inside ``GPTModel`` which requires Megatron's parallel state to be
    set up. Tear it down at the end so other tests in the same process
    aren't polluted.
    """
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29900")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(backend="nccl", world_size=1, rank=0)
        torch.cuda.set_device(0)
    from megatron.core import parallel_state as mpu
    if not mpu.model_parallel_is_initialized():
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
    # Required by ColumnParallelLinear's init under tensor-parallel — the
    # 'model-parallel-rng' tracker must be set up even at TP=1.
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    model_parallel_cuda_manual_seed(42)
    yield
    # Leave the process group up — tearing down here breaks pytest's
    # cross-test sharing on the same worker.


def _make_mcore_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HIDDEN_SIZE // NUM_HEADS,
        add_bias_linear=False,
        add_qkv_bias=False,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_cpu_initialization=True,
    )


def _make_llm_config():
    """Minimal HF-style config object that BagelMCoreModel requires.

    BagelMCoreModel accesses ``llm_config.layer_module`` (string match against
    'Mo*' to enable MoT) and ``freeze_und``; everything else is unused here
    because we go through the standard (non-MoT) path.
    """
    cfg = types.SimpleNamespace()
    cfg.layer_module = "Qwen2DecoderLayer"  # not 'Mo*' → use_mo=False, no MoT decoder
    cfg.freeze_und = False
    return cfg


def _build_mcore_model(*, post_process: bool, vp_stage=None,
                       virtual_pipeline_model_parallel_size=None) -> BagelMCoreModel:
    """Build a tiny BagelMCoreModel exercising the requested stage flags.

    Uses the standard (non-MoT) GPT layer spec so we don't need to thread
    MoT-specific machinery just to validate the shape contract.
    """
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    cfg = _make_mcore_config()
    if virtual_pipeline_model_parallel_size is not None:
        cfg.virtual_pipeline_model_parallel_size = virtual_pipeline_model_parallel_size
    std_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    model = BagelMCoreModel(
        config=cfg,
        transformer_layer_spec=std_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        pre_process=True,
        post_process=post_process,
        # Use untied output layer so BagelMCoreModel.forward's
        # `self.output_layer.weight` lookup gets a real tensor
        # (BagelMCoreModel doesn't currently route through
        # shared_embedding_or_output_weight when sharing).
        share_embeddings_and_output_weights=False,
        position_embedding_type="rope",
        rotary_base=int(ROPE_THETA),
        rotary_percent=1.0,
        llm_config=_make_llm_config(),
        use_flex_attention=False,
        vp_stage=vp_stage,
    )
    return model.cuda().to(torch.bfloat16).eval()


def _make_inputs(seq_len: int):
    """Build the minimum inputs BagelMCoreModel.forward needs."""
    device = torch.device("cuda")
    text_idx = torch.arange(seq_len, device=device)
    psp = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=text_idx,
        packed_gen_token_indexes=torch.zeros(0, dtype=torch.long, device=device),
        local_und_token_indexes=text_idx,
        local_gen_token_indexes=torch.zeros(0, dtype=torch.long, device=device),
        padded_und_seqlen=seq_len,
        padded_gen_seqlen=0,
    )
    decoder_input = torch.randn(seq_len, 1, HIDDEN_SIZE, dtype=torch.bfloat16, device=device)
    packed_position_ids = text_idx
    labels = torch.zeros(seq_len, dtype=torch.long, device=device)
    loss_mask = torch.ones(seq_len, dtype=torch.float, device=device)
    return dict(
        decoder_input=decoder_input,
        packed_position_ids=packed_position_ids,
        labels=labels,
        loss_mask=loss_mask,
        packed_seq_params=psp,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────


def test_post_process_true_returns_dict():
    """At PP=1 (post_process=True) BagelMCoreModel must return a dict
    with `last_hidden_state` and `ce`. Regression guard for the standard
    PP=1 contract.
    """
    seq_len = 16
    model = _build_mcore_model(post_process=True)
    inputs = _make_inputs(seq_len)
    with torch.no_grad():
        out = model(**inputs)
    assert isinstance(out, dict), f"expected dict at post_process=True, got {type(out).__name__}"
    assert "last_hidden_state" in out and "ce" in out, (
        f"expected keys 'last_hidden_state','ce' in dict, got {list(out.keys())}"
    )
    assert out["last_hidden_state"].dim() == 2, (
        f"last_hidden_state should be compacted [seq, hidden] (2-D), got "
        f"shape {tuple(out['last_hidden_state'].shape)}"
    )


def test_post_process_false_returns_3d_tensor():
    """Phase C smoking-gun fix: at non-last PP stage (post_process=False)
    BagelMCoreModel.forward must return a 3-D Tensor [seq, 1, hidden] so
    Megatron's _communicate_shapes (which hardcodes a 3-element shape
    buffer at p2p_communication.py:194) gets a matching shape.

    See examples/mimo_bagel/model/mcore_bagel_llm.py:254-263.
    """
    seq_len = 16
    model = _build_mcore_model(post_process=False)
    inputs = _make_inputs(seq_len)
    with torch.no_grad():
        out = model(**inputs)
    assert isinstance(out, torch.Tensor), (
        f"expected raw Tensor at post_process=False (so the pipeline schedule "
        f"can ship it as the activation), got {type(out).__name__}"
    )
    assert out.dim() == 3, (
        f"PP send/recv expects 3-D [seq, batch, hidden]; got {out.dim()}-D shape "
        f"{tuple(out.shape)}. Megatron's _communicate_shapes hardcodes a (3,) "
        f"int64 shape buffer (p2p_communication.py:194); a 2-D return causes "
        f"a (2,) vs (3,) mismatch and an indefinite NCCL hang."
    )
    assert out.shape == (seq_len, 1, HIDDEN_SIZE), (
        f"expected shape ({seq_len}, 1, {HIDDEN_SIZE}); got {tuple(out.shape)}"
    )


def test_vp_stage_propagation():
    """Phase D plumbing: vp_stage passed to BagelMCoreModel must reach
    GPTModel.vp_stage (and thus its inner TransformerMoTBlock).
    """
    model = _build_mcore_model(post_process=False, vp_stage=1,
                               virtual_pipeline_model_parallel_size=2)
    assert hasattr(model, "vp_stage"), "GPTModel.vp_stage attribute should exist"
    assert model.vp_stage == 1, f"expected vp_stage=1, got {model.vp_stage!r}"


def test_vp_stage_default_is_none():
    """Default vp_stage is None when no VP is configured (PP=1 or PP=2/VP=1)."""
    model = _build_mcore_model(post_process=True)
    assert getattr(model, "vp_stage", "MISSING") is None, (
        f"expected vp_stage to default to None, got {model.vp_stage!r}"
    )
