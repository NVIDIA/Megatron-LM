"""
End-to-end test: BagelMimoModel vs original Bagel (Qwen2Model) accuracy.

Tests run in three groups:

  1. get_packed_seq_params correctness (CP=1 and CP=2):
       Verify that Lund/Lgen, local slices, and padding produced by
       BagelMimoModel.get_packed_seq_params match manual computation.

  2. Forward accuracy vs Qwen2Model (CP=1):
       BagelMimoModel.forward with identity-RoPE matches
       Qwen2Model.forward_train on the same packed_sequence, for
       text-only and mixed (text + gen tokens) layouts.

  3. CP=2 parity (requires nproc_per_node=2):
       Per-rank BagelMimoModel.forward output matches slices of the
       CP=1 reference run (same model, A2A nulled out).

Usage
-----
# accuracy tests only (single GPU):
torchrun --nproc_per_node=1 test_bagel_mimo.py

# accuracy + CP=2 parity:
torchrun --nproc_per_node=2 test_bagel_mimo.py
"""

import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import create_block_mask

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC = os.path.join(_BAGEL_PKG, "bagel")

for p in [_ROOT, _BAGEL_PKG, _BAGEL_SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

from megatron.core.models.bagel.attention_mot import (
    SelfAttentionMoT,
    SelfAttentionMoTSubmodules,
)
from megatron.core.models.bagel.transformer_mot_layer import (
    MoTTransformerLayer,
    MoTTransformerLayerSubmodules,
)
from megatron.core.models.bagel.flex_attention import FlexAttention
from megatron.core.models.bagel.mot_packed_seq_params import MoTPackedSeqParams
from megatron.core.models.bagel.transformer_mot_block import (
    TransformerMoTBlock,
    TransformerMoTBlockSubmodules,
)
from megatron.core.models.bagel.mcore_bagel_llm import BagelMCoreModel
from megatron.core.models.bagel.bagel_mimo import BagelMimoModel

# ── Optional bagel-package imports ───────────────────────────────────────────
try:
    from bagel.modeling.bagel.qwen2_navit import (
        Qwen2Model,
        Qwen2MoTDecoderLayer,
        Qwen2Config as BagelQwen2Config,
    )
    HAVE_BAGEL_PKG = True
except ImportError:
    HAVE_BAGEL_PKG = False

try:
    from megatron.core.transformer.torch_norm import WrappedTorchNorm
    HAVE_WRAPPED_NORM = True
except ImportError:
    HAVE_WRAPPED_NORM = False

from megatron.core import parallel_state
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# ─────────────────────────────────────────────────────────────────────────────
# Test dimensions (same as test_mcore_bagel_llm.py)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE     = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS       = 4
NUM_KV_HEADS    = 4
HEAD_DIM        = HIDDEN_SIZE // NUM_HEADS
NUM_LAYERS      = 2
ROPE_THETA      = 10000.0
VOCAB_SIZE      = 256
MAX_SEQ_LEN     = 128

T_CLEAN, V_CLEAN, G_CLEAN = 8, 8, 16   # S=32, U=16 (V=0 in this test; ViT bypassed)
T_PAD,   V_PAD,   G_PAD   = 7, 0, 11   # S=18, U=7  (text-only und branch)


# ─────────────────────────────────────────────────────────────────────────────
# Config helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_mcore_config() -> TransformerConfig:
    return TransformerConfig(
        num_layers=NUM_LAYERS,
        hidden_size=HIDDEN_SIZE,
        ffn_hidden_size=FFN_HIDDEN_SIZE,
        num_attention_heads=NUM_HEADS,
        num_query_groups=NUM_KV_HEADS,
        kv_channels=HEAD_DIM,
        add_bias_linear=False,
        add_qkv_bias=True,
        normalization="RMSNorm",
        layernorm_epsilon=1e-6,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        gated_linear_unit=True,
        activation_func=F.silu,
        bias_dropout_fusion=False,
        fp16=True,
        params_dtype=torch.float16,
        use_cpu_initialization=True,
        apply_rope_fusion=False,
    )


def _make_bagel_ref_config() -> "BagelQwen2Config":
    cfg = BagelQwen2Config()
    cfg.torch_dtype           = torch.float16
    cfg.hidden_size           = HIDDEN_SIZE
    cfg.intermediate_size     = FFN_HIDDEN_SIZE
    cfg.num_hidden_layers     = NUM_LAYERS
    cfg.num_attention_heads   = NUM_HEADS
    cfg.num_key_value_heads   = NUM_KV_HEADS
    cfg.qk_norm               = True
    cfg.freeze_und            = False
    cfg.rms_norm_eps          = 1e-6
    cfg.hidden_act            = "silu"
    cfg.layer_module          = "Qwen2MoTDecoderLayer"
    cfg.rope_theta            = ROPE_THETA
    cfg.vocab_size            = VOCAB_SIZE
    return cfg


def _make_llm_config():
    class _LLMConfig:
        layer_module = "Qwen2MoTDecoderLayer"
        freeze_und   = False
    return _LLMConfig()


# ─────────────────────────────────────────────────────────────────────────────
# _TestBagelMimoModel — wraps a pre-built BagelMCoreModel for testing
#
# Bypasses MimoModel.__init__ (which requires MimoModelConfig) so we can reuse
# the same BagelMCoreModel instances that are already validated in other tests.
# ─────────────────────────────────────────────────────────────────────────────

class _TestBagelMimoModel(BagelMimoModel):
    """BagelMimoModel that accepts a pre-built BagelMCoreModel directly."""

    def __init__(self, language_model: BagelMCoreModel):
        # Call torch.nn.Module init directly — skip MimoModel.__init__ which
        # needs a full MimoModelConfig with ModuleSpec for the language model.
        torch.nn.Module.__init__(self)
        self.config = language_model.config      # required by MegatronModule
        self.language_model = language_model
        self.modality_submodules = torch.nn.ModuleDict()
        self.special_token_ids = {}

    def forward(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        loss_mask=None,
        labels=None,
        modality_inputs=None,
        sample_lens=None,
        packed_position_ids=None,
        ce_loss_indexes=None,
        packed_label_ids=None,
        sequence_length=None,
        packed_text_indexes=None,
        packed_vit_token_indexes=None,
        packed_vae_token_indexes=None,
        mse_loss_indexes=None,
        vis_gen_target=None,
        split_lens=None,
        attn_modes=None,
        # Extra param for testing: inject visual_latents directly
        visual_latents=None,
    ):
        """Forward pass that allows injecting visual_latents directly for testing."""
        packed_seq_params = None
        if self.language_model.use_mo and packed_text_indexes is not None:
            packed_seq_params = self.get_packed_seq_params(
                packed_text_indexes=packed_text_indexes,
                packed_vit_token_indexes=packed_vit_token_indexes,
                packed_vae_token_indexes=packed_vae_token_indexes,
            )

        lm_output = self.language_model(
            input_ids=input_ids,
            position_ids=position_ids,
            sample_lens=sample_lens,
            attention_mask=attention_mask,
            packed_position_ids=packed_position_ids,
            ce_loss_indexes=ce_loss_indexes,
            packed_label_ids=packed_label_ids,
            sequence_length=sequence_length,
            packed_text_indexes=packed_text_indexes,
            packed_vit_token_indexes=packed_vit_token_indexes,
            packed_vae_token_indexes=packed_vae_token_indexes,
            vision_embeddings=None,
            visual_latents=visual_latents,
            split_lens=split_lens,
            attn_modes=attn_modes,
            packed_seq_params=packed_seq_params,
        )
        lm_output['mse'] = None
        return lm_output, loss_mask


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_bagel_mcore_model(mcore_cfg: TransformerConfig) -> BagelMCoreModel:
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    std_spec = get_gpt_layer_local_spec(normalization="RMSNorm")
    model = BagelMCoreModel(
        config=mcore_cfg,
        transformer_layer_spec=std_spec,
        vocab_size=VOCAB_SIZE,
        max_sequence_length=MAX_SEQ_LEN,
        pre_process=True,
        post_process=True,
        share_embeddings_and_output_weights=True,
        position_embedding_type="rope",
        rotary_base=int(ROPE_THETA),
        rotary_percent=1.0,
        llm_config=_make_llm_config(),
        use_flex_attention=True,
    )
    return model.cuda().half()


def _build_packed_seq(model_cp, input_ids, text_idx, vae_idx, gen_emb, S, device):
    """Build packed_sequence [S, H] with text and gen embeddings placed at their indexes."""
    with torch.no_grad():
        text_emb = model_cp.embedding(
            input_ids=input_ids, position_ids=None
        ).squeeze(1)   # [T, H]
    H = text_emb.shape[-1]
    packed_seq = torch.zeros(S, H, dtype=text_emb.dtype, device=device)
    packed_seq[text_idx] = text_emb.detach()
    if gen_emb is not None and vae_idx is not None and len(vae_idx) > 0:
        packed_seq[vae_idx] = gen_emb
    return packed_seq


def _swap_cp_groups(model: BagelMCoreModel, cp_group, cp_size: int):
    saved = []
    for layer in model.decoder.layers:
        fa = layer.self_attention.core_attention
        saved.append((fa, fa.cp_group, fa.cp_size))
        fa.cp_group = cp_group
        fa.cp_size  = cp_size
    return saved


def _restore_cp_groups(saved):
    for fa, cg, cs in saved:
        fa.cp_group = cg
        fa.cp_size  = cs


# ─────────────────────────────────────────────────────────────────────────────
# Weight-copy helpers
# ─────────────────────────────────────────────────────────────────────────────

def _hf_to_mcore_qkv_weight(q_w, k_w, v_w, ng, np_, hn):
    h  = q_w.shape[1]
    nq = np_ // ng
    q  = q_w.view(ng, nq * hn, h)
    k  = k_w.view(ng, hn, h)
    v  = v_w.view(ng, hn, h)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn, h)


def _hf_to_mcore_qkv_bias(q_b, k_b, v_b, ng, np_, hn):
    nq = np_ // ng
    q  = q_b.view(ng, nq * hn)
    k  = k_b.view(ng, hn)
    v  = v_b.view(ng, hn)
    return torch.cat([q, k, v], dim=1).reshape(ng * (nq + 2) * hn)


def _copy_attn_weights(bagel_attn, mcore_attn):
    np_ = bagel_attn.num_heads
    ng  = bagel_attn.num_key_value_heads
    hn  = bagel_attn.head_dim
    for (q_proj, k_proj, v_proj, o_proj, q_norm, k_norm,
         lin_qkv, lin_proj, qln, kln) in [
        (bagel_attn.q_proj,         bagel_attn.k_proj,         bagel_attn.v_proj,
         bagel_attn.o_proj,         bagel_attn.q_norm,         bagel_attn.k_norm,
         mcore_attn.linear_qkv,    mcore_attn.linear_proj,
         mcore_attn.q_layernorm,   mcore_attn.k_layernorm),
        (bagel_attn.q_proj_moe_gen, bagel_attn.k_proj_moe_gen, bagel_attn.v_proj_moe_gen,
         bagel_attn.o_proj_moe_gen, bagel_attn.q_norm_moe_gen, bagel_attn.k_norm_moe_gen,
         mcore_attn.linear_qkv_gen, mcore_attn.linear_proj_gen,
         mcore_attn.q_layernorm_gen, mcore_attn.k_layernorm_gen),
    ]:
        lin_qkv.weight.data.copy_(
            _hf_to_mcore_qkv_weight(q_proj.weight.data, k_proj.weight.data,
                                     v_proj.weight.data, ng=ng, np_=np_, hn=hn))
        lin_qkv.bias.data.copy_(
            _hf_to_mcore_qkv_bias(q_proj.bias.data, k_proj.bias.data,
                                   v_proj.bias.data, ng=ng, np_=np_, hn=hn))
        lin_proj.weight.data.copy_(o_proj.weight.data)
        if qln is not None and hasattr(q_norm, "weight"):
            qln.weight.data.copy_(q_norm.weight.data)
        if kln is not None and hasattr(k_norm, "weight"):
            kln.weight.data.copy_(k_norm.weight.data)


def _copy_mlp_weights(bagel_mlp, mcore_mlp):
    ffn = bagel_mlp.gate_proj.weight.shape[0]
    mcore_mlp.linear_fc1.weight.data[:ffn].copy_(bagel_mlp.gate_proj.weight.data)
    mcore_mlp.linear_fc1.weight.data[ffn:].copy_(bagel_mlp.up_proj.weight.data)
    mcore_mlp.linear_fc2.weight.data.copy_(bagel_mlp.down_proj.weight.data)


def _copy_layer_weights(bagel_layer, mcore_layer):
    mcore_layer.input_layernorm.weight.data.copy_(bagel_layer.input_layernorm.weight.data)
    mcore_layer.input_layernorm_gen.weight.data.copy_(bagel_layer.input_layernorm_moe_gen.weight.data)
    mcore_layer.pre_mlp_layernorm.weight.data.copy_(bagel_layer.post_attention_layernorm.weight.data)
    mcore_layer.pre_mlp_layernorm_gen.weight.data.copy_(bagel_layer.post_attention_layernorm_moe_gen.weight.data)
    _copy_attn_weights(bagel_layer.self_attn, mcore_layer.self_attention)
    _copy_mlp_weights(bagel_layer.mlp, mcore_layer.mlp)
    _copy_mlp_weights(bagel_layer.mlp_moe_gen, mcore_layer.mlp_gen)


def _copy_block_weights(bagel_model: "Qwen2Model", mcore_block: TransformerMoTBlock):
    for b_layer, m_layer in zip(bagel_model.layers, mcore_block.layers):
        _copy_layer_weights(b_layer, m_layer)
    if mcore_block.final_layernorm is not None:
        mcore_block.final_layernorm.weight.data.copy_(bagel_model.norm.weight.data)
    if mcore_block.final_layernorm_gen is not None:
        mcore_block.final_layernorm_gen.weight.data.copy_(bagel_model.norm_moe_gen.weight.data)


def _copy_all_weights(bagel_model: "Qwen2Model", mcore_llm: BagelMCoreModel):
    """Copy all weights: embedding + decoder layers + final norms."""
    # Embedding: bagel embed_tokens → mcore word_embeddings
    mcore_llm.embedding.word_embeddings.weight.data.copy_(
        bagel_model.embed_tokens.weight.data
    )
    # Decoder layers + final norms
    _copy_block_weights(bagel_model, mcore_llm.decoder)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers (identity-RoPE for deterministic comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _identity_rope(seq_len: int):
    """Zero-frequency RoPE → cos=1, sin=0 (identity rotation)."""
    cos_id = torch.ones (1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    sin_id = torch.zeros(1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    return cos_id, sin_id


def _patch_bagel_rope(bagel_model, cos_id, sin_id):
    class _IdentityRope(torch.nn.Module):
        def forward(self, seq, pos_ids):
            return cos_id, sin_id
    object.__setattr__(bagel_model, "rotary_emb", _IdentityRope())


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_test_data(T: int, G: int, seed: int = 42):
    """
    Build inputs for both BagelMimoModel and Qwen2Model reference.

    Token layout in full sequence [S = T + G]:
      [0 .. T-1]   → text tokens
      [T .. S-1]   → VAE/gen tokens

    Returns:
        input_ids [1, T], text_idx [T], vae_idx [G], pos_ids [S],
        block_mask, gen_emb [G, H]  (random pre-projected gen embeddings)
    """
    torch.manual_seed(seed)
    device = "cuda"
    S = T + G
    H = HIDDEN_SIZE

    input_ids = torch.randint(0, VOCAB_SIZE, (1, T), device=device)
    text_idx  = torch.arange(0, T,   device=device)
    vae_idx   = torch.arange(T, S,   device=device)
    pos_ids   = torch.arange(0, S,   device=device)
    gen_emb   = torch.randn(G, H, dtype=torch.float16, device=device)  # mock vae2llm output

    block_mask = create_block_mask(
        lambda b, h, q, kv: q >= 0,
        B=1, H=1, Q_LEN=S, KV_LEN=S, device=device,
    )
    return input_ids, text_idx, vae_idx, pos_ids, block_mask, gen_emb


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — get_packed_seq_params correctness
# ─────────────────────────────────────────────────────────────────────────────

def test_get_packed_seq_params(cp_size: int, T: int, G: int, label: str):
    """Verify MoTPackedSeqParams fields match manual computation."""
    device = "cuda"
    text_idx = torch.arange(0, T, device=device)
    vae_idx  = torch.arange(T, T + G, device=device)

    cp_rank = parallel_state.get_context_parallel_rank()

    # Build a minimal mimo model (no language model needed for this test)
    class _MinimalMimo(BagelMimoModel):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.modality_submodules = torch.nn.ModuleDict()
            self.special_token_ids   = {}

    mimo = _MinimalMimo()
    psp = mimo.get_packed_seq_params(
        packed_text_indexes=text_idx,
        packed_vit_token_indexes=None,
        packed_vae_token_indexes=vae_idx,
    )

    U = T  # no ViT in this test
    Lund_expected = math.ceil(U / cp_size) if U > 0 else 0
    Lgen_expected = math.ceil(G / cp_size) if G > 0 else 0

    assert psp.padded_und_seqlen == Lund_expected, \
        f"[{label}] Lund: got {psp.padded_und_seqlen}, expected {Lund_expected}"
    assert psp.padded_gen_seqlen == Lgen_expected, \
        f"[{label}] Lgen: got {psp.padded_gen_seqlen}, expected {Lgen_expected}"
    assert len(psp.packed_und_token_indexes) == U, \
        f"[{label}] global und length mismatch"
    assert len(psp.packed_gen_token_indexes) == G, \
        f"[{label}] global gen length mismatch"

    # local_*_token_indexes stores the actual (unpadded) slice; len() == real token count
    actual_lund = min(Lund_expected, max(0, U - cp_rank * Lund_expected))
    actual_lgen = min(Lgen_expected, max(0, G - cp_rank * Lgen_expected))

    expected_local_und = text_idx[cp_rank * Lund_expected : cp_rank * Lund_expected + actual_lund]
    expected_local_gen = vae_idx [cp_rank * Lgen_expected : cp_rank * Lgen_expected + actual_lgen]
    assert len(psp.local_und_token_indexes) == actual_lund, \
        f"[{label}] local_und length mismatch: got {len(psp.local_und_token_indexes)}, expected {actual_lund}"
    assert len(psp.local_gen_token_indexes) == actual_lgen, \
        f"[{label}] local_gen length mismatch: got {len(psp.local_gen_token_indexes)}, expected {actual_lgen}"
    assert torch.equal(psp.local_und_token_indexes, expected_local_und), \
        f"[{label}] local_und tokens mismatch"
    assert torch.equal(psp.local_gen_token_indexes, expected_local_gen), \
        f"[{label}] local_gen tokens mismatch"

    print(f"  [psp {label:12s} cp={cp_size} rank={cp_rank}] PASS  "
          f"U={U} G={G}  Lund={Lund_expected} Lgen={Lgen_expected}  "
          f"actual=({actual_lund},{actual_lgen})")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — BagelMimoModel.forward accuracy vs Qwen2Model (CP=1)
#
# Uses identity-RoPE on both sides so the decoder outputs match exactly.
# Compares last_hidden_state at all valid (und + gen) token positions.
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_vs_qwen2(T: int, G: int, label: str):
    assert HAVE_BAGEL_PKG,    "skip: bagel-package not available"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"
    device = "cuda"

    input_ids, text_idx, vae_idx, pos_ids, block_mask, gen_emb = \
        _make_test_data(T, G)

    U = T  # no ViT
    S = T + G

    # ── Build models ──────────────────────────────────────────────────────────
    bagel_cfg   = _make_bagel_ref_config()
    mcore_cfg   = _make_mcore_config()
    bagel_model = Qwen2Model(bagel_cfg).cuda().half().train()
    mcore_llm   = _build_bagel_mcore_model(mcore_cfg)
    mcore_llm.train()

    # Copy weights: embedding + transformer layers + final norms
    _copy_all_weights(bagel_model, mcore_llm)

    # ── Identity RoPE ─────────────────────────────────────────────────────────
    seq_len_for_rope = U + G  # compact length
    cos_id, sin_id = _identity_rope(seq_len_for_rope)
    _patch_bagel_rope(bagel_model, cos_id, sin_id)
    # BagelRotaryEmbedding is already replaced; we force cos=1, sin=0 by
    # overriding inv_freq to zeros (outer product → all-zero → cos=1, sin=0)
    mcore_llm.rotary_pos_emb.inv_freq.zero_()

    # ── Reference: Qwen2Model.forward_train ──────────────────────────────────
    # Build packed_sequence manually: text_emb at text_idx, gen_emb at vae_idx
    with torch.no_grad():
        text_emb = bagel_model.embed_tokens(input_ids.squeeze(0))   # [T, H]
    packed_seq = torch.zeros(S, HIDDEN_SIZE, dtype=torch.float16, device=device)
    packed_seq[text_idx] = text_emb.detach()
    if G > 0:
        packed_seq[vae_idx] = gen_emb

    und_idx = text_idx
    gen_idx = vae_idx if G > 0 else torch.zeros(0, dtype=torch.long, device=device)

    with torch.no_grad():
        ref_out = bagel_model.forward_train(
            packed_sequence=packed_seq,
            sample_lens=[S],
            attention_mask=block_mask,   # FlexAttention BlockMask (not list → flex path)
            packed_position_ids=pos_ids,
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=gen_idx,
        )  # [S, H]: und positions normed by und LN, gen by gen LN

    # ── BagelMimoModel: build mimo wrapper and call forward ───────────────────
    mimo = _TestBagelMimoModel(mcore_llm)
    with torch.no_grad():
        lm_output, _ = mimo.forward(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=block_mask,
            packed_position_ids=pos_ids,
            sequence_length=S,
            packed_text_indexes=text_idx,
            packed_vit_token_indexes=None,
            packed_vae_token_indexes=vae_idx if G > 0 else None,
            visual_latents=gen_emb if G > 0 else None,
        )
    got = lm_output['last_hidden_state']   # [S, H]

    # ── Compare at valid token positions ─────────────────────────────────────
    atol = rtol = 1e-2
    valid_idx = torch.cat([und_idx, gen_idx]) if G > 0 else und_idx

    torch.testing.assert_close(
        got[valid_idx], ref_out[valid_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[mimo_vs_qwen2/{label}] {m}",
    )
    max_err = (got[valid_idx] - ref_out[valid_idx]).abs().max().item()
    print(f"  [mimo_vs_qwen2 {label:10s}] PASS  U={U} G={G}  max_err={max_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — CP=2 parity through BagelMimoModel (requires 2 processes)
#
# Each rank calls BagelMimoModel.forward with the full (shared) inputs.
# BagelMimoModel.get_packed_seq_params() automatically shards for this rank.
# Per-rank output is compared against slices of the CP=1 reference.
# ─────────────────────────────────────────────────────────────────────────────

def run_cp2_parity_mimo(model_cp: BagelMCoreModel, T: int, G: int,
                        cp_group, label: str, seed: int = 42):
    rank    = dist.get_rank()
    device  = "cuda"
    S       = T + G
    cp_size = cp_group.size()

    input_ids, text_idx, vae_idx, pos_ids, block_mask, gen_emb = \
        _make_test_data(T, G, seed=seed)

    U     = T
    Lund  = math.ceil(U / cp_size) if U > 0 else 0
    Lgen  = math.ceil(G / cp_size) if G > 0 else 0
    actual_lund = min(Lund, max(0, U - rank * Lund))
    actual_lgen = min(Lgen, max(0, G - rank * Lgen))

    und_idx = text_idx
    gen_idx = vae_idx if G > 0 else torch.zeros(0, dtype=torch.long, device=device)

    # ── Build packed_seq_params for both CP=1 ref and CP=N run ───────────────
    # psp_cp1: full sequence (Lund=U, Lgen=G), used for the reference
    psp_cp1 = MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx,
        packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=und_idx,
        local_gen_token_indexes=gen_idx,
        padded_und_seqlen=U,
        padded_gen_seqlen=G,
    )

    # psp_cpN: this rank's local slice (built by mimo.get_packed_seq_params)
    mimo = _TestBagelMimoModel(model_cp)
    psp_cpN = mimo.get_packed_seq_params(
        packed_text_indexes=text_idx,
        packed_vit_token_indexes=None,
        packed_vae_token_indexes=vae_idx if G > 0 else None,
    )

    # ── CP=1 reference: call model directly with psp_cp1, no A2A ─────────────
    # Build the packed_sequence (same as mcore_bagel_llm.forward would build it)
    packed_seq = _build_packed_seq(model_cp, input_ids, text_idx, vae_idx,
                                   gen_emb if G > 0 else None, S, device)
    saved = _swap_cp_groups(model_cp, cp_group=None, cp_size=1)
    with torch.no_grad():
        ref_lm = model_cp.forward(
            input_ids=input_ids,
            packed_position_ids=pos_ids,
            sequence_length=S,
            packed_text_indexes=text_idx,
            packed_vit_token_indexes=None,
            packed_vae_token_indexes=vae_idx if G > 0 else None,
            visual_latents=gen_emb if G > 0 else None,
            attention_mask=block_mask,
            packed_seq_params=psp_cp1,
        )
    _restore_cp_groups(saved)
    ref_hs = ref_lm['last_hidden_state']   # [S, H]

    # ── CP=N run (A2A active via cp_group) ────────────────────────────────────
    with torch.no_grad():
        cpN_lm = model_cp.forward(
            input_ids=input_ids,
            packed_position_ids=pos_ids,
            sequence_length=S,
            packed_text_indexes=text_idx,
            packed_vit_token_indexes=None,
            packed_vae_token_indexes=vae_idx if G > 0 else None,
            visual_latents=gen_emb if G > 0 else None,
            attention_mask=block_mask,
            packed_seq_params=psp_cpN,
        )
    cpN_hs = cpN_lm['last_hidden_state']   # [S, H] — only local positions filled

    dist.barrier()

    # ── Compare at this rank's local token positions ──────────────────────────
    local_und = text_idx[rank * Lund : rank * Lund + actual_lund]
    local_gen = gen_idx [rank * Lgen : rank * Lgen + actual_lgen]

    atol = rtol = 1e-2
    und_err = gen_err = 0.0
    if actual_lund > 0:
        torch.testing.assert_close(
            cpN_hs[local_und], ref_hs[local_und], atol=atol, rtol=rtol,
            msg=lambda m: f"[cp2_mimo/{label} rank={rank}] UND: {m}")
        und_err = (cpN_hs[local_und] - ref_hs[local_und]).abs().max().item()
    if actual_lgen > 0:
        torch.testing.assert_close(
            cpN_hs[local_gen], ref_hs[local_gen], atol=atol, rtol=rtol,
            msg=lambda m: f"[cp2_mimo/{label} rank={rank}] GEN: {m}")
        gen_err = (cpN_hs[local_gen] - ref_hs[local_gen]).abs().max().item()

    print(f"  [cp2_mimo {label:8s} rank={rank}] PASS  "
          f"U={U} G={G}  Lund={Lund} Lgen={Lgen}  "
          f"actual=({actual_lund},{actual_lgen})  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)

    Utils.initialize_model_parallel(1, 1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    mcore_cfg = _make_mcore_config()

    # ── Test 1: get_packed_seq_params ─────────────────────────────────────────
    if rank == 0:
        print("\n=== Test 1: get_packed_seq_params ===")
    test_get_packed_seq_params(cp_size=1, T=T_CLEAN, G=G_CLEAN, label="clean")
    test_get_packed_seq_params(cp_size=1, T=T_PAD,   G=G_PAD,   label="padding")
    test_get_packed_seq_params(cp_size=1, T=T_CLEAN, G=0,       label="und-only")
    test_get_packed_seq_params(cp_size=1, T=0,       G=G_CLEAN, label="gen-only")

    # ── Test 2: forward accuracy vs Qwen2Model ────────────────────────────────
    if HAVE_BAGEL_PKG and HAVE_WRAPPED_NORM:
        if rank == 0:
            print("\n=== Test 2: BagelMimoModel.forward vs Qwen2Model ===")
        test_forward_vs_qwen2(T=T_CLEAN, G=G_CLEAN, label="mixed")
        test_forward_vs_qwen2(T=T_CLEAN, G=0,       label="und-only")
        test_forward_vs_qwen2(T=0,       G=G_CLEAN, label="gen-only")
    else:
        if rank == 0:
            print("\n=== Test 2: SKIPPED (bagel-package not available) ===")

    # ── Test 3: CP=2 parity through BagelMimoModel ───────────────────────────
    if world_size >= 2:
        if rank == 0:
            print("\n=== Test 3: CP=2 parity via BagelMimoModel ===")
        Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
        model_parallel_cuda_manual_seed(7)

        # Re-run get_packed_seq_params with cp=2
        if rank == 0:
            print("  [psp CP=2 check]")
        test_get_packed_seq_params(cp_size=2, T=T_CLEAN, G=G_CLEAN, label="clean")
        test_get_packed_seq_params(cp_size=2, T=T_PAD,   G=G_PAD,   label="padding")

        model_cp  = _build_bagel_mcore_model(mcore_cfg)
        model_cp.train()
        cp_group  = dist.new_group(list(range(world_size)))

        run_cp2_parity_mimo(model_cp, T=T_CLEAN, G=G_CLEAN, cp_group=cp_group, label="clean",   seed=42)
        run_cp2_parity_mimo(model_cp, T=T_PAD,   G=G_PAD,   cp_group=cp_group, label="padding", seed=77)

        dist.barrier()
        if rank == 0:
            print("  [CP=2 parity] PASS  all ranks agree")
    else:
        if rank == 0:
            print("\n=== Test 3: CP=2 parity — SKIPPED (need nproc_per_node=2) ===")

    dist.barrier()
    if rank == 0:
        print("\nAll tests passed.")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
