"""
BagelMCoreModel._forward_decoder accuracy and CP parity tests.

Two test groups:

  1. Accuracy (CP=1, single process per rank):
       a. Scatter correctness — _forward_decoder on full [S,H] input gives
          *exactly* the same output as model.decoder run directly on the
          manually-compacted [U+G,H] input (with identical RoPE).  Verifies
          that the scatter-in / compact-rearrangement logic is correct.
       b. Qwen2 reference — model.decoder weights (copied to a
          TransformerMoTBlock with BagelMatchingAttention) match Qwen2Model
          for und-only, gen-only, and mixed token layouts.

  2. CP=2 parity (requires nproc_per_node=2):
       _forward_decoder with per-rank MoTPackedSeqParams (local Lund+Lgen
       tokens each) produces outputs that exactly match the corresponding
       slices of the CP=1 reference run.  Tested for both clean splits
       (U and G divisible by 2) and padding cases (odd counts).

Usage
-----
# accuracy tests only (single GPU):
torchrun --nproc_per_node=1 test_mcore_bagel_llm.py

# accuracy + CP=2 parity:
torchrun --nproc_per_node=2 test_mcore_bagel_llm.py
"""

import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.functional import scaled_dot_product_attention as F_sdpa

# ─────────────────────────────────────────────────────────────────────────────
# Path setup
# ─────────────────────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_BAGEL_PKG  = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC  = os.path.join(_BAGEL_PKG, "bagel")

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
from megatron.core.models.bagel.bagel_mimo import align_bagel_embeddings


def _align_and_shard(
    psp: "MoTPackedSeqParams",
    language_model,
    input_ids,
    vision_embeddings,
    visual_latents,
    sequence_length: int,
    packed_text_indexes,
    packed_vit_token_indexes,
    packed_vae_token_indexes,
    packed_position_ids,
    labels_full,
    loss_mask_full,
) -> dict:
    """Wrapper around align_bagel_embeddings that also computes CP-sharded
    labels, loss_mask, and packed_position_ids for the given psp shard."""
    psp.packed_text_indexes      = packed_text_indexes
    psp.packed_vit_token_indexes = packed_vit_token_indexes
    psp.packed_vae_token_indexes = packed_vae_token_indexes

    aligned = align_bagel_embeddings(
        language_model=language_model,
        input_ids=input_ids,
        vision_embeddings=vision_embeddings,
        visual_latents=visual_latents,
        sequence_length=sequence_length,
        packed_seq_params=psp,
    )

    und  = psp.local_und_token_indexes
    gen  = psp.local_gen_token_indexes
    Lund = psp.padded_und_seqlen
    Lgen = psp.padded_gen_seqlen

    def _gp(src, idx, tlen):
        g = src[idx]; n = tlen - len(g)
        return torch.cat([g, g[-1:].expand(n)]) if n > 0 else g

    aligned['labels']    = labels_full[und]    if labels_full    is not None else None
    aligned['loss_mask'] = loss_mask_full[und] if loss_mask_full is not None else None
    aligned['packed_position_ids'] = (
        torch.cat([_gp(packed_position_ids, und, Lund),
                   _gp(packed_position_ids, gen, Lgen)])
        if packed_position_ids is not None else None
    )
    return aligned

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

from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.tensor_parallel.layers import ColumnParallelLinear, RowParallelLinear
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# ─────────────────────────────────────────────────────────────────────────────
# Test dimensions
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE     = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS       = 4
NUM_KV_HEADS    = 4
HEAD_DIM        = HIDDEN_SIZE // NUM_HEADS   # 64
NUM_LAYERS      = 2
ROPE_THETA      = 10000.0
VOCAB_SIZE      = 256
MAX_SEQ_LEN     = 128

# For CP=2 clean split: U=16 (div by 2), G=16 (div by 2)
T_CLEAN, V_CLEAN, G_CLEAN = 8, 8, 16   # S=32, U=16

# For CP=2 padding: U=13 (not div by 2), G=11 (not div by 2)
T_PAD, V_PAD, G_PAD = 7, 6, 11         # S=24, U=13

# For CE tests: no ViT tokens so CE positions are unambiguously within text_idx
T_CE = 8
G_CE_MIX, L_CE_MIX = 16, 4   # mixed:    S=24, U=8, 4 CE positions
G_CE_UND, L_CE_UND =  0, 3   # und-only: S=8,  U=8, 3 CE positions


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
    cfg.torch_dtype      = torch.float16
    cfg.hidden_size      = HIDDEN_SIZE
    cfg.intermediate_size = FFN_HIDDEN_SIZE
    cfg.num_hidden_layers = NUM_LAYERS
    cfg.num_attention_heads = NUM_HEADS
    cfg.num_key_value_heads = NUM_KV_HEADS
    cfg.qk_norm          = True
    cfg.freeze_und       = False
    cfg.rms_norm_eps     = 1e-6
    cfg.hidden_act       = "silu"
    cfg.layer_module     = "Qwen2MoTDecoderLayer"
    cfg.rope_theta       = ROPE_THETA
    cfg.vocab_size       = VOCAB_SIZE
    return cfg


def _make_llm_config():
    """Minimal llm_config to enable use_mo=True in BagelMCoreModel."""
    class _LLMConfig:
        layer_module = "Qwen2MoTDecoderLayer"
        freeze_und   = False
    return _LLMConfig()


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_bagel_mcore_model(mcore_cfg: TransformerConfig) -> BagelMCoreModel:
    """Build a small BagelMCoreModel (FlexAttention, cp picked from parallel state)."""
    # Standard GPTModel layer spec — will be replaced by MoT spec in _setup_mot_decoder.
    # Pass normalization="RMSNorm" so WrappedTorchNorm is used (compatible with RMSNorm config).
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


def _swap_cp_groups(model: BagelMCoreModel, cp_group, cp_size: int):
    """
    Replace the cp_group / cp_size on every FlexAttention layer in model.
    Returns a list of (fa, old_cp_group, old_cp_size) so the caller can restore.
    """
    saved = []
    for layer in model.decoder.layers:
        fa = layer.self_attention.core_attention
        saved.append((fa, fa.cp_group, fa.cp_size))
        fa.cp_group = cp_group
        fa.cp_size  = cp_size
    return saved


def _restore_cp_groups(saved):
    """Restore cp_group / cp_size saved by _swap_cp_groups."""
    for fa, cg, cs in saved:
        fa.cp_group = cg
        fa.cp_size  = cs


# ─────────────────────────────────────────────────────────────────────────────
# BagelMatchingAttention — identical to test_transformer_mot_block.py
# Used for accurate Qwen2 comparison (SDPA-based, fp16).
# ─────────────────────────────────────────────────────────────────────────────

class BagelMatchingAttention(MegatronModule):
    def __init__(self, config, layer_number, attn_mask_type, attention_type,
                 softmax_scale=None, cp_comm_type=None, pg_collection=None, **kwargs):
        super().__init__(config=config)
        self.num_heads    = config.num_attention_heads
        self.num_kv_heads = config.num_query_groups
        self.head_dim     = config.kv_channels

    def forward(self, query, key, value, attention_mask,
                attn_mask_type=None, attention_bias=None, packed_seq_params=None):
        seq_len, batch_size, num_heads, head_dim = query.shape
        num_kv_heads = key.shape[2]
        if num_kv_heads != num_heads:
            n_groups = num_heads // num_kv_heads
            key   = key.repeat_interleave(n_groups, dim=2)
            value = value.repeat_interleave(n_groups, dim=2)
        q = query.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        k = key.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        v = value.squeeze(1).permute(1, 0, 2).unsqueeze(0).to(torch.float16)
        mask = torch.zeros(1, 1, seq_len, seq_len, dtype=torch.float16, device=query.device)
        with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
            out = F_sdpa(q, k, v, mask)
        out = out.squeeze(0).permute(1, 0, 2).contiguous()
        return out.reshape(seq_len, batch_size, num_heads * head_dim)


def _make_sdpa_block(mcore_cfg: TransformerConfig) -> TransformerMoTBlock:
    """Build TransformerMoTBlock with BagelMatchingAttention for Qwen2 comparison."""
    assert HAVE_WRAPPED_NORM, "WrappedTorchNorm not available"
    attn_sub = SelfAttentionMoTSubmodules(
        linear_qkv=ColumnParallelLinear,
        core_attention=BagelMatchingAttention,
        linear_proj=RowParallelLinear,
        q_layernorm=WrappedTorchNorm,
        k_layernorm=WrappedTorchNorm,
        linear_qkv_gen=ColumnParallelLinear,
        linear_proj_gen=RowParallelLinear,
        q_layernorm_gen=WrappedTorchNorm,
        k_layernorm_gen=WrappedTorchNorm,
    )
    mlp_spec = ModuleSpec(
        module=MLP,
        submodules=MLPSubmodules(linear_fc1=ColumnParallelLinear,
                                 linear_fc2=RowParallelLinear),
    )
    layer_sub = MoTTransformerLayerSubmodules(
        input_layernorm=WrappedTorchNorm,
        input_layernorm_gen=WrappedTorchNorm,
        self_attention=ModuleSpec(module=SelfAttentionMoT,
                                  params={"attn_mask_type": AttnMaskType.padding},
                                  submodules=attn_sub),
        self_attn_bda=get_bias_dropout_add,
        pre_mlp_layernorm=WrappedTorchNorm,
        pre_mlp_layernorm_gen=WrappedTorchNorm,
        mlp=mlp_spec, mlp_gen=mlp_spec,
        mlp_bda=get_bias_dropout_add,
    )
    block_sub = TransformerMoTBlockSubmodules(
        layer_specs=[ModuleSpec(module=MoTTransformerLayer, submodules=layer_sub)] * NUM_LAYERS,
        layer_norm=WrappedTorchNorm,
        layer_norm_gen=WrappedTorchNorm,
    )
    return TransformerMoTBlock(config=mcore_cfg, spec=block_sub).cuda().half()


# ─────────────────────────────────────────────────────────────────────────────
# Weight copy helpers (reused from test_transformer_mot_block.py)
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
        (bagel_attn.q_proj,        bagel_attn.k_proj,        bagel_attn.v_proj,
         bagel_attn.o_proj,        bagel_attn.q_norm,        bagel_attn.k_norm,
         mcore_attn.linear_qkv,   mcore_attn.linear_proj,
         mcore_attn.q_layernorm,  mcore_attn.k_layernorm),
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


def _copy_model_weights(bagel_model: "Qwen2Model", mcore_block: TransformerMoTBlock):
    for b_layer, m_layer in zip(bagel_model.layers, mcore_block.layers):
        _copy_layer_weights(b_layer, m_layer)
    if mcore_block.final_layernorm is not None:
        mcore_block.final_layernorm.weight.data.copy_(bagel_model.norm.weight.data)
    if mcore_block.final_layernorm_gen is not None:
        mcore_block.final_layernorm_gen.weight.data.copy_(bagel_model.norm_moe_gen.weight.data)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers (identity-RoPE for deterministic Qwen2 comparison)
# ─────────────────────────────────────────────────────────────────────────────

def _identity_rope(seq_len: int):
    """Zero-frequency RoPE → cos=1, sin=0 (identity rotation)."""
    mcore_rope = torch.zeros(seq_len, 1, 1, HEAD_DIM, dtype=torch.float32, device="cuda")
    pos_ids    = torch.arange(seq_len, dtype=torch.long, device="cuda")
    cos_id     = torch.ones (1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    sin_id     = torch.zeros(1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    return pos_ids, mcore_rope, (cos_id, sin_id)


def _patch_bagel_rope(bagel_model, cos_sin):
    cos_id, sin_id = cos_sin

    class _IdentityRope(torch.nn.Module):
        def forward(self, seq, pos_ids):
            return cos_id, sin_id

    object.__setattr__(bagel_model, "rotary_emb", _IdentityRope())


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_packed_data(T, V, G, H, device, seed=42):
    """
    Build a random full packed sequence and index arrays.

    Token layout in the full sequence of length S = T + V + G:
      [0 .. T-1]       → text tokens      (packed_text_indexes)
      [T .. T+V-1]     → ViT tokens       (packed_vit_indexes)
      [T+V .. S-1]     → VAE/gen tokens   (packed_vae_indexes)

    Returns:
        packed_seq [S, H], pos_ids [S], block_mask, text_idx, vit_idx, vae_idx
    """
    torch.manual_seed(seed)
    S = T + V + G
    packed_seq = torch.randn(S, H, dtype=torch.float16, device=device)
    pos_ids    = torch.arange(S, dtype=torch.long, device=device)
    text_idx   = torch.arange(0,     T,   device=device)
    vit_idx    = torch.arange(T,     T+V, device=device)
    vae_idx    = torch.arange(T+V,   S,   device=device)
    # Full attention over all S tokens (simplest mask)
    block_mask = create_block_mask(
        lambda b, h, q, kv: q >= 0,
        B=1, H=1, Q_LEN=S, KV_LEN=S, device=device,
    )
    return packed_seq, pos_ids, block_mask, text_idx, vit_idx, vae_idx


def _make_psp(und_idx, gen_idx,
              local_und_idx=None, local_gen_idx=None,
              Lund=None, Lgen=None) -> MoTPackedSeqParams:
    if local_und_idx is None:
        local_und_idx = und_idx
    if local_gen_idx is None:
        local_gen_idx = gen_idx
    if Lund is None:
        Lund = len(und_idx)
    if Lgen is None:
        Lgen = len(gen_idx)
    return MoTPackedSeqParams(
        qkv_format="thd",
        packed_und_token_indexes=und_idx,
        packed_gen_token_indexes=gen_idx,
        local_und_token_indexes=local_und_idx,
        local_gen_token_indexes=local_gen_idx,
        padded_und_seqlen=Lund,
        padded_gen_seqlen=Lgen,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Scatter correctness (CP=1, single rank)
#
# Verifies that _forward_decoder(packed_seq [S,1,H], psp_cp1) produces exactly
# the same output as calling model.decoder directly on the manually-compacted
# [U+G, 1, H] input with the same RoPE.
# ─────────────────────────────────────────────────────────────────────────────

def test_scatter_correctness(model: BagelMCoreModel, T: int, V: int, G: int, label: str):
    device = "cuda"
    H  = HIDDEN_SIZE
    S  = T + V + G

    packed_seq, pos_ids, block_mask, text_idx, vit_idx, vae_idx = \
        _make_packed_data(T, V, G, H, device)

    und_idx = torch.cat([text_idx, vit_idx])
    gen_idx = vae_idx
    U, Gsize = len(und_idx), len(gen_idx)

    psp = _make_psp(und_idx, gen_idx)

    # ── Reference: manual compact → model.decoder ────────────────────────────
    compact = torch.cat([packed_seq[und_idx], packed_seq[gen_idx]], dim=0).unsqueeze(1)
    compact_pos = torch.cat([pos_ids[und_idx], pos_ids[gen_idx]])
    rope_ref = model.rotary_pos_emb.forward_with_position_ids(compact_pos)

    with torch.no_grad():
        ref_out, _ = model.decoder(
            hidden_states=compact,
            attention_mask=block_mask,
            rotary_pos_emb=rope_ref,
            packed_seq_params=psp,
        )

    # ── Subject: _forward_decoder on full [S, 1, H] ──────────────────────────
    full_3d = packed_seq.unsqueeze(1)  # [S, 1, H]
    with torch.no_grad():
        dec_out = model._forward_decoder(
            decoder_input=full_3d,
            position_ids=None,
            attention_mask=block_mask,
            packed_position_ids=pos_ids,
            packed_seq_params=psp,
        )
    if isinstance(dec_out, tuple):
        dec_out = dec_out[0]

    # Must be exactly equal (same model, same computation path)
    assert dec_out.shape == ref_out.shape, \
        f"[{label}] shape mismatch: {dec_out.shape} vs {ref_out.shape}"
    assert torch.equal(dec_out, ref_out), \
        (f"[{label}] _forward_decoder != manual compact+decoder. "
         f"max_err={( dec_out - ref_out).abs().max().item():.3e}")

    print(f"  [scatter {label:12s}] PASS  S={S} U={U} G={Gsize}  shape={dec_out.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Qwen2 reference accuracy (CP=1, single rank)
#
# Builds a TransformerMoTBlock with BagelMatchingAttention (SDPA-based) and
# Qwen2Model, copies weights, runs both on the same compact sequence with
# identity-RoPE and asserts < 1e-3 error.
# ─────────────────────────────────────────────────────────────────────────────

def test_vs_qwen2(T: int, V: int, G: int, label: str):
    assert HAVE_BAGEL_PKG,   "skip: bagel-package not available"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"
    device = "cuda"
    H = HIDDEN_SIZE
    S = T + V + G

    packed_seq, _, _, text_idx, vit_idx, vae_idx = \
        _make_packed_data(T, V, G, H, device)

    und_idx = torch.cat([text_idx, vit_idx])
    gen_idx = vae_idx
    U, Gsize = len(und_idx), len(gen_idx)

    # Build models
    bagel_cfg  = _make_bagel_ref_config()
    mcore_cfg  = _make_mcore_config()
    bagel_model = Qwen2Model(bagel_cfg).cuda().half().train()
    mcore_block = _make_sdpa_block(mcore_cfg)
    _copy_model_weights(bagel_model, mcore_block)

    psp = _make_psp(und_idx, gen_idx)
    compact = torch.cat([packed_seq[und_idx], packed_seq[gen_idx]], dim=0)

    seq_len = U + Gsize
    _, mcore_rope, bagel_cos_sin = _identity_rope(seq_len)
    _patch_bagel_rope(bagel_model, bagel_cos_sin)

    with torch.no_grad():
        bagel_out = bagel_model.forward_train(
            packed_sequence=compact,
            sample_lens=[seq_len],
            attention_mask=[torch.zeros(1, seq_len, seq_len, dtype=torch.float16, device=device)],
            packed_position_ids=torch.arange(seq_len, device=device),
            packed_und_token_indexes=und_idx,
            packed_gen_token_indexes=gen_idx,
        )
        mcore_out, _ = mcore_block.forward_train(
            hidden_states=compact.unsqueeze(1),
            attention_mask=None,
            rotary_pos_emb=mcore_rope,
            packed_seq_params=psp,
        )

    mcore_flat = mcore_out.squeeze(1)
    assert not torch.any(torch.isnan(mcore_flat)),  f"[{label}] MCore has NaN"
    assert not torch.any(torch.isnan(bagel_out)),   f"[{label}] Bagel has NaN"
    torch.testing.assert_close(
        mcore_flat, bagel_out, atol=1e-3, rtol=1e-3,
        msg=lambda m: f"[vs_qwen2/{label}] {m}",
    )
    max_err = (mcore_flat - bagel_out).abs().max().item()
    print(f"  [qwen2   {label:12s}] PASS  U={U} G={Gsize}  max_err={max_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — CP=2 parity (requires 2 processes)
#
# Each rank calls _forward_decoder with its per-rank MoTPackedSeqParams.
# Outputs are compared against slices of the CP=1 reference (same model,
# cp_group nulled out in FlexAttention layers via deepcopy).
# ─────────────────────────────────────────────────────────────────────────────

def run_cp2_parity(model_cpN: BagelMCoreModel, T: int, V: int, G: int,
                   cp_group, label: str, seed: int = 42):
    rank    = dist.get_rank()
    device  = "cuda"
    H       = HIDDEN_SIZE
    S       = T + V + G
    cp_size = cp_group.size()  # 2

    packed_seq, pos_ids, block_mask, text_idx, vit_idx, vae_idx = \
        _make_packed_data(T, V, G, H, device, seed=seed)

    und_idx = torch.cat([text_idx, vit_idx])   # [U]
    gen_idx = vae_idx                           # [G]
    U = len(und_idx)
    Gsize = len(gen_idx)

    Lund = math.ceil(U      / cp_size)
    Lgen = math.ceil(Gsize  / cp_size)
    actual_lund = min(Lund, max(0, U     - rank * Lund))
    actual_lgen = min(Lgen, max(0, Gsize - rank * Lgen))

    local_und_idx = und_idx[rank * Lund : rank * Lund + actual_lund]
    local_gen_idx = gen_idx[rank * Lgen : rank * Lgen + actual_lgen]

    # Pad to Lund / Lgen if last rank has fewer tokens
    def _pad_idx(idx, target_len):
        n_pad = target_len - len(idx)
        if n_pad > 0:
            idx = torch.cat([idx, idx[-1:].expand(n_pad)])
        return idx

    local_und_idx_pad = _pad_idx(local_und_idx, Lund)
    local_gen_idx_pad = _pad_idx(local_gen_idx, Lgen)

    psp_cp1 = _make_psp(und_idx, gen_idx)
    psp_cpN = _make_psp(
        und_idx, gen_idx,
        local_und_idx=local_und_idx_pad,
        local_gen_idx=local_gen_idx_pad,
        Lund=Lund, Lgen=Lgen,
    )

    full_3d = packed_seq.unsqueeze(1)  # [S, 1, H]

    # ── CP=1 reference: temporarily null out CP groups (no A2A) ─────────────
    saved = _swap_cp_groups(model_cpN, cp_group=None, cp_size=1)
    with torch.no_grad():
        ref_out = model_cpN._forward_decoder(
            decoder_input=full_3d,
            position_ids=None,
            attention_mask=block_mask,
            packed_position_ids=pos_ids,
            packed_seq_params=psp_cp1,
        )
    _restore_cp_groups(saved)
    if isinstance(ref_out, tuple):
        ref_out = ref_out[0]
    # ref_out: [U+G, 1, H]  (und first, then gen)

    # ── CP=N run (restore cp_group, run with A2A) ─────────────────────────
    with torch.no_grad():
        cpN_out = model_cpN._forward_decoder(
            decoder_input=full_3d,
            position_ids=None,
            attention_mask=block_mask,
            packed_position_ids=pos_ids,
            packed_seq_params=psp_cpN,
        )
    if isinstance(cpN_out, tuple):
        cpN_out = cpN_out[0]
    # cpN_out: [Lund+Lgen, 1, H]

    dist.barrier()

    # ── Compare real-token slices ─────────────────────────────────────────────
    und_ref = ref_out[rank * Lund : rank * Lund + actual_lund]       # [actual_lund, 1, H]
    gen_ref = ref_out[U + rank * Lgen : U + rank * Lgen + actual_lgen]  # [actual_lgen, 1, H]

    und_got = cpN_out[:actual_lund]           # [actual_lund, 1, H]
    gen_got = cpN_out[Lund : Lund + actual_lgen]  # [actual_lgen, 1, H]

    atol = rtol = 1e-2
    if actual_lund > 0:
        torch.testing.assert_close(und_got, und_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[cp={cp_size}/{label} rank={rank}] UND: {m}")
    if actual_lgen > 0:
        torch.testing.assert_close(gen_got, gen_ref, atol=atol, rtol=rtol,
            msg=lambda m: f"[cp={cp_size}/{label} rank={rank}] GEN: {m}")

    und_err = (und_got - und_ref).abs().max().item() if actual_lund > 0 else 0.0
    gen_err = (gen_got - gen_ref).abs().max().item() if actual_lgen > 0 else 0.0
    print(f"  [cp={cp_size} {label:8s} rank={rank}] PASS  "
          f"U={U} G={Gsize}  Lund={Lund} Lgen={Lgen}  "
          f"actual=({actual_lund},{actual_lgen})  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — CE loss correctness (CP=1, single rank)
#
# Interface (Phase 6): forward() receives full-sequence labels [S] / loss_mask [S]
# and slices internally by local_und_token_indexes.
#
# Three sub-checks:
#   a. Values match:  ce[ce_mask] == F.cross_entropy(logits[ce_mask], packed_labels)
#                     and ce == 0 at non-CE positions.
#   b. None when no labels/loss_mask are supplied.
#   c. Zero CE when loss_mask is all-zero.
# ─────────────────────────────────────────────────────────────────────────────

def test_ce_loss(model: BagelMCoreModel, T: int, V: int, G: int, L_ce: int, label: str):
    """CE loss test.  V should be 0 so that ce_loss_indexes ⊆ text_idx is unambiguous."""
    device = "cuda"
    H = HIDDEN_SIZE
    S = T + V + G

    torch.manual_seed(7)
    _, pos_ids, block_mask, text_idx, vit_idx, vae_idx = \
        _make_packed_data(T, V, G, H, device)

    und_idx = torch.cat([text_idx, vit_idx])
    gen_idx = vae_idx
    psp     = _make_psp(und_idx, gen_idx)
    U       = len(psp.local_und_token_indexes)  # = T+V (CP=1)

    # CE-active positions: sparse subset of text tokens only
    ce_loss_indexes  = text_idx[:L_ce]                                      # [L_ce] ⊆ text_idx
    packed_label_ids = torch.randint(0, VOCAB_SIZE, (L_ce,), device=device, dtype=torch.long)

    # Full-sequence label tensors (0 / 0.0 at non-CE positions)
    labels_full    = torch.zeros(S, dtype=torch.long,    device=device)
    loss_mask_full = torch.zeros(S, dtype=torch.float16, device=device)
    if L_ce > 0:
        labels_full[ce_loss_indexes]    = packed_label_ids
        loss_mask_full[ce_loss_indexes] = 1.0

    input_ids   = torch.randint(0, VOCAB_SIZE, (1, T), device=device) if T > 0 else \
                  torch.zeros(1, 0, device=device, dtype=torch.long)
    vision_emb  = torch.randn(V, H, dtype=torch.float16, device=device) if V > 0 else None
    visual_lats = torch.randn(G, H, dtype=torch.float16, device=device) if G > 0 else None

    # Use align_bagel_embeddings to assemble compact decoder_input and pre-shard
    # CP-sensitive tensors before calling BagelMCoreModel.forward().
    und_idx = psp.local_und_token_indexes   # [U] when CP=1 == packed_und_token_indexes

    def _run(labels_arg, lm_arg):
        aligned = _align_and_shard(
            psp=psp,
            language_model=model,
            input_ids=input_ids,
            vision_embeddings=vision_emb,
            visual_latents=visual_lats,
            sequence_length=S,
            packed_text_indexes=text_idx,
            packed_vit_token_indexes=vit_idx if V > 0 else None,
            packed_vae_token_indexes=vae_idx if G > 0 else None,
            packed_position_ids=pos_ids,
            labels_full=labels_arg,
            loss_mask_full=lm_arg,
        )
        with torch.no_grad():
            return model.forward(
                decoder_input=aligned['decoder_input'],
                attention_mask=block_mask,
                packed_position_ids=aligned['packed_position_ids'],
                packed_seq_params=psp,
                labels=aligned['labels'],
                loss_mask=aligned['loss_mask'],
            )

    # ── (a) Value correctness ────────────────────────────────────────────────
    out      = _run(labels_full, loss_mask_full)
    ce_got   = out["ce"]
    last_hid = out["last_hidden_state"]

    assert ce_got is not None, f"[{label}] ce should not be None when labels provided"
    # ce_got is sparse: shape [L_ce] — model returns only active (loss_mask > 0) positions.
    assert ce_got.shape == (L_ce,), f"[{label}] ce shape {ce_got.shape} != ({L_ce},)"

    # Local slices (what align_embeddings_by_token_positions selects for this rank)
    local_labels = labels_full[und_idx]
    local_lm     = loss_mask_full[und_idx]
    ce_mask      = local_lm > 0                               # [U] bool, True at CE positions

    # CE at active positions must match manual recompute from last_hidden_state
    output_weight = model.shared_embedding_or_output_weight()
    und_hid    = last_hid[:U].to(output_weight.dtype)
    logits_all = F.linear(und_hid, output_weight)
    expected_active = (F.cross_entropy(logits_all[ce_mask], local_labels[ce_mask],
                                        reduction="none") * local_lm[ce_mask])
    torch.testing.assert_close(
        ce_got.float(), expected_active.float(), atol=1e-4, rtol=1e-4,
        msg=lambda m: f"[ce_loss/{label}] value mismatch: {m}",
    )
    if L_ce > 0:
        ref_logits = F.linear(und_hid[ce_mask].to(output_weight.dtype), output_weight)
        ref_ce = F.cross_entropy(ref_logits, packed_label_ids, reduction="none")
        torch.testing.assert_close(
            ce_got.float(), ref_ce.float(), atol=1e-4, rtol=1e-4,
            msg=lambda m: f"[ce_loss/{label}] CE at active positions: {m}",
        )

    # ── (b) CE is None when labels / loss_mask not supplied ──────────────────
    assert _run(None, None)["ce"] is None, f"[{label}] ce should be None with no labels"

    # ── (c) All-zero CE when loss_mask is all zeros ───────────────────────────
    zero_lm = torch.zeros(S, dtype=torch.float16, device=device)
    assert torch.all(_run(labels_full, zero_lm)["ce"] == 0), \
        f"[{label}] ce should be zero when loss_mask=0"

    print(f"  [ce_loss  {label:12s}] PASS  U={U} G={len(gen_idx)} L_ce={L_ce}"
          f"  sum_ce={ce_got.float().sum().item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4b — CE CP=2 parity (requires nproc_per_node=2)
#
# Both ranks forward the same full-sequence labels [S] / loss_mask [S].
# BagelMCoreModel slices by local_und_token_indexes internally.
#
# Two checks per rank:
#   i.  Per-rank CE at CE positions matches corresponding slice of CP=1 reference.
#   ii. AllReduce(sum of per-rank CE) == CP=1 reference total.
# ─────────────────────────────────────────────────────────────────────────────

def run_ce_cp2_parity(model_cpN: BagelMCoreModel, T: int, V: int, G: int,
                      L_ce: int, cp_group, label: str, seed: int = 42):
    rank    = dist.get_rank()
    device  = "cuda"
    H       = HIDDEN_SIZE
    S       = T + V + G
    cp_size = cp_group.size()

    torch.manual_seed(seed)
    _, pos_ids, block_mask, text_idx, vit_idx, vae_idx = \
        _make_packed_data(T, V, G, H, device, seed=seed)

    und_idx = torch.cat([text_idx, vit_idx])
    gen_idx = vae_idx
    U = len(und_idx)
    Gsize = len(gen_idx)

    Lund = math.ceil(U     / cp_size)
    Lgen = math.ceil(Gsize / cp_size)
    actual_lund = min(Lund, max(0, U     - rank * Lund))
    actual_lgen = min(Lgen, max(0, Gsize - rank * Lgen))

    def _pad_idx(idx, tlen):
        n = tlen - len(idx)
        return torch.cat([idx, idx[-1:].expand(n)]) if n > 0 else idx

    psp_cp1 = _make_psp(und_idx, gen_idx)
    psp_cpN = _make_psp(
        und_idx, gen_idx,
        local_und_idx=_pad_idx(und_idx[rank * Lund : rank * Lund + actual_lund], Lund),
        local_gen_idx=_pad_idx(gen_idx[rank * Lgen : rank * Lgen + actual_lgen], Lgen),
        Lund=Lund, Lgen=Lgen,
    )

    # CE data: sparse subset of first L_ce text tokens
    # Ensure the last und token (text_idx[-1]) is NOT a CE position so padding
    # slots (which repeat the last entry) carry loss_mask=0 and don't affect CE.
    ce_loss_indexes  = text_idx[:L_ce]
    packed_label_ids = torch.randint(0, VOCAB_SIZE, (L_ce,), device=device, dtype=torch.long)
    labels_full    = torch.zeros(S, dtype=torch.long,    device=device)
    loss_mask_full = torch.zeros(S, dtype=torch.float16, device=device)
    if L_ce > 0:
        labels_full[ce_loss_indexes]    = packed_label_ids
        loss_mask_full[ce_loss_indexes] = 1.0

    input_ids   = torch.randint(0, VOCAB_SIZE, (1, T), device=device) if T > 0 else \
                  torch.zeros(1, 0, device=device, dtype=torch.long)
    vision_emb  = torch.randn(V, H, dtype=torch.float16, device=device) if V > 0 else None
    visual_lats = torch.randn(G, H, dtype=torch.float16, device=device) if G > 0 else None

    embed_kwargs = dict(
        language_model=model_cpN,
        input_ids=input_ids,
        vision_embeddings=vision_emb,
        visual_latents=visual_lats,
        sequence_length=S,
        packed_text_indexes=text_idx,
        packed_vit_token_indexes=vit_idx if V > 0 else None,
        packed_vae_token_indexes=vae_idx if G > 0 else None,
        packed_position_ids=pos_ids,
        labels_full=labels_full,
        loss_mask_full=loss_mask_full,
    )

    # Use _align_and_shard to assemble + pre-shard, then call forward()
    # with clean pre-assembled inputs (BagelMCoreModel is CP-unaware).

    # ── CP=1 reference (null out CP groups in FlexAttention) ─────────────────
    saved = _swap_cp_groups(model_cpN, cp_group=None, cp_size=1)
    aligned_cp1 = _align_and_shard(psp=psp_cp1, **embed_kwargs)
    with torch.no_grad():
        ref_out = model_cpN.forward(
            decoder_input=aligned_cp1['decoder_input'],
            attention_mask=block_mask,
            packed_position_ids=aligned_cp1['packed_position_ids'],
            packed_seq_params=psp_cp1,
            labels=aligned_cp1['labels'],
            loss_mask=aligned_cp1['loss_mask'],
        )
    _restore_cp_groups(saved)
    ref_ce = ref_out["ce"]   # [U]: non-zero only at CE positions

    # ── CP=N run ─────────────────────────────────────────────────────────────
    aligned_cpN = _align_and_shard(psp=psp_cpN, **embed_kwargs)
    with torch.no_grad():
        cpN_out = model_cpN.forward(
            decoder_input=aligned_cpN['decoder_input'],
            attention_mask=block_mask,
            packed_position_ids=aligned_cpN['packed_position_ids'],
            packed_seq_params=psp_cpN,
            labels=aligned_cpN['labels'],
            loss_mask=aligned_cpN['loss_mask'],
        )
    local_ce = cpN_out["ce"]   # [actual_lund]: this rank's CE slice

    dist.barrier()

    # ── Check i: per-rank CE matches the same slice of the CP=1 reference ────
    # ref_ce[r*Lund : r*Lund + actual_lund] corresponds to this rank's real tokens.
    ref_slice   = ref_ce[rank * Lund : rank * Lund + actual_lund]   # [actual_lund]
    local_slice = local_ce[:actual_lund]                             # [actual_lund]
    torch.testing.assert_close(
        local_slice.float(), ref_slice.float(), atol=1e-2, rtol=1e-2,
        msg=lambda m: f"[ce_cp2/{label} rank={rank}] per-rank CE: {m}",
    )

    # ── Check ii: AllReduce(sum) == CP=1 total ────────────────────────────────
    local_sum = local_ce.float().sum()
    dist.all_reduce(local_sum, op=dist.ReduceOp.SUM, group=cp_group)
    ref_sum = ref_ce.float().sum()
    torch.testing.assert_close(
        local_sum, ref_sum, atol=1e-2, rtol=1e-2,
        msg=lambda m: f"[ce_cp2/{label} rank={rank}] total CE: {m}",
    )

    err = (local_slice - ref_slice).abs().max().item()
    print(f"  [ce_cp2   {label:8s} rank={rank}] PASS  "
          f"U={U} G={Gsize} L_ce={L_ce}  Lund={Lund}  "
          f"actual_lund={actual_lund}  err={err:.4f}  sum={local_sum.item():.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    world_size = dist.get_world_size()
    cp_size    = min(world_size, 2)
    torch.cuda.set_device(rank)

    # Initialize Megatron parallel state (CP group is registered here)
    Utils.initialize_model_parallel(1, 1, context_parallel_size=cp_size)
    model_parallel_cuda_manual_seed(rank)

    mcore_cfg = _make_mcore_config()

    # ── Test 1: Scatter correctness ───────────────────────────────────────────
    if rank == 0:
        print("\n=== Test 1: Scatter correctness ===")
    # Reinit with cp=1 for accuracy tests (single-rank model)
    Utils.initialize_model_parallel(1, 1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    model_acc = _build_bagel_mcore_model(mcore_cfg)
    model_acc.train()

    test_scatter_correctness(model_acc, T_CLEAN, V_CLEAN, G_CLEAN, "clean")
    test_scatter_correctness(model_acc, T_PAD,   V_PAD,   G_PAD,   "padding")

    # ── Test 2: vs Qwen2 reference ────────────────────────────────────────────
    if HAVE_BAGEL_PKG and HAVE_WRAPPED_NORM:
        if rank == 0:
            print("\n=== Test 2: Qwen2 accuracy ===")
        test_vs_qwen2(T_CLEAN, V_CLEAN, G_CLEAN, "clean")
        test_vs_qwen2(0,       0,       G_CLEAN, "gen-only")
        test_vs_qwen2(T_CLEAN, V_CLEAN, 0,       "und-only")
    else:
        if rank == 0:
            print("\n=== Test 2: Qwen2 accuracy — SKIPPED (bagel-package not available) ===")

    # ── Test 4a: CE loss correctness (CP=1) ──────────────────────────────────
    if rank == 0:
        print("\n=== Test 4a: CE loss correctness (CP=1) ===")
    test_ce_loss(model_acc, T_CE, 0, G_CE_MIX, L_CE_MIX, "mixed")
    test_ce_loss(model_acc, T_CE, 0, G_CE_UND, L_CE_UND, "und-only")
    test_ce_loss(model_acc, T_CE, 0, G_CE_MIX, 0,        "no-CE")

    # ── Test 3: CP=2 parity ──────────────────────────────────────────────────
    if world_size >= 2:
        if rank == 0:
            print("\n=== Test 3: CP=2 parity ===")
        # Reinitialize with cp=2
        Utils.initialize_model_parallel(1, 1, context_parallel_size=2)
        model_parallel_cuda_manual_seed(7)
        model_cp = _build_bagel_mcore_model(mcore_cfg)
        model_cp.train()

        cp_group = dist.new_group(list(range(world_size)))

        run_cp2_parity(model_cp, T_CLEAN, V_CLEAN, G_CLEAN, cp_group, "clean", seed=42)
        run_cp2_parity(model_cp, T_PAD,   V_PAD,   G_PAD,   cp_group, "padding", seed=77)

        dist.barrier()
        if rank == 0:
            print("  [CP=2 parity] PASS  all ranks agree")

        # ── Test 4b: CE loss CP=2 parity ─────────────────────────────────────
        if rank == 0:
            print("\n=== Test 4b: CE loss CP=2 parity ===")
        run_ce_cp2_parity(model_cp, T_CE, 0, G_CE_MIX, L_CE_MIX, cp_group, "mixed",    seed=42)
        run_ce_cp2_parity(model_cp, T_CE, 0, G_CE_UND, L_CE_UND, cp_group, "und-only", seed=77)

        dist.barrier()
        if rank == 0:
            print("  [CE CP=2 parity] PASS  all ranks agree")
    else:
        if rank == 0:
            print("\n=== Test 3: CP=2 parity — SKIPPED (need nproc_per_node=2) ===")

    dist.barrier()
    if rank == 0:
        print("\nAll tests passed.")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
