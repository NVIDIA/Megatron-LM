"""
Unit test: BagelMimoModel.forward vs original Bagel (bagel.py) using
bagel_packed_batch_to_mimo_batch() for input preparation.

The original Bagel model (bagel-package/bagel/modeling/bagel/bagel.py) takes a packed
batch dict and internally assembles the packed_sequence before calling its LLM.
BagelMimoModel takes the same packed batch (converted via bagel_packed_batch_to_mimo_batch)
and should produce identical last_hidden_state at every token position.

Test cases
----------
A. Text-only:    T text tokens, no ViT, no VAE.
B. Text + gen:   T text tokens + G pre-embedded gen tokens (mock diffusion submodule).

Usage
-----
torchrun --nproc_per_node=1 test_bagel_mimo_vs_original.py
"""

import math
import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# Path setup: bagel-package and examples/mimo_bagel still need explicit
# sys.path entries (bagel-package for the original Bagel reference imports
# and examples/mimo_bagel for examples.mimo_bagel.utils.data_helpers).
_ROOT        = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
_BAGEL_PKG   = os.path.join(_ROOT, "bagel-package")
_BAGEL_SRC   = os.path.join(_BAGEL_PKG, "bagel")
_MIMO_BAGEL  = os.path.join(_ROOT, "examples", "mimo_bagel")

for p in [_ROOT, _BAGEL_PKG, _BAGEL_SRC, _MIMO_BAGEL]:
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
from examples.mimo_bagel.utils.data_helpers import (
    bagel_packed_batch_to_mimo_batch,
    get_packed_seq_params,
)

# ── Optional: original Bagel package ─────────────────────────────────────────
try:
    from bagel.modeling.bagel.bagel import Bagel, BagelConfig
    from bagel.modeling.bagel.qwen2_navit import (
        Qwen2ForCausalLM,
        Qwen2Model,
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
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.test_utilities import Utils

# ─────────────────────────────────────────────────────────────────────────────
# Test dimensions  (small for fast CPU-based unit tests)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE     = 256
FFN_HIDDEN_SIZE = 512
NUM_HEADS       = 4
NUM_KV_HEADS    = 4
HEAD_DIM        = HIDDEN_SIZE // NUM_HEADS
NUM_LAYERS      = 2
ROPE_THETA      = 10000.0
VOCAB_SIZE      = 256
MAX_SEQ_LEN     = 512

T_TOKENS   = 128  # text tokens (must be >= BLOCK_SIZE=128 used in Bagel's create_block_mask)
G_TOKENS   = 128  # gen (VAE) tokens
V_TOKENS   = 128  # ViT tokens (must be >= 128 for block_mask)
LATENT_DIM = 16   # vae latent dimension (C * patch_size^2)
VIT_HIDDEN = 64   # ViT encoder output hidden size (before LLM projection)


# ─────────────────────────────────────────────────────────────────────────────
# Config factories
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


def _make_bagel_llm_config() -> "BagelQwen2Config":
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


def _make_llm_config_stub():
    """Minimal llm_config stub for BagelMCoreModel."""
    class _Stub:
        layer_module = "Qwen2MoTDecoderLayer"
        freeze_und   = False
    return _Stub()


# ─────────────────────────────────────────────────────────────────────────────
# Minimal BagelMimoModel wrapper (no MimoModelConfig required)
# ─────────────────────────────────────────────────────────────────────────────

class _BagelMimoTestModel(BagelMimoModel):
    """BagelMimoModel with pre-built LLM and optional named submodules."""

    def __init__(self, language_model: BagelMCoreModel,
                 diffusion_submodule: nn.Module = None,
                 images_submodule: nn.Module = None):
        nn.Module.__init__(self)
        self.config              = language_model.config
        self.language_model      = language_model
        self.special_token_ids   = {}
        self.cp_size             = parallel_state.get_context_parallel_world_size()
        self.cp_rank             = parallel_state.get_context_parallel_rank()
        self.cp_group            = parallel_state.get_context_parallel_group()

        submodules = {}
        if diffusion_submodule is not None:
            submodules["diffusion"] = diffusion_submodule
        if images_submodule is not None:
            submodules["images"] = images_submodule
        self.modality_submodules = nn.ModuleDict(submodules)
    # Inherits BagelMimoModel.forward — no override


class _MockDiffusionSubmodule(nn.Module):
    """Minimal diffusion submodule that returns pre-defined visual_latents."""

    def __init__(self, visual_latents: torch.Tensor):
        super().__init__()
        self._visual_latents = visual_latents
        # llm2vae is accessed in BagelMimoModel.forward when gen_loss_mask is set;
        # in this test we don't provide gen_loss_mask, so it won't be called.

    def forward(self, encoder_inputs=None):
        return self._visual_latents


# ─────────────────────────────────────────────────────────────────────────────
# Model builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_mcore_llm(mcore_cfg: TransformerConfig) -> BagelMCoreModel:
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
        llm_config=_make_llm_config_stub(),
        use_flex_attention=True,
    )
    return model.cuda().half()


def _build_bagel_ref_model(llm_cfg) -> "Bagel":
    """Build original Bagel model (text-only, no ViT / VAE)."""
    llm = Qwen2ForCausalLM(llm_cfg)
    llm.init_moe()
    bagel_cfg = BagelConfig(
        visual_gen=False,
        visual_und=False,
        llm_config=llm_cfg,
        vit_config=None,
        vae_config=None,
    )
    model = Bagel(language_model=llm, vit_model=None, config=bagel_cfg)
    return model.cuda().half().train()


# ─────────────────────────────────────────────────────────────────────────────
# Weight-copy helpers  (Bagel HF → BagelMCoreModel)
# ─────────────────────────────────────────────────────────────────────────────

def _hf_to_mcore_qkv_weight(q_w, k_w, v_w, ng, np_, hn):
    h  = q_w.shape[1]
    nq = np_ // ng
    return torch.cat([
        q_w.view(ng, nq * hn, h),
        k_w.view(ng, hn, h),
        v_w.view(ng, hn, h),
    ], dim=1).reshape(ng * (nq + 2) * hn, h)


def _hf_to_mcore_qkv_bias(q_b, k_b, v_b, ng, np_, hn):
    nq = np_ // ng
    return torch.cat([
        q_b.view(ng, nq * hn),
        k_b.view(ng, hn),
        v_b.view(ng, hn),
    ], dim=1).reshape(ng * (nq + 2) * hn)


def _copy_attn_weights(hf_attn, mc_attn):
    np_ = hf_attn.num_heads
    ng  = hf_attn.num_key_value_heads
    hn  = hf_attn.head_dim
    for (q_p, k_p, v_p, o_p, q_n, k_n, lqkv, lproj, qln, kln) in [
        (hf_attn.q_proj,         hf_attn.k_proj,         hf_attn.v_proj,
         hf_attn.o_proj,         hf_attn.q_norm,         hf_attn.k_norm,
         mc_attn.linear_qkv,    mc_attn.linear_proj,
         mc_attn.q_layernorm,   mc_attn.k_layernorm),
        (hf_attn.q_proj_moe_gen, hf_attn.k_proj_moe_gen, hf_attn.v_proj_moe_gen,
         hf_attn.o_proj_moe_gen, hf_attn.q_norm_moe_gen, hf_attn.k_norm_moe_gen,
         mc_attn.linear_qkv_gen, mc_attn.linear_proj_gen,
         mc_attn.q_layernorm_gen, mc_attn.k_layernorm_gen),
    ]:
        lqkv.weight.data.copy_(
            _hf_to_mcore_qkv_weight(q_p.weight.data, k_p.weight.data,
                                     v_p.weight.data, ng=ng, np_=np_, hn=hn))
        lqkv.bias.data.copy_(
            _hf_to_mcore_qkv_bias(q_p.bias.data, k_p.bias.data,
                                   v_p.bias.data, ng=ng, np_=np_, hn=hn))
        lproj.weight.data.copy_(o_p.weight.data)
        if qln is not None and hasattr(q_n, "weight"):
            qln.weight.data.copy_(q_n.weight.data)
        if kln is not None and hasattr(k_n, "weight"):
            kln.weight.data.copy_(k_n.weight.data)


def _copy_mlp_weights(hf_mlp, mc_mlp):
    ffn = hf_mlp.gate_proj.weight.shape[0]
    mc_mlp.linear_fc1.weight.data[:ffn].copy_(hf_mlp.gate_proj.weight.data)
    mc_mlp.linear_fc1.weight.data[ffn:].copy_(hf_mlp.up_proj.weight.data)
    mc_mlp.linear_fc2.weight.data.copy_(hf_mlp.down_proj.weight.data)


def _copy_layer_weights(hf_layer, mc_layer):
    mc_layer.input_layernorm.weight.data.copy_(hf_layer.input_layernorm.weight.data)
    mc_layer.input_layernorm_gen.weight.data.copy_(hf_layer.input_layernorm_moe_gen.weight.data)
    mc_layer.pre_mlp_layernorm.weight.data.copy_(hf_layer.post_attention_layernorm.weight.data)
    mc_layer.pre_mlp_layernorm_gen.weight.data.copy_(hf_layer.post_attention_layernorm_moe_gen.weight.data)
    _copy_attn_weights(hf_layer.self_attn, mc_layer.self_attention)
    _copy_mlp_weights(hf_layer.mlp, mc_layer.mlp)
    _copy_mlp_weights(hf_layer.mlp_moe_gen, mc_layer.mlp_gen)


def _copy_all_weights(bagel_ref: "Bagel", mcore_llm: BagelMCoreModel):
    """Copy weights: embedding + all MoT transformer layers + final norms."""
    hf_qwen = bagel_ref.language_model.model  # Qwen2Model
    mcore_llm.embedding.word_embeddings.weight.data.copy_(hf_qwen.embed_tokens.weight.data)
    for hf_layer, mc_layer in zip(hf_qwen.layers, mcore_llm.decoder.layers):
        _copy_layer_weights(hf_layer, mc_layer)
    if mcore_llm.decoder.final_layernorm is not None:
        mcore_llm.decoder.final_layernorm.weight.data.copy_(hf_qwen.norm.weight.data)
    if mcore_llm.decoder.final_layernorm_gen is not None:
        mcore_llm.decoder.final_layernorm_gen.weight.data.copy_(hf_qwen.norm_moe_gen.weight.data)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers  (identity RoPE → exact numerical match)
# ─────────────────────────────────────────────────────────────────────────────

def _identity_rope(seq_len: int):
    """Return cos=1, sin=0 tensors (no rotation)."""
    cos = torch.ones (1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    sin = torch.zeros(1, seq_len, HEAD_DIM, dtype=torch.float16, device="cuda")
    return cos, sin


def _patch_bagel_rope(bagel_ref: "Bagel", cos, sin):
    """Replace Bagel's rotary_emb with identity rotation."""
    class _IdentityRope(nn.Module):
        def forward(self, seq, pos_ids):
            return cos, sin
    object.__setattr__(bagel_ref.language_model.model, "rotary_emb", _IdentityRope())


# ─────────────────────────────────────────────────────────────────────────────
# Packed-batch constructors (matching Bagel.forward input format)
# ─────────────────────────────────────────────────────────────────────────────

def _make_text_only_batch(T: int, seed: int = 42):
    """Packed batch dict for T text-only tokens (no ViT, no VAE)."""
    torch.manual_seed(seed)
    S = T
    return {
        "sequence_length":      S,
        "packed_text_ids":      torch.randint(0, VOCAB_SIZE, (T,), dtype=torch.long),
        "packed_text_indexes":  torch.arange(T, dtype=torch.long),
        "packed_position_ids":  torch.arange(S, dtype=torch.long),
        "sample_lens":          [S],
        "split_lens":           [S],
        "attn_modes":           ["und"],
    }


def _make_text_and_gen_batch(T: int, G: int, seed: int = 42):
    """Packed batch dict for T text + G gen tokens.

    Text at positions [0, T-1], gen at [T, T+G-1].
    Also returns pre-projected gen embeddings [G, H] (mock vae2llm output).
    """
    torch.manual_seed(seed)
    S = T + G
    batch = {
        "sequence_length":          S,
        "packed_text_ids":          torch.randint(0, VOCAB_SIZE, (T,), dtype=torch.long),
        "packed_text_indexes":      torch.arange(T, dtype=torch.long),
        "packed_vae_token_indexes": torch.arange(T, S, dtype=torch.long),
        "packed_position_ids":      torch.arange(S, dtype=torch.long),
        "sample_lens":              [S],
        "split_lens":               [T, G],
        "attn_modes":               ["und", "gen"],
    }
    # Simulates the projected latent embeddings that would come from
    # vae2llm(latent) + time_embedder(ts) + latent_pos_embed(pos_ids)
    gen_emb = torch.randn(G, HIDDEN_SIZE, dtype=torch.float16)
    return batch, gen_emb


# ─────────────────────────────────────────────────────────────────────────────
# Helper: get reference hidden states from original Bagel
#
# Captures the Qwen2Model (LLM) output via a forward hook so we can compare
# token-level hidden states without needing Bagel to expose them directly.
# ─────────────────────────────────────────────────────────────────────────────

def _bagel_hidden_states(bagel_ref: "Bagel", packed_batch: dict) -> torch.Tensor:
    """Run Bagel.forward and return last_hidden_state [S, H] via hook."""
    captured = {}

    def _hook(module, inp, out):
        # Qwen2Model.forward_train returns last_hidden_state directly
        captured["hidden"] = out.detach()

    handle = bagel_ref.language_model.model.register_forward_hook(_hook)
    device = "cuda"

    with torch.no_grad():
        bagel_ref.forward(
            sequence_length    = packed_batch["sequence_length"],
            packed_text_ids    = packed_batch["packed_text_ids"].to(device),
            packed_text_indexes= packed_batch["packed_text_indexes"].to(device),
            sample_lens        = packed_batch["sample_lens"],
            packed_position_ids= packed_batch["packed_position_ids"].to(device),
            split_lens         = packed_batch["split_lens"],
            attn_modes         = packed_batch["attn_modes"],
        )
    handle.remove()
    return captured["hidden"]   # [S, H]


# ─────────────────────────────────────────────────────────────────────────────
# Test A — text-only parity
# ─────────────────────────────────────────────────────────────────────────────

def test_text_only_parity(T: int, label: str):
    assert HAVE_BAGEL_PKG,    "skip: bagel-package not importable"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"

    device = "cuda"

    # ── 1. Construct packed batch (same format as Bagel dataloader) ───────────
    packed_batch = _make_text_only_batch(T)

    # ── 2. Build models ───────────────────────────────────────────────────────
    llm_cfg   = _make_bagel_llm_config()
    mcore_cfg = _make_mcore_config()

    bagel_ref  = _build_bagel_ref_model(llm_cfg)       # original Bagel (HF)
    mcore_llm  = _build_mcore_llm(mcore_cfg)           # Megatron Core LLM
    mcore_llm.train()

    # Copy weights: Bagel LLM → BagelMCoreModel
    _copy_all_weights(bagel_ref, mcore_llm)

    # ── 3. Identity RoPE (both models) ────────────────────────────────────────
    S = T
    cos_id, sin_id = _identity_rope(S)
    _patch_bagel_rope(bagel_ref, cos_id, sin_id)
    mcore_llm.rotary_pos_emb.inv_freq.zero_()

    # ── 4. Reference: hidden states from original Bagel ───────────────────────
    ref_hidden = _bagel_hidden_states(bagel_ref, packed_batch)   # [S=T, H]

    # ── 5. BagelMimoModel: convert packed batch → MIMO batch, then forward ────
    # bagel_packed_batch_to_mimo_batch handles: input_ids, position_ids,
    # packed_position_ids, packed_seq_params, sample_lens, etc.
    mimo_batch = bagel_packed_batch_to_mimo_batch(packed_batch)

    # Move index tensors to CUDA (bagel_packed_batch_to_mimo_batch keeps them on CPU)
    def _to_cuda(batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    mimo_batch = _to_cuda(mimo_batch)

    # Move packed_seq_params tensors to CUDA
    psp = mimo_batch["packed_seq_params"]
    for attr in ("packed_text_indexes", "packed_und_token_indexes",
                 "packed_gen_token_indexes", "local_und_token_indexes",
                 "local_gen_token_indexes"):
        t = getattr(psp, attr, None)
        if t is not None:
            setattr(psp, attr, t.to(device))

    mimo_model = _BagelMimoTestModel(mcore_llm)

    with torch.no_grad():
        lm_output, _ = mimo_model.forward(
            input_ids          = mimo_batch["input_ids"],
            packed_position_ids= mimo_batch["packed_position_ids"],
            sequence_length    = mimo_batch["sequence_length"],
            sample_lens        = mimo_batch["sample_lens"],
            split_lens         = mimo_batch.get("split_lens"),
            attn_modes         = mimo_batch.get("attn_modes"),
            packed_seq_params  = psp,
        )

    # last_hidden_state is compact [Lund+Lgen=T, H] for text-only CP=1
    got = lm_output["last_hidden_state"]   # [T, H]

    # ── 6. Compare ────────────────────────────────────────────────────────────
    text_idx = packed_batch["packed_text_indexes"].to(device)
    # ref_hidden[text_idx] == ref_hidden (since text_idx = [0..T-1] = all positions)
    atol = rtol = 1e-2
    torch.testing.assert_close(
        got, ref_hidden[text_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[text_only/{label}] {m}",
    )
    max_err = (got - ref_hidden[text_idx]).abs().max().item()
    print(f"  [text_only  {label:8s}] PASS  T={T}  max_err={max_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test B — text + gen parity
#
# Gen embeddings are injected via a _MockDiffusionSubmodule.  The Bagel
# reference builds packed_sequence[vae_idx] = gen_emb directly (bypassing
# the real VAE/noise pipeline) so both sides see identical inputs.
# ─────────────────────────────────────────────────────────────────────────────

def test_text_and_gen_parity(T: int, G: int, label: str):
    assert HAVE_BAGEL_PKG,    "skip: bagel-package not importable"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"

    device = "cuda"

    # ── 1. Packed batch + pre-computed gen embeddings ─────────────────────────
    packed_batch, gen_emb = _make_text_and_gen_batch(T, G)
    gen_emb = gen_emb.to(device)   # [G, H]

    S = T + G
    text_idx = packed_batch["packed_text_indexes"].to(device)   # [T]
    vae_idx  = packed_batch["packed_vae_token_indexes"].to(device)  # [G]

    # ── 2. Build models ───────────────────────────────────────────────────────
    llm_cfg   = _make_bagel_llm_config()
    mcore_cfg = _make_mcore_config()

    bagel_ref = _build_bagel_ref_model(llm_cfg)
    mcore_llm = _build_mcore_llm(mcore_cfg)
    mcore_llm.train()
    _copy_all_weights(bagel_ref, mcore_llm)

    # ── 3. Identity RoPE ──────────────────────────────────────────────────────
    cos_id, sin_id = _identity_rope(S)
    _patch_bagel_rope(bagel_ref, cos_id, sin_id)
    mcore_llm.rotary_pos_emb.inv_freq.zero_()

    # ── 4. Reference: Bagel's Qwen2Model directly with manually assembled seq ─
    # We replicate what Bagel.forward does internally for text + gen tokens.
    hf_qwen = bagel_ref.language_model.model   # Qwen2Model
    with torch.no_grad():
        text_emb = hf_qwen.embed_tokens(
            packed_batch["packed_text_ids"].to(device)
        )  # [T, H]

    packed_seq = torch.zeros(S, HIDDEN_SIZE, dtype=torch.float16, device=device)
    packed_seq[text_idx] = text_emb.detach()
    packed_seq[vae_idx]  = gen_emb

    pos_ids = packed_batch["packed_position_ids"].to(device)

    # Build the flex-attention block mask for text+gen (same as Bagel.forward does)
    from bagel.data.data_utils import create_sparse_mask
    from torch.nn.attention.flex_attention import create_block_mask as _cbm
    _sparse = create_sparse_mask(
        packed_batch["sample_lens"], packed_batch["split_lens"],
        packed_batch["attn_modes"], device,
    )
    _block_mask = _cbm(
        _sparse, B=1, H=NUM_HEADS, Q_LEN=S, KV_LEN=S,
        device=device, BLOCK_SIZE=128, _compile=True,
    )

    with torch.no_grad():
        ref_hidden = hf_qwen.forward_train(
            packed_sequence         = packed_seq,
            sample_lens             = packed_batch["sample_lens"],
            attention_mask          = _block_mask,
            packed_position_ids     = pos_ids,
            packed_und_token_indexes= text_idx,
            packed_gen_token_indexes= vae_idx,
        )   # [S, H]

    # ── 5. BagelMimoModel with mock diffusion submodule ───────────────────────
    # Build a text-only MIMO batch, then extend with vae token info.
    text_only_batch = _make_text_only_batch(T)
    mimo_batch = bagel_packed_batch_to_mimo_batch(text_only_batch)

    # Extend mimo_batch with gen token indexes so packed_seq_params includes them
    psp = get_packed_seq_params(
        packed_text_indexes     = packed_batch["packed_text_indexes"],
        packed_vit_token_indexes= None,
        packed_vae_token_indexes= packed_batch["packed_vae_token_indexes"],
    )

    def _to_cuda(batch):
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                out[k] = v.to(device)
            else:
                out[k] = v
        return out
    mimo_batch = _to_cuda(mimo_batch)

    for attr in ("packed_text_indexes", "packed_und_token_indexes",
                 "packed_gen_token_indexes", "local_und_token_indexes",
                 "local_gen_token_indexes"):
        t = getattr(psp, attr, None)
        if t is not None:
            setattr(psp, attr, t.to(device))

    mock_diffusion = _MockDiffusionSubmodule(gen_emb)
    mimo_model = _BagelMimoTestModel(mcore_llm, diffusion_submodule=mock_diffusion)

    with torch.no_grad():
        lm_output, _ = mimo_model.forward(
            input_ids          = mimo_batch["input_ids"],
            packed_position_ids= torch.cat([
                pos_ids[text_idx],
                pos_ids[vae_idx],
            ]),                    # compact: und positions then gen positions
            sequence_length    = S,
            sample_lens        = packed_batch["sample_lens"],
            split_lens         = packed_batch.get("split_lens"),
            attn_modes         = packed_batch.get("attn_modes"),
            modality_inputs    = {"diffusion": {}},   # triggers mock submodule
            packed_seq_params  = psp,
        )

    got = lm_output["last_hidden_state"]   # [T+G, H] compact

    # ── 6. Compare at text positions (und) and gen positions ──────────────────
    atol = rtol = 1e-2
    torch.testing.assert_close(
        got[:T], ref_hidden[text_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[text_gen/{label}/und] {m}",
    )
    torch.testing.assert_close(
        got[T:], ref_hidden[vae_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[text_gen/{label}/gen] {m}",
    )
    und_err = (got[:T] - ref_hidden[text_idx]).abs().max().item()
    gen_err = (got[T:] - ref_hidden[vae_idx]).abs().max().item()
    print(f"  [text+gen   {label:8s}] PASS  T={T} G={G}  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers for Tests C and D
# ─────────────────────────────────────────────────────────────────────────────

def _make_text_and_vit_batch(T: int, V: int, seed: int = 42):
    """Packed batch with T text tokens + V ViT tokens (all understanding tokens).

    Text at positions [0, T-1], ViT at [T, T+V-1].
    Since both are understanding tokens, split_lens=[T+V], attn_modes=["und"].
    """
    torch.manual_seed(seed)
    S = T + V
    return {
        "sequence_length":          S,
        "packed_text_ids":          torch.randint(0, VOCAB_SIZE, (T,), dtype=torch.long),
        "packed_text_indexes":      torch.arange(T, dtype=torch.long),
        "packed_vit_token_indexes": torch.arange(T, S, dtype=torch.long),
        "packed_position_ids":      torch.arange(S, dtype=torch.long),
        "sample_lens":              [S],
        "split_lens":               [S],   # all und
        "attn_modes":               ["und"],
    }


def _build_block_mask(packed_batch: dict, S: int, device: str):
    """Build a flex-attention BlockMask from packed_batch split_lens/attn_modes."""
    from bagel.data.data_utils import create_sparse_mask
    from torch.nn.attention.flex_attention import create_block_mask as _cbm
    sparse = create_sparse_mask(
        packed_batch["sample_lens"], packed_batch["split_lens"],
        packed_batch["attn_modes"], device,
    )
    return _cbm(
        sparse, B=1, H=NUM_HEADS, Q_LEN=S, KV_LEN=S,
        device=device, BLOCK_SIZE=128, _compile=True,
    )


def _move_psp_to_device(psp, device):
    """Move all tensor fields of a MoTPackedSeqParams to the given device."""
    for attr in (
        "packed_text_indexes", "packed_vit_token_indexes", "packed_vae_token_indexes",
        "packed_und_token_indexes", "packed_gen_token_indexes",
        "local_und_token_indexes", "local_gen_token_indexes",
    ):
        t = getattr(psp, attr, None)
        if t is not None:
            setattr(psp, attr, t.to(device))
    return psp


# ─────────────────────────────────────────────────────────────────────────────
# Test C — diffusion (visual generation) module parity
#
# Verifies that BagelMimoModel's diffusion submodule correctly encodes
# latents + timesteps + position IDs and produces identical hidden states
# to the Bagel reference when the same encoding computation is applied.
#
# Submodule interface tested:
#   _TestDiffusionSubmodule.forward(encoder_inputs) → [G, H]
#   where encoder_inputs = {
#       "latents":             [G, LATENT_DIM],
#       "shifted_timesteps":   [G],
#       "latent_position_ids": [G],
#   }
# ─────────────────────────────────────────────────────────────────────────────

class _TestDiffusionSubmodule(nn.Module):
    """Diffusion submodule with simple trainable layers for unit testing.

    Computes: vae2llm(latents) + timestep_proj(timesteps) + pos_embed(pos_ids)

    This mirrors Bagel's own computation:
        packed_latent = vae2llm(packed_latent) + time_embedder(ts) + latent_pos_embed(pos)
    using simpler (un-fourier) versions suitable for small test dimensions.
    """

    def __init__(self, latent_dim: int, hidden_size: int, max_pos: int):
        super().__init__()
        self.vae2llm      = nn.Linear(latent_dim, hidden_size, bias=False)
        self.timestep_proj = nn.Linear(1, hidden_size, bias=False)
        self.pos_embed    = nn.Embedding(max_pos, hidden_size)

    def forward(self, encoder_inputs: dict) -> torch.Tensor:
        latents   = encoder_inputs["latents"]            # [G, LATENT_DIM]
        timesteps = encoder_inputs["shifted_timesteps"]  # [G]
        pos_ids   = encoder_inputs["latent_position_ids"]  # [G]
        return (
            self.vae2llm(latents)
            + self.timestep_proj(timesteps.unsqueeze(-1).to(latents.dtype))
            + self.pos_embed(pos_ids)
        )  # [G, H]


def test_diffusion_module_parity(T: int, G: int, label: str):
    """BagelMimoModel with a real diffusion-encoding submodule matches Bagel reference."""
    assert HAVE_BAGEL_PKG,    "skip: bagel-package not importable"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"

    device = "cuda"
    S = T + G

    # ── 1. Packed batch (text + gen layout) ──────────────────────────────────
    packed_batch, _ = _make_text_and_gen_batch(T, G)
    text_idx = packed_batch["packed_text_indexes"].to(device)
    vae_idx  = packed_batch["packed_vae_token_indexes"].to(device)

    # Random latents, pre-shifted timesteps, and position IDs for VAE tokens
    torch.manual_seed(123)
    latents          = torch.randn(G, LATENT_DIM, dtype=torch.float16, device=device)
    shifted_timesteps = torch.rand(G, dtype=torch.float32, device=device)
    latent_pos_ids   = torch.arange(G, dtype=torch.long, device=device)

    # ── 2. Build models ───────────────────────────────────────────────────────
    llm_cfg   = _make_bagel_llm_config()
    mcore_cfg = _make_mcore_config()
    bagel_ref = _build_bagel_ref_model(llm_cfg)
    mcore_llm = _build_mcore_llm(mcore_cfg)
    mcore_llm.train()
    _copy_all_weights(bagel_ref, mcore_llm)

    # ── 3. Identity RoPE ──────────────────────────────────────────────────────
    cos_id, sin_id = _identity_rope(S)
    _patch_bagel_rope(bagel_ref, cos_id, sin_id)
    mcore_llm.rotary_pos_emb.inv_freq.zero_()

    # ── 4. Build diffusion submodule (shared weights for reference & subject) ─
    test_diff = _TestDiffusionSubmodule(LATENT_DIM, HIDDEN_SIZE, max_pos=G).cuda().half()

    # ── 5. Reference: compute gen_emb with the test submodule, inject manually ─
    hf_qwen = bagel_ref.language_model.model   # Qwen2Model
    with torch.no_grad():
        text_emb = hf_qwen.embed_tokens(packed_batch["packed_text_ids"].to(device))
        gen_emb_ref = test_diff.forward({
            "latents":             latents,
            "shifted_timesteps":   shifted_timesteps,
            "latent_position_ids": latent_pos_ids,
        })  # [G, H]

    packed_seq = torch.zeros(S, HIDDEN_SIZE, dtype=torch.float16, device=device)
    packed_seq[text_idx] = text_emb.detach()
    packed_seq[vae_idx]  = gen_emb_ref.detach()

    pos_ids = packed_batch["packed_position_ids"].to(device)
    block_mask = _build_block_mask(packed_batch, S, device)

    with torch.no_grad():
        ref_hidden = hf_qwen.forward_train(
            packed_sequence         = packed_seq,
            sample_lens             = packed_batch["sample_lens"],
            attention_mask          = block_mask,
            packed_position_ids     = pos_ids,
            packed_und_token_indexes= text_idx,
            packed_gen_token_indexes= vae_idx,
        )  # [S, H]

    # ── 6. BagelMimoModel with _TestDiffusionSubmodule ────────────────────────
    psp = get_packed_seq_params(
        packed_text_indexes     = packed_batch["packed_text_indexes"],
        packed_vit_token_indexes= None,
        packed_vae_token_indexes= packed_batch["packed_vae_token_indexes"],
    )
    _move_psp_to_device(psp, device)

    mimo_model = _BagelMimoTestModel(mcore_llm, diffusion_submodule=test_diff)

    with torch.no_grad():
        lm_output, _ = mimo_model.forward(
            input_ids          = packed_batch["packed_text_ids"].unsqueeze(0).to(device),
            packed_position_ids= pos_ids,
            sequence_length    = S,
            sample_lens        = packed_batch["sample_lens"],
            split_lens         = packed_batch["split_lens"],
            attn_modes         = packed_batch["attn_modes"],
            modality_inputs    = {
                "diffusion": {
                    "latents":             latents,
                    "shifted_timesteps":   shifted_timesteps,
                    "latent_position_ids": latent_pos_ids,
                }
            },
            packed_seq_params  = psp,
        )

    got = lm_output["last_hidden_state"]   # [T+G, H] compact

    # ── 7. Compare ────────────────────────────────────────────────────────────
    atol = rtol = 1e-2
    torch.testing.assert_close(
        got[:T], ref_hidden[text_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[diffusion/{label}/und] {m}",
    )
    torch.testing.assert_close(
        got[T:], ref_hidden[vae_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[diffusion/{label}/gen] {m}",
    )
    und_err = (got[:T] - ref_hidden[text_idx]).abs().max().item()
    gen_err = (got[T:] - ref_hidden[vae_idx]).abs().max().item()
    print(f"  [diffusion  {label:10s}] PASS  T={T} G={G}  "
          f"und_err={und_err:.4f}  gen_err={gen_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Test D — ViT (visual understanding) module parity
#
# Verifies that BagelMimoModel's vision submodule correctly encodes
# pre-extracted ViT features through a connector + position embeddings
# and produces identical hidden states to the Bagel reference.
#
# Submodule interface tested:
#   _TestVitSubmodule.forward(encoder_inputs) → [V, H]
#   where encoder_inputs = {
#       "packed_vit_features":    [V, VIT_HIDDEN],  # pre-encoded ViT features
#       "packed_vit_position_ids": [V],
#   }
# ─────────────────────────────────────────────────────────────────────────────

class _TestVitSubmodule(nn.Module):
    """ViT submodule with a simple connector + position embedding for unit testing.

    Computes: connector(vit_features) + pos_embed(pos_ids)

    This mirrors Bagel's own computation:
        packed_vit_embed = connector(vit_model_output) + vit_pos_embed(pos_ids)
    using a single linear layer instead of the SigLIP encoder + MLP connector.
    """

    def __init__(self, vit_hidden: int, hidden_size: int, max_pos: int):
        super().__init__()
        self.connector = nn.Linear(vit_hidden, hidden_size, bias=False)
        self.pos_embed = nn.Embedding(max_pos, hidden_size)

    def forward(self, encoder_inputs: dict) -> torch.Tensor:
        features = encoder_inputs["packed_vit_features"]      # [V, VIT_HIDDEN]
        pos_ids  = encoder_inputs["packed_vit_position_ids"]  # [V]
        return self.connector(features) + self.pos_embed(pos_ids)  # [V, H]


def test_vit_module_parity(T: int, V: int, label: str):
    """BagelMimoModel with a real ViT-projection submodule matches Bagel reference."""
    assert HAVE_BAGEL_PKG,    "skip: bagel-package not importable"
    assert HAVE_WRAPPED_NORM, "skip: WrappedTorchNorm not available"

    device = "cuda"
    S = T + V  # all understanding tokens

    # ── 1. Packed batch (text + vit, all und) ─────────────────────────────────
    packed_batch = _make_text_and_vit_batch(T, V)
    text_idx = packed_batch["packed_text_indexes"].to(device)
    vit_idx  = packed_batch["packed_vit_token_indexes"].to(device)

    # Pre-encoded ViT features (simulate SigLIP encoder output)
    torch.manual_seed(456)
    vit_features = torch.randn(V, VIT_HIDDEN, dtype=torch.float16, device=device)
    vit_pos_ids  = torch.arange(T, S, dtype=torch.long, device=device)  # positions [T..T+V-1]

    # ── 2. Build models ───────────────────────────────────────────────────────
    llm_cfg   = _make_bagel_llm_config()
    mcore_cfg = _make_mcore_config()
    bagel_ref = _build_bagel_ref_model(llm_cfg)
    mcore_llm = _build_mcore_llm(mcore_cfg)
    mcore_llm.train()
    _copy_all_weights(bagel_ref, mcore_llm)

    # ── 3. Identity RoPE ──────────────────────────────────────────────────────
    cos_id, sin_id = _identity_rope(S)
    _patch_bagel_rope(bagel_ref, cos_id, sin_id)
    mcore_llm.rotary_pos_emb.inv_freq.zero_()

    # ── 4. Build ViT submodule (shared for reference & subject) ───────────────
    test_vit = _TestVitSubmodule(VIT_HIDDEN, HIDDEN_SIZE, max_pos=S).cuda().half()

    # ── 5. Reference: compute vit_emb, inject manually, run hf_qwen ──────────
    hf_qwen = bagel_ref.language_model.model   # Qwen2Model
    with torch.no_grad():
        text_emb = hf_qwen.embed_tokens(packed_batch["packed_text_ids"].to(device))
        vit_emb_ref = test_vit.forward({
            "packed_vit_features":     vit_features,
            "packed_vit_position_ids": vit_pos_ids,
        })  # [V, H]

    packed_seq = torch.zeros(S, HIDDEN_SIZE, dtype=torch.float16, device=device)
    packed_seq[text_idx] = text_emb.detach()
    packed_seq[vit_idx]  = vit_emb_ref.detach()

    pos_ids = packed_batch["packed_position_ids"].to(device)
    block_mask = _build_block_mask(packed_batch, S, device)
    und_idx = torch.cat([text_idx, vit_idx])  # [T+V] all und tokens

    with torch.no_grad():
        ref_hidden = hf_qwen.forward_train(
            packed_sequence         = packed_seq,
            sample_lens             = packed_batch["sample_lens"],
            attention_mask          = block_mask,
            packed_position_ids     = pos_ids,
            packed_und_token_indexes= und_idx,
            packed_gen_token_indexes= torch.zeros(0, dtype=torch.long, device=device),
        )  # [S, H]

    # ── 6. BagelMimoModel with _TestVitSubmodule ──────────────────────────────
    psp = get_packed_seq_params(
        packed_text_indexes     = packed_batch["packed_text_indexes"],
        packed_vit_token_indexes= packed_batch["packed_vit_token_indexes"],
        packed_vae_token_indexes= None,
    )
    _move_psp_to_device(psp, device)

    mimo_model = _BagelMimoTestModel(mcore_llm, images_submodule=test_vit)

    with torch.no_grad():
        lm_output, _ = mimo_model.forward(
            input_ids          = packed_batch["packed_text_ids"].unsqueeze(0).to(device),
            packed_position_ids= pos_ids,
            sequence_length    = S,
            sample_lens        = packed_batch["sample_lens"],
            split_lens         = packed_batch["split_lens"],
            attn_modes         = packed_batch["attn_modes"],
            modality_inputs    = {
                "images": {
                    "packed_vit_features":     vit_features,
                    "packed_vit_position_ids": vit_pos_ids,
                }
            },
            packed_seq_params  = psp,
        )

    got = lm_output["last_hidden_state"]   # [T+V, H] compact (all und, no gen)

    # ── 7. Compare at text positions and ViT positions ────────────────────────
    atol = rtol = 1e-2
    torch.testing.assert_close(
        got[:T], ref_hidden[text_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[vit/{label}/text] {m}",
    )
    torch.testing.assert_close(
        got[T:], ref_hidden[vit_idx],
        atol=atol, rtol=rtol,
        msg=lambda m: f"[vit/{label}/vit] {m}",
    )
    txt_err = (got[:T] - ref_hidden[text_idx]).abs().max().item()
    vit_err = (got[T:] - ref_hidden[vit_idx]).abs().max().item()
    print(f"  [vit        {label:10s}] PASS  T={T} V={V}  "
          f"txt_err={txt_err:.4f}  vit_err={vit_err:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)

    Utils.initialize_model_parallel(1, 1, context_parallel_size=1)
    model_parallel_cuda_manual_seed(42)

    if not (HAVE_BAGEL_PKG and HAVE_WRAPPED_NORM):
        if rank == 0:
            print("SKIP: bagel-package or WrappedTorchNorm not available")
        dist.barrier()
        Utils.destroy_model_parallel()
        return

    # ── Test A: text-only parity ──────────────────────────────────────────────
    if rank == 0:
        print("\n=== Test A: text-only parity (BagelMimoModel vs Bagel) ===")

    test_text_only_parity(T=T_TOKENS,     label="T=8")
    test_text_only_parity(T=T_TOKENS * 2, label="T=16")

    # ── Test B: text + gen parity ─────────────────────────────────────────────
    if rank == 0:
        print("\n=== Test B: text+gen parity (BagelMimoModel vs Bagel) ===")

    test_text_and_gen_parity(T=T_TOKENS, G=G_TOKENS, label="T=8,G=16")

    # ── Test C: diffusion module parity ───────────────────────────────────────
    if rank == 0:
        print("\n=== Test C: diffusion module parity (BagelMimoModel vs Bagel) ===")

    test_diffusion_module_parity(T=T_TOKENS, G=G_TOKENS, label="T=128,G=128")

    # ── Test D: ViT module parity ─────────────────────────────────────────────
    if rank == 0:
        print("\n=== Test D: ViT module parity (BagelMimoModel vs Bagel) ===")

    test_vit_module_parity(T=T_TOKENS, V=V_TOKENS, label="T=128,V=128")

    dist.barrier()
    if rank == 0:
        print("\nAll tests PASSED.")

    Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
