# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""MiniMax-M3 (text / LM) model configuration.

Mapped from the Hugging Face ``config.json`` of ``MiniMaxAI/MiniMax-M3``
(``model_type: minimax_m3_vl`` with a nested ``text_config``) or from the
text-only config emitted by ``tools/minimax_m3/slice_ckpt.py``
(``model_type: minimax_m3_vl_text``).

Architecture facts pinned in P0 (transformers 5.16.1):
* 60 layers, hidden 6144, 64 q heads x 128, 4 kv heads, per-head Gemma QK-norm
  (``(1 + w)`` scaling, eps 1e-6), partial RoPE on the first 64 dims
  (NeoX ``rotate_half``), theta 5e6, untied embeddings, vocab 200064.
* Layers 0-2: dense MLP (12288) + full causal attention.
  Layers 3-59: MoE (128 experts top-4, sigmoid router with ``e_score_correction_bias``,
  top-k weights renormalised then x ``routed_scaling_factor`` 2.0, one shared expert 3072)
  + MiniMax Sparse Attention (block 128, top-16, index 4 heads x 128, local block forced).
* Activation everywhere: clamped SwiGLU ``(up + 1) * gate * sigmoid(1.702 * gate)``,
  clamp limit 7.0 (HF ``swigluoai``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from megatron.lite.primitive.config import load_hf_config_dict

_TEXT_MODEL_TYPES = {"minimax_m3_vl", "minimax_m3_vl_text", "minimax_m3", "minimax_m3_text"}

# text_config keys consumed directly (same spelling in the dataclass)
_HF_FIELDS = frozenset(
    {
        "num_hidden_layers",
        "hidden_size",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "vocab_size",
        "rms_norm_eps",
        "max_position_embeddings",
        "rope_theta",
        "rotary_dim",
        "num_local_experts",
        "num_experts_per_tok",
        "n_shared_experts",
        "intermediate_size",
        "shared_intermediate_size",
        "dense_intermediate_size",
        "routed_scaling_factor",
        "scoring_func",
        "use_routing_bias",
        "swiglu_alpha",
        "swiglu_limit",
        "tie_word_embeddings",
        "router_aux_loss_coef",
        "qk_norm_type",
        "use_gemma_norm",
        "use_qk_norm",
        "attention_output_gate",
    }
)
# text_config keys we knowingly ignore (inference-only / unused / vision)
_HF_IGNORED = frozenset(
    {
        "model_type",
        "architectures",
        "torch_dtype",
        "dtype",
        "transformers_version",
        "hidden_act",  # 'swigluoai' is expressed through swiglu_alpha/limit
        "num_mtp_modules",
        "num_nextn_predict_layers",  # no MTP weights in the checkpoint
        "attention_dropout",
        "use_cache",
        "bos_token_id",
        "eos_token_id",
        "pad_token_id",
        "initializer_range",
        "output_router_logits",
        "router_jitter_noise",
        "rope_parameters",
        "partial_rotary_factor",  # derived from rotary_dim / head_dim and cross-checked
        "sparse_attention_config",  # unpacked below
        "moe_layer_freq",  # unpacked below
        "layer_types",
        "mlp_layer_types",
        "index_n_heads",
        "index_head_dim",
        "index_block_size",
        "index_topk_blocks",
        "index_local_blocks",
        "_source",
        "base_config_key",
        "auto_map",
    }
)


def _generic_hf_config_keys() -> frozenset[str]:
    """Boilerplate keys ``PretrainedConfig.to_dict()`` adds to every config (not architecture)."""
    try:
        from transformers import PretrainedConfig

        return frozenset(PretrainedConfig().to_dict().keys())
    except Exception:
        return frozenset(
            {
                "_name_or_path", "chunk_size_feed_forward", "id2label", "label2id", "is_encoder_decoder",
                "output_attentions", "output_hidden_states", "problem_type", "return_dict", "tie_encoder_decoder",
                "tokenizer_class", "prefix", "sep_token_id", "decoder_start_token_id", "task_specific_params",
                "torchscript", "tf_legacy_loss", "use_bfloat16", "pruned_heads", "is_decoder", "add_cross_attention",
                "finetuning_task", "architectures", "torch_dtype", "dtype",
            }
        )


@dataclass
class MiniMaxM3Config:
    """MiniMax-M3 text-model architecture parameters."""

    model_type: str = "minimax_m3_vl"
    num_hidden_layers: int = 60
    hidden_size: int = 6144
    num_attention_heads: int = 64
    num_key_value_heads: int = 4
    head_dim: int = 128
    vocab_size: int = 200064
    rms_norm_eps: float = 1e-6
    max_position_embeddings: int = 1_048_576
    rope_theta: float = 5_000_000.0
    rotary_dim: int = 64
    tie_word_embeddings: bool = False
    # MLP / MoE
    dense_intermediate_size: int = 12288
    intermediate_size: int = 3072  # routed expert intermediate
    shared_intermediate_size: int = 3072
    n_shared_experts: int = 1
    num_local_experts: int = 128
    num_experts_per_tok: int = 4
    routed_scaling_factor: float = 2.0
    scoring_func: str = "sigmoid"
    use_routing_bias: bool = True
    router_aux_loss_coef: float = 0.0  # HF default is off (output_router_logits=False); no aux loss in SFT
    swiglu_alpha: float = 1.702
    swiglu_limit: float = 7.0
    swiglu_up_offset: float = 1.0
    # per-layer dispatch
    mlp_layer_types: list[str] = field(default_factory=lambda: ["dense"] * 3 + ["sparse"] * 57)
    layer_types: list[str] = field(default_factory=lambda: ["full_attention"] * 3 + ["minimax_m3_sparse"] * 57)
    # MSA
    index_n_heads: int = 4
    index_head_dim: int = 128
    index_block_size: int = 128
    index_topk_blocks: int = 16
    index_local_blocks: int = 1
    # sanity flags carried from HF (validated, not used as switches)
    qk_norm_type: str = "per_head"
    use_gemma_norm: bool = True
    use_qk_norm: bool = True
    attention_output_gate: bool = False
    hf_text_prefix: str = "language_model.model"
    hf_head_name: str = "language_model.lm_head"

    # ------------------------------------------------------------------ derived
    @property
    def partial_rotary_factor(self) -> float:
        return self.rotary_dim / self.head_dim

    @property
    def num_experts(self) -> int:  # Experts / TokenDispatcher / SigmoidTopKRouter aliases
        return self.num_local_experts

    @property
    def n_routed_experts(self) -> int:
        return self.num_local_experts

    @property
    def moe_intermediate_size(self) -> int:
        return self.intermediate_size

    @property
    def shared_expert_intermediate_size(self) -> int:
        return self.shared_intermediate_size

    @property
    def aux_loss_alpha(self) -> float:
        return self.router_aux_loss_coef

    @property
    def n_group(self) -> None:
        return None

    def is_moe_layer(self, layer_idx: int) -> bool:
        return self.mlp_layer_types[layer_idx] == "sparse"

    def is_sparse_attention_layer(self, layer_idx: int) -> bool:
        return self.layer_types[layer_idx] == "minimax_m3_sparse"

    @property
    def has_moe(self) -> bool:
        return any(t == "sparse" for t in self.mlp_layer_types)

    # ------------------------------------------------------------------ validation
    def __post_init__(self):
        errors: list[str] = []

        def _check(cond: bool, msg: str):
            if not cond:
                errors.append(msg)

        _check(self.model_type in _TEXT_MODEL_TYPES, f"unsupported model_type {self.model_type!r}")
        _check(len(self.layer_types) == self.num_hidden_layers, "len(layer_types) != num_hidden_layers")
        _check(len(self.mlp_layer_types) == self.num_hidden_layers, "len(mlp_layer_types) != num_hidden_layers")
        _check(self.num_attention_heads % self.num_key_value_heads == 0, "heads not divisible by kv heads")
        _check(self.index_n_heads == self.num_key_value_heads, "MSA requires index_n_heads == num_key_value_heads")
        _check(0 < self.rotary_dim <= self.head_dim and self.rotary_dim % 2 == 0, "bad rotary_dim")
        _check(self.head_dim == 128, "MSA flex backend / cuDNN paths validated for head_dim 128 only")
        _check(self.scoring_func == "sigmoid", "only sigmoid routing is supported")
        _check(self.use_gemma_norm and self.use_qk_norm and self.qk_norm_type == "per_head", "norm flags differ from M3")
        _check(not self.attention_output_gate, "attention_output_gate is not supported")
        _check(not self.tie_word_embeddings, "tied embeddings are not supported")
        _check(1 <= self.num_experts_per_tok <= self.num_local_experts, "bad num_experts_per_tok")
        _check(self.index_topk_blocks % 2 == 0, "index_topk_blocks must be even (kernel contract)")
        for i, t in enumerate(self.layer_types):
            _check(t in {"full_attention", "minimax_m3_sparse"}, f"layer_types[{i}]={t!r}")
        for i, t in enumerate(self.mlp_layer_types):
            _check(t in {"dense", "sparse"}, f"mlp_layer_types[{i}]={t!r}")
        if errors:
            raise ValueError("Invalid MiniMaxM3Config:\n  " + "\n  ".join(errors))

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_hf(cls, path: str, **overrides) -> MiniMaxM3Config:
        return cls._from_hf_dict(load_hf_config_dict(path), **overrides)

    @classmethod
    def from_hf_config(cls, hf_config, **overrides) -> MiniMaxM3Config:
        return cls._from_hf_dict(hf_config.to_dict(), **overrides)

    @classmethod
    def _from_hf_dict(cls, hf: dict, *, strict: bool = True, **overrides) -> MiniMaxM3Config:
        model_type = hf.get("model_type")
        text = hf.get("text_config")
        if isinstance(text, dict):
            hf = dict(text)
            hf.setdefault("model_type", model_type)
        else:
            hf = dict(hf)
        model_type = hf.get("model_type") or model_type
        if model_type not in _TEXT_MODEL_TYPES:
            raise ValueError(f"Unsupported MiniMax-M3 model_type: {model_type!r}")

        kwargs = {k: v for k, v in hf.items() if k in _HF_FIELDS}
        kwargs["model_type"] = model_type
        n = int(hf["num_hidden_layers"])

        # per-layer dispatch: checkpoint spelling (moe_layer_freq / sparse_attention_config) or HF flat spelling
        if "mlp_layer_types" in hf:
            kwargs["mlp_layer_types"] = list(hf["mlp_layer_types"])
        elif "moe_layer_freq" in hf:
            kwargs["mlp_layer_types"] = ["sparse" if f else "dense" for f in hf["moe_layer_freq"]]
        sparse = hf.get("sparse_attention_config") or {}
        if "layer_types" in hf:
            kwargs["layer_types"] = list(hf["layer_types"])
        elif "sparse_attention_freq" in sparse:
            kwargs["layer_types"] = ["minimax_m3_sparse" if f else "full_attention" for f in sparse["sparse_attention_freq"]]
        for flat, legacy in {
            "index_n_heads": "sparse_num_index_heads",
            "index_head_dim": "sparse_index_dim",
            "index_block_size": "sparse_block_size",
            "index_topk_blocks": "sparse_topk_blocks",
            "index_local_blocks": "sparse_local_block",
        }.items():
            if flat in hf:
                kwargs[flat] = hf[flat]
            elif legacy in sparse:
                kwargs[flat] = sparse[legacy]
        if sparse:
            if sparse.get("sparse_score_type", "max") != "max":
                raise ValueError("only max block pooling is supported")
            if int(sparse.get("sparse_init_block", 0)) != 0:
                raise ValueError("sparse_init_block (sink block) != 0 is not supported")
            if "sparse_disable_index_value" in sparse:
                flags = sparse["sparse_disable_index_value"]
                if any(not f for f, t in zip(flags, kwargs.get("layer_types", [])) if t == "minimax_m3_sparse"):
                    raise ValueError("index-value branch enabled in checkpoint; not supported")
            if "use_sparse_attention" in sparse and not sparse["use_sparse_attention"]:
                kwargs["layer_types"] = ["full_attention"] * n

        rp = hf.get("rope_parameters")
        if isinstance(rp, dict):
            kwargs.setdefault("rope_theta", float(rp.get("rope_theta", 5e6)))
            if "partial_rotary_factor" in rp and "rotary_dim" not in kwargs:
                kwargs["rotary_dim"] = int(round(float(rp["partial_rotary_factor"]) * int(hf.get("head_dim", 128))))
        if "partial_rotary_factor" in hf and "rotary_dim" in kwargs:
            expect = float(hf["partial_rotary_factor"]) * int(kwargs.get("head_dim", 128))
            if abs(expect - kwargs["rotary_dim"]) > 1e-6:
                raise ValueError("partial_rotary_factor and rotary_dim disagree")
        if hf.get("hidden_act", "swigluoai") not in ("swigluoai", "silu"):
            raise ValueError(f"unsupported hidden_act {hf.get('hidden_act')!r}")
        if hf.get("hidden_act") == "swigluoai" or "swiglu_alpha" in hf:
            kwargs.setdefault("swiglu_up_offset", 1.0)
        # aux loss is only active in HF when output_router_logits=True (default False)
        if not hf.get("output_router_logits", False):
            kwargs["router_aux_loss_coef"] = 0.0
        if hf.get("num_mtp_modules") or hf.get("num_nextn_predict_layers"):
            pass  # MTP modules are declared but absent from the checkpoint; nothing to build

        if strict:
            unknown = set(hf) - _HF_FIELDS - _HF_IGNORED - _generic_hf_config_keys()
            if unknown:
                raise ValueError(f"MiniMaxM3Config: unmapped HF text_config fields {sorted(unknown)} (no silent defaults)")
        kwargs.update(overrides)
        return cls(**kwargs)
