# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The 4-layer Qwen3.8-Flash-Next proxy, described once for both implementations.

The HF config below is the primary description; ``mcore_argv()`` is the same model expressed
as Megatron arguments. Every structural knob of the real 48-layer model is present -- gated
residual over 4 streams, GDN with a sigmoid output gate, QSA with its indexer, 512-expert-style
MoE with a gated shared expert, and the PLE n-gram memory -- only the sizes are small.

Layer layout: HF has 4 blocks (GDN, GDN, GDN, QSA), each block being attention-then-MoE. Megatron
expands that into 8 hybrid layers, so HF block ``i`` becomes Megatron layers ``2i`` (the attention
or GDN half, plus PLE where present) and ``2i+1`` (the MoE half). The PLE sits on HF block 1 =
Megatron layer 2, which is why ``--engram-layer-ids`` is 3 (Megatron counts from 1).

Size constraints that are easy to violate when editing this file:
  * ``ple_embed_dim % ((ngram_size - 1) * heads_per_ngram) == 0``  ->  128 % 16
  * Megatron ``--engram-memory-dim`` = ``ple_embed_dim / (ngram_size - 1)`` = 64, itself divisible
    by the 8 hash heads
  * ``hidden_size % num_attention_heads``, ``num_attention_heads % num_key_value_heads``,
    ``hidden_size % hc_count``
  * ``partial_rotary_factor * head_dim`` must be even  ->  0.25 * 64 = 16
  * indexer budget 64 / compress 4 -> 16 blocks: selection is genuinely sparse past token 64,
    so the sparse path is actually exercised rather than degenerating to dense
  * vocab 4096 on both sides. Megatron's ``NullTokenizer`` keeps ``vocab_size`` as given and uses
    id 4095 as EOD, which must equal HF's ``eos_token_id`` because that token is the PLE n-gram
    boundary -- a mismatch silently changes which n-grams are looked up.
"""

from __future__ import annotations

SEED = 20260912
SEQ = 512  # HF's QSA indexer is a Python double loop over (batch, seq); longer is slow, not wrong
BATCH = 2
NUM_BATCHES = 24  # 1 for the forward/gradient checks, up to 20 for the trajectory, plus spare
VOCAB = 4096
EOS = VOCAB - 1

HF_CONFIG = {
    "architectures": ["Qwen4ExpForCausalLM"],
    "model_type": "qwen4_exp_text",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "eos_token_id": EOS,
    "pad_token_id": None,
    "bos_token_id": None,
    "hidden_act": "silu",
    "hidden_size": 256,
    "initializer_range": 0.02,
    "intermediate_size": 512,
    "max_position_embeddings": 1024,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 64,
    "num_hidden_layers": 4,
    "layer_types": [
        "linear_attention",
        "linear_attention",
        "linear_attention",
        "qwen_sparse_attention",
    ],
    "rms_norm_eps": 1e-06,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000000,
        "partial_rotary_factor": 0.25,
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    },
    "tie_word_embeddings": False,
    "use_cache": False,
    "vocab_size": VOCAB,
    # Gated DeltaNet. sigmoid is Qwen3.8; Qwen3-Next uses silu here, and the two are not
    # interchangeable -- Megatron needs --gdn-output-gate-activation to match.
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 64,
    "linear_value_head_dim": 64,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "output_gate_type": "sigmoid",
    # QSA: a parameter-free mean-pool indexer selecting blocks for a GQA attention.
    "indexer_n_heads": 4,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 64,
    "indexer_budget": 64,
    "indexer_compress_ratio": 4,
    # MoE on every block, with a gated shared expert.
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.0,
    "output_router_logits": False,
    # Gated residual over 4 streams.
    "hc_count": 4,
    "hc_lowrank": 32,
    # PLE / n-gram memory on block 1.
    "ple_layer_ids": [2],
    "ple_embed_dim": 128,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 8,
    "ngram_vocab_size_base": 20000,
    "make_ngram_vocab_size_divisible_by": 128,
    "seed": 1234,
    "split_ngram_parts": 4,
}


def mcore_argv(*, seq_length: int = SEQ, micro_batch: int = BATCH) -> list[str]:
    """The same model as Megatron arguments.

    Kept explicit rather than derived, so this file alone tells you what each HF field becomes.
    The non-obvious correspondences:

      ``layer_types``          -> ``--hybrid-layer-pattern GEGEGEQE`` (G=GDN, Q=QSA, E=MoE)
      ``output_gate_type``     -> ``--gdn-output-gate-activation``
      ``norm_topk_prob: true`` -> **no flag**. HF's "softmax over all, take top-k, renormalize"
                                  is arithmetically Megatron's default post-softmax routing
                                  ("take top-k, softmax over those k"), so the correct
                                  expression is simply to leave ``--moe-router-pre-softmax``
                                  off. Setting a renormalization knob on top of the pre-softmax
                                  path would be a different function.
      ``hc_count``             -> ``--mhc-num-residual-streams``
      ``ple_layer_ids: [2]``   -> ``--engram-layer-ids 3`` (HF block 1 = Megatron layer 2,
                                  and Megatron counts layers from 1)
      ``ple_embed_dim 128``    -> ``--engram-memory-dim 64`` (= 128 / (ngram_size - 1))

    ``--attention-backend unfused`` and the ``--no-*-fusion`` flags are here for parity, not for
    performance: fused kernels reorder reductions and put ~1e-6 between the two sides, which is
    the same order as the quantity being measured.
    """
    return [
        # --- structure -------------------------------------------------------------------
        "--hybrid-layer-pattern", "GEGEGEQE",
        "--spec", "megatron.core.models.hybrid.hybrid_layer_specs",
        "gated_residual_hybrid_stack_spec",
        "--hidden-size", "256",
        "--num-attention-heads", "4",
        "--group-query-attention",
        "--num-query-groups", "2",
        "--kv-channels", "64",
        "--qk-layernorm",
        "--attention-output-gate",
        "--position-embedding-type", "rope",
        "--rotary-percent", "0.25",
        "--rotary-base", "10000000",
        "--no-rope-fusion",
        # --- GDN -------------------------------------------------------------------------
        "--linear-num-key-heads", "2",
        "--linear-num-value-heads", "4",
        "--linear-key-head-dim", "64",
        "--linear-value-head-dim", "64",
        "--linear-conv-kernel-dim", "4",
        "--gdn-output-gate-activation", "sigmoid",
        # --- QSA -------------------------------------------------------------------------
        "--qsa-indexer-n-heads", "4",
        "--qsa-indexer-head-dim", "64",
        "--qsa-indexer-budget", "64",
        "--qsa-indexer-compress-ratio", "4",
        # --- MoE -------------------------------------------------------------------------
        "--num-experts", "8",
        "--moe-router-topk", "2",
        "--moe-router-score-function", "softmax",
        "--moe-router-dtype", "fp32",
        "--moe-router-load-balancing-type", "none",
        "--moe-ffn-hidden-size", "128",
        "--moe-shared-expert-intermediate-size", "128",
        "--moe-shared-expert-gate",
        "--moe-grouped-gemm",
        "--moe-token-dispatcher-type", "alltoall",
        # --- gated residual --------------------------------------------------------------
        "--enable-mhc-connections",
        "--mhc-connection-variant", "gated_residual",
        "--mhc-num-residual-streams", "4",
        "--hc-lowrank", "32",
        # --- PLE / n-gram memory ---------------------------------------------------------
        "--engram-variant", "qwen",
        "--engram-vocab-sizes", "20000", "20000",
        "--engram-layer-ids", "3",
        "--engram-num-hash-heads", "8",
        "--engram-memory-dim", "64",
        "--engram-max-ngram-order", "3",
        "--engram-kernel-size", "4",
        "--engram-hash-seed", "1234",
        "--engram-eos-token-id", str(EOS),
        "--engram-unigram-vocab-size", str(VOCAB),
        # --- norms / activation ----------------------------------------------------------
        "--normalization", "RMSNorm",
        "--apply-layernorm-1p",  # zero-centered gamma, matching Qwen4ExpTextRMSNorm
        "--norm-epsilon", "1e-06",
        "--swiglu",
        "--disable-bias-linear",
        "--untie-embeddings-and-output-weights",
        "--init-method-std", "0.02",
        # --- runtime ---------------------------------------------------------------------
        "--transformer-impl", "transformer_engine",
        "--attention-backend", "unfused",
        "--attention-dropout", "0.0",
        "--hidden-dropout", "0.0",
        "--seq-length", str(seq_length),
        "--max-position-embeddings", "1024",
        "--micro-batch-size", str(micro_batch),
        "--global-batch-size", str(micro_batch),
        "--train-iters", "1",
        "--lr", "0.0001",
        "--tokenizer-type", "NullTokenizer",
        "--vocab-size", str(VOCAB),
        "--make-vocab-size-divisible-by", "1",
        "--mock-data",
        "--seed", "1234",
        "--no-gradient-accumulation-fusion",
        "--no-masked-softmax-fusion",
        "--no-bias-dropout-fusion",
        "--no-bias-swiglu-fusion",
        "--distributed-backend", "nccl",
        "--use-cpu-initialization",
    ]
