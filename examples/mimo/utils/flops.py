# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""
Multimodal FLOPS calculation for MIMO/Bagel models.

Extends Megatron's LLM-only FLOPS counting to include vision encoder,
projector, diffusion submodules, and CE logits.
"""


def num_floating_point_operations_multimodal(args, batch_size):
    """Calculate total FLOPS for a multimodal (Bagel) training iteration.

    Accounts for: LLM decoder, ViT encoder, vision projector,
    diffusion submodules (vae2llm, llm2vae, timestep embedder), and CE logits.
    """
    from megatron.training.training import num_floating_point_operations as llm_flops_fn

    # ── LLM Decoder FLOPS (existing Megatron formula) ──
    llm_flops = llm_flops_fn(args, batch_size)

    fwd_bwd = 3   # forward + backward (wgrad + dgrad)
    fma = 2        # multiply-add = 2 flops

    def _tokens_per_iteration(avg_tokens):
        """Convert avg token stats to per-iteration token count.

        Historical args use *per micro-batch* naming, while this function is
        invoked with iteration-level `batch_size` (global number of sequences).
        Scale by `batch_size / micro_batch_size` so units stay consistent.
        """
        if avg_tokens <= 0:
            return 0.0
        micro_batch_size = max(getattr(args, "micro_batch_size", 1), 1)
        return float(avg_tokens) * (float(batch_size) / float(micro_batch_size))

    # ── Vision Encoder (SigLIP ViT) FLOPS ──
    vit_flops = 0
    vit_h = getattr(args, 'vit_hidden_size', 0)
    if vit_h > 0:
        vit_layers = args.vit_num_layers           # 26
        vit_ffn = args.vit_intermediate_size        # 4304
        vit_tokens = _tokens_per_iteration(getattr(args, "avg_vit_tokens_per_batch", 0))

        # Per layer: QKV proj + attn output proj + MLP (2 linears)
        linear_per_layer = (
            fma * vit_tokens * (
            vit_h * 3 * vit_h          # QKV projection
            + vit_h * vit_h             # output projection
            + vit_h * vit_ffn           # MLP fc1
            + vit_ffn * vit_h           # MLP fc2
            )
        )

        # Attention (QK^T + attn@V): quadratic in per-sample seq length.
        # Must use num_samples * s^2, NOT total_tokens^2.
        # Full attention (no causal mask), so no /2 factor.
        avg_vit_seq = getattr(args, 'avg_vit_seq_per_sample', 0)
        if avg_vit_seq > 0 and vit_tokens > 0:
            num_vit_samples = vit_tokens / float(avg_vit_seq)
            attn_per_layer = fma * 2 * num_vit_samples * (avg_vit_seq ** 2) * vit_h
        else:
            attn_per_layer = 0

        per_layer = linear_per_layer + attn_per_layer
        vit_fwd_bwd = 1 if getattr(args, 'freeze_vit', False) else fwd_bwd
        vit_flops = vit_fwd_bwd * vit_layers * per_layer

    # ── Vision Projector (2-layer MLP: vit_h -> llm_h -> llm_h) ──
    proj_flops = 0
    if vit_h > 0:
        llm_h = args.hidden_size
        vit_tokens = _tokens_per_iteration(getattr(args, "avg_vit_tokens_per_batch", 0))
        proj_flops = fwd_bwd * fma * vit_tokens * (
            vit_h * llm_h       # layer 1
            + llm_h * llm_h     # layer 2
        )

    # ── Diffusion Submodules FLOPS ──
    diff_flops = 0
    latent_patch_size = getattr(args, 'latent_patch_size', 0)
    if latent_patch_size > 0:
        llm_h = args.hidden_size
        latent_channels = getattr(args, "vae_latent_channels", 16)
        latent_dim = latent_patch_size ** 2 * latent_channels  # 2*2*16=64
        latent_tokens = _tokens_per_iteration(getattr(args, "avg_latent_tokens_per_batch", 0))

        diff_flops = fwd_bwd * fma * latent_tokens * (
            latent_dim * llm_h              # vae2llm
            + 256 * llm_h + llm_h * llm_h   # timestep embedder MLP
            + llm_h * latent_dim             # llm2vae (output projection)
        )

    # ── Optional CE logits correction (lm_head) ──
    # Megatron LLM FLOPs include logits on full sequence length. For multimodal,
    # CE is often computed on a token subset. If user provides
    # avg_ce_tokens_per_batch, replace full-seq logits estimate with subset one.
    ce_tokens = _tokens_per_iteration(getattr(args, "avg_ce_tokens_per_batch", 0))
    if ce_tokens > 0 and hasattr(args, "padded_vocab_size") and hasattr(args, "seq_length"):
        mtp_num_layers = getattr(args, "mtp_num_layers", 0) or 0
        logits_factor = (1 + mtp_num_layers)
        full_logits_flops = (
            fwd_bwd * fma * float(batch_size) * float(args.seq_length)
            * float(args.hidden_size) * float(args.padded_vocab_size) * logits_factor
        )
        subset_logits_flops = (
            fwd_bwd * fma * ce_tokens
            * float(args.hidden_size) * float(args.padded_vocab_size) * logits_factor
        )
        llm_flops = llm_flops - full_logits_flops + subset_logits_flops

    return llm_flops + vit_flops + proj_flops + diff_flops
