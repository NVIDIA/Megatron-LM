# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""MFU (Model FLOP Utilization) calculation for MIMO-Bagel.

The MIMO-Bagel model has four components that contribute to total FLOPs:

  1. LLM backbone (Qwen2 dense or Qwen3 MoE) with MoT (Mixture-of-Transformers)
  2. SigLIP ViT vision encoder
  3. Vision→language projection MLP
  4. (Optional) diffusion VAE linear projections

MoT note
--------
Each token is processed by exactly ONE path (und OR gen) for dense linear ops
(QKV projections and MLP), so dense FLOPs per token equal a standard LLM despite
having double the parameters.  The attention softmax, however, operates over the
full packed sequence (N_und + N_gen) because both token types attend to each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import time
from typing import Optional, Sequence, Union
from megatron.core import parallel_state
import torch
from megatron.core.parallel_state import get_data_parallel_group


# ---------------------------------------------------------------------------
# GPU peak-TFLOPs detection
# ---------------------------------------------------------------------------

def detect_peak_tflops(default_tflops: float = 989.0) -> float:
    """Return per-device BF16 TFLOPs based on the current GPU name.

    Falls back to *default_tflops* when the GPU is unknown or CUDA is
    unavailable.
    """
    try:
        import torch
        name = torch.cuda.get_device_name().upper()
    except (ImportError, RuntimeError):
        return default_tflops

    if any(tag in name for tag in ("H100", "H800", "H200")):
        return 989.0
    if any(tag in name for tag in ("A100", "A800")):
        return 312.0
    if "L40" in name:
        return 181.05
    if "L20" in name:
        return 119.5
    if "H20" in name:
        return 148.0
    if "910B" in name:
        return 354.0
    return default_tflops


# ---------------------------------------------------------------------------
# Per-component FLOP coefficient helpers
# ---------------------------------------------------------------------------

def llm_flop_coefficients(
    language_config,
    vocab_size: int,
) -> tuple[float, float]:
    """Compute FLOP coefficients for the Qwen2/Qwen3 LLM backbone.

    Works for both dense (Qwen2) and MoE (Qwen3) configs derived from
    ``get_bagel_language_model_config`` / ``get_bagel_language_model_config_qwen3_30b``.

    Returns
    -------
    dense_token_factor : float
        FLOPs per token for all dense linear operations across all layers
        (attention projections, MLP / MoE, embedding + LM head).
    attn_factor : float
        FLOPs per unit of Σ(seqlen²) for quadratic attention operations.

    Usage
    -----
    total_llm_flops = dense_token_factor * N_tokens + attn_factor * sum(seqlen_i ** 2)

    Notes — why factor 6 for dense ops
    ------------------------------------
    For a linear layer  y = x W  with x:[S,m], W:[m,n]:

      Forward          y  = x W          →  2·S·m·n FLOPs
      Backward (input) dx = dy · Wᵀ      →  2·S·m·n FLOPs
      Backward (weight) dW = xᵀ · dy     →  2·S·m·n FLOPs
      Total                               =  6·S·m·n FLOPs

    Per token: 6 × (m×n) = 6 × (number of parameters in W).
    Hence ``dense_token_factor = 6 × dense_N`` where ``dense_N`` is the
    total parameter count of all linear layers (attention projections,
    MLP, embedding, LM head).

    Notes — why factor 12 for attention
    ------------------------------------
    Attention has two matrix multiplications; each gets the same 6×
    treatment as above (fwd + two backward matmuls = ×3, times 2 FLOPs
    per MAC = ×2 → ×6 total):

      QKᵀ:  fwd A=QKᵀ, bwd dQ=dA·K, bwd dK=Qᵀ·dA  →  6·S²·h·d
      AV:   fwd O=A·V,  bwd dA=dO·Vᵀ, bwd dV=Aᵀ·dO →  6·S²·h·d
      Total                                           = 12·S²·h·d

    Softmax (element-wise O(S²)) is excluded — it is dominated by the
    matmuls for large d and carries no large constant.
    Hence ``attn_factor = 12 × head_dim × num_heads × num_layers``.
    """
    hidden = language_config.hidden_size
    num_layers = language_config.num_layers
    num_heads = language_config.num_attention_heads
    num_kv_heads = getattr(language_config, "num_query_groups", num_heads) or num_heads
    head_dim = getattr(language_config, "kv_channels", None) or (hidden // num_heads)

    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim
    out_size = num_heads * head_dim
    attn_linear_N = hidden * (q_size + k_size + v_size + out_size)

    # MoE vs dense MLP.  For MoE each token activates *topk* experts; each
    # expert is a SwiGLU block (3 matrices: gate, up, down).
    num_moe_experts = getattr(language_config, "num_moe_experts", None)
    is_moe = num_moe_experts not in (None, 0)
    if is_moe:
        topk = getattr(language_config, "moe_router_topk", 1)
        moe_ffn = getattr(language_config, "moe_ffn_hidden_size",
                          language_config.ffn_hidden_size)
        # SwiGLU = 3 weight matrices (gate + up + down)
        mlp_N = hidden * moe_ffn * 3 * topk
    else:
        mlp_N = hidden * language_config.ffn_hidden_size * 3  # SwiGLU

    emb_N = vocab_size * hidden * 2  # token embedding + LM head
    dense_N = (attn_linear_N + mlp_N) * num_layers + emb_N
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_heads * num_layers
    return dense_token_factor, attn_factor


def effective_seqlen_sq_from_block_mask(block_mask) -> int:
    """Count effective (q, k) attention pairs from a BlockMask efficiently.

    Avoids ``to_dense()`` which allocates a [Q_LEN × KV_LEN] bool tensor —
    prohibitively large for sequences of 50 K+ tokens (50K² ≈ 2.5 GB).

    BlockMask internals
    -------------------
    PyTorch's ``create_block_mask`` divides the (Q, KV) space into blocks of
    size ``BLOCK_SIZE × BLOCK_SIZE`` and classifies each block as:

    * **Full**    — entirely unmasked; all BLOCK_SIZE² pairs are computed.
                   Counted in ``full_kv_num_blocks[b, h, q_block]``.
    * **Partial** — mask_mod is applied element-wise within the block; only
                   some pairs are computed.  Counted in ``kv_num_blocks``.
                   Diagonal blocks in causal attention contribute
                   BLOCK_SIZE×(BLOCK_SIZE+1)/2 ≈ BLOCK_SIZE²/2 pairs;
                   other boundary blocks vary.
    * **Empty**   — entirely masked; 0 pairs.  Not stored.

    Formula
    -------
    pairs ≈ full_blocks × BLOCK_SIZE²  +  partial_blocks × BLOCK_SIZE² / 2

    The ``/ 2`` approximation for partial blocks is exact for causal diagonal
    blocks and a reasonable average for segment-boundary blocks.  The error
    is bounded by ``partial_blocks × BLOCK_SIZE² / 2``, which is small
    compared to the full-block contribution for typical BAGEL sequences.

    Parameters
    ----------
    block_mask : BlockMask
        The block mask built by ``create_block_mask`` for this step.

    Returns
    -------
    int
        Approximate number of (q, k) pairs actually computed.

    Complexity
    ----------
    Time and memory O(Q_LEN / BLOCK_SIZE) — independent of Q_LEN for the
    summation step.
    """
    bs = block_mask.BLOCK_SIZE
    if isinstance(bs, (list, tuple)):
        bs_q, bs_kv = bs
    else:
        bs_q = bs_kv = bs
    block_area = bs_q * bs_kv
    # All heads share the same mask; use head 0.
    full_pairs    = int(block_mask.full_kv_num_blocks[0, 0].sum().item()) * block_area
    partial_pairs = int(block_mask.kv_num_blocks[0, 0].sum().item()) * block_area // 2
    return full_pairs + partial_pairs


def vit_flop_coefficients(vit_config) -> tuple[float, float]:
    """Compute FLOP coefficients for the SigLIP ViT vision encoder.

    SigLIP uses standard GELU (not SwiGLU), so the MLP has 2 weight matrices
    (fc1: hidden→ffn_hidden, fc2: ffn_hidden→hidden).

    Returns
    -------
    dense_token_factor : float
        FLOPs per vision token for all linear ops across ViT layers.
    attn_factor : float
        FLOPs per unit of Σ(N_vision_tokens²) for ViT self-attention.

    Usage
    -----
    total_vit_flops = dense_token_factor * N_vision_tokens
                      + attn_factor * sum(N_vit_tokens_i ** 2)
    """
    hidden = vit_config.hidden_size
    num_layers = vit_config.num_hidden_layers
    num_heads = vit_config.num_attention_heads
    head_dim = hidden // num_heads
    ffn_hidden = vit_config.intermediate_size

    # Full self-attention (no GQA in ViT): Q + K + V + O projections
    attn_linear_N = hidden * hidden * 4
    # Standard GELU MLP: 2 matrices
    mlp_N = hidden * ffn_hidden * 2

    dense_N = (attn_linear_N + mlp_N) * num_layers
    dense_token_factor = 6.0 * dense_N
    attn_factor = 12.0 * head_dim * num_heads * num_layers
    return dense_token_factor, attn_factor


def projection_mlp_flops(vit_hidden: int, llm_hidden: int) -> float:
    """FLOPs per vision token for the 2-layer vision→language projection MLP.

    The projection is:  vit_hidden → llm_hidden → llm_hidden  (2 linear layers,
    no activation between in terms of FLOP cost).

    Returns a scalar: FLOPs per vision token (forward + backward).
    """
    # FC1: vit_hidden × llm_hidden,  FC2: llm_hidden × llm_hidden
    dense_N = vit_hidden * llm_hidden + llm_hidden * llm_hidden
    return 6.0 * dense_N


def diffusion_projection_flops(
    latent_patch_size: int,
    latent_channels: int,
    llm_hidden: int,
) -> float:
    """FLOPs per diffusion/VAE token for the vae2llm and llm2vae projections.

    The diffusion submodule adds two small linear layers:
      vae2llm: (latent_patch_size² × latent_channels) → llm_hidden
      llm2vae: llm_hidden → (latent_patch_size² × latent_channels)

    Returns a scalar: FLOPs per diffusion token (forward + backward).
    """
    vae_dim = latent_patch_size ** 2 * latent_channels
    dense_N = vae_dim * llm_hidden * 2  # both directions
    return 6.0 * dense_N


# ---------------------------------------------------------------------------
# Main MFU computation
# ---------------------------------------------------------------------------

def compute_bagel_mfu(
    # Token counts for the measurement window
    llm_token_window: int,
    llm_seqlen_sq_window: float,
    vision_token_window: int,
    vision_seqlen_sq_window: float,
    # Model configs
    language_config,
    vit_config,
    vocab_size: int,
    # Timing
    elapsed_seconds: float,
    # Hardware
    world_size: int,
    peak_tflops_per_gpu: float,
    # Optional modality configs
    vit_hidden: Optional[int] = None,
    n_diffusion_tokens: int = 0,
    latent_patch_size: int = 2,
    latent_channels: int = 16,
    # Flags
    include_vit: bool = True,
    include_projection: bool = True,
    include_diffusion: bool = False,
) -> dict:
    """Compute MFU for a MIMO-Bagel training window.

    Parameters
    ----------
    llm_token_window : int
        Total tokens (text + image) fed to the LLM in the window.
    llm_seqlen_sq_window : float
        Sum of effective attention pairs over all steps in the window.
        Use ``packed_mot_effective_seqlen_sq(sample_lens, split_lens, attn_modes)``
        to correctly account for the MoT mixed mask and packed-sample boundaries.
    vision_token_window : int
        Total vision (patch) tokens processed by the ViT in the window.
    vision_seqlen_sq_window : float
        Sum of (N_vit_tokens_i)² for ViT self-attention.
    language_config : TransformerConfig
        Megatron TransformerConfig for the LLM backbone (Qwen2 or Qwen3).
    vit_config : SiglipVisionConfig
        HuggingFace SigLIP config for the vision encoder.
    vocab_size : int
        Vocabulary size (used for embedding + LM-head FLOPs).
    elapsed_seconds : float
        Wall-clock seconds elapsed over the measurement window.
    world_size : int
        Total number of GPUs.
    peak_tflops_per_gpu : float
        Per-GPU peak BF16 TFLOPs (use ``detect_peak_tflops()`` if unknown).
    vit_hidden : int, optional
        ViT hidden size for projection-MLP FLOPs; inferred from vit_config if None.
    n_diffusion_tokens : int
        Total diffusion/VAE tokens in the window (0 = skip).
    latent_patch_size : int
        VAE latent patch size (default 2).
    latent_channels : int
        Number of VAE latent channels.
    include_vit : bool
        Whether to include ViT FLOPs in the total.
    include_projection : bool
        Whether to include projection MLP FLOPs.
    include_diffusion : bool
        Whether to include diffusion projection FLOPs.

    Returns
    -------
    dict with keys:
        ``mfu``            – MFU value in [0, 1]
        ``mfu_pct``        – MFU as a percentage
        ``actual_tflops``  – Actual TFLOPs/s across all GPUs
        ``llm_flops``      – FLOPs attributed to the LLM backbone
        ``vit_flops``      – FLOPs attributed to the ViT encoder
        ``proj_flops``     – FLOPs attributed to the projection MLP
        ``diff_flops``     – FLOPs attributed to diffusion projections
        ``total_flops``    – Sum of all component FLOPs
    """
    if elapsed_seconds <= 0:
        raise ValueError("elapsed_seconds must be > 0")

    # --- LLM backbone ---
    # print("llm_token_window", llm_token_window)
    # print("llm_seqlen_sq_window", llm_seqlen_sq_window)
    # print("vision_token_window", vision_token_window)
    # print("vision_seqlen_sq_window", vision_seqlen_sq_window)
    # print("n_diffusion_tokens", n_diffusion_tokens)
    llm_dense_factor, llm_attn_factor = llm_flop_coefficients(language_config, vocab_size)
    llm_flops = (
        llm_dense_factor * llm_token_window
        + llm_attn_factor * llm_seqlen_sq_window
    )

    # --- SigLIP ViT ---
    vit_flops = 0.0
    if include_vit and vision_token_window > 0:
        vit_dense_factor, vit_attn_factor = vit_flop_coefficients(vit_config)
        vit_flops = (
            vit_dense_factor * vision_token_window
            + vit_attn_factor * vision_seqlen_sq_window
        )

    # --- Vision→language projection MLP ---
    proj_flops = 0.0
    if include_projection and vision_token_window > 0:
        _vit_hidden = vit_hidden or vit_config.hidden_size
        proj_flops = projection_mlp_flops(_vit_hidden, language_config.hidden_size) * vision_token_window

    # --- Diffusion VAE projections ---
    diff_flops = 0.0
    if include_diffusion and n_diffusion_tokens > 0:
        diff_flops = diffusion_projection_flops(
            latent_patch_size, latent_channels, language_config.hidden_size
        ) * n_diffusion_tokens

    dp_flops = llm_flops + vit_flops + proj_flops + diff_flops

    global_flops = 0.0
    world_size = torch.distributed.get_world_size()
    try:
        dp_group = get_data_parallel_group()
        t = torch.tensor(
            [dp_flops],
            dtype=torch.float64,
            device='cuda',
        )
        torch.distributed.all_reduce(t, group=dp_group, op=torch.distributed.ReduceOp.SUM)
        # After reduce: t.item() == DP × sample_FLOPs / T == global / T
        global_flops = t.item()

    except Exception as e:
        try:
            import torch.distributed as _td
            if _td.is_initialized() and _td.get_rank() == 0:
                print(f"[MFU] warning: DP all-reduce failed, "
                      f"reporting per-rank fallback: {e}", flush=True)
        except Exception:
            pass

    actual_tflops = global_flops / world_size / elapsed_seconds / 1e12  # sample_FLOPs / T
    mfu = actual_tflops / peak_tflops_per_gpu if peak_tflops_per_gpu > 0 else 0.0

    return {
        "mfu": mfu,
        "mfu_pct": mfu * 100.0,
        "actual_tflops": actual_tflops,
        "llm_flops": llm_flops,
        "vit_flops": vit_flops,
        "proj_flops": proj_flops,
        "diff_flops": diff_flops,
        "dp_flops": dp_flops,
        "global_flops": global_flops,
    }


# ---------------------------------------------------------------------------
# Convenience tracker for use inside a training loop
# ---------------------------------------------------------------------------

@dataclass
class BagelMFUTracker:
    """Accumulates per-step statistics and computes MFU over a logging window.

    Typical usage in a training loop::

        tracker = BagelMFUTracker(
            language_config=language_config,
            vit_config=vit_config,
            vocab_size=151936,
            world_size=dist.get_world_size(),
        )

        for step, batch in enumerate(dataloader):
            t0 = time.time()
            loss = train_step(batch)
            elapsed = time.time() - t0

            # Tracker rebuilds the attention BlockMask internally from the raw
            # ingredients, so the caller does not need access to the one
            # constructed inside BagelMCoreModel.forward().
            tracker.update(
                sample_lens=batch["sample_lens"],
                split_lens=batch["split_lens"],
                attn_modes=batch["attn_modes"],
                vit_token_seqlens=batch.get("vit_token_seqlens", []),
                elapsed=elapsed,
            )

            if (step + 1) % log_interval == 0:
                metrics = tracker.compute_and_reset()
                print(f"MFU: {metrics['mfu_pct']:.1f}%")
    """

    language_config: object
    vit_config: object
    vocab_size: int
    world_size: int
    peak_tflops_per_gpu: float = field(default_factory=lambda: detect_peak_tflops())

    # Include flags
    include_vit: bool = True
    include_projection: bool = True
    include_diffusion: bool = False
    latent_patch_size: int = 2
    latent_channels: int = 16

    # BlockMask construction params (match mcore_bagel_llm.py)
    block_size: int = 128

    # Logging cadence for step_and_log()
    log_interval: int = 10

    # Accumulated window state (not set by caller)
    _llm_tokens: int = field(default=0, init=False)
    _llm_seqlen_sq: float = field(default=0.0, init=False)
    _vision_tokens: int = field(default=0, init=False)
    _vision_seqlen_sq: float = field(default=0.0, init=False)
    _elapsed: float = field(default=0.0, init=False)
    _last_step_time: Optional[float] = field(default=None, init=False)
    _step_count: int = field(default=0, init=False)

    def update(
        self,
        sample_lens: Sequence[int],
        split_lens: Sequence[int],
        attn_modes: Sequence[str],
        vit_token_seqlens: Union[Sequence[int], "torch.Tensor"] = (),
        elapsed: float = 0.0,
        device: Union[str, "torch.device"] = "cuda",
    ) -> None:
        """Accumulate stats for one training step.

        The tracker rebuilds the FlexAttention BlockMask internally from
        ``sample_lens`` / ``split_lens`` / ``attn_modes`` so the caller does not
        need access to the BlockMask built inside ``BagelMCoreModel.forward()``.
        Rebuilding duplicates the model's mask construction, but
        ``_compile=True`` keeps the amortised cost negligible.

        Parameters
        ----------
        sample_lens : Sequence[int]
            Per-sample lengths of the packed sequence — defines cross-sample
            isolation boundaries.
        split_lens : Sequence[int]
            Per-segment lengths within each sample (flat concatenation across
            samples).  Identifies text / ViT / VAE segments.
        attn_modes : Sequence[str]
            Per-segment attention mode — one of ``'causal'``, ``'full'``,
            ``'noise'``.  Must be the same length as ``split_lens``.
        vit_token_seqlens : Sequence[int] | Tensor
            Per-image ViT patch counts for this step.  ViT uses
            ``flash_attn_varlen_func`` with ``cu_seqlens`` (block-diagonal
            per-image attention), so we need per-image lengths to compute
            ``Σ s_i²`` — NOT ``(Σ s_i)²``.  Pass an empty sequence for
            text-only batches.
        elapsed : float
            Wall-clock seconds for this step.
        device : str | torch.device
            Device on which to construct the BlockMask.
        """
        import torch
        from bagel.data.data_utils import create_sparse_mask
        from torch.nn.attention.flex_attention import create_block_mask

        sparse_mask = create_sparse_mask(
            list(sample_lens), list(split_lens), list(attn_modes), device
        )
        seqlen = int(sum(sample_lens))
        block_mask = create_block_mask(
            sparse_mask,
            B=1, H=1,  # mask is head-invariant; H=1 is cheapest
            Q_LEN=seqlen, KV_LEN=seqlen,
            device=device,
            BLOCK_SIZE=self.block_size,
            _compile=True,
        )

        if isinstance(vit_token_seqlens, torch.Tensor):
            vit_seqlens = vit_token_seqlens.flatten().tolist()
        else:
            vit_seqlens = list(vit_token_seqlens)

        self._llm_tokens       += seqlen
        self._llm_seqlen_sq    += effective_seqlen_sq_from_block_mask(block_mask)
        self._vision_tokens    += sum(vit_seqlens)
        self._vision_seqlen_sq += sum(s * s for s in vit_seqlens)
        self._elapsed          += elapsed

    def compute_and_reset(self) -> dict:
        """Compute MFU over the accumulated window, then reset counters.

        Each DP rank first computes its own ``actual_tflops`` from its local
        token/seqlen² counters.  The results are then all-reduced (summed)
        across data-parallel ranks so that the final MFU reflects the total
        FLOPs computed cluster-wide, not just on this rank.  The wall-clock
        ``elapsed`` is not reduced because all ranks run in lock-step.

        Falls back to the local value silently if Megatron parallel state is
        not initialised (e.g. in unit tests).

        Returns the same dict as ``compute_bagel_mfu``.
        """
        result = compute_bagel_mfu(
            llm_token_window=self._llm_tokens,
            llm_seqlen_sq_window=self._llm_seqlen_sq,
            vision_token_window=self._vision_tokens,
            vision_seqlen_sq_window=self._vision_seqlen_sq,
            language_config=self.language_config,
            vit_config=self.vit_config,
            vocab_size=self.vocab_size,
            elapsed_seconds=self._elapsed,
            world_size=self.world_size,
            peak_tflops_per_gpu=self.peak_tflops_per_gpu,
            latent_patch_size=self.latent_patch_size,
            latent_channels=self.latent_channels,
            include_vit=self.include_vit,
            include_projection=self.include_projection,
            include_diffusion=self.include_diffusion,
        )
        self._reset()



        return result

    def _reset(self) -> None:
        self._llm_tokens       = 0
        self._llm_seqlen_sq    = 0.0
        self._vision_tokens    = 0
        self._vision_seqlen_sq = 0.0
        self._elapsed          = 0.0

    def step_and_log(self, data_batch: dict, prefix: str = "[MFU]") -> None:
        """One-call helper for MIMO-Bagel training-loop integration.

        Intended use inside ``forward_step`` — extracts mask ingredients and
        per-image ViT seqlens from the MIMO batch, times one iteration
        (wall-clock between consecutive calls), computes **per-step** MFU
        (the accumulator is reset every call — no windowed averaging), and
        prints on global rank 0 every ``self.log_interval`` steps.

        Silently no-ops on ranks that don't hold the Python-list metadata
        (e.g. TP/CP rank 1+, where ``sample_lens`` etc. are not broadcast),
        so those ranks never enter the DP all-reduce collective.
        """
        import time
        import torch

        now = time.time()
        if self._last_step_time is None:
            self._last_step_time = now
            return
        elapsed = now - self._last_step_time
        self._last_step_time = now

        sample_lens = data_batch.get('sample_lens')
        split_lens  = data_batch.get('split_lens')
        attn_modes  = data_batch.get('attn_modes')
        if sample_lens is None or split_lens is None or attn_modes is None:
            return

        vit_seqlens: Sequence = ()
        vision_encoder_input = (
            (data_batch.get('modality_inputs') or {}).get('images', {}) or {}
        ).get('vision_encoder', {}) or {}
        if 'vit_token_seqlens' in vision_encoder_input:
            vit_seqlens = vision_encoder_input['vit_token_seqlens']

        try:
            self.update(
                sample_lens=sample_lens,
                split_lens=split_lens,
                attn_modes=attn_modes,
                vit_token_seqlens=vit_seqlens,
                elapsed=elapsed,
                device='cuda',
            )
        except Exception as e:
            if _is_global_rank_0():
                print(f"{prefix} tracker.update failed: {e}", flush=True)
            return

        metrics = self.compute_and_reset()
        self._step_count += 1

        if self._step_count % self.log_interval == 0 and _is_global_rank_0():
            print(
                f"{prefix} iter={self._step_count} "
                f"mfu={metrics['mfu_pct']:.2f}% "
                f"actual={metrics['actual_tflops']:.1f} TFLOPs/s "
                f"elapsed={elapsed*1000:.0f}ms "
                f"(llm={metrics['llm_flops']/1e12:.1f}, "
                f"vit={metrics['vit_flops']/1e12:.1f}, "
                f"proj={metrics['proj_flops']/1e12:.2f}, "
                f"diff={metrics['diff_flops']/1e12:.2f}, "
                f"dp_flops={metrics['dp_flops']/1e12:.2f}, "
                f"global_flops={metrics['global_flops']/1e12:.2f})",
                flush=True,
            )


def _is_global_rank_0() -> bool:
    """True on global rank 0, or always in non-distributed contexts."""
    try:
        import torch
        if not torch.distributed.is_initialized():
            return True
        return torch.distributed.get_rank() == 0
    except Exception:
        return True
