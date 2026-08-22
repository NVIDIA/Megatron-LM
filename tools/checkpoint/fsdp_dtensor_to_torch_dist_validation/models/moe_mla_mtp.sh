# (Extra archetype) MoE + Multi-Latent Attention + MTP (deepseek-like). Exercises
# the MLA and MTP passthrough paths on top of the grouped-expert restack — a
# realistic multi-feature combination. MTP requires RoPE.
MODEL_LABEL="MoE + MLA + MTP (deepseek-like)"
MODEL_TRANSFORM="MLA + MTP passthrough over grouped-expert restack"
NUM_LAYERS=12
ARCH="--swiglu --num-experts 8 --moe-grouped-gemm --disable-bias-linear \
  --multi-latent-attention --q-lora-rank 512 --kv-lora-rank 256 \
  --mtp-num-layers 1 --position-embedding-type rope"
# Load-side reshard not part of the validated coverage for this extra archetype.
RESHARD_LAYOUTS=""
