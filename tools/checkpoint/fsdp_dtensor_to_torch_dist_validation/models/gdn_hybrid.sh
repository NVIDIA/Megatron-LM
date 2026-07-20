# Hybrid Gated-DeltaNet + shared-expert MoE + GQA (Qwen-Next-like). The most
# complex family here. Gates the converter's split of the fused GatedDeltaNet
# in_proj/conv1d projections into the per-component sub-keys the classic
# checkpoint stores, on top of the non-grouped expert restack. A GDN layer is
# placed every 3rd layer (--linear-attention-freq 3), so 6 layers is enough.
#
# Layers are interleaved (GDN vs attention), so the converter keeps a per-layer
# layout; the load-side reshard sweep is not part of the default coverage here.
#
# GatedDeltaNet needs the real flash-linear-attention package (the mcore dev
# image ships an insufficient `fla` stub), installed via EXTRA_SETUP.
MODEL_LABEL="Hybrid Gated-DeltaNet + MoE (Qwen-Next-like)"
MODEL_TRANSFORM="GDN in_proj/conv1d factory split + non-grouped experts"
NUM_LAYERS=6
ARCH="--group-query-attention --num-query-groups 2 --swiglu --disable-bias-linear \
  --rotary-percent 0.5 --no-rope-fusion --apply-layernorm-1p --apply-wd-to-qk-layernorm \
  --attention-output-gate --experimental-attention-variant gated_delta_net \
  --linear-attention-freq 3 --linear-conv-kernel-dim 4 \
  --linear-key-head-dim 64 --linear-value-head-dim 64 \
  --linear-num-key-heads 4 --linear-num-value-heads 8 \
  --untie-embeddings-and-output-weights \
  --num-experts 32 --moe-ffn-hidden-size 64 --moe-shared-expert-intermediate-size 64 \
  --moe-shared-expert-gate --moe-router-load-balancing-type aux_loss --moe-router-topk 8 \
  --moe-router-dtype fp32 --attention-softmax-in-fp32 --attention-backend unfused"
RESHARD_LAYOUTS=""
EXTRA_SETUP="pip install --no-input -q flash-linear-attention"
