# MoE with grouped-GEMM experts (mixtral-like). Gates the converter's re-stack of
# the per-expert FSDP tensors into the grouped
# experts.experts.linear_fc{1,2}.weight (num_experts, ...) layout mcore expects.
# ETP>1 forbids bias, so keep --disable-bias-linear.
MODEL_LABEL="MoE, grouped-GEMM (mixtral-like)"
MODEL_TRANSFORM="grouped-expert restack"
NUM_LAYERS=12
ARCH="--swiglu --num-experts 8 --moe-grouped-gemm --disable-bias-linear"
RESHARD_LAYOUTS="EP2 TP2SP"
