# Non-grouped (SequentialMLP) MoE with a shared expert and shared-expert gate.
# Gates the converter's re-stack of the mlp.experts.local_experts.{i}.linear_fc*
# layout (models without --moe-grouped-gemm) into the grouped key the factory
# expects, while leaving the shared expert untouched.
MODEL_LABEL="MoE, non-grouped shared-expert + gate"
MODEL_TRANSFORM="non-grouped local_experts restack"
NUM_LAYERS=12
ARCH="--swiglu --num-experts 8 --disable-bias-linear \
  --moe-shared-expert-intermediate-size 128 --moe-shared-expert-gate \
  --moe-router-load-balancing-type aux_loss --moe-router-topk 2"
RESHARD_LAYOUTS="EP2"
