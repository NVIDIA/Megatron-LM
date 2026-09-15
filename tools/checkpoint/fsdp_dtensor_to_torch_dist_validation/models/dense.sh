# Dense GPT (GELU) — the baseline. Gates the converter's per-layer stacking of a
# homogeneous decoder into mcore's native stacked torch_dist layout.
MODEL_LABEL="Dense GPT (GELU)"
MODEL_TRANSFORM="dense layer-stack"
NUM_LAYERS=12
ARCH=""
RESHARD_LAYOUTS="TP2 PP2"
