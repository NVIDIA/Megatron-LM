# Dense GPT + SwiGLU. Gates the converter's merge of the FSDP-split fc1 _w/_v
# halves back into a single gate_up_proj tensor.
MODEL_LABEL="Dense GPT + SwiGLU"
MODEL_TRANSFORM="SwiGLU fc1 _w/_v merge"
NUM_LAYERS=12
ARCH="--swiglu"
RESHARD_LAYOUTS="TP2 PP2"
