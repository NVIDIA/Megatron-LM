# Dense GPT + Multi-Token Prediction. Gates the converter's rename of the FSDP
# mtp_model_layer keys back to transformer_layer. MTP forbids learned_absolute
# position embeddings, so RoPE is required.
MODEL_LABEL="GPT + Multi-Token Prediction"
MODEL_TRANSFORM="MTP key-rename"
NUM_LAYERS=12
ARCH="--mtp-num-layers 1 --position-embedding-type rope \
  --untie-embeddings-and-output-weights"
RESHARD_LAYOUTS="TP2 PP2"
