export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama32_1b_fp8"
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp4'/" \
    megatron/core/tensor_parallel/layers.py
sed -i "s/^\([[:space:]]*custom_quant_type[[:space:]]*=[[:space:]]*\)'[^']*'/\1'mxfp4'/" \
    megatron/core/transformer/dot_product_attention.py
bash examples/llama/train_llama32_1b_h100_fp8.sh \
	checkpoints/llama32_1b/wikipedia_all_fp4 \
	tensorboard_logs/llama32_1b_fp8 \
	model/llama3.2-1b \
	dataset/wikipedia_processed/wikipedia_processed_text_document \
	bf16 \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_wikipedia_all_fp4_$(date +'%y-%m-%d_%H-%M-%S').log"
