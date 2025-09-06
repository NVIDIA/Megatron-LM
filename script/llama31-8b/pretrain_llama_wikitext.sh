export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama3_8b_fp8"
bash examples/llama/train_llama3_8b_h100_fp8.sh \
	checkpoints/llama3_8b/wikitext_bf16 \
	tensorboard_logs/llama3_8b_fp8 \
	model/llama3 \
	dataset/wikitext_processed/wikitext_processed_text_document \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/raining_wikitext_$(date +'%y-%m-%d_%H-%M-%S').log"
