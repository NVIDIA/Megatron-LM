export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama3_8b_fp8"
bash examples/llama/train_llama3_8b_h100_fp8.sh \
	checkpoints/llama3_8b/wikipedia_fp8 \
	tensorboard_logs/llama3_8b_fp8 \
	model/llama3 \
	dataset/wikipedia_processed/wikipedia_processed_text_document \
	bf16 \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_wikipedia_fp8_$(date +'%y-%m-%d_%H-%M-%S').log"
