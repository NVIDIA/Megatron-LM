export HOST_TENSORBOARD_LOGS_PATH="./tensorboard_logs/llama3_8b_fp8"
bash examples/llama/train_llama3_8b_h100_fp8.sh \
	checkpoints/llama3_8b/mock \
	tensorboard_logs/llama3_8b_fp8 \
	MOCK \
	MOCK \
  2>&1 | tee "${HOST_TENSORBOARD_LOGS_PATH}/training_mock_$(date +'%y-%m-%d_%H-%M-%S').log"
