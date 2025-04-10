# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=Qwen/Qwen2.5-1.5B
export TOKENIZER=$MODEL
export NAME=Qwen2.5-1.5B
export ARGS="--size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-mainv1 --wandb-id $NAME --bs 64 --consumed-tokens 18000000000000 --tasks scripts/evaluation/swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS
