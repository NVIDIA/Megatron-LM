# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=HuggingFaceTB/SmolLM2-1.7B-intermediate-checkpoints
export TOKENIZER=$MODEL
export NAME=SmolLM2-1.7B
export ARGS="--size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-mainv1 --wandb-id $NAME --bs 64 --tokens-per-iter 2097152 --tasks scripts/evaluation/swissai_eval"

ITS="125000,1000000,2000000,3000000,4000000,5000000"
REVS=$(echo "$ITS" | sed 's/\([^,]*\)/step-\1/g')
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations $ITS --revisions $REVS

# Now do the final SmolLM2 checkpoint.
export MODEL=HuggingFaceTB/SmolLM2-1.7B
export TOKENIZER=$MODEL
bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations 5240000
