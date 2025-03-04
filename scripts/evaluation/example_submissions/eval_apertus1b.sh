# Make sure to set your WANDB_API_KEY.
export LOGS_ROOT=$SCRATCH/eval-logs/main-v1
export MODEL=/capstor/store/cscs/swissai/a06/main_run_megatron/Megatron-LM/logs/Meg-Runs/main-runs-v1/apertus3-1b-21-nodes/checkpoints/
export NAME=Apertus3-1.5B
export ARGS="--convert-to-hf --size 1 --wandb-entity epflmlo-epfl --wandb-project swissai-eval-mainv1 --wandb-id $NAME --bs 64 --tokens-per-iter 2064384 --tasks scripts/evaluation/swissai_eval"

bash scripts/evaluation/submit_evaluation.sh $MODEL $ARGS --iterations "40000,100000,140000,200000,240000,300000"
