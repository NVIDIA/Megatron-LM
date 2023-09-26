DATA_PATH=/lustre/fsw/adlr/adlr-nlp/adlr_ci/megatron/data
MEGATRON_LM_PATH=/lustre/fsw/joc/shanmugamr/megatron_core/megatron-lm

srun -t 120 --container-image nvcr.io/nvidia/pytorch:23.04-py3 --container-mounts $MEGATRON_LM_PATH:/workspace/megatron-lm,$DATA_PATH:/workspace/data --account coreai_dlalgo_genai -N 1 -J coreai_dlalgo_genai-multimodal:bert_core  -p interactive --no-container-mount-home --pty /bin/bash


mkdir logs
mkdir checkpoints
cd megatron-lm

export CUDA_DEVICE_MAX_CONNECTIONS=1

torchrun --nproc_per_node 8 --nnodes 1 pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --log-params-norm --log-num-zeros-in-grad --log-validation-ppl-to-tensorboard --log-timers-to-tensorboard --tensorboard-dir /workspace/logs --micro-batch-size 4 --global-batch-size 128 --seq-length 512 --max-position-embeddings 512 --train-iters 50 --timing-log-level 2 --lr-decay-iters 990000 --save /workspace/checkpoints --load /workspace/checkpoints --data-path /workspace/data/bert_data/my-bert_00_text_sentence --vocab-file /workspace/data/bert_data/vocab.txt --split 949,50,1 --distributed-backend nccl --lr 0.0001 --min-lr 0.00001 --lr-warmup-fraction 0.01 --log-interval 1 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 2 --no-gradient-accumulation-fusion --fp16 --use-mcore-models