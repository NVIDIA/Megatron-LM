export MEGATRON_PATH=.
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
    --capture-range=cudaProfilerApi \
    --capture-range-end=stop \
    --cuda-graph-trace=node \
    -f true -x true \
    -o mhc_profile \
uv run --no-sync python -m torch.distributed.run --nproc-per-node=8 $MEGATRON_PATH/pretrain_gpt.py \
    --exit-duration-in-mins 225 \
    --distributed-timeout-minutes 60 \
    \
    --pipeline-model-parallel-size 1 \
    --tensor-model-parallel-size 2 \
    \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather \
    --use-mcore-models \
    --sequence-parallel \
    --disable-bias-linear \
    --transformer-impl transformer_engine \
    \
    --micro-batch-size 1 \
    --global-batch-size 8 \
    --train-samples 51200 \
    \
    --tokenizer-type NullTokenizer \
    --mock-data \
    --vocab-size 32000 \
    --split 99,1,0 \
    --no-mmap-bin-files \
    --num-workers 6 \
    \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --no-rope-fusion \
    --rotary-percent 1.0 \
    --normalization RMSNorm \
    --swiglu \
    \
    --num-layers 8 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --num-attention-heads 32 \
    --group-query-attention \
    --num-query-groups 8 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    --make-vocab-size-divisible-by 128 \
    \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --clip-grad 1.0 \
    --weight-decay 0.1 \
    \
    --lr-decay-samples 255126953 \
    --lr-warmup-samples 162761 \
    --lr 1.2e-5 \
    --min-lr 1.2e-6 \
    --lr-decay-style cosine \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --eval-iters 32 \
    --eval-interval 200 \
    \
    --auto-detect-ckpt-format \
    --no-ckpt-fully-parallel-save \
    --save-interval 200 \
    --init-method-std 0.008 \
    \
    --log-timers-to-tensorboard \
    --log-memory-to-tensorboard \
    --log-num-zeros-in-grad \
    --log-params-norm \
    --log-validation-ppl-to-tensorboard \
    --log-throughput \
    --log-interval 1 \
    \
    --bf16 \
    --dist-ckpt-strictness log_all \
    --exit-interval 10 \
    \
    --enable-hyper-connections \
    --recompute-granularity selective \
    --recompute-hyper-connections \
    --profile --profile-step-start 3 --profile-step-end 4 --profile-ranks 0