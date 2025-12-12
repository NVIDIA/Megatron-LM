echo "Loading common options"

export UB_TIMEOUT=720
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16
export NCCL_P2P_NET_CHUNKSIZE=2097152
export NCCL_DEBUG=WARN



COMMON_OPTIONS="\
    --tensor-model-parallel-size $TP  \
    --pipeline-model-parallel-size $PP  \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --${PRECISION:-bf16} \
    --te-rng-tracker \
    --rl-offload-optimizer-during-inference \
    --inference-dynamic-batching-buffer-size-gb 20 \
    --data-parallel-random-init \
    --attention-backend flash \
    --timing-log-level 1 \
    --log-timers-to-tensorboard \
    --save-retain-interval 120 \
    --inference-dynamic-batching-num-cuda-graphs 1 \
    --adam-beta1 0.9 \
    --adam-beta2 ${ADAM_BETA2:-0.95} \
    --adam-eps 1e-8 \
    "

if [ ${LOWER_PRECISION:-false} == true ]; then
    echo "Lower precision experiments, disabling cuda graphs."
    ENABLE_CUDA_GRAPH=false
    COMMON_OPTIONS="${COMMON_OPTIONS} --no-gradient-accumulation-fusion"
else 
    COMMON_OPTIONS="${COMMON_OPTIONS}"
fi

if [ ${ENABLE_CUDA_GRAPH:-true} == true ]; then
    COMMON_OPTIONS="${COMMON_OPTIONS} --cuda-graph-impl=local"
fi
