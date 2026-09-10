export  PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export  NVTE_FUSED_ATTN=0
#export  NVTE_CPU_OFFLOAD_V1=1
export  NVTE_ALLOW_NONDETERMINISTIC_ALGO=0

  # FSDP Perf settings
export  CUDA_DEVICE_MAX_CONNECTIONS=32
export  NVTE_NORM_FWD_USE_CUDNN=1
export  NVTE_NORM_BWD_USE_CUDNN=1
export  NVTE_FWD_LAYERNORM_SM_MARGIN=16
export  NVTE_BWD_LAYERNORM_SM_MARGIN=16

# Required by GroupedMLP._is_fused_impl_supported() whenever the TE op fuser is
# combined with a GLU activation; without it the fused path is rejected.
export  NVTE_CUTEDSL_FUSED_GROUPED_MLP=1

export  PYTHONWARNINGS=ignore
export  NCCL_DEBUG=VERSION
export  NCCL_GRAPH_REGISTER=0
export  NCCL_ALGO=Ring

#Torch Compile stuff
# Dynamo reads TORCH_COMPILE_DISABLE at import time, so it also neutralizes the
# @jit_fuser (torch.compile) decorators that are bound before arg parsing.
export NO_TORCH_COMPILE=1
export TORCH_COMPILE_DISABLE=1


MEGATRON_LM_DIR="$LUSTRE_ROOT/Megatron-LM"
LOGS_DIR="${MEGATRON_LM_DIR}/logs"
#SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#cd "$SCRIPT_DIR"
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

GPUS_PER_NODE=${GPUS_PER_NODE:-4}
MASTER_PORT=${MASTER_PORT:-6000}

USE_MOCK_DATA=${USE_MOCK_DATA:-1}

DATA_CACHE_PATH=${DATA_CACHE_PATH:-"${MEGATRON_LM_DIR}/data_cache"}
TENSORBOARD_PATH=${TENSORBOARD_PATH:-"${MEGATRON_LM_DIR}/tensorboard_logs"}
mkdir -p "$TENSORBOARD_PATH"

DISTRIBUTED_ARGS=(
    --nproc_per_node "$GPUS_PER_NODE"
    --nnodes 1
    --master_addr localhost
    --master_port "$MASTER_PORT"
)


if [[ "$USE_MOCK_DATA" == "1" ]]; then
    DATA_ARGS=(
        --mock-data
        --tokenizer-type NullTokenizer
        --vocab-size 65536
        --split 949,50,1
    )
else
    DATA_PATH=${DATA_PATH:-""}
    mkdir -p "$DATA_CACHE_PATH"

    if [[ -z "$DATA_PATH" ]]; then
        echo "ERROR: DATA_PATH must be set when USE_MOCK_DATA=0"
        exit 1
    fi
    DATA_ARGS=(
        --data-path "${DATA_PATH}/text/the_pile/shard00/my-gpt3_00_text_document"
        --vocab-file "${DATA_PATH}/text/the_pile/shard00/bpe/vocab.json"
        --merge-file "${DATA_PATH}/text/the_pile/shard00/bpe/merges.txt"
        --data-cache-path "$DATA_CACHE_PATH"
        --split 949,50,1
    )
fi

MODEL_ARGS=(
  --distributed-timeout-minutes=60
  --tensor-model-parallel-size=1
  --pipeline-model-parallel-size=1
  --expert-model-parallel-size=1
  --context-parallel-size=1
  --expert-tensor-parallel-size=1
  
  #--use-distributed-optimizer
  --overlap-grad-reduce
  --overlap-param-gather

  

  # FSDP args
  --megatron-fsdp-version=2
  --use-megatron-fsdp 
  --data-parallel-sharding-strategy="optim_grads_params"
  --outer-dp-sharding-strategy="no_shard"
  --ckpt-format="fsdp_dtensor"
  --num-distributed-optimizer-instances=1

  # Precision
  #--fp8-format="hybrid"
  #--fp8-recipe="mxfp8"
  #--fp8-param-gather
  --bf16


  --megatron-fsdp-main-params-dtype=fp32
  --megatron-fsdp-main-grads-dtype=fp32
  --megatron-fsdp-grad-comm-dtype=fp32

  # Training args
  --sequence-parallel
  --disable-bias-linear
  --no-gradient-accumulation-fusion
  --micro-batch-size=4
  --global-batch-size=32
  --train-iters=10
  --exit-duration-in-mins=60
  --no-check-for-nan-in-loss-and-grad 
  --no-rope-fusion
  # --cross-entropy-loss-fusion: true  # Not deterministic yet.
  #--cross-entropy-fusion-impl=te
  --manual-gc 
  --manual-gc-interval=100
  --deterministic-mode 
  --seed=1234

  --disable-jit-fuser


  # Transformer Engine args
  --transformer-impl=transformer_engine
  #--transformer-impl=local

  # The local backend builds norms via WrappedTorchNorm, which rejects the Apex
  # persistent fused LayerNorm kernel.
  #--no-persist-layer-norm

  # Recompute args
  #--recompute-granularity=selective
  #--recompute-modules layernorm mla_up_proj moe_act

  # CPU Offloading
  #--fine-grained-activation-offloading 
  #--offload-modules core_attn attn_proj


  # Data args
  --seq-length=1024


  # Network
  --num-layers=2
  --hidden-size=512
  --ffn-hidden-size=2048
  --num-attention-heads=8
  --kv-channels=128
  --max-position-embeddings=1024
  --position-embedding-type=rope
  --rotary-base=10000
  --make-vocab-size-divisible-by=64
  --normalization=RMSNorm
  --norm-epsilon=1e-6
  --swiglu 
  --untie-embeddings-and-output-weights
  --multi-latent-attention



  # Regularization args
  --attention-dropout=0.0
  --hidden-dropout=0.0
  --clip-grad=1.0
  --weight-decay=0.1
  --qk-layernorm 

  # Learning Rate args
  --lr-warmup-fraction=.01
  --lr=0.00015
  --min-lr=1.0e-5
  --lr-decay-style=cosine
  --adam-beta1=0.9
  --adam-beta2=0.95
  
  # MoE args (DeepSeek-style)
  --num-experts=8
  --moe-layer-freq="[0,1]"
  --moe-ffn-hidden-size=1024
  --moe-shared-expert-intermediate-size=1024
  --moe-router-load-balancing-type=seq_aux_loss
  --moe-router-topk=4
  --moe-grouped-gemm 
  --moe-aux-loss-coeff=1e-4
  --moe-router-group-topk=2
  --moe-router-num-groups=4
  --moe-router-topk-scaling-factor=2.0
  --moe-router-score-function=sigmoid
  --moe-router-enable-expert-bias 
  --moe-router-bias-update-rate=1e-3
  --moe-router-dtype=fp32
  --moe-permute-fusion 
  #--moe-router-fusion 
  --moe-router-pre-softmax 
  
  # MoE Dispatcher
  --moe-token-dispatcher-type=flex
  --moe-flex-dispatcher-backend=hybridep
  
  # MLA args (DeepSeek-style)
  --q-lora-rank=256
  --kv-lora-rank=128
  --qk-head-dim=128
  --qk-pos-emb-head-dim=64
  --v-head-dim=128
  --rotary-scaling-factor=1.0
  --mscale=1.0
  --mscale-all-dim=1.0
  --attention-backend=unfused

  # Cuda Graphs
  --cuda-graph-impl=local
  --cuda-graph-scope=full_iteration

  #--megatron-fsdp-cuda-graph-mode 
  #--overlap-dispatch-backward-with-experts-wgrad
  #--moe-paged-stash

  # Full-iteration capture includes the HybridEP token dispatch, so the dispatch has to be
  # sync-free: the rank capacity factor gives it a static permuted-token budget, and the
  # grouped-tensor expert path keeps the GEMMs off host-side token counts.
  --moe-expert-rank-capacity-factor=1
  --moe-use-grouped-tensor
  --use-transformer-engine-op-fuser

  --moe-router-force-load-balancing



  # Validation
  --eval-iters=10
  --eval-interval=200

  # Checkpointing
  # --save=${CHECKPOINT_SAVE_PATH}
  # --load=${CHECKPOINT_LOAD_PATH}

  # Logging
  #--save-interval=25
  --init-method-std=0.02
  --log-timers-to-tensorboard 
  --log-memory-to-tensorboard 
  --log-num-zeros-in-grad 
  --log-params-norm 
  --log-validation-ppl-to-tensorboard 
  --log-throughput 
  --log-interval=1
  #--logging-level=10
  --tensorboard-dir=${TENSORBOARD_PATH}
)

echo "=== DeepSeek Proxy FSDP EP2 — Single Node ==="
echo "GPUs per node : $GPUS_PER_NODE"
echo "Mock data     : $USE_MOCK_DATA"
echo "Tensorboard   : $TENSORBOARD_PATH"
echo "==============================================="


run_cmd="python -m torch.distributed.launch --nproc_per_node=4 ${MEGATRON_LM_DIR}/pretrain_gpt.py \
         ${MODEL_ARGS[@]} \
         ${DATA_ARGS[@]}"

${run_cmd}
