export NCCL_SOCKET_IFNAME="ib,bond"
export NCCL_IB_CUDA_SUPPORT=1

MASTER_ADDR=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | head -n 1)
MASTER_PORT=5${LSB_JOBID: -5:-1}
NNODES=$(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | wc -w)
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -w)
NODE_RANK=$(($(echo ${LSB_MCPU_HOSTS} | tr ' ' '\n' | sed 'n; d' | grep -n -m1 $HOSTNAME | cut -d':' -f1)-1))



DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=checkpoints  # Adjust: Directory to store the checkpoints
DATA_PATH=data/debug_text_document  # Adjust: Prefix of the preprocessed dataset.
TOKENIZER_FILE=tokenizer.json  # Adjust

GPT_ARGS="\
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--recompute-activations \
--num-layers 24 \
--hidden-size 2048 \
--num-attention-heads 16 \
--attention-head-type multiquery \
--init-method-std 0.022 \
--seq-length 2048 \
--max-position-embeddings 2048 \
--attention-dropout 0.1 \
--hidden-dropout 0.1 \
--micro-batch-size 2 \
--global-batch-size 192 \
--lr 0.0002 \
--train-iters 3000 \
--lr-decay-iters 600000 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.02 \
--weight-decay .1 \
--adam-beta2 .95 \
--clip-grad 1.0 \
--fp16 \
--log-interval 10 \
--save-interval 4000 \
--eval-interval 200 \
--eval-iters 10 \
--initial-loss-scale 65536 \
--fim-rate 0.5 \
"

TENSORBOARD_ARGS="--tensorboard-dir ${CHECKPOINT_PATH}/tensorboard"

torchrun $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       $GPT_ARGS \
       --tokenizer-type TokenizerFromFile \
       --tokenizer-file $TOKENIZER_FILE \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       $TENSORBOARD_ARGS
