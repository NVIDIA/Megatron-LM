#!/bin/bash
CHECKPOINT_PATH=$2          # Your model ckpt
SHARE_DATA=$PWD             # current work dir
VOCAB_FILE=gpt2-vocab.json  # Your gpt-2 vocab
MERGE_FILE=gpt2-merges.txt  # Your gpt-2 merge file

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=$(($RANDOM + 1024))
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
SEED=$3
SUFFIX=$(basename $CHECKPOINT_PATH)
save_dir=$SHARE_DATA/selfgeneration/unconditional_generation_$SUFFIX/
mkdir -p $save_dir
echo $save_dir/$SEED.out

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.run $DISTRIBUTED_ARGS examples/detxoify_lm/generate_samples_gpt.py \
       --tensor-model-parallel-size 1 \
       --num-layers 24 \
       --hidden-size 2048 \
       --load $CHECKPOINT_PATH \
       --num-attention-heads 32 \
       --max-position-embeddings 2048 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 150 \
       --seq-length 2048 \
       --out-seq-length 1000 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --num-samples $1 \
       --top_p 0.9 \
       --max-tokens-to-oom 1200000 \
       --genfile $save_dir/$SEED.out  \
       --seed $SEED

