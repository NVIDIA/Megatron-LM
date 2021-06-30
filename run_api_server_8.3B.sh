#!/bin/bash
WORLD_SIZE=8
DISTRIBUTED_ARGS="--nproc_per_node $WORLD_SIZE \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"
CHECKPOINT="/home/universal-lm-data.cosmos549/chkpts/gpt2/8.3B_no_rng"
DATA_PATH="/home/universal-lm-data.cosmos549/scratch/mshoeybi/data/gpt2"
VOCAB_FILE="${DATA_PATH}/bpe/gpt2-vocab.json"
MERGE_FILE="${DATA_PATH}/bpe/gpt2-merges.txt"
python -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_api_server.py \
       --tensor-model-parallel-size 8 \
       --pipeline-model-parallel-size 1 \
       --num-layers 72 \
       --hidden-size 3072 \
       --load $CHECKPOINT \
       --num-attention-heads 24 \
       --max-position-embeddings 1024 \
       --tokenizer-type GPT2BPETokenizer \
       --fp16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --out-seq-length 1024 \
       --temperature 1.0 \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --top_p 0.9 \
	   --seed 42
