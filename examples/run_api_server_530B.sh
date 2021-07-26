#!/bin/bash
DISTRIBUTED_ARGS="--nproc_per_node 16 \
                  --nnodes 3 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT=<Path to checkpoint (e.g /gpt3-530b-megatron_tp16_pp3)>
VOCAB_FILE=<Path to vocab.json (e.g. /gpt2-vocab.json)>
MERGE_FILE=<Path to merges.txt (e.g. /gpt2-merges.txt)>

pip install flask-restful

python -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_api_server.py   /
       --tensor-model-parallel-size 16  /
       --pipeline-model-parallel-size 3  /
       --num-layers 105  /
       --hidden-size 20480  /
       --load ${CHECKPOINT}  /
       --num-attention-heads 128  /
       --max-position-embeddings 2048  /
       --tokenizer-type GPT2BPETokenizer  /
       --fp16  /
       --micro-batch-size 1  /
       --seq-length 2048  /
       --out-seq-length 2048  /
       --temperature 1.0  /
       --vocab-file $VOCAB_FILE  /
       --merge-file $MERGE_FILE  /
       --top_p 0.9  /
	   --seed 42
