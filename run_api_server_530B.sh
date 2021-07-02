#!/bin/bash
CHECKPOINT="/home/universal-lm-data.cosmos549/scratch/jcasper/gpt3-530b-megatron_tp16_pp3"
DATA_PATH="/home/universal-lm-data.cosmos549/scratch/mshoeybi/data/gpt2"
VOCAB_FILE="${DATA_PATH}/bpe/gpt2-vocab.json"
MERGE_FILE="${DATA_PATH}/bpe/gpt2-merges.txt"
RUN_CMD=(
python -m cProfile -s cumtime tools/run_api_server.py 
       --tensor-model-parallel-size 16 
       --pipeline-model-parallel-size 3 
       --num-layers 105 
       --hidden-size 20480 
       --load ${CHECKPOINT} 
       --num-attention-heads 128 
       --max-position-embeddings 2048 
       --tokenizer-type GPT2BPETokenizer 
       --fp16 
       --micro-batch-size 1 
       --seq-length 2048 
       --out-seq-length 2048 
       --temperature 1.0 
       --vocab-file $VOCAB_FILE 
       --merge-file $MERGE_FILE 
       --top_p 0.9 
	   --seed 42
)

submit_job --nodes 3 --gpu 16 --reservation adlr-530b --partition batch_UN_dgx2_singlenode --mounts /home/universal-lm-data.cosmos549,/home/dcg-adlr-rprenger-source.cosmos352,/home/dcg-adlr-sgodil-data.cosmos233,/home/dcg-adlr-rprenger-output.cosmos349,/home/dcg-adlr-mchrzanowski-chidesign-data --image gitlab-master.nvidia.com/adlr/rprenger/megatron:latest --skip_ib_check --tasks_per_node 16 -c "${RUN_CMD[*]}"
