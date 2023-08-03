#!/bin/bash

set -x -e

export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCHELASTIC_ERROR_FILE=/tmp/torch-elastic-error.json
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_SOCKET_IFNAME=ib0
export NCCL_DEBUG=INFO
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=60234

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]:-${(%):-%x}}" )" &> /dev/null && pwd )


export CMD=" \
       $SCRIPT_DIR/../tools/run_text_generation_server.py \
       --load /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-17_16-38-00/output_dir/340M_monolingual_en_sp_bpe_32768_10.sbatch/checkpoints \
       --tokenizer-model /p/scratch/opengptx-elm/data/datasources_opgptx/data_quality_experiments_datasets/ablations_studies/monolingual_en/70B_10/tokenizer_training/bpe/sp/32768_10/bpe_tokenizer.model \
       --tokenizer-type SentencePieceTokenizer \
       --pipeline-model-parallel-size 1 \
       --tensor-model-parallel-size 1 \
       --num-layers 24  \
       --hidden-size 1024  \
       --num-attention-heads 16  \
       --max-position-embeddings 2048  \
       --bf16  \
       --micro-batch-size 1  \
       --seq-length 2048  \
       --out-seq-length 2048  \
       --temperature 0.8  \
       --top_p 0.5  \
       --seed 42 \
       --position-embedding-type alibi \
       --no-position-embedding \
       "


export LAUNCHER="python -u -m torch.distributed.run \
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6000"

bash -c "$LAUNCHER $CMD" 
