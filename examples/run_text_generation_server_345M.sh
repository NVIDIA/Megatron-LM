#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"


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

# pip install flask-restful

python tools/run_text_generation_server.py   \
       --load /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-27_18-00-00/output_dir/340M_meglm_8105626.sbatch/checkpoints \
       --tokenizer-model /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-27_18-00-00/output_dir/340M_meglm_8105626.sbatch/converted_checkpoints/iter_0015000/tokenizer.model \
       --tokenizer-type OpenGPTX-SPTokenizer \
       --pipeline-model-parallel-size 1  \
       --tensor-model-parallel-size 1  \
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
       --no-position-embedding \
       --position-embedding-type rotary \
       --use-flash-attn \
       --reset-attention-mask \
       --reset-position-ids



#       --load /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-17_16-38-00/output_dir/340M_monolingual_en_sp_bpe_32768_10.sbatch/checkpoints \
#       --tokenizer-model /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-17_16-38-00/output_dir/340M_monolingual_en_sp_bpe_32768_10.sbatch/converted_checkpoints/iter_0001525/tokenizer.model \
#       --tokenizer-type SentencePieceTokenizer \
#       --pipeline-model-parallel-size 1  \
#       --tensor-model-parallel-size 1  \
#       --num-layers 24  \
#       --hidden-size 1024  \
#       --num-attention-heads 16  \
#       --max-position-embeddings 2048  \
#       --position-embedding-type alibi \
#       --no-position-embedding \
#       --bf16  \
#       --micro-batch-size 1  \
#       --seq-length 2048  \
#       --out-seq-length 2048  \
#       --temperature 0.8  \
#       --top_p 0.5  \
#       --seed 42 

#       --load /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-13_13-00-00/output_dir/2_6B_multilingual-unigram_sp_32768_10.sbatch/checkpoints \
#       --vocab-file='/p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-13_13-00-00/output_dir/2_6B_multilingual-unigram_sp_32768_10.sbatch/converted_checkpoints/iter_0009537/vocab.json' \
#       --merge-file='/p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-13_13-00-00/output_dir/2_6B_multilingual-unigram_sp_32768_10.sbatch/converted_checkpoints/iter_0009537/merges.txt' \
#       --tokenizer-type GPT2BPETokenizer \
#       --pipeline-model-parallel-size 2  \
#       --tensor-model-parallel-size 2  \
#       --num-layers 12  \
#       --hidden-size 768  \
#       --num-attention-heads 12  \
#       --max-position-embeddings 2048  \
#       --position-embedding-type alibi \
#       --bf16  \
#       --micro-batch-size 32  \
#       --seq-length 2048  \
#       --out-seq-length 2048  \
#       --temperature 0.8  \
#       --top_p 0.5  \
#       --seed 42 



#       --load /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-27_18-00-00/output_dir/340M_meglm_8105626.sbatch/checkpoints/iter_0015000 \
#       --tokenizer-model /p/project/opengptx-elm/thellmann1/workdir/checkpoint_conversion_meglm_test/meglm/2023-07-27_18-00-00/output_dir/340M_meglm_8105626.sbatch/converted_checkpoints/iter_0015000/tokenizer.mode
#       --tokenizer-type OpenGPTX-SPTokenizer \
#       --pipeline-model-parallel-size 1  \
#       --tensor-model-parallel-size 1  \
#       --num-layers 24  \
#       --hidden-size 1024  \
#       --num-attention-heads 16  \
#       --max-position-embeddings 2048  \
#       --bf16  \
#       --micro-batch-size 1  \
#       --seq-length 2048  \
#       --out-seq-length 2048  \
#       --temperature 0.8  \
#       --top_p 0.5  \
#       --seed 42 \
#       --distributed-backend nccl
#       --position-embedding-type rotary \
#       --use-flash-attn \
#       --reset-attention-mask \
#       --reset-position-ids \
#       --no-position-embedding \

