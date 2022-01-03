TENSOR_MODEL_PARALLEL_SIZE=2
PIPELINE_MODEL_PARALLEL_SIZE=2
TARGET_PIPELINE_MODEL_PARALLEL_SIZE=1

export PYTHONPATH=$PYTHONPATH:/gpfs/alpine/med106/world-shared/irl1/rhel8/Megatron-LM/megatron/fused_kernels/build

VOCAB_FILE=/gpfs/alpine/world-shared/med106/g8o/pubmed_bert-vocab.txt
CHECKPOINT_PATH=/gpfs/alpine/med106/world-shared/irl1/rhel8/Megatron-LM/chkptt

WORLD_SIZE=$TENSOR_MODEL_PARALLEL_SIZE jsrun -r 1 -g 1 -a 1 -c 1 python tools/merge_mp_partitions.py \
        --model-type BERT \
        --tensor-model-parallel-size $TENSOR_MODEL_PARALLEL_SIZE \
        --pipeline-model-parallel-size $PIPELINE_MODEL_PARALLEL_SIZE \
        --target-pipeline-model-parallel-size $TARGET_PIPELINE_MODEL_PARALLEL_SIZE \
        --tokenizer-type BertWordPieceLowerCase \
        --vocab-file $VOCAB_FILE \
        --num-layers 24 \
        --hidden-size 1024 \
        --num-attention-heads 16 \
        --seq-length 512 \
        --max-position-embeddings 512 \
        --load $CHECKPOINT_PATH \
        --save $CHECKPOINT_PATH/merged
