PRETRAINED_MODEL=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6


DISTRIBUTED_ARGS=(
    --nproc_per_node 1 
    --nnodes 1
    --master_addr localhost 
    --master_port 60006
)


GPT_MODEL_ARGS=(
    --use-mcore-models
    --seq-length 2048 
    --max-position-embeddings 2048 
    --num-layers 24
    --hidden-size 2048
    --ffn-hidden-size 3072
    --num-attention-heads 16
    --hidden-dropout 0.0
    --attention-dropout 0.0
    --make-vocab-size-divisible-by 128
    --norm-epsilon 1.0e-05
    --disable-bias-linear
    --swiglu
    --tokenizer-type Llama2Tokenizer
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-query-key-layer-scaling
    --use-distributed-optimizer
    --overlap-grad-reduce
)

LOGISTICS_ARGS=(
    --load $PRETRAINED_MODEL
    --resume-with-new-dataset
    --load-iteration 40
    --override-dataloader-consumed-samples 0
    --save $CHECKPOINT_DIR 
    --tokenizer-model $TOKENIZER_MODEL
    --split 1,0,0 
    --log-interval 10
    --save-interval 10
    --eval-interval 10 
    --eval-iters 0
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 10
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
    --seed 1234
)

TRAINING_ARGS=(
    --micro-batch-size 8
    --global-batch-size 16
    --train-iters 100
    --lr 0.0002 
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.01
    --clip-grad 1.0 
    --min-lr 2.0e-05
    --lr-warmup-iters 10
    --use-flash-attn
    --bf16
)


MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
    --pipeline-model-parallel-size 1
    --sequence-parallel
    --no-async-tensor-model-parallel-allreduce
)

DATA_PATH=(
    --data-path 1.0 $BIN_IDX_PATH/en_Pile-ArXiv_0000_text_document_dc\=1823931_sc\=1823931_tc\=28911020643
1.0 $BIN_IDX_PATH/en_Pile-ArXiv_0001_text_document_dc\=553810_sc\=553810_tc\=8792719530
1.0 $BIN_IDX_PATH/en_Pile-books2-books3-pgtbrg_0000_text_document_dc\=169830_sc\=169830_tc\=23873854486
)

# torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
