
PRETRAINED_LLAMA_MODEL_PATH=$1
TOKENIZER_MODEL=$2
BIN_IDX_PATH=$3
DATA_CACHE=$4
CHECKPOINT_DIR=$5
TENSORBOARD_LOGS_PATH=$6

# DISTRIBUTED_ARGS=(
#     --nproc_per_node $GPUS_PER_NODE 
#     --nnodes $NUM_NODES 
#     --master_addr $MASTER_ADDR 
#     --master_port $MASTER_PORT
# )

GPT_MODEL_ARGS=(
    --seq-length 4096 
    --max-position-embeddings 4096 
    --tokenizer-type Llama2Tokenizer
    --exit-on-missing-checkpoint
    --use-checkpoint-args
    --untie-embeddings-and-output-weights
    --use-rotary-position-embeddings
    --normalization RMSNorm
    --no-position-embedding
    --no-masked-softmax-fusion
    --no-query-key-layer-scaling
)

LOGISTICS_ARGS=(
    --save $CHECKPOINT_DIR 
    --load $PRETRAINED_LLAMA_MODEL_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --split 9998,1,1 
    --log-interval 100
    --save-interval 10000 
    --eval-interval 1000
    --eval-iters 50
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 100
    --data-cache-path $DATA_CACHE
    --log-validation-ppl-to-tensorboard 
)

TRAINING_ARGS=(
    --no-initialization
    --no-load-optim
    --no-load-rng
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 160_000
    --lr 0.000001 
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr 1.0e-6 
    --min-lr 1.0e-6
    --lr-warmup-iters 10000
    --use-flash-attn
    --bf16
)
# --use-mcore-models

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)

DATA_PATH=(
    --data-path 0.02320759714533172 en_books_books_split_00_text_document_dc=102105_sc=102105_tc=14473665967
    0.02321307164143844 en_books_books_split_01_text_document_dc=102718_sc=102718_tc=14477080195
    0.0002053932783944309 en_books_books_split_02_text_document_dc=937_sc=937_tc=128095713
    0.04834963433501935 en_code_github_split_00_text_document_dc=6919454_sc=6919454_tc=15076883070
    0.048331386838829725 en_code_github_split_01_text_document_dc=6787019_sc=6787019_tc=15071192947
    0.048338489858301845 en_code_github_split_02_text_document_dc=6791613_sc=6791613_tc=15073407884
    0.048365048309650965 en_code_github_split_03_text_document_dc=6645201_sc=6645201_tc=15081689615
    0.0117744060917995 en_code_github_split_04_text_document_dc=1650889_sc=1650889_tc=3671617093
    0.047373979121856744 en_code_stackexchange_split_00_text_document_dc=19981970_sc=19981970_tc=14772644170
    0.023166379192111904 en_code_stackexchange_split_01_text_document_dc=9843118_sc=9843118_tc=7223979975
    0.057916327734856464 en_reasoning_open-web-math_split_00_text_document_dc=5157493_sc=5157493_tc=12040045571
    0.013058044094589058 en_reasoning_open-web-math_split_01_text_document_dc=1157740_sc=1157740_tc=2714596248
    0.05033917923646424 en_reasoning_peS2o_split_00_text_document_dc=34104559_sc=34104559_tc=10464855693
    0.05557502534611291 en_reasoning_peS2o_split_01_text_document_dc=14452182_sc=14452182_tc=11553319486
    0.05895261692226231 en_reasoning_peS2o_split_02_text_document_dc=1721917_sc=1721917_tc=12255476513
    0.0589572623617823 en_reasoning_peS2o_split_03_text_document_dc=1720379_sc=1720379_tc=12256442239
    0.05895563541459449 en_reasoning_peS2o_split_04_text_document_dc=1719262_sc=1719262_tc=12256104018
    0.05896402799747416 en_reasoning_peS2o_split_05_text_document_dc=1721575_sc=1721575_tc=12257848726
    0.05895909786468449 en_reasoning_peS2o_split_06_text_document_dc=1722370_sc=1722370_tc=12256823816
    0.058951712541062344 en_reasoning_peS2o_split_07_text_document_dc=1719665_sc=1719665_tc=12255288504
    0.05895948833662746 en_reasoning_peS2o_split_08_text_document_dc=1721188_sc=1721188_tc=12256904990
    0.058959318157550435 en_reasoning_peS2o_split_09_text_document_dc=1719879_sc=1719879_tc=12256869612
    0.02912687817920471 en_reasoning_peS2o_split_10_text_document_dc=850041_sc=850041_tc=6055096280
)

# torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
