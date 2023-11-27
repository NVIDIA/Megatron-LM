
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
for LR_RATE in 1e-4 5e-5 1e-5 5e-6 1e-6; do


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

mkdir -p $TENSORBOARD_LOGS_PATH/$LR_RATE/tensorboard
mkdir -p $DATA_CACHE/$LR_RATE/cache
mkdir -p $CHECKPOINT_DIR/$LR_RATE

LOGISTICS_ARGS=(
    --save $CHECKPOINT_DIR/$LR_RATE/
    --load $PRETRAINED_LLAMA_MODEL_PATH 
    --tokenizer-model $TOKENIZER_MODEL
    --split 9998,1,1 
    --log-interval 10
    --save-interval 50 
    --eval-interval 50
    --eval-iters 50
    --tensorboard-dir $TENSORBOARD_LOGS_PATH/$LR_RATE/tensorboard
    --tensorboard-log-interval 10
    --data-cache-path $DATA_CACHE/$LR_RATE/cache
    --log-validation-ppl-to-tensorboard 
)

TRAINING_ARGS=(
    --no-initialization
    --no-load-optim
    --no-load-rng
    --micro-batch-size 1 
    --global-batch-size 1024
    --train-iters 750
    --lr-decay-style cosine 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --lr $LR_RATE
    --min-lr $LR_RATE
    --lr-warmup-iters 10
    --use-flash-attn
    --bf16
)
# --use-mcore-models

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 2
    --pipeline-model-parallel-size 2
    --no-async-tensor-model-parallel-allreduce
)
# en 66 ar 34 
DATA_PATH=(
    --data-path 
    0.05813941824342718 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.05125618834833857 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.002790780002279952 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.01469598339124225 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.01621773198006719 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.01714338602796367 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.015972162026233513 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.010016256294430571 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.003662869488166829 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.003562777996795149 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.017724030268232417 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.01787488861086862 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.017908926767509242 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.01779463136278764 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.01783699994705954 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.017894366099115537 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.01779827517885532 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.01782595437294879 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.017811763633543087 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.01784825082983638 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.00695438123551208 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.01649938701286036 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.011169369769859938 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.017853885996981938 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.017714830452445283 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.018095346704538915 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.01817628567987582 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.017869912596478937 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.01773002565309148 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.017767161898518038 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.01766342177493956 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.017670472447237646 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.017724873350631182 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.01789350453159017 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.01781628754702729 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.005407175680796228 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.0169441606748402 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.01695371567891181 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.0003201604441617155 $BIN_IDX_PATH/ar_web_oscar2301_split_02_text_document_dc\=84865_sc\=84865_tc\=121585660
    0.007194806862031795 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.007196307460815953 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.014986030232041946 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.014980425456841275 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.01498265093185284 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.014990709528985659 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.003649475353757181 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.014686404070109453 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.007181762051189771 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.017954809300244238 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.004048172379868544 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.015615471552475536 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.01723339087029703 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.018277083014291925 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.018278521193657877 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.01827802630131604 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.01828061476498449 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.018279103333797116 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.018276796756690552 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.018279207029289674 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.01827916877262779 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.009030209724140145 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.015469279367446964 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.014571573691246231 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)

# $BIN_IDX_PATH/$BIN_IDX_PATH/torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
done