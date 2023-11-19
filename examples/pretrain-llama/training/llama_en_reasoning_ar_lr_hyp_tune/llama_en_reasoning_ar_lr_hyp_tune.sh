
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
        --data-path 
        0.05465975063602466 $BIN_IDX_PATH/ar_books_split_00_text_document_dc=74697_sc=74697_tc=23946784716
        0.047924016635759036 $BIN_IDX_PATH/ar_books_split_01_text_document_dc=182478_sc=182478_tc=20995816771
        0.0026508979041214285 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc=1134657_sc=1134657_tc=1742062871
        0.016209294013801764 $BIN_IDX_PATH/ar_news_split_00_text_document_dc=13366967_sc=13366967_tc=21304184686
        0.017120344050967663 $BIN_IDX_PATH/ar_news_split_01_text_document_dc=12454060_sc=12454060_tc=22501595149
        0.017588499649541713 $BIN_IDX_PATH/ar_news_split_02_text_document_dc=8106915_sc=8106915_tc=23116900993
        0.016983352672476606 $BIN_IDX_PATH/ar_news_split_03_text_document_dc=11173000_sc=11173000_tc=22321544764
        0.011486430071655335 $BIN_IDX_PATH/ar_news_split_04_text_document_dc=10090583_sc=10090583_tc=15096834410
        0.0037937153153474617 $BIN_IDX_PATH/ar_others_split_00_text_document_dc=927554_sc=927554_tc=4986152491
        0.003326725849489915 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc=86178_sc=86178_tc=1748951727
        0.0178440341704459 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc=5122708_sc=5122708_tc=23452754894
        0.01780947225844737 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc=5575027_sc=5575027_tc=23407329513
        0.017814976373096158 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc=5521485_sc=5521485_tc=23414563676
        0.017809468270831266 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc=5408044_sc=5408044_tc=23407324272
        0.01780595757163551 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc=5351784_sc=5351784_tc=23402710093
        0.017859309071367527 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc=5170226_sc=5170226_tc=23472830988
        0.017829224464773256 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc=5294345_sc=5294345_tc=23433290215
        0.017830545526000136 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc=5443921_sc=5443921_tc=23435026511
        0.017810412148160575 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc=5271931_sc=5271931_tc=23408564828
        0.0178306481631757 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc=5273864_sc=5273864_tc=23435161409
        0.00685196875957389 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc=1971786_sc=1971786_tc=9005673399
        0.01769484747041879 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc=84634264_sc=84634264_tc=23256675965
        0.012000207365216004 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc=27533033_sc=27533033_tc=15772101719
        0.01776568329723433 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc=6188667_sc=6188667_tc=23349776845
        0.01772756670813537 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc=5901018_sc=5901018_tc=23299679484
        0.017822722426559263 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc=6071497_sc=6071497_tc=23424744462
        0.01780401965335977 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc=6668426_sc=6668426_tc=23400163050
        0.017730563602121205 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc=6592093_sc=6592093_tc=23303618359
        0.017700071052736747 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc=5826549_sc=5826549_tc=23263541419
        0.017701912655741562 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc=6652064_sc=6652064_tc=23265961873
        0.01770276802640595 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc=6979539_sc=6979539_tc=23267086103
        0.017703633690613226 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc=7026762_sc=7026762_tc=23268223862
        0.017675813851383844 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc=7050626_sc=7050626_tc=23231659716
        0.01772636907583854 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc=6488044_sc=6488044_tc=23298105413
        0.01768576149805563 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc=6992450_sc=6992450_tc=23244734098
        0.005459052614697585 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc=2365853_sc=2365853_tc=7174937108
        0.017460244233282542 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc=4544790_sc=4544790_tc=22948332450
        0.01747002790607588 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc=4488706_sc=4488706_tc=22961191318
        0.00032969129543092333 $BIN_IDX_PATH/ar_web_oscar2301_split_02_text_document_dc=84865_sc=84865_tc=433319566
        0.007193537806733607 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc=102105_sc=102105_tc=14473665967
        0.007195234707729857 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc=102718_sc=102718_tc=14477080195
        6.366468291080727e-$BIN_IDX_PATH/05 en_books_books_split_02_text_document_dc=937_sc=937_tc=128095713
        0.014986683901511495 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc=6919454_sc=6919454_tc=15076883070
        0.014981027820319792 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc=6787019_sc=6787019_tc=15071192947
        0.014983229506207164 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc=6791613_sc=6791613_tc=15073407884
        0.01499146169080912 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc=6645201_sc=6645201_tc=15081689615
        0.003649651225966398 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc=1650889_sc=1650889_tc=3671617093
        0.01468426514534855 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc=19981970_sc=19981970_tc=14772644170
        0.007180761692819437 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc=9843118_sc=9843118_tc=7223979975
        0.01795202194259345 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc=5157493_sc=5157493_tc=12040045571
        0.004047533800599255 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc=1157740_sc=1157740_tc=2714596248
        0.01560337275461048 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc=34104559_sc=34104559_tc=10464855693
        0.017226300656371867 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc=14452182_sc=14452182_tc=11553319486
        0.018273235095408482 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc=1721917_sc=1721917_tc=12255476513
        0.018274675018059964 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc=1720379_sc=1720379_tc=12256442239
        0.018274170721728388 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc=1719262_sc=1719262_tc=12256104018
        0.01827677212685719 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc=1721575_sc=1721575_tc=12257848726
        0.018275243959318233 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc=1722370_sc=1722370_tc=12256823816
        0.018272954769086337 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc=1719665_sc=1719665_tc=12255288504
        0.018275364991869193 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc=1721188_sc=1721188_tc=12256904990
        0.018275312242348563 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc=1719879_sc=1719879_tc=12256869612
        0.009028306466289202 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc=850041_sc=850041_tc=6055096280
        0.015466372845621787 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc=805220_sc=805220_tc=15559459080
        0.014568844428881407 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc=753086_sc=753086_tc=14656528780
    )

    # torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
    python pretrain_gpt.py \
        ${GPT_MODEL_ARGS[@]} \
        ${LOGISTICS_ARGS[@]} \
        ${TRAINING_ARGS[@]} \
        ${MODEL_PARALLEL_ARGS[@]} \
        ${DATA_PATH[@]}
    
  
done
  
  
