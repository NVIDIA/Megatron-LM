
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
LR_RATE=1e-5

for ENG_LANG_PROB in 99 75 95 90 50; do

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
    --save-interval 150 
    --eval-interval 150
    --eval-iters 150
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
    --train-iters 2250
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

if [ "$ENG_LANG_PROB" -eq 50 ]; then

# en 50 ar 50 
DATA_PATH=(
    --data-path 
    0.04404501382077816 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.03883044571843831 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.002114227274454509 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.011133320750941098 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.012286160590959992 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.012987413657548234 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.012100122747146601 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.0075880729503261905 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.0027749011273991127 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.002699074239996325 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.01342729565775183 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.013541582280961076 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.013567368763264576 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.01348078133544518 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.013512878747772378 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.013556337953875407 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.013483541802163122 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.01350451088859757 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.013493760328441731 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.013521402143815439 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.005268470632963697 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.012499535615803302 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.008461643765045407 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.0135256712098348 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.013420326100337335 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.013708595988287055 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.013769913393845317 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.013537812573090103 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.013431837615978394 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.013459971135240936 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.01338138013252997 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.013386721550937609 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.013427934356538775 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.013555685251204673 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.013497187535626735 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.004096345212724415 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.012836485359727425 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.012843723999175613 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.010580598326517344 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.010582805089435224 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.02203827975300286 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.022030037436531286 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.022033310193901232 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.02204516107203773 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.005366875520231148 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.021597653044278606 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.010561414781161427 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.026404131323888583 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.0059531946762772705 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.022963928753640492 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.025343221868083868 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.026878063256311653 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.026880178225967465 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.02687945044311182 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.02688325700733013 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.026881034314407522 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.02687764228925081 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.02688118680777893 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.026881130547982045 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.013279720182559036 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.022748940246245533 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.021428784840067987 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)

elif [ "$ENG_LANG_PROB" -eq 75 ]; then

# en 75 ar 25 
DATA_PATH=(
    --data-path 
    0.02202250691038908 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.019415222859219154 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.0010571136372272544 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.005566660375470549 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.006143080295479996 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.006493706828774117 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.006050061373573301 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.0037940364751630953 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.0013874505636995564 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.0013495371199981625 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.006713647828875915 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.006770791140480538 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.006783684381632288 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.00674039066772259 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.006756439373886189 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.006778168976937704 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.006741770901081561 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.006752255444298785 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.0067468801642208654 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.006760701071907719 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.0026342353164818485 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.006249767807901651 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.004230821882522703 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.0067628356049174 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.006710163050168668 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.0068542979941435276 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.006884956696922659 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.0067689062865450515 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.006715918807989197 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.006729985567620468 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.006690690066264985 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.006693360775468804 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.0067139671782693875 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.0067778426256023365 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.006748593767813367 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.0020481726063622074 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.006418242679863712 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.006421861999587807 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.015870897489776017 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.015874207634152836 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.03305741962950429 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.03304505615479693 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.03304996529085185 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.033067741608056596 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.008050313280346721 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.03239647956641791 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.01584212217174214 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.03960619698583288 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.008929792014415905 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.03444589313046074 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.0380148328021258 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.04031709488446748 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.04032026733895119 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.04031917566466773 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.040324885510995195 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.04032155147161128 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.040316463433876217 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.04032178021166839 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.040321695821973064 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.019919580273838555 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.0341234103693683 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.03214317726010198 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)

elif [ "$ENG_LANG_PROB" -eq 90 ]; then
# en 90 ar 10 
DATA_PATH=(
    --data-path 
    0.008809002764155633 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.007766089143687662 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.0004228454548909018 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.0022266641501882197 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.0024572321181919985 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.002597482731509647 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.0024200245494293204 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.0015176145900652382 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.0005549802254798226 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.000539814847999265 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.002685459131550366 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.002708316456192215 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.0027134737526529154 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.002696156267089036 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.002702575749554476 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.0027112675907750815 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.0026967083604326246 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.0027009021777195143 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.0026987520656883463 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.002704280428763088 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.0010536941265927395 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.0024999071231606605 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.0016923287530090814 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.00270513424196696 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.002684065220067467 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.0027417191976574114 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.002753982678769064 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.0027075625146180207 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.002686367523195679 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.002691994227048187 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.002676276026505994 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.002677344310187522 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.0026855868713077553 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.002711137050240935 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.002699437507125347 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.0008192690425448831 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.0025672970719454852 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.002568744799835123 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.01904507698773122 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.019049049160983402 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.039668903555405144 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.03965406738575632 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.039659958349022216 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.039681289929667914 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.009660375936416067 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.038875775479701495 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.01901054660609057 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.04752743638299945 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.010715750417299087 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.041335071756552884 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.04561779936255096 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.048380513861360976 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.04838432080674144 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.04838301079760128 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.04838986261319424 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.04838586176593354 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.048379756120651464 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.04838613625400207 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.04838603498636768 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.023903496328606263 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.04094809244324196 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.03857181271212238 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)

elif [ "$ENG_LANG_PROB" -eq 95 ]; then
# en 95 ar 5 
DATA_PATH=(
    --data-path 
    0.04404501382077816 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.03883044571843831 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.002114227274454509 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.011133320750941098 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.012286160590959992 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.012987413657548234 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.012100122747146601 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.0075880729503261905 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.0027749011273991127 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.002699074239996325 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.01342729565775183 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.013541582280961076 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.013567368763264576 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.01348078133544518 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.013512878747772378 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.013556337953875407 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.013483541802163122 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.01350451088859757 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.013493760328441731 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.013521402143815439 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.005268470632963697 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.012499535615803302 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.008461643765045407 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.0135256712098348 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.013420326100337335 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.013708595988287055 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.013769913393845317 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.013537812573090103 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.013431837615978394 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.013459971135240936 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.01338138013252997 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.013386721550937609 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.013427934356538775 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.013555685251204673 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.013497187535626735 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.004096345212724415 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.012836485359727425 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.012843723999175613 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.020103136820382953 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.020107329669926923 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.04187273153070543 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.041857071129409444 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.04186328936841234 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.041885806036871684 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.01019706348843918 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.04103554078412935 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.02006668808420671 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.05016784951538831 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.011311069884926814 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.04363146463191693 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.04815212154935935 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.05106832018699214 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.05107233862933818 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.05107095584191246 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.05107818831392724 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.05107396519737429 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.05106752034957654 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.05107425493477996 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.05107414804116588 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.025231468346862167 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.043222986467866514 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.04071469119612917 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)

elif [ "$ENG_LANG_PROB" -eq 99 ]; then
# en 99 ar 1 
DATA_PATH=(
    --data-path 
    0.008809002764155633 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=7359767754
    0.007766089143687662 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=6488431663
    0.0004228454548909018 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=529919974
    0.0022266641501882197 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=5581016870
    0.0024572321181919985 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=6158923385
    0.002597482731509647 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=6510454189
    0.0024200245494293204 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=6065664566
    0.0015176145900652382 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=3803821348
    0.0005549802254798226 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=1391028818
    0.000539814847999265 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=541207038
    0.002685459131550366 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=6730962420
    0.002708316456192215 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=6788253105
    0.0027134737526529154 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=6801179598
    0.002696156267089036 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=6757774229
    0.002702575749554476 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=6773864325
    0.0027112675907750815 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=6795649969
    0.0026967083604326246 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=6759158022
    0.0027009021777195143 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=6769669605
    0.0026987520656883463 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=6764280462
    0.002704280428763088 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=6778137014
    0.0010536941265927395 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=2641029046
    0.0024999071231606605 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=6265886046
    0.0016923287530090814 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=4241733231
    0.00270513424196696 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=6780277052
    0.002684065220067467 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=6727468654
    0.0027417191976574114 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=6871975324
    0.002753982678769064 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=6902713096
    0.0027075625146180207 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=6786363390
    0.002686367523195679 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=6733239256
    0.002691994227048187 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=6747342294
    0.002676276026505994 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=6707945449
    0.002677344310187522 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=6710623046
    0.0026855868713077553 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=6731282593
    0.002711137050240935 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=6795322776
    0.002699437507125347 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=6765998485
    0.0008192690425448831 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=2053454872
    0.0025672970719454852 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=6434795417
    0.002568744799835123 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=6438424071
    0.020949584686504343 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473430615
    0.020953954077081744 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14476449294
    0.04363579391094566 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15073321141
    0.043619474124331946 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15067683719
    0.04362595418392444 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15069922157
    0.04364941892263471 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15078027694
    0.010626413530057673 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3670732886
    0.04276335302767164 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14771949711
    0.020911601266699626 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223594513
    0.052280180021299395 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12039595206
    0.011787325459028995 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714501500
    0.045468578932208174 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10470952562
    0.05017957929880606 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11555848165
    0.053218565247497074 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255695806
    0.05322275288741558 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256660177
    0.053221311877361406 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256328327
    0.05322884887451366 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12258064021
    0.05322444794252689 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12257050531
    0.05321773173271661 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255503856
    0.053224749879402275 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12257120064
    0.05322463848500445 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12257094411
    0.026293845961466892 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055206039
    0.04504290168756615 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559385115
    0.04242899398333461 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656450466
)
fi

# $BIN_IDX_PATH/$BIN_IDX_PATH/torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  

done