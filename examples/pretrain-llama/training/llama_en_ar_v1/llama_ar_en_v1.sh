
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
    --train-iters 250000
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
    --data-path 0.05060197557342783 $BIN_IDX_PATH/ar_books_split_00_text_document_dc\=74697_sc\=74697_tc\=23946784716
    0.04436628215396483 $BIN_IDX_PATH/ar_books_split_01_text_document_dc\=182478_sc\=182478_tc\=20995816771
    0.0036811548561180773 $BIN_IDX_PATH/ar_encyclopedias_split_00_text_document_dc\=1134657_sc\=1134657_tc\=1742062871
    0.02250894735718906 $BIN_IDX_PATH/ar_news_split_00_text_document_dc\=13366967_sc\=13366967_tc\=21304184686
    0.023774072001659784 $BIN_IDX_PATH/ar_news_split_01_text_document_dc\=12454060_sc\=12454060_tc\=22501595149
    0.024424173709624613 $BIN_IDX_PATH/ar_news_split_02_text_document_dc\=8106915_sc\=8106915_tc\=23116900993
    0.02358383967419269 $BIN_IDX_PATH/ar_news_split_03_text_document_dc\=11173000_sc\=11173000_tc\=22321544764
    0.01595056821011312 $BIN_IDX_PATH/ar_news_split_04_text_document_dc\=10090583_sc\=10090583_tc\=15096834410
    0.00702416266031151 $BIN_IDX_PATH/ar_others_split_00_text_document_dc\=927554_sc\=927554_tc\=4986152491
    0.0036957117048631184 $BIN_IDX_PATH/ar_transcribed_split_00_text_document_dc\=86178_sc\=86178_tc\=1748951727
    0.006894124575150024 $BIN_IDX_PATH/ar_translated_En_wikipedia_split_translated_split_00_text_document_dc\=20745266_sc\=20745266_tc\=9787688038
    0.016519346881711004 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_00_text_document_dc\=5122708_sc\=5122708_tc\=23452754894
    0.01648735074183897 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_01_text_document_dc\=5575027_sc\=5575027_tc\=23407329513
    0.016492446247613708 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_02_text_document_dc\=5521485_sc\=5521485_tc\=23414563676
    0.016487347050251462 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_03_text_document_dc\=5408044_sc\=5408044_tc\=23407324272
    0.01648409697477761 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_04_text_document_dc\=5351784_sc\=5351784_tc\=23402710093
    0.016533487820049157 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_05_text_document_dc\=5170226_sc\=5170226_tc\=23472830988
    0.016505636603929334 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_06_text_document_dc\=5294345_sc\=5294345_tc\=23433290215
    0.016506859593554347 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_07_text_document_dc\=5443921_sc\=5443921_tc\=23435026511
    0.016488220856974076 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_08_text_document_dc\=5271931_sc\=5271931_tc\=23408564828
    0.016506954611255495 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_09_text_document_dc\=5273864_sc\=5273864_tc\=23435161409
    0.0063432992607422075 $BIN_IDX_PATH/ar_web_arabicweb16_v2_split_10_text_document_dc\=1971786_sc\=1971786_tc\=9005673399
    0.01638123535242648 $BIN_IDX_PATH/ar_web_arabicweb22_split_00_text_document_dc\=84634264_sc\=84634264_tc\=23256675965
    0.011109348156640117 $BIN_IDX_PATH/ar_web_arabicweb22_split_01_text_document_dc\=27533033_sc\=27533033_tc\=15772101719
    0.016446812541062263 $BIN_IDX_PATH/ar_web_metadialog_split_00_text_document_dc\=6188667_sc\=6188667_tc\=23349776845
    0.016411525612598732 $BIN_IDX_PATH/ar_web_metadialog_split_01_text_document_dc\=5901018_sc\=5901018_tc\=23299679484
    0.016499617257425672 $BIN_IDX_PATH/ar_web_metadialog_split_02_text_document_dc\=6071497_sc\=6071497_tc\=23424744462
    0.01648230292171093 $BIN_IDX_PATH/ar_web_metadialog_split_03_text_document_dc\=6668426_sc\=6668426_tc\=23400163050
    0.01641430002621209 $BIN_IDX_PATH/ar_web_metadialog_split_04_text_document_dc\=6592093_sc\=6592093_tc\=23303618359
    0.01638607115174469 $BIN_IDX_PATH/ar_web_metadialog_split_05_text_document_dc\=5826549_sc\=5826549_tc\=23263541419
    0.016387776039695715 $BIN_IDX_PATH/ar_web_metadialog_split_06_text_document_dc\=6652064_sc\=6652064_tc\=23265961873
    0.01638856791022132 $BIN_IDX_PATH/ar_web_metadialog_split_07_text_document_dc\=6979539_sc\=6979539_tc\=23267086103
    0.01638936931012823 $BIN_IDX_PATH/ar_web_metadialog_split_08_text_document_dc\=7026762_sc\=7026762_tc\=23268223862
    0.01636361473187346 $BIN_IDX_PATH/ar_web_metadialog_split_09_text_document_dc\=7050626_sc\=7050626_tc\=23231659716
    0.016410416888912198 $BIN_IDX_PATH/ar_web_metadialog_split_10_text_document_dc\=6488044_sc\=6488044_tc\=23298105413
    0.01637282389525312 $BIN_IDX_PATH/ar_web_metadialog_split_11_text_document_dc\=6992450_sc\=6992450_tc\=23244734098
    0.0050537890101701914 $BIN_IDX_PATH/ar_web_metadialog_split_12_text_document_dc\=2365853_sc\=2365853_tc\=7174937108
    0.016164048352177135 $BIN_IDX_PATH/ar_web_oscar2301_split_00_text_document_dc\=4544790_sc\=4544790_tc\=22948332450
    0.01617310571460463 $BIN_IDX_PATH/ar_web_oscar2301_split_01_text_document_dc\=4488706_sc\=4488706_tc\=22961191318
    0.00030521600783103576 $BIN_IDX_PATH/ar_web_oscar2301_split_02_text_document_dc\=84865_sc\=84865_tc\=433319566
    0.009853714273288728 $BIN_IDX_PATH/en_books_books_split_00_text_document_dc\=102105_sc\=102105_tc\=14473665967
    0.009856038689732536 $BIN_IDX_PATH/en_books_books_split_01_text_document_dc\=102718_sc\=102718_tc\=14477080195
    0.008553655206306458 $BIN_IDX_PATH/en_code_github_split_00_text_document_dc\=6919454_sc\=6919454_tc\=15076883070
    0.008550426995939801 $BIN_IDX_PATH/en_code_github_split_01_text_document_dc\=6787019_sc\=6787019_tc\=15071192947
    0.00855168360894885 $BIN_IDX_PATH/en_code_github_split_02_text_document_dc\=6791613_sc\=6791613_tc\=15073407884
    0.008556382131259893 $BIN_IDX_PATH/en_code_github_split_03_text_document_dc\=6645201_sc\=6645201_tc\=15081689615
    0.0020830397448391986 $BIN_IDX_PATH/en_code_github_split_04_text_document_dc\=1650889_sc\=1650889_tc\=3671617093
    0.008381049592874057 $BIN_IDX_PATH/en_code_stackexchange_split_00_text_document_dc\=19981970_sc\=19981970_tc\=14772644170
    0.004098422308942955 $BIN_IDX_PATH/en_code_stackexchange_split_01_text_document_dc\=9843118_sc\=9843118_tc\=7223979975
    0.009557251854363144 $BIN_IDX_PATH/en_encyclopedia_m_wikipedia_split_00_text_document_dc\=14935020_sc\=14935020_tc\=14038206007
    0.006118047480921474 $BIN_IDX_PATH/en_encyclopedia_m_wikipedia_split_01_text_document_dc\=13560334_sc\=13560334_tc\=8986517485
    0.0007677042219232239 $BIN_IDX_PATH/en_encyclopedia_m_wikipedia_split_02_text_document_dc\=1338817_sc\=1338817_tc\=1127645288
    0.0039365502483162795 $BIN_IDX_PATH/en_encyclopedia_wikipedia_split_00_text_document_dc\=6630656_sc\=6630656_tc\=5782216916
    0.008196898364554433 $BIN_IDX_PATH/en_reasoning_open-web-math_split_00_text_document_dc\=5157493_sc\=5157493_tc\=12040045571
    0.0018481050934933208 $BIN_IDX_PATH/en_reasoning_open-web-math_split_01_text_document_dc\=1157740_sc\=1157740_tc\=2714596248
    0.007124504472130942 $BIN_IDX_PATH/en_reasoning_peS2o_split_00_text_document_dc\=34104559_sc\=34104559_tc\=10464855693
    0.007865533817253046 $BIN_IDX_PATH/en_reasoning_peS2o_split_01_text_document_dc\=14452182_sc\=14452182_tc\=11553319486
    0.008343564382198711 $BIN_IDX_PATH/en_reasoning_peS2o_split_02_text_document_dc\=1721917_sc\=1721917_tc\=12255476513
    0.008344221851294102 $BIN_IDX_PATH/en_reasoning_peS2o_split_03_text_document_dc\=1720379_sc\=1720379_tc\=12256442239
    0.008343991589444558 $BIN_IDX_PATH/en_reasoning_peS2o_split_04_text_document_dc\=1719262_sc\=1719262_tc\=12256104018
    0.008345179391772008 $BIN_IDX_PATH/en_reasoning_peS2o_split_05_text_document_dc\=1721575_sc\=1721575_tc\=12257848726
    0.008344481630035702 $BIN_IDX_PATH/en_reasoning_peS2o_split_06_text_document_dc\=1722370_sc\=1722370_tc\=12256823816
    0.00834343638511967 $BIN_IDX_PATH/en_reasoning_peS2o_split_07_text_document_dc\=1719665_sc\=1719665_tc\=12255288504
    0.008344536893533164 $BIN_IDX_PATH/en_reasoning_peS2o_split_08_text_document_dc\=1721188_sc\=1721188_tc\=12256904990
    0.008344512808086922 $BIN_IDX_PATH/en_reasoning_peS2o_split_09_text_document_dc\=1719879_sc\=1719879_tc\=12256869612
    0.004122327320280175 $BIN_IDX_PATH/en_reasoning_peS2o_split_10_text_document_dc\=850041_sc\=850041_tc\=6055096280
    0.01059292541162788 $BIN_IDX_PATH/en_scientific_arxiv_split_00_text_document_dc\=805220_sc\=805220_tc\=15559459080
    0.009978207813116173 $BIN_IDX_PATH/en_scientific_arxiv_split_01_text_document_dc\=753086_sc\=753086_tc\=14656528780
    0.0027913617153658846 $BIN_IDX_PATH/en_web_c4_split_01_text_document_dc\=22603224_sc\=22603224_tc\=12300307054
    0.0027913110885902813 $BIN_IDX_PATH/en_web_c4_split_03_text_document_dc\=22602978_sc\=22602978_tc\=12300083964
    0.002790896665008083 $BIN_IDX_PATH/en_web_c4_split_05_text_document_dc\=22614260_sc\=22614260_tc\=12298257781
    0.0027909940152891287 $BIN_IDX_PATH/en_web_c4_split_07_text_document_dc\=22608892_sc\=22608892_tc\=12298686761
    0.0027911158041262047 $BIN_IDX_PATH/en_web_c4_split_09_text_document_dc\=22602542_sc\=22602542_tc\=12299223431
    0.002791032826502858 $BIN_IDX_PATH/en_web_c4_split_11_text_document_dc\=22613813_sc\=22613813_tc\=12298857785
    0.0027907420977790207 $BIN_IDX_PATH/en_web_c4_split_13_text_document_dc\=22612514_sc\=22612514_tc\=12297576671
    0.002790991887553022 $BIN_IDX_PATH/en_web_c4_split_15_text_document_dc\=22616256_sc\=22616256_tc\=12298677385
    0.002795540958755593 $BIN_IDX_PATH/en_web_cc_split_01_text_document_dc\=5560954_sc\=5560954_tc\=12318723147
    0.0028018603422546263 $BIN_IDX_PATH/en_web_cc_split_03_text_document_dc\=5598998_sc\=5598998_tc\=12346569899
    0.002797259229567952 $BIN_IDX_PATH/en_web_cc_split_05_text_document_dc\=5549841_sc\=5549841_tc\=12326294813
    0.0028799322253720483 $BIN_IDX_PATH/en_web_cc_split_07_text_document_dc\=5678451_sc\=5678451_tc\=12690598453
    0.002881662491932125 $BIN_IDX_PATH/en_web_cc_split_09_text_document_dc\=5663768_sc\=5663768_tc\=12698222979
    0.002880856512711388 $BIN_IDX_PATH/en_web_cc_split_11_text_document_dc\=5662932_sc\=5662932_tc\=12694671382
    0.0028822521153840735 $BIN_IDX_PATH/en_web_cc_split_13_text_document_dc\=5676090_sc\=5676090_tc\=12700821191
    0.0028040447724527683 $BIN_IDX_PATH/en_web_cc_split_15_text_document_dc\=5162978_sc\=5162978_tc\=12356195725
    0.0028128077172210183 $BIN_IDX_PATH/en_web_cc_split_17_text_document_dc\=5169213_sc\=5169213_tc\=12394810180
    0.0028042716132675614 $BIN_IDX_PATH/en_web_cc_split_19_text_document_dc\=5161905_sc\=5161905_tc\=12357195313
    0.002812849946746285 $BIN_IDX_PATH/en_web_cc_split_21_text_document_dc\=5164876_sc\=5164876_tc\=12394996267
    0.0028940177396825028 $BIN_IDX_PATH/en_web_cc_split_23_text_document_dc\=5246670_sc\=5246670_tc\=12752667138
    0.002889963861614623 $BIN_IDX_PATH/en_web_cc_split_25_text_document_dc\=5266480_sc\=5266480_tc\=12734803475
    0.0028938142531315983 $BIN_IDX_PATH/en_web_cc_split_27_text_document_dc\=5243484_sc\=5243484_tc\=12751770462
    0.002888433860499831 $BIN_IDX_PATH/en_web_cc_split_29_text_document_dc\=5265289_sc\=5265289_tc\=12728061431
    0.002790783058060682 $BIN_IDX_PATH/en_web_cc_split_33_text_document_dc\=5779417_sc\=5779417_tc\=12297757165
    0.002794244464424067 $BIN_IDX_PATH/en_web_cc_split_35_text_document_dc\=5768229_sc\=5768229_tc\=12313010065
    0.002794165593177091 $BIN_IDX_PATH/en_web_cc_split_37_text_document_dc\=5798426_sc\=5798426_tc\=12312662514
    0.0028260739026729144 $BIN_IDX_PATH/en_web_cc_split_39_text_document_dc\=5886477_sc\=5886477_tc\=12453268442
    0.0028713772563692575 $BIN_IDX_PATH/en_web_cc_split_41_text_document_dc\=5962779_sc\=5962779_tc\=12652900456
    0.002872640196420386 $BIN_IDX_PATH/en_web_cc_split_43_text_document_dc\=5968099_sc\=5968099_tc\=12658465679
    0.0028700890140501084 $BIN_IDX_PATH/en_web_cc_split_45_text_document_dc\=5947770_sc\=5947770_tc\=12647223737
    0.0028751125980904773 $BIN_IDX_PATH/en_web_cc_split_47_text_document_dc\=5930752_sc\=5930752_tc\=12669360469
    0.0027744262156385036 $BIN_IDX_PATH/en_web_cc_split_49_text_document_dc\=5951616_sc\=5951616_tc\=12225679733
    0.002769627670100624 $BIN_IDX_PATH/en_web_cc_split_51_text_document_dc\=5938206_sc\=5938206_tc\=12204534647
    0.0027715268261680453 $BIN_IDX_PATH/en_web_cc_split_53_text_document_dc\=5944074_sc\=5944074_tc\=12212903395
    0.0027694771393521646 $BIN_IDX_PATH/en_web_cc_split_55_text_document_dc\=5926865_sc\=5926865_tc\=12203871324
    0.0028613932491000286 $BIN_IDX_PATH/en_web_cc_split_57_text_document_dc\=6237580_sc\=6237580_tc\=12608905314
    0.0028589875565863438 $BIN_IDX_PATH/en_web_cc_split_59_text_document_dc\=6274946_sc\=6274946_tc\=12598304482
    0.0028614085769242455 $BIN_IDX_PATH/en_web_cc_split_61_text_document_dc\=6237973_sc\=6237973_tc\=12608972857
    0.0028590373729795206 $BIN_IDX_PATH/en_web_cc_split_63_text_document_dc\=6277037_sc\=6277037_tc\=12598524001
    0.0027685969705393025 $BIN_IDX_PATH/en_web_cc_split_65_text_document_dc\=6045112_sc\=6045112_tc\=12199992806
    0.00276508333456327 $BIN_IDX_PATH/en_web_cc_split_67_text_document_dc\=6044979_sc\=6044979_tc\=12184509753
    0.002765894442953321 $BIN_IDX_PATH/en_web_cc_split_69_text_document_dc\=6106624_sc\=6106624_tc\=12188083952
    0.002768759264427811 $BIN_IDX_PATH/en_web_cc_split_71_text_document_dc\=6048110_sc\=6048110_tc\=12200707964
    0.0028502793034410894 $BIN_IDX_PATH/en_web_cc_split_73_text_document_dc\=6559407_sc\=6559407_tc\=12559931029
    0.0028521756595929725 $BIN_IDX_PATH/en_web_cc_split_75_text_document_dc\=6593170_sc\=6593170_tc\=12568287439
    0.00285175055466659 $BIN_IDX_PATH/en_web_cc_split_77_text_document_dc\=6553152_sc\=6553152_tc\=12566414188
    0.0028541271802989565 $BIN_IDX_PATH/en_web_cc_split_79_text_document_dc\=6596841_sc\=6596841_tc\=12576886935
    0.00190855527699637 $BIN_IDX_PATH/en_web_cc_split_81_text_document_dc\=4396657_sc\=4396657_tc\=8410166195
)

# torchrun ${\DISTRIBUTED_ARGS[@]}\ pretrain_gpt.py\ \
python pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${LOGISTICS_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_PATH[@]}
  
  
  
  
  
