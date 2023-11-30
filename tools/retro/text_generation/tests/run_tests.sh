CKPT_43B=/lustre/fsw/adlr/adlr-nlp/boxinw/no-hack-open-instructretro-megatron/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6
CKPT_800M=/lustre/fsw/adlr/adlr-nlp/boxinw/no-hack-open-instructretro-megatron/checkpoints/applications/retro-qc_pp1_same_format_ctx1_843m_128_5e-6

# minimal tests

## 800M
bash tools/retro/text_generation/tests/retro_generate.sh nq 843m greedy test  0 20000 1000 5 pp1 $CKPT_800M 2 1
bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 843m greedy test  0 20000 1000 1 pp1 $CKPT_800M 1 0


## 43B
bash tools/retro/text_generation/tests/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 $CKPT_43B 2 1

bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 43b greedy test  0 2000 1000 1 pp1 $CKPT_43B 1 0
bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 43b greedy test  2000 20000 1000 1 pp1 $CKPT_43B 1 0


# full tests

### 800M
bash tools/retro/text_generation/tests/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 843m greedy test  0 20000 1000 5 pp1 $CKPT_800M 2 1

CKPT_800M=/lustre/fsw/adlr/adlr-nlp/boxinw/no-hack-open-instructretro-megatron/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6
#### open inst acc
bash tools/retro/text_generation/tests/retro_generate.sh nq 843m greedy test  0 20000 1000 5 pp1 $CKPT_800M 2 1
bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 843m greedy test  0 20000 1000 1 pp1 $CKPT_800M 1 0
bash tools/retro/text_generation/tests/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 843m greedy test  0 20000 1000 5 pp1 $CKPT_800M 2 1

## 43B
bash tools/retro/text_generation/tests/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 20000 1000 5 pp1 $CKPT_43B 2 1

#### open inst acc
CKPT_43B=/lustre/fsw/adlr/adlr-nlp/boxinw/no-hack-open-instructretro-megatron/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6
bash tools/retro/text_generation/tests/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 $CKPT_43B 2 1
bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 43b greedy test  0 2000 1000 1 pp1 $CKPT_43B 1 0
bash tools/retro/text_generation/tests/retro_generate.sh doc2dial 43b greedy test  2000 20000 1000 1 pp1 $CKPT_43B 1 0
bash tools/retro/text_generation/tests/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 20000 1000 5 pp1 $CKPT_43B 2 1
#


## see whether the numbers match or not

# short format for foundation models
CKPT_800M=/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting-github-mr-no-hacks
bash tools/retro/text_generation/tests/retro_generate_short_format.sh nq 843m greedy test  0 200 195312 5 pp1 $CKPT_800M 2 1
bash tools/retro/text_generation/tests/retro_generate_short_format.sh tqa 843m greedy test  0 200 195312 5 pp1 $CKPT_800M 2 1

CKPT_43B=/lustre/fsw/adlr/adlr-nlp/boxinw/no-hack-open-instructretro-megatron/checkpoints/applications/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed
bash tools/retro/text_generation/tests/retro_generate_short_format.sh nq 43b greedy test  0 200 32000 5 pp1 $CKPT_43B 2 1
bash tools/retro/text_generation/tests/retro_generate_short_format.sh tqa 43b greedy test  0 200 32000 5 pp1 $CKPT_43B 2 1

CKPT_800M=/lustre/fsw/adlr/adlr-nlp/boxinw/checkpoints/retro-nvllm/gpt3-800m-pretraining-retro-fitting
bash tools/retro/text_generation/tests/retro_generate_short_format.sh nq 843m greedy test  0 200 195312 5 pp1 $CKPT_800M 2 1
bash tools/retro/text_generation/tests/retro_generate_short_format.sh tqa 843m greedy test  0 200 195312 5 pp1 $CKPT_800M 2 1

#python tools/retro/text_generation/tests/truncate_qa_output.py