# minimal tests

## 800M
bash tools/retro/text_generation/retro_generate.sh nq 843m greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_843m_128_5e-6 2 1

bash tools/retro/text_generation/retro_generate.sh doc2dial 843m greedy test  0 20000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_843m_128_5e-6 1 0


## 43B
bash tools/retro/text_generation/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6 2 1

bash tools/retro/text_generation/retro_generate.sh doc2dial 43b greedy test  0 2000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6 1 0
bash tools/retro/text_generation/retro_generate.sh doc2dial 43b greedy test  2000 20000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6 1 0


# full tests

## 800M
bash tools/retro/text_generation/retro_generate.sh nq 843m greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 2 1

bash tools/retro/text_generation/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 843m greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_843m_128_5e-6 2 1
bash tools/retro/text_generation/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 843m greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 2 1

bash tools/retro/text_generation/retro_generate.sh doc2dial 843m greedy test  0 20000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_843m_128_5e-6 1 0

## 43B
bash tools/retro/text_generation/retro_generate.sh nq 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6 2 1

bash tools/retro/text_generation/retro_generate.sh doc2dial 43b greedy test  0 2000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6 1 0
bash tools/retro/text_generation/retro_generate.sh doc2dial 43b greedy test  2000 20000 1000 1 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6 1 0

bash tools/retro/text_generation/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-open_inst_pp1_same_format_ctx1_43b_128_5e-6 2 1
bash tools/retro/text_generation/retro_generate.sh ford_tasb_ftmsmarcominilm_chunkbysents150_benzlandroverford_retrieved 43b greedy test  0 20000 1000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/retro-qc_pp1_same_format_ctx1_43b_128_5e-6 2 1


## see whether the numbers match or not

# short format for foundation models

#bash tools/retro/text_generation/tests/retro_generate_short_format.sh nq 843m greedy test  0 20000 195312 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/gpt3-800m-pretraining-retro-fitting 2 1
#bash tools/retro/text_generation/tests/retro_generate_short_format.sh nq 43b greedy  test  0 20000 32000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2 1 # unable to finish

#bash tools/retro/text_generation/tests/retro_generate_short_format.sh tqa 843m greedy test  0 20000 195312 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/gpt3-800m-pretraining-retro-fitting 2 1  # unable to finish
#bash tools/retro/text_generation/tests/retro_generate_short_format.sh tqa 43b greedy  test  0 20000 32000 5 pp1 /lustre/fsw/adlr/adlr-nlp/boxinw/github-version/retro/Megatron-LM/checkpoints/applications/gpt3-43b-pretraining-retro-fitting-noseqpar-pp1-distributed 2 1  # unable to finish

#python tools/retro/text_generation/tests/truncate_qa_output.py