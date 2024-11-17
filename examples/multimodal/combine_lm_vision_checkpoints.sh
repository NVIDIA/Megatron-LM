#/bin/bash
MCORE_LM=$1    # <path_to_mcore_lm_model_folder>
MCORE_VISION=$2   # <path_to_mcore_vision_model_folder>
OUTPUT_DIR=$3   # <path_to_output_folder_for_combined_checkpoint>
MODEL_TYPE=$4   # Model type. Default: Mistral CLIP example.

if [[ $MODEL_TYPE == "nvlm" ]]; then
    # NVLM TP=8
    python examples/multimodal/combine_state_dicts.py \
        --input \
        ${MCORE_LM}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_03/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_03/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_04/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_04/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_05/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_05/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_06/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_06/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_07/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_07/model_optim_rng.pt \
        --prefixes language_model vision_model language_model vision_model language_model vision_model language_model vision_model language_model vision_model language_model vision_model language_model vision_model language_model vision_model \
        --output \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_03/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_04/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_05/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_06/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_07/model_optim_rng.pt
else
    # Mistral CLIP example TP=4.
    python examples/multimodal/combine_state_dicts.py \
        --input \
        ${MCORE_LM}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${MCORE_LM}/iter_0000001/mp_rank_03/model_optim_rng.pt \
        ${MCORE_VISION}/iter_0000001/mp_rank_03/model_optim_rng.pt \
        --prefixes language_model vision_model language_model vision_model language_model vision_model language_model vision_model \
        --output \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_00/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_01/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_02/model_optim_rng.pt \
        ${OUTPUT_DIR}/iter_0000001/mp_rank_03/model_optim_rng.pt
fi

echo 1 > ${OUTPUT_DIR}/latest_checkpointed_iteration.txt
