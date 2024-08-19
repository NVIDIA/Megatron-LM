
#MCORE_MISTRAL=<path_to_mcore_mistral_model_folder>
#MCORE_CLIP=<path_to_mcore_clip_model_folder>
#OUTPUT_DIR=<path_to_output_folder_for_combined_checkpoint>

MCORE_MISTRAL=$1
MCORE_CLIP=$2
OUTPUT_DIR=$3

python examples/multimodal/combine_state_dicts.py \
    --input \
    ${MCORE_MISTRAL}/iter_0000001/mp_rank_00/model_optim_rng.pt \
    ${MCORE_CLIP}/state_dict_tp_0.pt \
    ${MCORE_MISTRAL}/iter_0000001/mp_rank_01/model_optim_rng.pt \
    ${MCORE_CLIP}/state_dict_tp_1.pt \
    ${MCORE_MISTRAL}/iter_0000001/mp_rank_02/model_optim_rng.pt \
    ${MCORE_CLIP}/state_dict_tp_2.pt \
    ${MCORE_MISTRAL}/iter_0000001/mp_rank_03/model_optim_rng.pt \
    ${MCORE_CLIP}/state_dict_tp_3.pt \
    --prefixes language_model vision_model language_model vision_model language_model vision_model language_model vision_model \
    --output \
    ${OUTPUT_DIR}/mistral_instruct_clip336_tp4_combined_mcore/iter_0000001/mp_rank_00/model_optim_rng.pt \
    ${OUTPUT_DIR}/mistral_instruct_clip336_tp4_combined_mcore/iter_0000001/mp_rank_01/model_optim_rng.pt \
    ${OUTPUT_DIR}/mistral_instruct_clip336_tp4_combined_mcore/iter_0000001/mp_rank_02/model_optim_rng.pt \
    ${OUTPUT_DIR}/mistral_instruct_clip336_tp4_combined_mcore/iter_0000001/mp_rank_03/model_optim_rng.pt
