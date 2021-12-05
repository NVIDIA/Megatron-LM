
# process WoW train
python tasks/knwl_dialo/preprocessing.py --func process_wow_dataset --input_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.json --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.txt

# process WoW test
python tasks/knwl_dialo/preprocessing.py --func process_wow_dataset --input_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_random_split.json --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_seen.txt
python tasks/knwl_dialo/preprocessing.py --func process_wow_dataset --input_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_topic_split.json --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_unseen.txt

# process WoI test
python tasks/knwl_dialo/preprocessing.py --func process_woi_dataset --input_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_internet/data/test.jsonl --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_internet/data/test.txt


# get knowledge generation prompts
# WoW seen
python tasks/knwl_dialo/preprocessing.py --func get_knwl_gen_prompts --test_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_seen.txt --train_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.txt --model_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/dpr_wow/best_question_encoder.pt --data_type wow_seen --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/knowledge_prompts_test_seen.json
# WoW unseen
python tasks/knwl_dialo/preprocessing.py --func get_knwl_gen_prompts --test_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_unseen.txt --train_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.txt --model_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/dpr_wow_ctrl/best_question_encoder.pt --data_type wow_unseen --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/knowledge_prompts_test_unseen.json
# WoI
python tasks/knwl_dialo/preprocessing.py --func get_knwl_gen_prompts --test_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_internet/data/test.txt --train_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.txt --model_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/checkpoints/dpr_wow_ctrl/best_question_encoder.pt --data_type woi --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_internet/data/knowledge_prompts_test.json


# get response generation prompts --seed 147
python tasks/knwl_dialo/preprocessing.py --func get_resp_gen_prompts --train_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/train.txt --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/response_generation_prompts_temp.txt --seed 1234



# prepare response generation inputs
# WoW seen
python tasks/knwl_dialo/preprocessing.py --func prepare_input --test_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_seen.txt --knowledge_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/output_testseen_knowledge_357m.txt --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_seen_resp_gen_input.txt
# WoW unseen
python tasks/knwl_dialo/preprocessing.py --func prepare_input --test_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_unseen.txt --knowledge_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/output_testunseen_knowledge_357m.txt --output_file /gpfs/fs1/projects/gpu_adlr/datasets/zihanl/dialog_datasets/wizard_of_wikipedia/data/test_unseen_resp_gen_input.txt
