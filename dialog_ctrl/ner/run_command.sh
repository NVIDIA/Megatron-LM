
# train_ner.py command
CUDA_VISIBLE_DEVICES=0 python train_ner.py --exp_name conll2003 --exp_id 1 --model_name roberta-large --lr 3e-5 --seed 111

# gen_entityctrl_data.py command (by default is to process training data)
CUDA_VISIBLE_DEVICES=0 python gen_entityctrl_data.py

CUDA_VISIBLE_DEVICES=0 python gen_entityctrl_data.py --infer_dataname valid_random_split.txt --output_dataname valid_random_split_entity_based_control.txt

CUDA_VISIBLE_DEVICES=0 python gen_entityctrl_data.py --infer_dataname valid_topic_split.txt --output_dataname valid_topic_split_entity_based_control.txt

CUDA_VISIBLE_DEVICES=0 python gen_entityctrl_data.py --infer_dataname test_random_split_seen.txt --output_dataname test_random_split_entity_based_control.txt

CUDA_VISIBLE_DEVICES=0 python gen_entityctrl_data.py --infer_dataname test_topic_split_unseen.txt --output_dataname test_topic_split_entity_based_control.txt

